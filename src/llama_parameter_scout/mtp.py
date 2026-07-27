from __future__ import annotations

import json
import math
import os
import re
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from .models import (
    CandidateConfig,
    IncompatibleDraftError,
    MemoryEvidence,
    ModelFingerprint,
    MtpBenefit,
    MtpComparison,
    MtpDraftSettings,
    MtpEvidence,
    MtpMeasurement,
    MtpMeasurementError,
    MtpProfileObservation,
    MtpProfileReport,
    MtpSearchProfile,
    NctxComposition,
    ResourceBudget,
    ServerCapabilityError,
    ValidatedCandidate,
    ValidatedRecommendation,
)
from .runner import ProcessRunner

_SYNTHETIC_PROMPT = "The"
_N_PREDICT = 128
_WARMUP_REPETITIONS = 1
_MIN_REPETITIONS = 3
_MAX_PORT_RETRIES = 5
_SERVER_READY_TIMEOUT = 60.0
_COMPLETION_TIMEOUT = 30.0

_MTP_CAVEAT = (
    "Measured synthetic generation throughput only. "
    "Does not cover prompt processing, output quality, "
    "representative workloads, acceptance under real prompts, "
    "or universal end-to-end speedup."
)


class ServerStartError(ServerCapabilityError):
    """The llama-server process could not be started."""


class ServerReadyError(ServerCapabilityError):
    """The llama-server process started but did not become ready."""


class ServerLifecycle(Protocol):
    def start(self, argv: Sequence[str], *, port: int) -> None: ...

    def wait_for_ready(self, timeout: float = _SERVER_READY_TIMEOUT) -> bool: ...

    def stop(self) -> None: ...

    @property
    def diagnostics(self) -> str: ...


class ProductionServerLifecycle:
    def __init__(self) -> None:
        self._process: subprocess.Popen[Any] | None = None
        self._port: int | None = None
        self._captured_stderr: str = ""

    def start(self, argv: Sequence[str], *, port: int) -> None:
        self._port = port
        self._captured_stderr = ""
        base_env = os.environ.copy()
        start_new_session = os.name != "nt"
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        try:
            self._process = subprocess.Popen(
                [str(a) for a in argv],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=base_env,
                start_new_session=start_new_session,
                creationflags=creationflags,
            )
        except OSError as e:
            raise ServerStartError(str(e)) from e

    def wait_for_ready(self, timeout: float = _SERVER_READY_TIMEOUT) -> bool:
        if self._port is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                return False
            try:
                req = Request(f"http://127.0.0.1:{self._port}/health", method="GET")
                with urlopen(req, timeout=2):
                    return True
            except (URLError, OSError):
                time.sleep(0.5)
        return False

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.returncode is not None:
            self._capture_stderr()
            self._process = None
            return
        _terminate_process_tree(self._process)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._capture_stderr()
        self._process = None

    def _capture_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            data = self._process.stderr.read()
            if data:
                self._captured_stderr = data
        except OSError:
            pass

    @property
    def diagnostics(self) -> str:
        if self._process is None:
            return self._captured_stderr[:500] if self._captured_stderr else ""
        if self._process.returncode is None:
            return f"Server process (PID {self._process.pid}) still running; not ready"
        stderr_text = self._captured_stderr
        if not stderr_text:
            try:
                if self._process.stderr:
                    stderr_text = self._process.stderr.read()
            except OSError:
                pass
        return (
            f"Process exited with code {self._process.returncode}"
            + (f": {stderr_text[:500]}" if stderr_text else "")
        )


@dataclass
class FakeServerLifecycle:
    _port: int = 0
    _start_fail: bool = False
    _start_fail_with_server_error: bool = False
    _ready: bool = True
    _ready_timeout: bool = False
    start_calls: list[tuple[str, ...]] = field(default_factory=list)
    stop_called: bool = False
    _diagnostics: str = ""

    def start(self, argv: Sequence[str], *, port: int) -> None:
        self.start_calls.append(tuple(argv))
        self._port = port
        if self._start_fail_with_server_error:
            raise ServerStartError("Simulated ServerStartError")
        if self._start_fail:
            raise OSError("Simulated start failure")

    def wait_for_ready(self, timeout: float = _SERVER_READY_TIMEOUT) -> bool:
        if self._ready_timeout:
            return False
        return self._ready

    def stop(self) -> None:
        self.stop_called = True

    @property
    def port(self) -> int:
        return self._port

    @property
    def diagnostics(self) -> str:
        return self._diagnostics


def _terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.returncode is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        except (OSError, ProcessLookupError):
            pass


# ── Input validation ─────────────────────────────────────────────


def validate_mtp_inputs(
    candidate: CandidateConfig,
    n_ctx: NctxComposition,
    cpu_only: bool,
    device_id: int | None,
) -> None:
    if n_ctx.total <= 0:
        raise ValueError("n_ctx.total must be > 0")
    if n_ctx.prompt < 0 or n_ctx.generation < 0 or n_ctx.depth < 0:
        raise ValueError("n_ctx components must be >= 0")
    if n_ctx.prompt + n_ctx.generation + n_ctx.depth != n_ctx.total:
        raise ValueError("n_ctx components do not sum to total")
    if candidate.threads <= 0:
        raise ValueError("threads must be > 0")
    if candidate.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if candidate.ubatch_size <= 0:
        raise ValueError("ubatch_size must be > 0")
    if candidate.n_gpu_layers < -1:
        raise ValueError("n_gpu_layers must be >= -1")
    if cpu_only and device_id is not None:
        raise ValueError("device_id must be None when cpu_only=True")
    if not cpu_only and device_id is None:
        raise ValueError("device_id is required when cpu_only=False")


def _validate_server_exists(server_path: Path) -> None:
    if not server_path.exists():
        raise FileNotFoundError(f"Server not found: {server_path}")


# ── Server capability inspection ─────────────────────────────────


_VERSION_RE = re.compile(r"^version:\s*(\S+)", re.MULTILINE)


def _parse_help_flags(help_output: str) -> set[str]:
    return set(re.findall(r"(?:^|\s)(--?[\w-]+)", help_output, re.MULTILINE))


def _parse_supported_flags(version_output: str) -> set[str]:
    m = re.search(r"^supported_flags:[ \t]*(.*)$", version_output, re.MULTILINE)
    if m:
        raw = m.group(1).strip()
        if raw:
            return {f.strip() for f in raw.split(",")}
    return set()


def _get_required_flags(
    cpu_only: bool, device_id: int | None, has_n_cpu_moe: bool,
    draft_settings: MtpDraftSettings | None = None,
    draft_is_moe: bool = False,
) -> set[str]:
    flags: set[str] = set()
    flags.add("-m")
    flags.add("--host")
    flags.add("--port")
    flags.add("-c")
    flags.add("-t")
    flags.add("-ngl")
    flags.add("-fa")
    flags.add("-ctk")
    flags.add("-ctv")
    flags.add("-b")
    flags.add("--ubatch-size")
    if not cpu_only and device_id is not None:
        flags.add("--split-mode")
        flags.add("--main-gpu")
    if has_n_cpu_moe:
        flags.add("--n-cpu-moe")
    if draft_settings is not None:
        flags.add("-md")
        flags.add("--gpu-layers-draft")
        flags.add("--draft-max")
        if draft_is_moe:
            flags.add("--n-cpu-moe-draft")
    return flags


def inspect_server_capabilities(
    runner: ProcessRunner,
    server_path: Path,
    *,
    cpu_only: bool = True,
    device_id: int | None = None,
    has_n_cpu_moe: bool = False,
    draft_settings: MtpDraftSettings | None = None,
    draft_is_moe: bool = False,
) -> tuple[str, frozenset[str]]:
    version_output = _run_server_command(runner, server_path, "--version")
    help_output = _run_server_command(runner, server_path, "--help")
    return _validate_server_capabilities(
        version_output, help_output, server_path,
        cpu_only=cpu_only, device_id=device_id, has_n_cpu_moe=has_n_cpu_moe,
        draft_settings=draft_settings, draft_is_moe=draft_is_moe,
    )


def _run_server_command(runner: ProcessRunner, server_path: Path, flag: str) -> str:
    result = runner.run([str(server_path), flag])
    if result.returncode != 0:
        raise ServerCapabilityError(
            f"llama-server {flag} exited with code {result.returncode}:\n{result.stderr}",
        )
    if not result.stdout.strip():
        raise ServerCapabilityError(f"llama-server {flag} produced no output.")
    return result.stdout


def _validate_server_capabilities(
    version_output: str,
    help_output: str,
    server_path: Path,
    *,
    cpu_only: bool = True,
    device_id: int | None = None,
    has_n_cpu_moe: bool = False,
    draft_settings: MtpDraftSettings | None = None,
    draft_is_moe: bool = False,
) -> tuple[str, frozenset[str]]:
    m = _VERSION_RE.search(version_output)
    if m is None:
        raise ServerCapabilityError(
            "Could not find 'version:' line in llama-server --version output.",
        )
    version = m.group(1)

    help_flags = _parse_help_flags(help_output)
    version_flags = _parse_supported_flags(version_output)
    available = help_flags | version_flags

    required = _get_required_flags(
        cpu_only, device_id, has_n_cpu_moe,
        draft_settings=draft_settings, draft_is_moe=draft_is_moe,
    )
    missing = required - available
    if missing:
        raise ServerCapabilityError(
            f"llama-server at {server_path} is missing required flags: "
            f"{', '.join(sorted(missing))}",
        )

    return version, frozenset(version_flags)


# ── Draft compatibility ──────────────────────────────────────────


def validate_draft_compatibility(
    main: ModelFingerprint,
    draft: ModelFingerprint,
) -> None:
    reasons: list[str] = []

    if main.architecture != draft.architecture:
        reasons.append(
            f"Architecture mismatch: main={main.architecture}, "
            f"draft={draft.architecture}",
        )

    if main.context_length is not None and draft.context_length is not None:
        if draft.context_length < main.context_length:
            reasons.append(
                f"Draft context length ({draft.context_length}) is less than "
                f"main context length ({main.context_length})",
            )

    if main.parameter_count is not None and draft.parameter_count is not None:
        if draft.parameter_count >= main.parameter_count:
            reasons.append(
                f"Draft parameter count ({draft.parameter_count}) is not less than "
                f"main parameter count ({main.parameter_count})",
            )

    if reasons:
        raise IncompatibleDraftError(
            "Draft model is not compatible with main model:\n" + "\n".join(reasons),
        )


# ── Port selection ───────────────────────────────────────────────


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


# ── Server argv builder ──────────────────────────────────────────


def _build_server_argv(
    server_path: Path,
    main_model_path: Path,
    draft_model_path: Path | None,
    port: int,
    n_ctx: NctxComposition,
    candidate: CandidateConfig,
    cpu_only: bool,
    device_id: int | None,
    draft_settings: MtpDraftSettings | None = None,
) -> list[str]:
    argv = [
        str(server_path),
        "-m", str(main_model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(n_ctx.total),
        "-t", str(candidate.threads),
        "-ngl", str(candidate.n_gpu_layers),
        "-fa", "on" if candidate.flash_attn else "off",
        "-ctk", candidate.cache_type_k,
        "-ctv", candidate.cache_type_v,
        "-b", str(candidate.batch_size),
        "--ubatch-size", str(candidate.ubatch_size),
    ]

    if draft_model_path is not None:
        argv.extend(["-md", str(draft_model_path)])

    if draft_settings is not None:
        argv.extend(["--gpu-layers-draft", str(draft_settings.n_gpu_layers_draft)])
        if draft_settings.n_cpu_moe_draft is not None:
            argv.extend(["--n-cpu-moe-draft", str(draft_settings.n_cpu_moe_draft)])
        argv.extend(["--draft-max", str(draft_settings.draft_max)])

    if not cpu_only and device_id is not None:
        argv.extend(["--split-mode", "none", "--main-gpu", str(device_id)])

    if candidate.n_cpu_moe is not None:
        argv.extend(["--n-cpu-moe", str(candidate.n_cpu_moe)])

    return argv


# ── HTTP completion ──────────────────────────────────────────────


def _send_completion(port: int, prompt: str, n_predict: int) -> dict[str, Any]:
    try:
        body = json.dumps({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "stream": False,
            "cache_prompt": False,
        }).encode()
        req = Request(
            f"http://127.0.0.1:{port}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=_COMPLETION_TIMEOUT) as resp:
            raw = resp.read()
            data: Any = json.loads(raw)
    except (json.JSONDecodeError, OSError, URLError) as e:
        raise MtpMeasurementError(f"Completion request failed: {e}") from e
    if not isinstance(data, dict):
        raise MtpMeasurementError("Completion response is not a JSON object")
    return data


def _measure_one_completion(port: int) -> float:
    start = time.monotonic()
    response = _send_completion(port, _SYNTHETIC_PROMPT, _N_PREDICT)
    elapsed = time.monotonic() - start
    tokens = response.get("tokens_predicted")
    if not isinstance(tokens, int) or tokens <= 0:
        raise MtpMeasurementError(
            f"Invalid tokens_predicted in completion response: {tokens!r}",
        )
    tps = tokens / elapsed
    if not math.isfinite(tps) or tps <= 0:
        raise MtpMeasurementError(f"Invalid throughput: {tps}")
    return tps


def _warmup_server(port: int, n: int = _WARMUP_REPETITIONS) -> None:
    for _ in range(n):
        _send_completion(port, _SYNTHETIC_PROMPT, _N_PREDICT)


def _compute_stats(samples: list[float]) -> tuple[float, float]:
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    if n % 2 == 1:
        median = sorted_samples[n // 2]
    else:
        median = (sorted_samples[n // 2 - 1] + sorted_samples[n // 2]) / 2.0
    mean = sum(sorted_samples) / n
    variance = sum((x - mean) ** 2 for x in sorted_samples) / n
    dispersion = variance ** 0.5
    return median, dispersion


def _start_server_with_retry(
    lifecycle: ServerLifecycle,
    server_path: Path,
    main_model_path: Path,
    draft_model_path: Path | None,
    n_ctx: NctxComposition,
    candidate: CandidateConfig,
    cpu_only: bool,
    device_id: int | None,
    draft_settings: MtpDraftSettings | None = None,
) -> tuple[int, tuple[str, ...]]:
    port = _find_free_port()
    for attempt in range(_MAX_PORT_RETRIES):
        argv = _build_server_argv(
            server_path, main_model_path, draft_model_path,
            port, n_ctx, candidate, cpu_only, device_id,
            draft_settings=draft_settings,
        )
        try:
            lifecycle.start(argv, port=port)
        except (OSError, ServerStartError):
            if attempt == _MAX_PORT_RETRIES - 1:
                raise
            port = _find_free_port()
            continue
        if lifecycle.wait_for_ready(_SERVER_READY_TIMEOUT):
            return port, tuple(argv)
        diag = lifecycle.diagnostics
        lifecycle.stop()
        if attempt == _MAX_PORT_RETRIES - 1:
            raise ServerReadyError(
                f"Server did not become ready after {_MAX_PORT_RETRIES} attempts."
                + (f" Diagnostics: {diag}" if diag else ""),
            )
        port = _find_free_port()
    raise ServerReadyError(
        f"Server did not become ready after {_MAX_PORT_RETRIES} attempts",
    )


def _measure_mode(
    port: int,
    mode: str,
    *,
    _warmup_fn: Callable[[int], None] | None = None,
    _measure_fn: Callable[[int], float] | None = None,
) -> MtpMeasurement:
    warmup = _warmup_fn or _warmup_server
    measure = _measure_fn or _measure_one_completion

    warmup(port)

    samples: list[float] = []
    for _ in range(_MIN_REPETITIONS):
        tps = measure(port)
        samples.append(tps)

    median, dispersion = _compute_stats(samples)
    return MtpMeasurement(
        mode=mode,
        samples_tps=tuple(samples),
        median_tps=median,
        dispersion_tps=dispersion,
    )


# ── Rich single-execution evidence collector ────────────────────


def collect_mtp_evidence(
    *,
    candidate: CandidateConfig,
    main_model_path: Path,
    draft_model_path: Path,
    server_path: Path,
    n_ctx: NctxComposition,
    cpu_only: bool = True,
    device_id: int | None = None,
    runner: ProcessRunner | None = None,
    server_lifecycle: ServerLifecycle | None = None,
    main_fingerprint: ModelFingerprint | None = None,
    draft_fingerprint: ModelFingerprint | None = None,
    draft_settings: MtpDraftSettings | None = None,
    _warmup_fn: Callable[[int], None] | None = None,
    _measure_fn: Callable[[int], float] | None = None,
) -> MtpEvidence:
    validate_mtp_inputs(candidate, n_ctx, cpu_only, device_id)

    if main_fingerprint is None:
        from .model_inspector import inspect_model
        main_fingerprint = inspect_model(main_model_path)
    if draft_fingerprint is None:
        from .model_inspector import inspect_model
        draft_fingerprint = inspect_model(draft_model_path)

    validate_draft_compatibility(main_fingerprint, draft_fingerprint)

    if runner is None:
        from .runner import ProductionRunner
        runner = ProductionRunner()
        _validate_server_exists(server_path)

    server_version, server_supported_flags = inspect_server_capabilities(
        runner, server_path,
        cpu_only=cpu_only,
        device_id=device_id,
        has_n_cpu_moe=candidate.n_cpu_moe is not None,
        draft_settings=draft_settings,
        draft_is_moe=draft_fingerprint.is_moe,
    )

    lifecycle = server_lifecycle or ProductionServerLifecycle()

    try:
        port_off, off_argv = _start_server_with_retry(
            lifecycle, server_path, main_model_path, None,
            n_ctx, candidate, cpu_only, device_id,
        )
        try:
            off_measurement = _measure_mode(
                port_off, "off",
                _warmup_fn=_warmup_fn, _measure_fn=_measure_fn,
            )
        finally:
            lifecycle.stop()

        port_on, on_argv = _start_server_with_retry(
            lifecycle, server_path, main_model_path, draft_model_path,
            n_ctx, candidate, cpu_only, device_id,
            draft_settings=draft_settings,
        )
        try:
            on_measurement = _measure_mode(
                port_on, "on",
                _warmup_fn=_warmup_fn, _measure_fn=_measure_fn,
            )
        finally:
            lifecycle.stop()
    except BaseException:
        lifecycle.stop()
        raise

    comparison = _compute_comparison(off_measurement, on_measurement)

    return MtpEvidence(
        main_fingerprint=main_fingerprint,
        draft_fingerprint=draft_fingerprint,
        server_path=str(server_path),
        server_version=server_version,
        server_supported_flags=tuple(sorted(server_supported_flags)),
        candidate=candidate,
        n_ctx=n_ctx,
        cpu_only=cpu_only,
        device_id=device_id,
        off_argv=off_argv,
        on_argv=on_argv,
        comparison=comparison,
    )


# ── MTP Profile type alias ──────────────────────────────────────

MtpProfileEvaluator = Callable[
    [CandidateConfig, MtpDraftSettings],
    tuple[MtpComparison, MemoryEvidence],
]

# ── Draft reservation helpers ────────────────────────────────────


def _compute_draft_reservation_bytes(draft_fingerprint: ModelFingerprint) -> int:
    if draft_fingerprint.shards:
        return sum(s.size_bytes for s in draft_fingerprint.shards)
    return draft_fingerprint.path.stat().st_size if draft_fingerprint.path.exists() else 0


# ── Memory fit check ─────────────────────────────────────────────


def compute_combined_preflight(
    main_memory: MemoryEvidence,
    draft_fingerprint: ModelFingerprint,
    draft_settings: MtpDraftSettings,
    memory_unified: bool,
) -> MemoryEvidence:
    draft_reservation = _compute_draft_reservation_bytes(draft_fingerprint)

    if main_memory.quality != "complete" or main_memory.ram_bytes is None:
        return MemoryEvidence(quality="incomparable")

    if memory_unified:
        return MemoryEvidence(
            ram_bytes=main_memory.ram_bytes + draft_reservation,
            quality="complete",
        )

    if draft_settings.n_gpu_layers_draft != 0:
        if main_memory.vram_bytes is not None:
            return MemoryEvidence(
                ram_bytes=main_memory.ram_bytes,
                vram_bytes=main_memory.vram_bytes + draft_reservation,
                quality="complete",
            )
        return MemoryEvidence(
            ram_bytes=main_memory.ram_bytes + draft_reservation,
            vram_bytes=main_memory.vram_bytes,
            quality="complete" if main_memory.vram_bytes is not None else "partial",
        )

    return MemoryEvidence(
        ram_bytes=main_memory.ram_bytes + draft_reservation,
        vram_bytes=main_memory.vram_bytes,
        quality="complete",
    )


def check_combined_memory_fit(
    combined_memory: MemoryEvidence,
    budget: ResourceBudget,
    memory_unified: bool,
) -> bool:
    if combined_memory.quality != "complete":
        return False
    if combined_memory.ram_bytes is None:
        return False

    headroom = budget.headroom_ratio
    ram_available = int(budget.ram_bytes * (1 - headroom))

    if memory_unified or budget.vram_bytes is None:
        return combined_memory.ram_bytes <= ram_available

    if combined_memory.vram_bytes is None:
        return False

    vram_available = int(budget.vram_bytes * (1 - headroom))
    return (
        combined_memory.ram_bytes <= ram_available
        and combined_memory.vram_bytes <= vram_available
    )


def check_mtp_memory_fit(
    main_memory: MemoryEvidence,
    draft_fingerprint: ModelFingerprint,
    draft_settings: MtpDraftSettings,
    budget: ResourceBudget,
    memory_unified: bool,
) -> bool:
    if main_memory.quality != "complete":
        return False
    if main_memory.ram_bytes is None:
        return False

    draft_reservation = _compute_draft_reservation_bytes(draft_fingerprint)
    headroom = budget.headroom_ratio

    ram_available = int(budget.ram_bytes * (1 - headroom))

    if memory_unified or budget.vram_bytes is None:
        ram_needed = main_memory.ram_bytes + draft_reservation
        return ram_needed <= ram_available

    vram_available = int(budget.vram_bytes * (1 - headroom))

    if draft_settings.n_gpu_layers_draft != 0:
        if main_memory.vram_bytes is None:
            return False
        vram_needed = main_memory.vram_bytes + draft_reservation
        ram_ok = main_memory.ram_bytes <= ram_available
        vram_ok = vram_needed <= vram_available
        return ram_ok and vram_ok
    else:
        ram_needed = main_memory.ram_bytes + draft_reservation
        vram_ok = main_memory.vram_bytes is None or main_memory.vram_bytes <= vram_available
        return ram_needed <= ram_available and vram_ok


# ── MTP main candidate selection ─────────────────────────────────


def select_mtp_main_candidate(
    validation_result: ValidatedRecommendation,
    draft_fingerprint: ModelFingerprint,
    budget: ResourceBudget,
    memory_unified: bool,
    default_draft_settings: MtpDraftSettings = MtpDraftSettings(n_gpu_layers_draft=0),
    draft_settings_seq: Sequence[MtpDraftSettings] | None = None,
) -> CandidateConfig | None:
    candidates: list[ValidatedCandidate] = [
        validation_result.recommendation,
        *validation_result.alternatives,
    ]
    settings_to_test: Sequence[MtpDraftSettings]
    if draft_settings_seq is not None:
        settings_to_test = draft_settings_seq
    else:
        settings_to_test = (default_draft_settings,)
    for vc in candidates:
        me = vc.memory_evidence
        if me is None or me.quality != "complete":
            continue
        for ds in settings_to_test:
            if check_mtp_memory_fit(
                me, draft_fingerprint, ds, budget, memory_unified,
            ):
                return vc.candidate
    return None


# ── Draft settings grid ──────────────────────────────────────────


def generate_mtp_draft_grid(
    draft_fingerprint: ModelFingerprint,
    profile: MtpSearchProfile,
) -> tuple[MtpDraftSettings, ...]:
    grid: list[MtpDraftSettings] = []
    for ngl in profile.draft_gpu_layers_values:
        for nmax in profile.draft_n_max_values:
            if draft_fingerprint.is_moe:
                for moe in profile.draft_n_cpu_moe_values:
                    grid.append(MtpDraftSettings(
                        n_gpu_layers_draft=ngl,
                        n_cpu_moe_draft=moe,
                        draft_max=nmax,
                    ))
            else:
                grid.append(MtpDraftSettings(
                    n_gpu_layers_draft=ngl,
                    draft_max=nmax,
                ))
    return tuple(grid)


# ── Profile orchestrator ─────────────────────────────────────────


def collect_mtp_profile(
    *,
    validation_result: ValidatedRecommendation,
    draft_fingerprint: ModelFingerprint,
    budget: ResourceBudget,
    memory_unified: bool,
    profile: MtpSearchProfile,
    evaluator: MtpProfileEvaluator,
) -> MtpProfileReport:
    normal_winner = validation_result.recommendation

    draft_grid = generate_mtp_draft_grid(draft_fingerprint, profile)

    if len(draft_grid) > profile.max_attempts:
        draft_grid = draft_grid[:profile.max_attempts]

    main_candidate = select_mtp_main_candidate(
        validation_result, draft_fingerprint, budget, memory_unified,
        draft_settings_seq=draft_grid,
    )

    if main_candidate is None:
        return MtpProfileReport(
            normal_winner=normal_winner,
            mtp_main_candidate=None,
            draft_fingerprint=draft_fingerprint,
            observations=(),
            recommendation=None,
            benefit="not_established",
            note="No validated candidate has complete memory evidence "
                 "with sufficient headroom for draft model reservation.",
        )

    observations: list[MtpProfileObservation] = []
    eligible: list[tuple[float, MtpComparison, MemoryEvidence, MtpDraftSettings]] = []

    for ds in draft_grid:
        try:
            comparison, combined_memory = evaluator(main_candidate, ds)
        except Exception as exc:
            observations.append(MtpProfileObservation(
                main_candidate=main_candidate,
                draft_settings=ds,
                comparison=None,
                combined_memory=None,
                disposition="not_attempted",
                note=f"Evaluator raised: {exc}",
            ))
            continue

        if comparison.benefit != "established":
            observations.append(MtpProfileObservation(
                main_candidate=main_candidate,
                draft_settings=ds,
                comparison=comparison,
                combined_memory=combined_memory,
                disposition="not_established",
            ))
            continue

        if combined_memory is None:
            observations.append(MtpProfileObservation(
                main_candidate=main_candidate,
                draft_settings=ds,
                comparison=comparison,
                combined_memory=combined_memory,
                disposition="incomplete_evidence",
                note="Combined memory evidence is missing; cannot verify budget fit.",
            ))
            continue

        if combined_memory.quality != "complete":
            observations.append(MtpProfileObservation(
                main_candidate=main_candidate,
                draft_settings=ds,
                comparison=comparison,
                combined_memory=combined_memory,
                disposition="incomplete_evidence",
                note=f"Combined memory quality is {combined_memory.quality}, "
                     f"not complete; cannot verify budget fit.",
            ))
            continue

        if not check_combined_memory_fit(combined_memory, budget, memory_unified):
            observations.append(MtpProfileObservation(
                main_candidate=main_candidate,
                draft_settings=ds,
                comparison=comparison,
                combined_memory=combined_memory,
                disposition="no_fit",
                note="Combined memory exceeds available budget after headroom.",
            ))
            continue

        observations.append(MtpProfileObservation(
            main_candidate=main_candidate,
            draft_settings=ds,
            comparison=comparison,
            combined_memory=combined_memory,
            disposition="established",
        ))
        eligible.append((comparison.on.median_tps, comparison, combined_memory, ds))

    if not eligible:
        return MtpProfileReport(
            normal_winner=normal_winner,
            mtp_main_candidate=main_candidate,
            draft_fingerprint=draft_fingerprint,
            observations=tuple(observations),
            recommendation=None,
            benefit="not_established",
            note=(
                "No MTP candidate established benefit with complete "
                "memory evidence within budget."
            ),
        )

    eligible.sort(key=lambda x: x[0], reverse=True)
    best_comparison = eligible[0][1]
    best_draft_settings = eligible[0][3]

    return MtpProfileReport(
        normal_winner=normal_winner,
        mtp_main_candidate=main_candidate,
        draft_fingerprint=draft_fingerprint,
        observations=tuple(observations),
        recommendation=best_comparison,
        benefit="established",
        note="",
        winning_draft_settings=best_draft_settings,
    )


# ── Convenience wrapper ──────────────────────────────────────────


def compare_mtp(
    *,
    candidate: CandidateConfig,
    main_model_path: Path,
    draft_model_path: Path,
    server_path: Path,
    n_ctx: NctxComposition,
    cpu_only: bool = True,
    device_id: int | None = None,
    runner: ProcessRunner | None = None,
    server_lifecycle: ServerLifecycle | None = None,
    main_fingerprint: ModelFingerprint | None = None,
    draft_fingerprint: ModelFingerprint | None = None,
    _warmup_fn: Callable[[int], None] | None = None,
    _measure_fn: Callable[[int], float] | None = None,
) -> MtpComparison:
    evidence = collect_mtp_evidence(
        candidate=candidate,
        main_model_path=main_model_path,
        draft_model_path=draft_model_path,
        server_path=server_path,
        n_ctx=n_ctx,
        cpu_only=cpu_only,
        device_id=device_id,
        runner=runner,
        server_lifecycle=server_lifecycle,
        main_fingerprint=main_fingerprint,
        draft_fingerprint=draft_fingerprint,
        _warmup_fn=_warmup_fn,
        _measure_fn=_measure_fn,
    )
    return evidence.comparison


def _compute_comparison(
    off: MtpMeasurement,
    on: MtpMeasurement,
) -> MtpComparison:
    improvement = on.median_tps - off.median_tps
    tie_bound = max(off.dispersion_tps, on.dispersion_tps)

    if improvement > tie_bound and off.median_tps > 0:
        benefit: MtpBenefit = "established"
        improvement_pct = (improvement / off.median_tps) * 100
    else:
        benefit = "not_established"
        improvement_pct = None

    return MtpComparison(
        off=off,
        on=on,
        benefit=benefit,
        improvement_pct=improvement_pct,
        tie_bound_tps=tie_bound,
        caveat=_MTP_CAVEAT,
    )



