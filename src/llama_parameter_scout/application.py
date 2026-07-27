from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from typing import Any

from .journal import write_observation
from .model_inspector import inspect_model
from .models import (
    BenchmarkObservation,
    BinaryNotFoundError,
    CandidateConfig,
    CandidateError,
    ContextAdvisory,
    ContextExceededWithoutAcknowledgementError,
    InconsistentNglError,
    InvalidHeadroomRatioError,
    InvalidNctxError,
    JournalError,
    MalformedOutputError,
    ModelFingerprint,
    ModelFingerprintMismatchError,
    ModelNotFoundError,
    MultipleAcceleratorsSelectedError,
    NctxComposition,
    NoAcceleratorSelectedError,
    NonpositiveBudgetError,
    ResourceBudget,
    SessionSetupError,
    SessionSpecification,
    SingleRunRequest,
    SubprocessError,
    SystemTopology,
    ToolchainFingerprint,
    TrainingContextStatus,
    UnknownDeviceError,
)
from .runner import ProcessRunner
from .toolchain import inspect_toolchain
from .topology import DetectTopologyFn, detect_topology

# ── Defaults ─────────────────────────────────────────────────────

DEFAULT_HEADROOM_RATIO = 0.1
_ESTIMATED_BYTES_PER_TOKEN_PER_PARAM = 0.000075


def _cache_element_size(cache_type: str) -> int:
    mapping = {"f16": 2, "f32": 4, "q8_0": 1, "q4_0": 1}
    return mapping.get(cache_type, 2)


def _build_argv(
    request: SingleRunRequest,
    *,
    repetitions: int,
    session_spec: SessionSpecification | None = None,
) -> list[str]:
    argv = [
        str(request.binary_path),
        "-m", str(request.model_path),
        "-p", str(request.n_ctx.prompt),
        "-n", str(request.n_ctx.generation),
        "-d", str(request.n_ctx.depth),
        "-t", str(request.candidate.threads),
        "-b", str(request.candidate.batch_size),
        "-ub", str(request.candidate.ubatch_size),
        "-ngl", str(request.candidate.n_gpu_layers),
        "-fa", "on" if request.candidate.flash_attn else "off",
        "-ctk", request.candidate.cache_type_k,
        "-ctv", request.candidate.cache_type_v,
        "-o", "json",
        "-r", str(repetitions),
    ]
    if session_spec is not None and not session_spec.cpu_only:
        if session_spec.selected_device_ids:
            argv.extend(["--split-mode", "none"])
            argv.extend(["--main-gpu", str(session_spec.selected_device_ids[0])])
    return argv


def _validate_request(request: SingleRunRequest) -> None:
    bp = request.binary_path
    if not bp.exists():
        raise BinaryNotFoundError(f"Binary not found: {bp}")
    if not bp.is_file():
        raise BinaryNotFoundError(f"Binary is not a file: {bp}")
    if not os.access(str(bp), os.X_OK):
        raise BinaryNotFoundError(f"Binary is not executable: {bp}")

    mp = request.model_path
    if not mp.exists():
        raise ModelNotFoundError(f"Model not found: {mp}")
    if not mp.is_file():
        raise ModelNotFoundError(f"Model is not a file: {mp}")

    nc = request.n_ctx
    if nc.total <= 0:
        raise InvalidNctxError("n_ctx.total must be > 0")
    if nc.prompt < 0:
        raise InvalidNctxError("n_ctx.prompt must be >= 0")
    if nc.generation < 0:
        raise InvalidNctxError("n_ctx.generation must be >= 0")
    if nc.depth < 0:
        raise InvalidNctxError("n_ctx.depth must be >= 0")
    if nc.prompt + nc.generation + nc.depth != nc.total:
        raise InvalidNctxError(
            f"n_ctx components ({nc.prompt} + {nc.generation} + {nc.depth}) "
            f"do not sum to total ({nc.total})",
        )

    cc = request.candidate
    if cc.threads <= 0:
        raise CandidateError("threads must be > 0")
    if cc.batch_size <= 0:
        raise CandidateError("batch_size must be > 0")
    if cc.ubatch_size <= 0:
        raise CandidateError("ubatch_size must be > 0")
    if cc.n_gpu_layers < -1:
        raise CandidateError("n_gpu_layers must be >= -1")


def _parse_measurements(stdout: str) -> list[float]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        data = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                pass
        if not data:
            raise MalformedOutputError(
                f"Cannot parse llama-bench output as JSON or JSONL.\n"
                f"Raw output:\n{stdout}",
            )

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise MalformedOutputError(
            f"Top-level JSON is not an array or object.\n"
            f"Raw output:\n{stdout}",
        )

    samples: list[float] | None = None
    for record in data:
        if not isinstance(record, dict):
            continue
        if "samples_ts" not in record:
            continue
        s = record["samples_ts"]
        if not isinstance(s, list) or len(s) == 0:
            continue
        converted: list[float] = []
        for x in s:
            if isinstance(x, bool):
                raise MalformedOutputError(
                    f"Samples contain boolean values (bool must not become number).\n"
                    f"Raw output:\n{stdout}",
                )
            if not isinstance(x, (int, float)):
                raise MalformedOutputError(
                    f"Samples contain non-numeric value: {type(x).__name__}.\n"
                    f"Raw output:\n{stdout}",
                )
            val = float(x)
            if not math.isfinite(val) or val <= 0:
                raise MalformedOutputError(
                    f"samples_ts contains non-finite or non-positive values: {s}\n"
                    f"Raw output:\n{stdout}",
                )
            converted.append(val)
        if len(converted) >= 3:
            samples = converted
            break

    if samples is None or len(samples) < 3:
        raise MalformedOutputError(
            f"Could not find samples_ts with >= 3 entries in llama-bench output.\n"
            f"Raw output:\n{stdout}",
        )

    return samples


def _validate_warmup_output(stdout: str) -> None:
    try:
        json.loads(stdout)
        return
    except json.JSONDecodeError:
        pass
    for line in stdout.strip().splitlines():
        try:
            json.loads(line)
            return
        except json.JSONDecodeError:
            continue
    raise MalformedOutputError(
        f"Warmup output is not valid JSON or JSONL:\n{stdout}",
    )


def _compute_stats(samples: list[float]) -> tuple[float, float]:
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    if n % 2 == 1:
        median = sorted_samples[n // 2]
    else:
        median = (sorted_samples[n // 2 - 1] + sorted_samples[n // 2]) / 2.0

    mean = sum(sorted_samples) / n
    variance = sum((x - mean) ** 2 for x in sorted_samples) / n
    dispersion = math.sqrt(variance)

    return median, dispersion


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _compute_session_fingerprint(spec: SessionSpecification) -> str:
    model = spec.model
    canon = _canonical_json({
        "cpu_only": spec.cpu_only,
        "selected_device_ids": list(spec.selected_device_ids),
        "n_ctx": {
            "total": spec.n_ctx.total,
            "prompt": spec.n_ctx.prompt,
            "generation": spec.n_ctx.generation,
            "depth": spec.n_ctx.depth,
        },
        "budget": {
            "ram_bytes": spec.budget.ram_bytes,
            "vram_bytes": spec.budget.vram_bytes,
            "headroom_ratio": spec.budget.headroom_ratio,
        },
        "advisory": {
            "training_context_status": spec.advisory.training_context_status,
            "training_context_tokens": spec.advisory.training_context_tokens,
            "model_weight_bytes": spec.advisory.model_weight_bytes,
            "kv_cache_bytes_estimate": spec.advisory.kv_cache_bytes_estimate,
        },
        "topology": {
            "cpu_count": spec.topology.cpu_count,
            "system_memory_bytes": spec.topology.system_memory_bytes,
            "memory_unified": spec.topology.memory_unified,
            "accelerators": [
                {
                    "kind": a.kind,
                    "backend": a.backend,
                    "device_id": a.device_id,
                    "description": a.description,
                    "memory_bytes": a.memory_bytes,
                }
                for a in spec.topology.accelerators
            ],
        },
        "model": {
            "path": str(model.path),
            "architecture": model.architecture,
            "parameter_count": model.parameter_count,
            "context_length": model.context_length,
            "quantization": model.quantization,
            "diagnostics": list(model.diagnostics),
            "shards": [
                {
                    "path": str(s.path),
                    "size_bytes": s.size_bytes,
                    "modified_time": s.modified_time,
                    "digest_sha256": s.digest_sha256,
                }
                for s in model.shards
            ],
        },
        "acknowledged": spec.acknowledged,
    })
    return hashlib.sha256(canon.encode()).hexdigest()


def _validate_device_selection(
    topology: SystemTopology,
    cpu_only: bool,
    device_ids: tuple[int, ...],
) -> None:
    if cpu_only:
        return

    if len(device_ids) == 0:
        raise NoAcceleratorSelectedError(
            "Acceleration requested (cpu_only=False) but no --device specified. "
            "Use --device <id> to select exactly one accelerator, "
            "or use --cpu-only for CPU-only operation."
            if topology.accelerators
            else "Acceleration requested (cpu_only=False) but no accelerators detected. "
            "Use --cpu-only for CPU-only operation.",
        )

    if len(device_ids) > 1:
        raise MultipleAcceleratorsSelectedError(
            f"{len(device_ids)} device IDs provided. "
            "Exactly one accelerator device may be selected at a time.",
        )

    device_id = device_ids[0]
    if not any(a.device_id == device_id for a in topology.accelerators):
        known = ", ".join(str(a.device_id) for a in topology.accelerators)
        raise UnknownDeviceError(
            f"Device {device_id} not found among detected accelerators "
            f"(IDs: {known}).",
        )


def _validate_ngl_consistency(
    cpu_only: bool,
    n_gpu_layers: int,
) -> None:
    if cpu_only and n_gpu_layers > 0:
        raise InconsistentNglError(
            "CPU-only mode selected but n_gpu_layers > 0. "
            "Set --n-gpu-layers 0 for CPU-only operation.",
        )
    if cpu_only and n_gpu_layers == -1:
        raise InconsistentNglError(
            "CPU-only mode selected but n_gpu_layers == -1 (all layers). "
            "Set --n-gpu-layers 0 for CPU-only operation.",
        )


def _compute_model_weight_bytes(model: ModelFingerprint) -> int:
    if model.shards:
        return sum(s.size_bytes for s in model.shards)
    return model.path.stat().st_size if model.path.exists() else 0


def _estimate_advisory(
    model: ModelFingerprint,
    n_ctx: NctxComposition,
    budget: ResourceBudget,
    cache_type: str = "f16",
) -> ContextAdvisory:
    model_weight = _compute_model_weight_bytes(model)
    element_size = _cache_element_size(cache_type)
    param_count = model.parameter_count or 7_000_000_000
    bytes_per_token = int(param_count * _ESTIMATED_BYTES_PER_TOKEN_PER_PARAM * element_size // 2)
    kv_cache_bytes_estimate = bytes_per_token * n_ctx.total

    # Keep RAM and VRAM separate and conservative.
    # Formula: available_for_kv = min(ram_after_headroom, vram_after_headroom) - model_weight
    #          then max_tokens = available_for_kv / bytes_per_token.
    # RAM and VRAM limits are evaluated independently; the tighter bound dominates.
    ram_after_headroom = int(budget.ram_bytes * (1 - budget.headroom_ratio))
    if budget.vram_bytes is not None and budget.vram_bytes > 0:
        vram_after_headroom = int(budget.vram_bytes * (1 - budget.headroom_ratio))
    else:
        vram_after_headroom = ram_after_headroom

    available_for_kv = max(0, min(ram_after_headroom, vram_after_headroom) - model_weight)

    # Determine training-context status and bound
    ctx = model.context_length
    has_conflict = any("conflict" in d.lower() for d in model.diagnostics)
    if ctx is not None and not has_conflict:
        training_context_status: TrainingContextStatus = "known"
        training_context_tokens: int | None = ctx
    elif has_conflict:
        training_context_status = "conflicted"
        training_context_tokens = ctx  # still record numeric value
    else:
        training_context_status = "unknown"
        training_context_tokens = None

    if bytes_per_token > 0:
        max_tokens_raw = available_for_kv // bytes_per_token
        uncertain_maximum = max(1, int(max_tokens_raw * 0.95))

        if training_context_status == "known" and training_context_tokens is not None:
            upper = min(max_tokens_raw, training_context_tokens)
        else:
            upper = max_tokens_raw
        upper = max(1, int(upper * 0.8))
        likely_fit_lower = max(1, int(upper * 0.5))
    else:
        uncertain_maximum = available_for_kv
        upper = available_for_kv
        likely_fit_lower = 1

    status_label = {
        "known": "training context is known",
        "unknown": "training context is unknown",
        "conflicted": "training context is conflicted",
    }[training_context_status]

    return ContextAdvisory(
        likely_fit_lower=likely_fit_lower,
        likely_fit_upper=upper,
        uncertain_maximum=uncertain_maximum,
        model_weight_bytes=model_weight,
        kv_cache_bytes_estimate=kv_cache_bytes_estimate,
        training_context_status=training_context_status,
        training_context_tokens=training_context_tokens,
        note=(
            f"Status: {status_label}. "
            f"Likely-fit range based on model size ({model_weight} bytes model weight), "
            f"KV cache heuristics ({kv_cache_bytes_estimate} bytes at {n_ctx.total} tokens), "
            f"resource budget (RAM={budget.ram_bytes}, VRAM={budget.vram_bytes}), "
            f"and headroom ({budget.headroom_ratio * 100:.0f}%). "
            "It is not a tested guarantee. Actual fit depends on runtime overhead, "
            "memory fragmentation, and the specific build of llama.cpp."
        ),
    )


def _validate_budget(
    budget: ResourceBudget,
) -> None:
    if budget.ram_bytes <= 0:
        raise NonpositiveBudgetError(
            f"RAM budget must be positive, got {budget.ram_bytes}",
        )
    if budget.vram_bytes is not None and budget.vram_bytes <= 0:
        raise NonpositiveBudgetError(
            f"VRAM budget must be positive, got {budget.vram_bytes}",
        )
    if budget.headroom_ratio < 0.0 or budget.headroom_ratio >= 1.0:
        raise InvalidHeadroomRatioError(
            f"Headroom ratio must be in [0.0, 1.0), got {budget.headroom_ratio}",
        )


def _check_context_acknowledgement(
    model: ModelFingerprint,
    n_ctx: NctxComposition,
    acknowledged: bool,
) -> None:
    ctx = model.context_length
    has_conflict = any("conflict" in d.lower() for d in model.diagnostics)
    needs_ack = False

    if ctx is not None and n_ctx.total > ctx:
        needs_ack = True
    elif ctx is None:
        needs_ack = True

    if has_conflict:
        needs_ack = True

    if needs_ack and not acknowledged:
        if ctx is not None and n_ctx.total > ctx:
            msg = (
                f"Requested context ({n_ctx.total}) exceeds the model's known "
                f"training context ({ctx}). Use --yes to acknowledge."
            )
        elif has_conflict:
            msg = (
                "Model context length metadata is internally conflicted. "
                "Use --yes to acknowledge."
            )
        else:
            msg = (
                "Model training context length is unknown. "
                "Use --yes to acknowledge."
            )
        raise ContextExceededWithoutAcknowledgementError(msg)


class Application:
    def __init__(
        self,
        runner: ProcessRunner,
        detect_topology_fn: DetectTopologyFn | None = None,
    ) -> None:
        self._runner = runner
        self._detect_topology = detect_topology_fn or detect_topology

    def setup_session(
        self,
        model: ModelFingerprint,
        n_ctx: NctxComposition,
        candidate: CandidateConfig,
        cpu_only: bool,
        device_ids: tuple[int, ...] = (),
        budget_overrides: tuple[int | None, int | None] | None = None,
        headroom_ratio: float = DEFAULT_HEADROOM_RATIO,
        acknowledged: bool = False,
    ) -> SessionSpecification:
        topology = self._detect_topology()

        _validate_device_selection(topology, cpu_only, device_ids)
        _validate_ngl_consistency(cpu_only, candidate.n_gpu_layers)

        if candidate.n_gpu_layers < -1:
            raise CandidateError("n_gpu_layers must be >= -1")

        if budget_overrides is not None and budget_overrides[0] is not None:
            ram = budget_overrides[0]
        else:
            ram = topology.system_memory_bytes

        if cpu_only or topology.memory_unified or not topology.accelerators:
            vram: int | None = None
        else:
            if budget_overrides is not None and budget_overrides[1] is not None:
                vram = budget_overrides[1]
            else:
                selected_device = None
                if device_ids:
                    device_id = device_ids[0]
                    for a in topology.accelerators:
                        if a.device_id == device_id:
                            selected_device = a
                            break
                if selected_device is not None and selected_device.memory_bytes is not None:
                    vram = selected_device.memory_bytes
                elif not topology.accelerators:
                    vram = None
                elif device_ids:
                    raise SessionSetupError(
                        f"Cannot determine VRAM for device {device_ids[0]}. "
                        "Specify --budget-vram to provide a VRAM budget."
                    )
                else:
                    vram = None

        budget = ResourceBudget(
            ram_bytes=ram, vram_bytes=vram,
            headroom_ratio=headroom_ratio,
        )
        _validate_budget(budget)
        _validate_nctx_at_setup(n_ctx)

        _check_context_acknowledgement(model, n_ctx, acknowledged)

        advisory = _estimate_advisory(model, n_ctx, budget, candidate.cache_type_k)

        spec = SessionSpecification(
            topology=topology,
            cpu_only=cpu_only,
            selected_device_ids=device_ids,
            n_ctx=n_ctx,
            budget=budget,
            advisory=advisory,
            model=model,
            acknowledged=acknowledged,
            fingerprint="",
        )
        spec_fingerprint = _compute_session_fingerprint(spec)
        spec = SessionSpecification(
            topology=topology,
            cpu_only=cpu_only,
            selected_device_ids=device_ids,
            n_ctx=n_ctx,
            budget=budget,
            advisory=advisory,
            model=model,
            acknowledged=acknowledged,
            fingerprint=spec_fingerprint,
        )
        return spec

    def run(
        self,
        request: SingleRunRequest,
        *,
        on_ready: Callable[[ToolchainFingerprint, ModelFingerprint], Any] | None = None,
        session_spec: SessionSpecification | None = None,
    ) -> BenchmarkObservation:
        _validate_request(request)

        toolchain = inspect_toolchain(self._runner, request.binary_path)
        model = inspect_model(request.model_path)

        if session_spec is not None:
            if session_spec.n_ctx != request.n_ctx:
                raise SessionSetupError(
                    f"Session spec n_ctx ({session_spec.n_ctx}) does not match "
                    f"request n_ctx ({request.n_ctx})",
                )
            if session_spec.model != model:
                raise ModelFingerprintMismatchError(
                    f"Session spec model fingerprint does not match freshly inspected model.\n"
                    f"Spec model: path={session_spec.model.path} "
                    f"arch={session_spec.model.architecture} "
                    f"shards={len(session_spec.model.shards)}\n"
                    f"Fresh model: path={model.path} "
                    f"arch={model.architecture} "
                    f"shards={len(model.shards)}",
                )

            _validate_ngl_consistency(
                session_spec.cpu_only,
                request.candidate.n_gpu_layers,
            )

            if not session_spec.cpu_only:
                if "--split-mode" not in toolchain.supported_flags:
                    raise SessionSetupError(
                        "Accelerated launch requested but llama-bench does not support "
                        "--split-mode. Cannot guarantee device isolation.",
                    )
                if "--main-gpu" not in toolchain.supported_flags:
                    raise SessionSetupError(
                        "Accelerated launch requested but llama-bench does not support "
                        "--main-gpu. Cannot guarantee device isolation.",
                    )

                device_id = session_spec.selected_device_ids[0]
                selected_accelerator = next(
                    (a for a in session_spec.topology.accelerators
                     if a.device_id == device_id),
                    None,
                )
                if selected_accelerator is not None:
                    if selected_accelerator.backend not in toolchain.device_capabilities:
                        raise SessionSetupError(
                            f"Selected device {device_id} requires backend "
                            f"'{selected_accelerator.backend}' but toolchain supports: "
                            f"{', '.join(toolchain.device_capabilities)}",
                        )

        if on_ready is not None:
            on_ready(toolchain, model)

        warmup_argv = _build_argv(request, repetitions=1, session_spec=session_spec)
        warmup_result = self._runner.run(warmup_argv)
        if warmup_result.returncode != 0:
            raise SubprocessError(
                f"Warmup failed (exit {warmup_result.returncode}):\n"
                f"{warmup_result.stderr}",
            )
        _validate_warmup_output(warmup_result.stdout)

        measure_argv = _build_argv(request, repetitions=3, session_spec=session_spec)
        measure_result = self._runner.run(measure_argv)
        if measure_result.returncode != 0:
            raise SubprocessError(
                f"Measurement failed (exit {measure_result.returncode}):\n"
                f"{measure_result.stderr}",
            )

        samples = _parse_measurements(measure_result.stdout)
        median, dispersion = _compute_stats(samples)

        observation = BenchmarkObservation(
            candidate=request.candidate,
            total_ctx=request.n_ctx.total,
            prompt_ctx=request.n_ctx.prompt,
            generation_ctx=request.n_ctx.generation,
            depth_ctx=request.n_ctx.depth,
            samples_tps=tuple(samples),
            median_tps=median,
            dispersion_tps=dispersion,
            binary_path=request.binary_path,
            binary_source=request.binary_source,
            model_path=request.model_path,
            toolchain=toolchain,
            model=model,
            session_spec=session_spec,
        )

        try:
            write_observation(request.journal_path, observation, session_spec=session_spec)
        except OSError as e:
            raise JournalError(f"Failed to write journal: {e}") from e

        return observation


def _validate_nctx_at_setup(n_ctx: NctxComposition) -> None:
    if n_ctx.total <= 0:
        raise InvalidNctxError("n_ctx.total must be > 0")
    if n_ctx.prompt < 0:
        raise InvalidNctxError("n_ctx.prompt must be >= 0")
    if n_ctx.generation < 0:
        raise InvalidNctxError("n_ctx.generation must be >= 0")
    if n_ctx.depth < 0:
        raise InvalidNctxError("n_ctx.depth must be >= 0")
    if n_ctx.prompt + n_ctx.generation + n_ctx.depth != n_ctx.total:
        raise InvalidNctxError(
            f"n_ctx components ({n_ctx.prompt} + {n_ctx.generation} + {n_ctx.depth}) "
            f"do not sum to total ({n_ctx.total})",
        )
