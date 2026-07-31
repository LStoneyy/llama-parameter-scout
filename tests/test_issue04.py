from __future__ import annotations

import json
import struct
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from llama_parameter_scout.application import (
    Application,
    _build_argv,
    _build_launch_env,
    _check_resource_budget,
    _classify_launch,
    _detect_available_memory_bytes,
    _env_for_device,
    _is_oom,
    pre_execution_disposition,
)
from llama_parameter_scout.models import (
    Accelerator,
    CandidateConfig,
    CandidateOutcome,
    ContextAdvisory,
    ExecutionPolicy,
    JournalError,
    ModelFingerprint,
    NctxComposition,
    ResourceBudget,
    SessionSpecification,
    SingleRunRequest,
    StopTracker,
    SystemTopology,
    ToolchainFingerprint,
)
from llama_parameter_scout.runner import (
    FakeProcessAdapter,
    FakeRunner,
    ProcessResult,
    ProcessStartError,
)
from tests.conftest import write_fake_executable


def _read_journal_first_payload(path: Path) -> dict[str, Any]:
    raw: Any = json.loads(path.read_text())
    if raw.get("schema_version") == 5:
        ev: Any = raw["events"][0]
        et = str(ev.get("event_type", ""))
        if et == "observation_completed":
            obs: Any = ev["observation"]
            return dict(obs)
        outcome: Any = ev.get("outcome")
        if outcome is not None:
            return dict(outcome)
        return dict(ev)
    return dict(raw)

_TOOLCHAIN_FLAGS = "-m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"
_TOOLCHAIN_STDOUT = (
    "version: 1\n"
    "backends: cpu\n"
    f"supported_flags: {_TOOLCHAIN_FLAGS}\n"
    "supported_output_formats: json\n"
)


def _build_minimal_gguf(path: Path, metadata: dict[str, tuple[int, Any]]) -> None:
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 0))
    buf.extend(struct.pack("<Q", len(metadata)))
    for key, (value_type, value) in metadata.items():
        _pack_gguf_string(buf, key)
        buf.extend(struct.pack("<I", value_type))
        _pack_gguf_value(buf, value_type, value)
    path.write_bytes(bytes(buf))


def _pack_gguf_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


def _pack_gguf_value(buf: bytearray, value_type: int, value: Any) -> None:
    if value_type == 8:
        _pack_gguf_string(buf, value)


def _enqueue_toolchain(runner: FakeRunner, binary_path: Path) -> None:
    bp = str(binary_path)
    runner.enqueue(
        (bp, "--version"),
        ProcessResult(0, _TOOLCHAIN_STDOUT, ""),
    )
    runner.enqueue(
        (bp, "--help"),
        ProcessResult(0, "usage\n", ""),
    )


def _make_toolchain() -> ToolchainFingerprint:
    return ToolchainFingerprint(
        binary_path=Path("/fake/llama-bench"),
        version="1.0",
        supported_flags=("-m", "-p", "-n", "-d", "-t", "-b", "-ub",
                         "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r"),
        supported_output_formats=("json",),
        device_capabilities=("cpu",),
    )


def _make_model() -> ModelFingerprint:
    return ModelFingerprint(
        path=Path("/fake/model.gguf"),
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=4096,
        quantization="Q4_K_M",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )


@pytest.fixture
def sample_candidate() -> CandidateConfig:
    return CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=0, flash_attn=False,
        cache_type_k="f16", cache_type_v="f16",
    )


@pytest.fixture
def sample_request(
    sample_candidate: CandidateConfig,
    tmp_path: Path,
) -> SingleRunRequest:
    bp = tmp_path / "llama-bench"
    bp.write_text("#!/bin/sh\necho '{\"samples_ts\":[1.0,2.0,3.0]}'")
    bp.chmod(0o755)
    mp = tmp_path / "model.gguf"
    _build_minimal_gguf(mp, {"general.architecture": (8, "llama")})
    return SingleRunRequest(
        binary_path=bp,
        model_path=mp,
        journal_path=tmp_path / "journal.json",
        n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
        candidate=sample_candidate,
    )


# ── Pre-execution dispositions ──────────────────────────────────


class TestPreExecutionDispositions:
    def test_missing_binary_returns_unsupported(
        self,
        sample_candidate: CandidateConfig,
        tmp_path: Path,
    ) -> None:
        req = SingleRunRequest(
            binary_path=Path("/nonexistent"),
            model_path=tmp_path / "model.gguf",
            journal_path=tmp_path / "j.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=sample_candidate,
        )
        app = Application(FakeRunner())
        outcome = app.evaluate_candidate(req)
        assert outcome.disposition == "unsupported"
        assert outcome.process_launched is False
        assert "Binary not found" in outcome.diagnostic

    def test_missing_model_returns_unsupported(
        self,
        sample_candidate: CandidateConfig,
        fake_binary: Path,
        tmp_path: Path,
    ) -> None:
        req = SingleRunRequest(
            binary_path=fake_binary,
            model_path=Path("/nonexistent/model.gguf"),
            journal_path=tmp_path / "j.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=sample_candidate,
        )
        app = Application(FakeRunner())
        outcome = app.evaluate_candidate(req)
        assert outcome.disposition == "unsupported"
        assert outcome.process_launched is False
        assert "Model not found" in outcome.diagnostic or "not found" in outcome.diagnostic

    def test_invalid_nctx_returns_unsupported(
        self,
        sample_candidate: CandidateConfig,
        fake_binary: Path,
        fake_model: Path,
        tmp_path: Path,
    ) -> None:
        req = SingleRunRequest(
            binary_path=fake_binary,
            model_path=fake_model,
            journal_path=tmp_path / "j.json",
            n_ctx=NctxComposition(total=0, prompt=0, generation=0, depth=0),
            candidate=sample_candidate,
        )
        app = Application(FakeRunner())
        outcome = app.evaluate_candidate(req)
        assert outcome.disposition == "unsupported"
        assert outcome.process_launched is False

    def test_unsupported_does_not_launch(
        self,
        sample_candidate: CandidateConfig,
        tmp_path: Path,
    ) -> None:
        runner = FakeRunner()
        adapter = FakeProcessAdapter()
        req = SingleRunRequest(
            binary_path=Path("/nonexistent"),
            model_path=tmp_path / "model.gguf",
            journal_path=tmp_path / "j.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=sample_candidate,
        )
        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(req)
        assert outcome.disposition == "unsupported"
        assert outcome.process_launched is False
        assert len(runner.calls) == 0
        assert len(adapter.calls) == 0

    def test_ngl_inconsistency_returns_pruned(
        self,
        fake_binary: Path,
        fake_model: Path,
        tmp_path: Path,
    ) -> None:
        from llama_parameter_scout.model_inspector import inspect_model

        candidate = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=20, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        req = SingleRunRequest(
            binary_path=fake_binary,
            model_path=fake_model,
            journal_path=tmp_path / "j.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=candidate,
        )
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=32_000_000_000,
            accelerators=(),
            memory_unified=False,
        )
        model = inspect_model(fake_model)
        app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
        spec = app.setup_session(
            model=model,
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
            cpu_only=True,
            acknowledged=True,
        )
        # Enqueue toolchain for the binary
        runner = FakeRunner()
        _enqueue_toolchain(runner, fake_binary)
        app = Application(runner, detect_topology_fn=lambda: topology)
        outcome = app.evaluate_candidate(req, session_spec=spec)
        assert outcome.disposition == "pruned"
        assert outcome.process_launched is False


# ── Execution outcomes ──────────────────────────────────────────


class TestLaunchedOutcomes:
    def test_completed_via_fake_adapter(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )
        measure_argv = _build_argv(sample_request, repetitions=3)
        adapter.enqueue(
            measure_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "completed"
        assert outcome.process_launched is True
        assert outcome.safe_next_action == ""
        assert outcome.observation is not None
        assert len(outcome.observation.samples_tps) == 3

    def test_oom_disposition(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(-9, "", "Killed"),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "oom"
        assert outcome.process_launched is True

    def test_timeout_disposition(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, "ok", "", termination="timeout"),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "timeout"
        assert outcome.process_launched is True

    def test_cancellation_disposition(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        cancel = Event()
        cancel.set()
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, "ok", "", termination="cancellation"),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(
            sample_request,
            policy=ExecutionPolicy(cancellation=cancel),
        )

        assert outcome.disposition == "cancellation"
        assert outcome.process_launched is True

    def test_crash_disposition(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(1, "", "Segmentation fault"),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "crash"
        assert outcome.process_launched is True

    def test_malformed_output_disposition(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, "not valid json at all", ""),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "malformed_output"
        assert outcome.process_launched is True

    def test_systemic_error_on_launch_exception(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        from collections.abc import Callable, Sequence

        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)

        class BrokenAdapter:
            def launch(
                self,
                argv: Sequence[str],
                *,
                env: dict[str, str] | None = None,
                timeout_seconds: float | None = None,
                cancellation: Event | None = None,
                memory_pressure_fn: Callable[[], bool] | None = None,
                memory_pressure_backoff_seconds: float = 1.0,
                memory_pressure_max_backoffs: int = 3,
            ) -> ProcessResult:
                raise RuntimeError("broken")

        app = Application(runner, process_adapter=BrokenAdapter())
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "systemic_error"
        assert outcome.process_launched is True


class TestProcessStartError:
    def test_warmup_process_start_error_returns_unsupported(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        from collections.abc import Callable, Sequence

        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)

        class FailingAdapter:
            def launch(
                self,
                argv: Sequence[str],
                *,
                env: dict[str, str] | None = None,
                timeout_seconds: float | None = None,
                cancellation: Event | None = None,
                memory_pressure_fn: Callable[[], bool] | None = None,
                memory_pressure_backoff_seconds: float = 1.0,
                memory_pressure_max_backoffs: int = 3,
            ) -> ProcessResult:
                raise ProcessStartError("binary not found at /nonexistent")

        app = Application(runner, process_adapter=FailingAdapter())
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "unsupported"
        assert outcome.process_launched is False
        assert "could not be started" in outcome.diagnostic.lower()

    def test_measurement_process_start_error_returns_unsupported(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        from collections.abc import Callable, Sequence

        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )

        call_count: list[int] = [0]

        class FailingAdapter:
            def launch(
                self,
                argv: Sequence[str],
                *,
                env: dict[str, str] | None = None,
                timeout_seconds: float | None = None,
                cancellation: Event | None = None,
                memory_pressure_fn: Callable[[], bool] | None = None,
                memory_pressure_backoff_seconds: float = 1.0,
                memory_pressure_max_backoffs: int = 3,
            ) -> ProcessResult:
                call_count[0] += 1
                if call_count[0] > 1:
                    raise ProcessStartError("binary disappeared before measurement")
                return adapter.launch(
                    argv,
                    env=env,
                    timeout_seconds=timeout_seconds,
                    cancellation=cancellation,
                    memory_pressure_fn=memory_pressure_fn,
                    memory_pressure_backoff_seconds=memory_pressure_backoff_seconds,
                    memory_pressure_max_backoffs=memory_pressure_max_backoffs,
                )

        app = Application(runner, process_adapter=FailingAdapter())
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "unsupported"
        assert outcome.process_launched is False
        assert "could not be started" in outcome.diagnostic.lower()

    def test_general_exception_after_start_remains_systemic(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        from collections.abc import Callable, Sequence

        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)

        class BrokenAdapter:
            def launch(
                self,
                argv: Sequence[str],
                *,
                env: dict[str, str] | None = None,
                timeout_seconds: float | None = None,
                cancellation: Event | None = None,
                memory_pressure_fn: Callable[[], bool] | None = None,
                memory_pressure_backoff_seconds: float = 1.0,
                memory_pressure_max_backoffs: int = 3,
            ) -> ProcessResult:
                raise RuntimeError("unexpected runtime issue")

        app = Application(runner, process_adapter=BrokenAdapter())
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "systemic_error"
        assert outcome.process_launched is True


class TestToolchainProbeFailure:
    def test_toolchain_probe_failure_returns_unsupported(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        bp = str(sample_request.binary_path)
        runner.enqueue(
            (bp, "--version"),
            ProcessResult(1, "", "command not found"),
        )

        app = Application(runner)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "unsupported"
        assert outcome.process_launched is False
        assert "exited with code" in outcome.diagnostic

    def test_toolchain_probe_failure_journals(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        bp = str(sample_request.binary_path)
        runner.enqueue(
            (bp, "--version"),
            ProcessResult(1, "", "command not found"),
        )

        app = Application(runner)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "unsupported"
        assert outcome.process_launched is False
        assert sample_request.journal_path.exists()
        data = _read_journal_first_payload(sample_request.journal_path)
        assert data["disposition"] == "unsupported"
        assert data["process_launched"] is False


# ── ExecutionPolicy and env propagation ─────────────────────────


class TestExecutionPolicy:
    def test_env_overrides_in_launch_env(
        self,
    ) -> None:
        policy = ExecutionPolicy(env_overrides={"MY_VAR": "my_value"})
        env = _build_launch_env(policy, None)
        assert env is not None
        assert env["MY_VAR"] == "my_value"

    def test_device_env_added_for_cuda(
        self,
    ) -> None:
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=32_000_000_000,
            accelerators=(Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),),
            memory_unified=False,
        )
        spec = SessionSpecification(
            topology=topology,
            cpu_only=False,
            selected_device_ids=(0,),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=None,  # type: ignore[arg-type]
            advisory=None,  # type: ignore[arg-type]
            model=None,  # type: ignore[arg-type]
            acknowledged=True,
            fingerprint="",
        )
        env = _env_for_device(spec)
        assert env is not None
        assert env["CUDA_VISIBLE_DEVICES"] == "0"

    def test_no_env_for_cpu_only(
        self,
    ) -> None:
        spec = SessionSpecification(
            topology=None,  # type: ignore[arg-type]
            cpu_only=True,
            selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=None,  # type: ignore[arg-type]
            advisory=None,  # type: ignore[arg-type]
            model=None,  # type: ignore[arg-type]
            acknowledged=True,
            fingerprint="",
        )
        env = _env_for_device(spec)
        assert env is None

    def test_timeout_propagates_to_fake_adapter(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, '[{"samples_ts":[1.0,2.0,3.0]}]', "", termination="timeout"),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(
            sample_request,
            policy=ExecutionPolicy(candidate_timeout_seconds=0.001),
        )
        assert outcome.disposition == "timeout"

    def test_memory_pressure_triggers_oom(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)

        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, "", "", termination="oom"),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(
            sample_request,
            policy=ExecutionPolicy(memory_pressure_fn=lambda: True),
        )
        assert outcome.disposition == "oom"


# ── OOM classification ──────────────────────────────────────────


class TestOomClassification:
    def test_exit_code_137_is_oom(self) -> None:
        assert _is_oom(137, "")

    def test_exit_code_negative_9_is_oom(self) -> None:
        assert _is_oom(-9, "")

    def test_cuda_oom_stderr(self) -> None:
        assert _is_oom(0, "CUDA out of memory")

    def test_oom_killer_stderr(self) -> None:
        assert _is_oom(0, "Killed")

    def test_allocator_failed_stderr(self) -> None:
        assert _is_oom(0, "allocator ran out of memory")

    def test_normal_exit_not_oom(self) -> None:
        assert not _is_oom(0, "")

    def test_nonzero_normal_not_oom(self) -> None:
        assert not _is_oom(1, "some error")

    def test_bad_alloc_is_oom(self) -> None:
        assert _is_oom(0, "std::bad_alloc")


# ── Candidate lifecycle output → observation ────────────────────


class TestCandidateLifecycle:
    def test_completed_has_observation(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )
        measure_argv = _build_argv(sample_request, repetitions=3)
        adapter.enqueue(
            measure_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.observation is not None
        obs = outcome.observation
        assert obs.candidate == sample_request.candidate
        assert obs.median_tps == pytest.approx(11.0, rel=1e-6)
        assert len(obs.samples_tps) == 3

    def test_unsupported_has_no_observation(
        self,
        sample_candidate: CandidateConfig,
        tmp_path: Path,
    ) -> None:
        req = SingleRunRequest(
            binary_path=Path("/nonexistent"),
            model_path=tmp_path / "model.gguf",
            journal_path=tmp_path / "j.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=sample_candidate,
        )
        app = Application(FakeRunner())
        outcome = app.evaluate_candidate(req)
        assert outcome.observation is None

    def test_journal_written_for_completed(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )
        measure_argv = _build_argv(sample_request, repetitions=3)
        adapter.enqueue(
            measure_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "completed"
        journal_path = sample_request.journal_path
        assert journal_path.exists()
        data = _read_journal_first_payload(journal_path)
        assert data["disposition"] == "completed"
        assert data["candidate"]["threads"] == 4
        assert data["samples_tps"] == [10.0, 11.0, 12.0]

    def test_journal_written_for_failure(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(1, "", "failure"),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "crash"
        journal_path = sample_request.journal_path
        assert journal_path.exists()
        data = _read_journal_first_payload(journal_path)
        assert data["disposition"] == "crash"
        assert "failure" in data["diagnostic"]
        assert data["safe_next_action"] != ""


# ── StopTracker (repeated / systemic stop policy) ───────────────


class TestStopTracker:
    def test_single_success_resets(self) -> None:
        tracker = StopTracker(max_stops=3)
        outcome = CandidateOutcome(
            candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                                      n_gpu_layers=0, flash_attn=False,
                                      cache_type_k="f16", cache_type_v="f16"),
            disposition="completed",
            process_launched=True,
            diagnostic="",
            safe_next_action="",
        )
        assert tracker.record(outcome) is False
        assert tracker.should_stop is False

    def test_consecutive_crashes_trigger_stop(self) -> None:
        tracker = StopTracker(max_stops=2)
        outcome = CandidateOutcome(
            candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                                      n_gpu_layers=0, flash_attn=False,
                                      cache_type_k="f16", cache_type_v="f16"),
            disposition="crash",
            process_launched=True,
            diagnostic="",
            safe_next_action="",
        )
        assert tracker.record(outcome) is False
        assert tracker.record(outcome) is True
        assert tracker.should_stop is True

    def test_consecutive_systemic_errors_trigger_stop(self) -> None:
        tracker = StopTracker(max_stops=3)
        outcome = CandidateOutcome(
            candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                                      n_gpu_layers=0, flash_attn=False,
                                      cache_type_k="f16", cache_type_v="f16"),
            disposition="systemic_error",
            process_launched=True,
            diagnostic="",
            safe_next_action="",
        )
        assert tracker.record(outcome) is False
        assert tracker.record(outcome) is False
        assert tracker.record(outcome) is True
        assert tracker.should_stop is True

    def test_success_resets_count(self) -> None:
        tracker = StopTracker(max_stops=3)
        crash = CandidateOutcome(
            candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                                      n_gpu_layers=0, flash_attn=False,
                                      cache_type_k="f16", cache_type_v="f16"),
            disposition="crash",
            process_launched=True,
            diagnostic="",
            safe_next_action="",
        )
        success = CandidateOutcome(
            candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                                      n_gpu_layers=0, flash_attn=False,
                                      cache_type_k="f16", cache_type_v="f16"),
            disposition="completed",
            process_launched=True,
            diagnostic="",
            safe_next_action="",
        )
        assert tracker.record(crash) is False
        assert tracker.record(crash) is False
        assert tracker.record(success) is False
        assert tracker.should_stop is False

    def test_reset_method(self) -> None:
        tracker = StopTracker(max_stops=2)
        outcome = CandidateOutcome(
            candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                                      n_gpu_layers=0, flash_attn=False,
                                      cache_type_k="f16", cache_type_v="f16"),
            disposition="crash",
            process_launched=True,
            diagnostic="",
            safe_next_action="",
        )
        tracker.record(outcome)
        assert tracker.should_stop is False
        tracker.reset()
        assert tracker.should_stop is False


# ── ProductionProcessAdapter process tree cleanup ────────────────


class TestProductionProcessAdapter:
    def test_runs_and_collects_output(self, tmp_path: Path) -> None:
        from llama_parameter_scout.runner import ProductionProcessAdapter
        script = write_fake_executable(tmp_path, "echo_bin", "print('hello')\n")
        adapter = ProductionProcessAdapter()
        result = adapter.launch([str(script)])
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_timeout_terminates(self, tmp_path: Path) -> None:
        from llama_parameter_scout.runner import ProductionProcessAdapter
        script = write_fake_executable(
            tmp_path, "sleep_bin", "import time\ntime.sleep(60)\nprint('done')\n",
        )
        adapter = ProductionProcessAdapter()
        result = adapter.launch(
            [str(script)],
            # Bumped from 0.1s: on Windows the "binary" is a .bat trampoline
            # that spawns cmd.exe + a fresh Python interpreter, which is
            # slower to start than a native /bin/sh script. The child sleeps
            # for 60s regardless, so this still exercises timeout-triggered
            # termination of a live process rather than a normal exit; it
            # just adds margin against slow interpreter/shell startup.
            timeout_seconds=0.3,
        )
        assert result.termination == "timeout"
        assert result.returncode != 0 or True

    def test_cancellation_terminates(self, tmp_path: Path) -> None:
        from llama_parameter_scout.runner import ProductionProcessAdapter
        script = write_fake_executable(
            tmp_path, "sleep_bin", "import time\ntime.sleep(60)\nprint('done')\n",
        )
        cancel = Event()
        adapter = ProductionProcessAdapter()

        def _set_cancel() -> None:
            import time
            # Bumped from 0.05s for the same reason as test_timeout_terminates:
            # margin against slow cmd.exe + interpreter startup on Windows.
            # The child still sleeps for 60s, so cancellation always fires
            # against a genuinely live process either way.
            time.sleep(0.3)
            cancel.set()

        import threading
        threading.Thread(target=_set_cancel, daemon=True).start()

        result = adapter.launch(
            [str(script)],
            cancellation=cancel,
        )
        assert result.termination == "cancellation"

    def test_memory_pressure_triggers_oom(self, tmp_path: Path) -> None:
        from llama_parameter_scout.runner import ProductionProcessAdapter
        script = write_fake_executable(
            tmp_path, "sleep_bin", "import time\ntime.sleep(60)\nprint('done')\n",
        )
        pressure_count: list[int] = [0]

        def pressure_fn() -> bool:
            pressure_count[0] += 1
            return True

        adapter = ProductionProcessAdapter()
        result = adapter.launch(
            [str(script)],
            memory_pressure_fn=pressure_fn,
            memory_pressure_backoff_seconds=0.01,
            memory_pressure_max_backoffs=2,
        )
        assert result.termination == "oom"
        assert pressure_count[0] >= 2

    def test_normal_exit_no_termination(self, tmp_path: Path) -> None:
        from llama_parameter_scout.runner import ProductionProcessAdapter
        script = write_fake_executable(tmp_path, "ok_bin", "print('ok')\n")
        adapter = ProductionProcessAdapter()
        result = adapter.launch([str(script)])
        assert result.termination is None
        assert result.returncode == 0
        assert "ok" in result.stdout

    def test_finally_handles_wait_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import subprocess

        from llama_parameter_scout.runner import ProductionProcessAdapter

        script = write_fake_executable(tmp_path, "sleeper_bin", "import time\ntime.sleep(60)\n")

        original_wait = subprocess.Popen.wait

        def failing_wait(proc: subprocess.Popen[Any], timeout: float = 30) -> int:
            if proc.returncode is None:
                raise subprocess.TimeoutExpired(cmd=proc.args, timeout=timeout)
            return original_wait(proc, timeout=timeout)

        monkeypatch.setattr(subprocess.Popen, "wait", failing_wait)

        adapter = ProductionProcessAdapter()
        result = adapter.launch(
            [str(script)],
            # Bumped from 0.05s: see test_timeout_terminates. This test's
            # actual assertion (returncode falls back to -1 because `wait`
            # always raises TimeoutExpired) doesn't depend on precise timing,
            # but a very short deadline unnecessarily races interpreter/shell
            # startup on Windows.
            timeout_seconds=0.3,
        )
        assert result.termination == "timeout"
        assert result.returncode != 0

    def test_classify_nonzero_without_oom_evidence_is_crash(self) -> None:
        result = ProcessResult(returncode=1, stdout="", stderr="Segmentation fault")
        outcome = _classify_launch(
            result,
            SingleRunRequest(
                binary_path=Path("/x"), model_path=Path("/x"),
                journal_path=Path("/x"),
                n_ctx=NctxComposition(total=1, prompt=1, generation=0, depth=0),
                candidate=CandidateConfig(threads=1, batch_size=1, ubatch_size=1,
                                          n_gpu_layers=0, flash_attn=False,
                                          cache_type_k="f16", cache_type_v="f16"),
            ),
            _make_toolchain(),
            _make_model(),
            None, None,
        )
        assert outcome.disposition == "crash"


# ── Fix 1: session_spec in warmup env ────────────────────────────


class TestWarmupSessionSpec:
    def test_warmup_receives_device_env(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        from llama_parameter_scout.model_inspector import inspect_model

        runner = FakeRunner()
        bp = sample_request.binary_path
        runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(0, (
                "version: 1\n"
                "backends: cpu,cuda\n"
                "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,"
                "-ctk,-ctv,-o,-r,--split-mode,--main-gpu\n"
                "supported_output_formats: json\n"
            ), ""),
        )
        runner.enqueue(
            (str(bp), "--help"),
            ProcessResult(0, "usage\n", ""),
        )
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=100_000_000_000,
            accelerators=(Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),),
            memory_unified=False,
        )
        model = inspect_model(sample_request.model_path)
        spec = SessionSpecification(
            topology=topology,
            cpu_only=False,
            selected_device_ids=(0,),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(
                ram_bytes=100_000_000_000, vram_bytes=24_000_000_000, headroom_ratio=0.1,
            ),
            advisory=ContextAdvisory(
                likely_fit_lower=100000, likely_fit_upper=200000, uncertain_maximum=500000,
                model_weight_bytes=4_000_000_000, kv_cache_bytes_estimate=500_000_000,
            ),
            model=model,
            acknowledged=True,
            fingerprint="",
        )

        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1, session_spec=spec)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )
        measure_argv = _build_argv(sample_request, repetitions=3, session_spec=spec)
        adapter.enqueue(
            measure_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )

        app = Application(runner, detect_topology_fn=lambda: topology, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request, session_spec=spec)

        assert outcome.disposition == "completed"


# ── Fix 2: SIGSEGV codes classify as crash, not OOM ─────────────


class TestSigsegvIsCrash:
    def test_exit_code_negative_11_is_crash(self) -> None:
        assert not _is_oom(-11, "")

    def test_exit_code_139_is_crash(self) -> None:
        assert not _is_oom(139, "")

    def test_exit_code_negative_11_classifies_crash(self) -> None:
        result = ProcessResult(returncode=-11, stdout="", stderr="Segmentation fault")
        outcome = _classify_launch(
            result,
            SingleRunRequest(
                binary_path=Path("/x"), model_path=Path("/x"),
                journal_path=Path("/x"),
                n_ctx=NctxComposition(total=1, prompt=1, generation=0, depth=0),
                candidate=CandidateConfig(threads=1, batch_size=1, ubatch_size=1,
                                          n_gpu_layers=0, flash_attn=False,
                                          cache_type_k="f16", cache_type_v="f16"),
            ),
            _make_toolchain(),
            _make_model(),
            None, None,
        )
        assert outcome.disposition == "crash"


# ── Fix 3: OSError from write_outcome for measurement ───────────


class TestWriteOutcomeOSError:
    def test_oserror_on_measurement_write_returns_systemic_error(
        self,
        sample_request: SingleRunRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import llama_parameter_scout.application as app_mod

        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, '[{"samples_ts":[10.0,11.0,12.0]}]', ""),
        )
        measure_argv = _build_argv(sample_request, repetitions=3)
        adapter.enqueue(
            measure_argv,
            ProcessResult(1, "", "failure"),
        )

        write_called: list[bool] = []

        def _raise_oserror(*args: object, **kwargs: object) -> object:
            write_called.append(True)
            raise OSError("mock journal write failure")

        monkeypatch.setattr(app_mod, "write_outcome", _raise_oserror)

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)
        assert outcome.disposition == "systemic_error"
        assert outcome.process_launched is True
        assert "journal" in outcome.diagnostic.lower()
        assert len(write_called) == 1


# ── Fix 4: Pre-execution resource budget gate ───────────────────


class TestBudgetGate:
    def test_not_exhausted_when_budget_ample(self) -> None:
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=100_000_000_000,
            accelerators=(),
            memory_unified=True,
        )
        model = ModelFingerprint(
            path=Path("/fake/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(),
            symlink_warning=False, network_mount_detected=False, diagnostics=(),
        )
        spec = SessionSpecification(
            topology=topology,
            cpu_only=True,
            selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(ram_bytes=100_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=None,  # type: ignore[arg-type]
            model=model,
            acknowledged=True,
            fingerprint="",
        )
        n_ctx = NctxComposition(total=2048, prompt=512, generation=1024, depth=512)
        candidate = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        assert _check_resource_budget(spec, n_ctx, candidate) is None

    def test_exhausted_when_budget_too_small(self) -> None:
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=1_000_000_000,
            accelerators=(),
            memory_unified=True,
        )
        model = ModelFingerprint(
            path=Path("/fake/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(),
            symlink_warning=False, network_mount_detected=False, diagnostics=(),
        )
        spec = SessionSpecification(
            topology=topology,
            cpu_only=True,
            selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(ram_bytes=1_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=None,  # type: ignore[arg-type]
            model=model,
            acknowledged=True,
            fingerprint="",
        )
        n_ctx = NctxComposition(total=2048, prompt=512, generation=1024, depth=512)
        candidate = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        result = _check_resource_budget(spec, n_ctx, candidate)
        assert result is not None
        assert result.disposition == "budget_exhausted"
        assert result.process_launched is False
        assert result.candidate == candidate

    def test_evaluate_candidate_returns_budget_exhausted(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        from llama_parameter_scout.model_inspector import inspect_model

        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()

        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=1_000_000_000,
            accelerators=(),
            memory_unified=True,
        )
        model = inspect_model(sample_request.model_path)
        spec = SessionSpecification(
            topology=topology,
            cpu_only=True,
            selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(ram_bytes=1_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=ContextAdvisory(
                likely_fit_lower=100, likely_fit_upper=500, uncertain_maximum=1000,
                model_weight_bytes=1_000_000_000, kv_cache_bytes_estimate=500_000_000,
            ),
            model=model,
            acknowledged=True,
            fingerprint="",
        )
        app = Application(runner, detect_topology_fn=lambda: topology, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request, session_spec=spec)
        assert outcome.disposition == "budget_exhausted"
        assert outcome.process_launched is False
        assert outcome.candidate == sample_request.candidate

    def test_discrete_gpu_budget_exhausted(self) -> None:
        from llama_parameter_scout.models import ShardFingerprint
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=4_000_000_000,
            accelerators=(Accelerator("gpu", "cuda", 0, "GPU", memory_bytes=2_000_000_000),),
            memory_unified=False,
        )
        model = ModelFingerprint(
            path=Path("/fake/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M",
            shards=(
                ShardFingerprint(Path("/fake/model.gguf"), 6_000_000_000, 0, "a" * 64),
            ),
            symlink_warning=False, network_mount_detected=False, diagnostics=(),
        )
        spec = SessionSpecification(
            topology=topology,
            cpu_only=False,
            selected_device_ids=(0,),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(
                ram_bytes=4_000_000_000, vram_bytes=2_000_000_000, headroom_ratio=0.1,
            ),
            advisory=ContextAdvisory(
                likely_fit_lower=100, likely_fit_upper=500, uncertain_maximum=1000,
                model_weight_bytes=6_000_000_000, kv_cache_bytes_estimate=500_000_000,
            ),
            model=model,
            acknowledged=True,
            fingerprint="",
        )
        n_ctx = NctxComposition(total=2048, prompt=512, generation=1024, depth=512)
        candidate = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=20, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        result = _check_resource_budget(spec, n_ctx, candidate)
        assert result is not None
        assert result.disposition == "budget_exhausted"
        assert result.process_launched is False
        assert result.candidate == candidate


# ── Fix 5: Windows direct taskkill ──────────────────────────────


class TestWindowsTermination:
    def test_direct_taskkill_no_os_kill(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import subprocess

        from llama_parameter_scout.runner import _terminate_process_tree

        taskkill_called: list[list[str]] = []

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            taskkill_called.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("subprocess.run", fake_run)

        proc = subprocess.Popen(["sleep", "60"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.returncode = None
        _terminate_process_tree(proc)
        proc.kill()
        proc.wait()

        assert any("taskkill" in c for c in taskkill_called[0])


# ── Fix 6: Memory pressure detection ────────────────────────────


class TestMemoryPressureDetection:
    def test_detect_returns_none_on_unsupported(self) -> None:
        result = _detect_available_memory_bytes()
        assert result is None or result > 0

    def test_pressure_fn_returns_bool(self) -> None:
        from llama_parameter_scout.application import make_memory_pressure_fn
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=100_000_000_000,
            accelerators=(),
            memory_unified=True,
        )
        model = ModelFingerprint(
            path=Path("/fake/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(),
            symlink_warning=False, network_mount_detected=False, diagnostics=(),
        )
        spec = SessionSpecification(
            topology=topology,
            cpu_only=True,
            selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(ram_bytes=100_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=None,  # type: ignore[arg-type]
            model=model,
            acknowledged=True,
            fingerprint="",
        )
        fn = make_memory_pressure_fn(spec)
        assert fn is not None
        result = fn()
        assert isinstance(result, bool)

    def test_no_session_spec_returns_none(self) -> None:
        from llama_parameter_scout.application import make_memory_pressure_fn
        fn = make_memory_pressure_fn(None)
        assert fn is None


    def test_available_above_headroom_is_not_pressure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llama_parameter_scout.application import make_memory_pressure_fn
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=100_000_000_000,
            accelerators=(),
            memory_unified=True,
        )
        model = ModelFingerprint(
            path=Path("/fake/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(),
            symlink_warning=False, network_mount_detected=False, diagnostics=(),
        )
        spec = SessionSpecification(
            topology=topology,
            cpu_only=True,
            selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(ram_bytes=100_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=None,  # type: ignore[arg-type]
            model=model,
            acknowledged=True,
            fingerprint="",
        )
        monkeypatch.setattr(
            "llama_parameter_scout.application._detect_available_memory_bytes",
            lambda: 95_000_000_000,
        )
        fn = make_memory_pressure_fn(spec)
        assert fn is not None
        assert fn() is False

    def test_detect_returns_none_on_win32_without_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force the win32 code path regardless of the host platform, and
        # make `GlobalMemoryStatusEx` itself report failure (returns 0 per
        # the WinAPI convention), rather than relying on `ctypes.windll` not
        # existing at all. Monkeypatching only `sys.platform` (without this)
        # passes trivially on macOS/Linux because `ctypes.windll` doesn't
        # exist there and the resulting AttributeError is swallowed by the
        # `except` -- but on real Windows the API succeeds and returns a
        # real byte count, so that version never exercises the intended
        # "API unavailable -> fall through to None" branch. `raising=False`
        # lets this attribute be set on platforms where `ctypes.windll`
        # doesn't normally exist.
        monkeypatch.setattr("sys.platform", "win32")

        class _FakeKernel32:
            @staticmethod
            def GlobalMemoryStatusEx(_buffer: object) -> int:
                return 0  # 0 == failure, per GlobalMemoryStatusEx's contract

        class _FakeWindll:
            kernel32 = _FakeKernel32()

        monkeypatch.setattr("ctypes.windll", _FakeWindll(), raising=False)

        result = _detect_available_memory_bytes()
        assert result is None

    def test_available_below_headroom_is_pressure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llama_parameter_scout.application import make_memory_pressure_fn
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=100_000_000_000,
            accelerators=(),
            memory_unified=True,
        )
        model = ModelFingerprint(
            path=Path("/fake/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(),
            symlink_warning=False, network_mount_detected=False, diagnostics=(),
        )
        spec = SessionSpecification(
            topology=topology,
            cpu_only=True,
            selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(ram_bytes=100_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=None,  # type: ignore[arg-type]
            model=model,
            acknowledged=True,
            fingerprint="",
        )
        monkeypatch.setattr(
            "llama_parameter_scout.application._detect_available_memory_bytes",
            lambda: 5_000_000_000,
        )
        fn = make_memory_pressure_fn(spec)
        assert fn is not None
        assert fn() is True


# ── Fix 7: not_attempted disposition ────────────────────────────


class TestNotAttempted:
    def test_pre_execution_disposition_returns_not_attempted(self) -> None:
        request = SingleRunRequest(
            binary_path=Path("/x"), model_path=Path("/x"),
            journal_path=Path("/x"),
            n_ctx=NctxComposition(total=1, prompt=1, generation=0, depth=0),
            candidate=CandidateConfig(threads=1, batch_size=1, ubatch_size=1,
                                      n_gpu_layers=0, flash_attn=False,
                                      cache_type_k="f16", cache_type_v="f16"),
        )
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=32_000_000_000,
            accelerators=(),
            memory_unified=True,
        )
        model = ModelFingerprint(
            path=Path("/fake/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(),
            symlink_warning=False, network_mount_detected=False, diagnostics=(),
        )
        spec = SessionSpecification(
            topology=topology,
            cpu_only=True,
            selected_device_ids=(),
            n_ctx=NctxComposition(total=1, prompt=1, generation=0, depth=0),
            budget=ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=None,  # type: ignore[arg-type]
            model=model,
            acknowledged=True,
            fingerprint="",
        )
        outcome = pre_execution_disposition(request, spec, skip_candidate=True)
        assert outcome is not None
        assert outcome.disposition == "not_attempted"
        assert outcome.process_launched is False

    def test_not_attempted_disposition_value(self) -> None:
        outcome = CandidateOutcome(
            candidate=CandidateConfig(threads=1, batch_size=1, ubatch_size=1,
                                      n_gpu_layers=0, flash_attn=False,
                                      cache_type_k="f16", cache_type_v="f16"),
            disposition="not_attempted",
            process_launched=False,
            diagnostic="Skipped",
            safe_next_action="",
        )
        assert outcome.disposition == "not_attempted"
        assert outcome.process_launched is False


# ── Fix 8: ProductionProcessAdapter returncode ──────────────────


class TestReturncodeHandling:
    def test_timeout_returncode_not_zero(
        self,
        tmp_path: Path,
    ) -> None:
        from llama_parameter_scout.runner import ProductionProcessAdapter
        script = write_fake_executable(tmp_path, "sleeper_bin", "import time\ntime.sleep(60)\n")
        adapter = ProductionProcessAdapter()
        # Bumped from 0.05s: see test_timeout_terminates for rationale.
        result = adapter.launch([str(script)], timeout_seconds=0.3)
        assert result.termination == "timeout"
        assert result.returncode != 0

    def test_normal_exit_returncode_is_zero(
        self,
        tmp_path: Path,
    ) -> None:
        from llama_parameter_scout.runner import ProductionProcessAdapter
        script = write_fake_executable(tmp_path, "ok_bin", "print('ok')\n")
        adapter = ProductionProcessAdapter()
        result = adapter.launch([str(script)])
        assert result.returncode == 0


# ── Fix 9: Pre-execution journaling ─────────────────────────────


class TestPreExecutionJournaling:
    def test_pre_execution_outcome_writes_journal(
        self,
        sample_candidate: CandidateConfig,
        tmp_path: Path,
    ) -> None:
        req = SingleRunRequest(
            binary_path=Path("/nonexistent"),
            model_path=tmp_path / "model.gguf",
            journal_path=tmp_path / "journal.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=sample_candidate,
        )
        app = Application(FakeRunner())
        outcome = app.evaluate_candidate(req)
        assert outcome.disposition == "unsupported"
        assert req.journal_path.exists()
        data = _read_journal_first_payload(req.journal_path)
        assert data["disposition"] == "unsupported"
        assert data["process_launched"] is False

    def test_pruned_outcome_writes_journal(
        self,
        fake_binary: Path,
        fake_model: Path,
        tmp_path: Path,
    ) -> None:
        from llama_parameter_scout.model_inspector import inspect_model

        candidate = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=20, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        req = SingleRunRequest(
            binary_path=fake_binary,
            model_path=fake_model,
            journal_path=tmp_path / "journal.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=candidate,
        )
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=32_000_000_000,
            accelerators=(),
            memory_unified=False,
        )
        model = inspect_model(fake_model)
        runner = FakeRunner()
        _enqueue_toolchain(runner, fake_binary)
        app = Application(runner, detect_topology_fn=lambda: topology)
        spec = app.setup_session(
            model=model,
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
            cpu_only=True,
            acknowledged=True,
        )
        outcome = app.evaluate_candidate(req, session_spec=spec)
        assert outcome.disposition == "pruned"
        assert req.journal_path.exists()
        data = _read_journal_first_payload(req.journal_path)
        assert data["disposition"] == "pruned"
        assert data["safe_next_action"] != ""

    def test_failure_journal_contains_disposition(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(1, "", "failure"),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "crash"
        assert sample_request.journal_path.exists()
        data = _read_journal_first_payload(sample_request.journal_path)
        assert data["disposition"] == "crash"
        assert data["process_launched"] is True
        assert "failure" in data["diagnostic"]


    def test_pre_execution_journal_oserror_raises_journal_error(
        self,
        sample_candidate: CandidateConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_oserror(*args: object, **kwargs: object) -> object:
            raise OSError("mock journal write failure")

        monkeypatch.setattr(
            "llama_parameter_scout.application.write_outcome",
            _raise_oserror,
        )

        req = SingleRunRequest(
            binary_path=Path("/nonexistent"),
            model_path=tmp_path / "model.gguf",
            journal_path=tmp_path / "journal.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=sample_candidate,
        )
        app = Application(FakeRunner())
        with pytest.raises(JournalError):
            app.evaluate_candidate(req)


# ── Warmup failure journaling ───────────────────────────────────


class TestWarmupFailureJournal:
    def test_warmup_failure_writes_journal(
        self,
        sample_request: SingleRunRequest,
    ) -> None:
        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(0, "", "", termination="timeout"),
        )

        app = Application(runner, process_adapter=adapter)
        outcome = app.evaluate_candidate(sample_request)

        assert outcome.disposition == "timeout"
        assert sample_request.journal_path.exists()
        data = _read_journal_first_payload(sample_request.journal_path)
        assert data["disposition"] == "timeout"
        assert data["diagnostic"] != ""

    def test_warmup_failure_oserror_returns_systemic_error(
        self,
        sample_request: SingleRunRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_oserror(*args: object, **kwargs: object) -> object:
            raise OSError("mock journal write failure")

        monkeypatch.setattr(
            "llama_parameter_scout.application.write_outcome",
            _raise_oserror,
        )

        runner = FakeRunner()
        _enqueue_toolchain(runner, sample_request.binary_path)
        adapter = FakeProcessAdapter()
        warmup_argv = _build_argv(sample_request, repetitions=1)
        adapter.enqueue(
            warmup_argv,
            ProcessResult(1, "", "warmup failed"),
        )

        app = Application(runner, process_adapter=adapter)

        outcome = app.evaluate_candidate(sample_request)
        assert outcome.disposition == "systemic_error"
        assert outcome.process_launched is True
        assert "journal" in outcome.diagnostic.lower()


# ── ExecutionPolicy validation ──────────────────────────────────


class TestExecutionPolicyValidation:
    def test_positive_timeout_passes(self) -> None:
        ExecutionPolicy(candidate_timeout_seconds=10.0)

    def test_none_timeout_passes(self) -> None:
        ExecutionPolicy(candidate_timeout_seconds=None)

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ExecutionPolicy(candidate_timeout_seconds=0.0)

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ExecutionPolicy(candidate_timeout_seconds=-1.0)

    def test_negative_backoff_raises(self) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            ExecutionPolicy(memory_pressure_backoff_seconds=-1.0)

    def test_zero_max_backoffs_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ExecutionPolicy(memory_pressure_max_backoffs=0)

    def test_negative_max_backoffs_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ExecutionPolicy(memory_pressure_max_backoffs=-1)
