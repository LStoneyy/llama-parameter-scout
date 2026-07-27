from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from llama_parameter_scout.application import (
    Application,
    _build_argv,
    _compute_stats,
    _parse_measurements,
    _validate_request,
)
from llama_parameter_scout.models import (
    Accelerator,
    BinaryNotFoundError,
    CandidateConfig,
    CandidateError,
    InvalidNctxError,
    JournalError,
    MalformedOutputError,
    ModelFingerprint,
    ModelFingerprintMismatchError,
    ModelNotFoundError,
    NctxComposition,
    ResourceBudget,
    SessionSetupError,
    SessionSpecification,
    SingleRunRequest,
    SubprocessError,
    SystemTopology,
    ToolchainError,
    ToolchainFingerprint,
)
from llama_parameter_scout.runner import FakeRunner, ProcessResult, ProcessStartError


def _enqueue_toolchain_with_gpu_flags(
    runner: FakeRunner,
    bp: Path,
    backends: tuple[str, ...] = ("cpu",),
) -> None:
    runner.enqueue(
        (str(bp), "--version"),
        ProcessResult(
            returncode=0,
            stdout=(
                "version: test123\n"
                f"backends: {','.join(backends)}\n"
                "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r,"
                "--split-mode,--main-gpu\n"
                "supported_output_formats: json\n"
            ),
            stderr="",
        ),
    )
    runner.enqueue(
        (str(bp), "--help"),
        ProcessResult(returncode=0, stdout="usage: llama-bench\n", stderr=""),
    )


def _enqueue_toolchain(runner: FakeRunner, bp: Path) -> None:
    runner.enqueue(
        (str(bp), "--version"),
        ProcessResult(
            returncode=0,
            stdout=(
                "version: test123\n"
                "backends: cpu\n"
                "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
                "supported_output_formats: json\n"
            ),
            stderr="",
        ),
    )
    runner.enqueue(
        (str(bp), "--help"),
        ProcessResult(returncode=0, stdout="usage: llama-bench\n", stderr=""),
    )


class TestBuildArgv:
    def test_builds_correct_args(self, valid_request: SingleRunRequest) -> None:
        argv = _build_argv(valid_request, repetitions=3)
        assert argv[0] == str(valid_request.binary_path)
        pairs = dict(zip(argv[1::2], argv[2::2]))
        assert pairs["-m"] == str(valid_request.model_path)
        assert pairs["-p"] == "512"
        assert pairs["-n"] == "1024"
        assert pairs["-d"] == "512"
        assert pairs["-t"] == "4"
        assert pairs["-b"] == "512"
        assert pairs["-ub"] == "512"
        assert pairs["-ngl"] == "0"
        assert pairs["-fa"] == "off"
        assert pairs["-ctk"] == "f16"
        assert pairs["-ctv"] == "f16"
        assert pairs["-o"] == "json"
        assert pairs["-r"] == "3"
        assert "-c" not in argv

    def test_flash_attn_true(self, valid_request: SingleRunRequest) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512, n_gpu_layers=0,
                flash_attn=True, cache_type_k="f16", cache_type_v="f16",
            ),
        )
        argv = _build_argv(req, repetitions=3)
        pairs = dict(zip(argv[1::2], argv[2::2]))
        assert pairs["-fa"] == "on"


class TestValidateRequest:
    def test_valid(self, valid_request: SingleRunRequest) -> None:
        _validate_request(valid_request)

    def test_missing_binary(self, valid_request: SingleRunRequest) -> None:
        req = SingleRunRequest(
            binary_path=Path("/nonexistent/binary"),
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=valid_request.candidate,
        )
        with pytest.raises(BinaryNotFoundError, match="Binary not found"):
            _validate_request(req)

    def test_binary_not_file(self, tmp_path: Path, valid_request: SingleRunRequest) -> None:
        d = tmp_path / "dir"
        d.mkdir()
        req = SingleRunRequest(
            binary_path=d,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=valid_request.candidate,
        )
        with pytest.raises(BinaryNotFoundError, match="not a file"):
            _validate_request(req)

    def test_binary_not_executable(self, tmp_path: Path, valid_request: SingleRunRequest) -> None:
        p = tmp_path / "llama-bench"
        p.write_text("data")
        p.chmod(0o644)
        req = SingleRunRequest(
            binary_path=p,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=valid_request.candidate,
        )
        with pytest.raises(BinaryNotFoundError, match="not executable"):
            _validate_request(req)

    def test_missing_model(self, valid_request: SingleRunRequest) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=Path("/nonexistent/model.gguf"),
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=valid_request.candidate,
        )
        with pytest.raises(ModelNotFoundError, match="Model not found"):
            _validate_request(req)

    def test_nctx_total_zero(self, valid_request: SingleRunRequest) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=NctxComposition(total=0, prompt=0, generation=0, depth=0),
            candidate=valid_request.candidate,
        )
        with pytest.raises(InvalidNctxError, match="total must be > 0"):
            _validate_request(req)

    def test_nctx_negative_component(self, valid_request: SingleRunRequest) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=NctxComposition(total=2048, prompt=-1, generation=1024, depth=1025),
            candidate=valid_request.candidate,
        )
        with pytest.raises(InvalidNctxError, match="must be >= 0"):
            _validate_request(req)

    def test_nctx_components_dont_sum(self, valid_request: SingleRunRequest) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=NctxComposition(total=2048, prompt=500, generation=1000, depth=500),
            candidate=valid_request.candidate,
        )
        with pytest.raises(InvalidNctxError, match="do not sum"):
            _validate_request(req)

    def test_candidate_threads_zero(self, valid_request: SingleRunRequest) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=CandidateConfig(
                threads=0, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
        )
        with pytest.raises(CandidateError, match="threads"):
            _validate_request(req)

    def test_candidate_n_gpu_layers_too_low(self, valid_request: SingleRunRequest) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=-2, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
        )
        with pytest.raises(CandidateError, match="n_gpu_layers"):
            _validate_request(req)


class TestParseMeasurements:
    def test_json_array(self, valid_json_output: str) -> None:
        samples = _parse_measurements(valid_json_output)
        assert samples == [45.5, 44.8, 45.3]

    def test_jsonl(self) -> None:
        output = (
            '{"n_ctx":2048,"samples_ts":[45.5,44.8,45.3]}\n'
            '{"n_ctx":4096,"samples_ts":[40.0,39.5,40.2]}\n'
        )
        samples = _parse_measurements(output)
        assert samples == [45.5, 44.8, 45.3]

    def test_single_object(self) -> None:
        output = '{"n_ctx":2048,"samples_ts":[10.0,11.0,12.0]}'
        samples = _parse_measurements(output)
        assert samples == [10.0, 11.0, 12.0]

    def test_malformed(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements("not json")

    def test_no_samples_key(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements('[{"n_ctx":2048}]')

    def test_fewer_than_3_samples(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements('[{"samples_ts":[1.0,2.0]}]')

    def test_empty_samples(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements('[{"samples_ts":[]}]')

    def test_non_positive_samples(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements('[{"samples_ts":[-1.0, 2.0, 3.0]}]')

    def test_scalar_null(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements("null")

    def test_scalar_number(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements("42")

    def test_scalar_bool(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements("true")

    def test_list_of_non_dicts(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements('["a", "b", 42]')

    def test_list_of_nulls(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements("[null, null, null]")

    def test_record_with_non_numeric_samples(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements('[{"samples_ts":["a","b","c"]}]')

    def test_record_with_bool_samples(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements('[{"samples_ts":[true, false, true]}]')

    def test_record_with_mixed_types_in_samples(self) -> None:
        with pytest.raises(MalformedOutputError):
            _parse_measurements('[{"samples_ts":[1.0, "x", 3.0]}]')


class TestComputeStats:
    def test_median_odd(self) -> None:
        median, dispersion = _compute_stats([40.0, 50.0, 60.0])
        assert median == 50.0
        assert dispersion == pytest.approx(8.1649658, rel=1e-6)

    def test_median_even(self) -> None:
        median, _ = _compute_stats([10.0, 20.0, 30.0, 40.0])
        assert median == 25.0

    def test_known_dispersion(self) -> None:
        _, dispersion = _compute_stats([1.0, 2.0, 3.0])
        assert dispersion == pytest.approx(0.81649658, rel=1e-6)


class TestApplication:
    def test_success(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
        valid_json_output: str,
    ) -> None:
        _enqueue_toolchain(fake_runner, valid_request.binary_path)
        warmup_argv = _build_argv(valid_request, repetitions=1)
        measure_argv = _build_argv(valid_request, repetitions=3)
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        fake_runner.enqueue(
            measure_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        app = Application(fake_runner)
        obs = app.run(valid_request)

        assert obs.candidate == valid_request.candidate
        assert obs.total_ctx == 2048
        assert obs.prompt_ctx == 512
        assert obs.generation_ctx == 1024
        assert obs.depth_ctx == 512
        assert obs.samples_tps == (45.5, 44.8, 45.3)
        assert obs.median_tps == pytest.approx(45.3, rel=1e-6)
        assert len(fake_runner.calls) == 4

        assert valid_request.journal_path.exists()
        journal = json.loads(valid_request.journal_path.read_text())
        assert journal["version"] == 4
        assert journal["candidate"]["threads"] == 4
        assert journal["total_ctx"] == 2048
        assert journal["samples_tps"] == [45.5, 44.8, 45.3]
        assert journal["median_tps"] == obs.median_tps
        assert journal["n_measurements"] == 3
        assert journal["binary_path"] == str(valid_request.binary_path)
        assert journal["binary_source"] == "explicit"
        assert journal["model_path"] == str(valid_request.model_path)

    def test_on_ready_called_before_warmup(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
        valid_json_output: str,
    ) -> None:
        _enqueue_toolchain(fake_runner, valid_request.binary_path)
        warmup_argv = _build_argv(valid_request, repetitions=1)
        measure_argv = _build_argv(valid_request, repetitions=3)
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        fake_runner.enqueue(
            measure_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        ready_calls: list[tuple[ToolchainFingerprint, ModelFingerprint]] = []

        def on_ready(tc: ToolchainFingerprint, m: ModelFingerprint) -> None:
            ready_calls.append((tc, m))

        app = Application(fake_runner)
        app.run(valid_request, on_ready=on_ready)

        assert len(ready_calls) == 1
        tc, m = ready_calls[0]
        assert tc.version == "test123"
        assert m.architecture == "llama"
        toolchain_calls = [c for c in fake_runner.calls if "--version" in c or "--help" in c]
        warmup_calls = [c for c in fake_runner.calls if "-r" in c and "1" in c[-1]]
        assert len(toolchain_calls) == 2
        assert len(warmup_calls) == 1
        toolchain_last_idx = max(
            fake_runner.calls.index(c) for c in toolchain_calls
        )
        warmup_idx = fake_runner.calls.index(warmup_calls[0])
        assert toolchain_last_idx < warmup_idx

    def test_toolchain_in_observation(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
        valid_json_output: str,
    ) -> None:
        _enqueue_toolchain(fake_runner, valid_request.binary_path)
        warmup_argv = _build_argv(valid_request, repetitions=1)
        measure_argv = _build_argv(valid_request, repetitions=3)
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        fake_runner.enqueue(
            measure_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        app = Application(fake_runner)
        obs = app.run(valid_request)

        assert obs.toolchain.binary_path == valid_request.binary_path
        assert obs.toolchain.version == "test123"

    def test_model_in_observation(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
        valid_json_output: str,
    ) -> None:
        _enqueue_toolchain(fake_runner, valid_request.binary_path)
        warmup_argv = _build_argv(valid_request, repetitions=1)
        measure_argv = _build_argv(valid_request, repetitions=3)
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        fake_runner.enqueue(
            measure_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        app = Application(fake_runner)
        obs = app.run(valid_request)

        assert obs.model.architecture == "llama"

    def test_toolchain_inspection_failure_before_warmup(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
    ) -> None:
        bp = valid_request.binary_path
        fake_runner.enqueue(
            (str(bp), "--version"),
            ProcessResult(returncode=1, stdout="", stderr="not found"),
        )
        app = Application(fake_runner)
        with pytest.raises(ToolchainError):
            app.run(valid_request)
        assert len(fake_runner.calls) == 1

    def test_warmup_failure(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
        valid_json_output: str,
    ) -> None:
        _enqueue_toolchain(fake_runner, valid_request.binary_path)
        warmup_argv = _build_argv(valid_request, repetitions=1)
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=1, stdout="", stderr="warmup failed"),
        )
        app = Application(fake_runner)
        with pytest.raises(SubprocessError, match="Warmup failed"):
            app.run(valid_request)

    def test_measurement_failure(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
        valid_json_output: str,
    ) -> None:
        _enqueue_toolchain(fake_runner, valid_request.binary_path)
        warmup_argv = _build_argv(valid_request, repetitions=1)
        measure_argv = _build_argv(valid_request, repetitions=3)
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        fake_runner.enqueue(
            measure_argv,
            ProcessResult(returncode=1, stdout="", stderr="measurement failed"),
        )
        app = Application(fake_runner)
        with pytest.raises(SubprocessError, match="Measurement failed"):
            app.run(valid_request)

    def test_malformed_measurement_output(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
    ) -> None:
        _enqueue_toolchain(fake_runner, valid_request.binary_path)
        warmup_argv = _build_argv(valid_request, repetitions=1)
        measure_argv = _build_argv(valid_request, repetitions=3)
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout='[{"samples_ts":[1.0,2.0,3.0]}]', stderr=""),
        )
        fake_runner.enqueue(
            measure_argv,
            ProcessResult(returncode=0, stdout="not json", stderr=""),
        )
        app = Application(fake_runner)
        with pytest.raises(MalformedOutputError):
            app.run(valid_request)

    def test_malformed_warmup_output(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
    ) -> None:
        _enqueue_toolchain(fake_runner, valid_request.binary_path)
        warmup_argv = _build_argv(valid_request, repetitions=1)
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout="not json either", stderr=""),
        )
        app = Application(fake_runner)
        with pytest.raises(MalformedOutputError):
            app.run(valid_request)

    def test_journal_error_on_write_failure(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
        valid_json_output: str,
    ) -> None:
        _enqueue_toolchain(fake_runner, valid_request.binary_path)
        warmup_argv = _build_argv(valid_request, repetitions=1)
        measure_argv = _build_argv(valid_request, repetitions=3)
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        fake_runner.enqueue(
            measure_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        read_only = valid_request.journal_path.parent / "readonly"
        read_only.mkdir(mode=0o444)
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=valid_request.model_path,
            journal_path=read_only / "journal.json",
            n_ctx=valid_request.n_ctx,
            candidate=valid_request.candidate,
        )
        app = Application(fake_runner)
        with pytest.raises(JournalError):
            app.run(req)

    def test_preflight_no_process_launch(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
    ) -> None:
        req = SingleRunRequest(
            binary_path=Path("/nonexistent"),
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=valid_request.candidate,
        )
        app = Application(fake_runner)
        with pytest.raises(BinaryNotFoundError):
            app.run(req)
        assert len(fake_runner.calls) == 0

    def test_preflight_no_process_launch_missing_model(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
    ) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=Path("/nonexistent/model.gguf"),
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=valid_request.candidate,
        )
        app = Application(fake_runner)
        with pytest.raises(ModelNotFoundError):
            app.run(req)
        assert len(fake_runner.calls) == 0

    def test_preflight_no_process_launch_invalid_nctx(
        self,
        valid_request: SingleRunRequest,
        fake_runner: FakeRunner,
    ) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=NctxComposition(total=0, prompt=0, generation=0, depth=0),
            candidate=valid_request.candidate,
        )
        app = Application(fake_runner)
        with pytest.raises(InvalidNctxError):
            app.run(req)
        assert len(fake_runner.calls) == 0

    def test_model_fingerprint_mismatch_before_warmup(
        self,
        fake_runner: FakeRunner,
        fake_binary: Path,
        fake_model: Path,
        tmp_path: Path,
        valid_json_output: str,
    ) -> None:
        from llama_parameter_scout.model_inspector import inspect_model
        from tests.conftest import _build_minimal_gguf

        model_fp = inspect_model(fake_model)
        _enqueue_toolchain(fake_runner, fake_binary)

        app = Application(fake_runner)
        spec = app.setup_session(
            model=model_fp,
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
            cpu_only=True,
            acknowledged=True,
        )

        _build_minimal_gguf(fake_model, {"general.architecture": (8, "deepseek")})

        req = SingleRunRequest(
            binary_path=fake_binary,
            model_path=fake_model,
            journal_path=tmp_path / "journal.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
        )
        with pytest.raises(ModelFingerprintMismatchError):
            app.run(req, session_spec=spec)

        assert len(fake_runner.calls) == 2

    def test_session_spec_accelerated_unsupported_backend(
        self,
        fake_runner: FakeRunner,
        fake_binary: Path,
        fake_model: Path,
        tmp_path: Path,
        valid_json_output: str,
    ) -> None:
        from llama_parameter_scout.model_inspector import inspect_model

        model_fp = inspect_model(fake_model)

        topology = SystemTopology(
            cpu_count=8,
            system_memory_bytes=32_000_000_000,
            accelerators=(
                Accelerator("gpu", "hip", 0, "AMD GPU", memory_bytes=16_000_000_000),
            ),
            memory_unified=False,
        )

        app = Application(fake_runner, detect_topology_fn=lambda: topology)
        spec = app.setup_session(
            model=model_fp,
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=20, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
            cpu_only=False,
            device_ids=(0,),
            acknowledged=True,
        )

        _enqueue_toolchain_with_gpu_flags(
            fake_runner, fake_binary, backends=("cpu", "cuda"),
        )

        req = SingleRunRequest(
            binary_path=fake_binary,
            model_path=fake_model,
            journal_path=tmp_path / "journal.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=20, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
        )
        with pytest.raises(SessionSetupError, match="hip"):
            app.run(req, session_spec=spec)

    def test_session_spec_accelerated_supported_backend(
        self,
        fake_runner: FakeRunner,
        fake_binary: Path,
        fake_model: Path,
        tmp_path: Path,
        valid_json_output: str,
    ) -> None:
        from llama_parameter_scout.model_inspector import inspect_model

        model_fp = inspect_model(fake_model)

        topology = SystemTopology(
            cpu_count=8,
            system_memory_bytes=32_000_000_000,
            accelerators=(
                Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),
            ),
            memory_unified=False,
        )

        app = Application(fake_runner, detect_topology_fn=lambda: topology)
        spec = app.setup_session(
            model=model_fp,
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=20, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
            cpu_only=False,
            device_ids=(0,),
            acknowledged=True,
        )

        _enqueue_toolchain_with_gpu_flags(
            fake_runner, fake_binary, backends=("cpu", "cuda"),
        )

        warmup_argv = _build_argv(
            SingleRunRequest(
                binary_path=fake_binary,
                model_path=fake_model,
                journal_path=tmp_path / "journal.json",
                n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
                candidate=CandidateConfig(
                    threads=4, batch_size=512, ubatch_size=512,
                    n_gpu_layers=20, flash_attn=False,
                    cache_type_k="f16", cache_type_v="f16",
                ),
            ),
            repetitions=1,
            session_spec=spec,
        )
        measure_argv = _build_argv(
            SingleRunRequest(
                binary_path=fake_binary,
                model_path=fake_model,
                journal_path=tmp_path / "journal.json",
                n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
                candidate=CandidateConfig(
                    threads=4, batch_size=512, ubatch_size=512,
                    n_gpu_layers=20, flash_attn=False,
                    cache_type_k="f16", cache_type_v="f16",
                ),
            ),
            repetitions=3,
            session_spec=spec,
        )
        fake_runner.enqueue(
            warmup_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )
        fake_runner.enqueue(
            measure_argv,
            ProcessResult(returncode=0, stdout=valid_json_output, stderr=""),
        )

        obs = app.run(
            SingleRunRequest(
                binary_path=fake_binary,
                model_path=fake_model,
                journal_path=tmp_path / "journal.json",
                n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
                candidate=CandidateConfig(
                    threads=4, batch_size=512, ubatch_size=512,
                    n_gpu_layers=20, flash_attn=False,
                    cache_type_k="f16", cache_type_v="f16",
                ),
            ),
            session_spec=spec,
        )

        assert obs is not None
        assert obs.median_tps == pytest.approx(45.3, rel=1e-6)
        warmup_calls = [
            c for c in fake_runner.calls
            if "-r" in c and c[c.index("-r") + 1] == "1"
        ]
        assert len(warmup_calls) == 1
        actual_warmup = warmup_calls[0]
        pairs = dict(zip(actual_warmup[1::2], actual_warmup[2::2]))
        assert pairs.get("--split-mode") == "none"
        assert pairs.get("--main-gpu") == "0"


# ── Application.screen_candidates production-path tests ──────────


class TestApplicationScreenCandidates:
    """Tests that exercise Application.screen_candidates with NO injected
    ScreeningEvaluator, using a dynamic fake ProcessAdapter."""

    def _build_request(self, tmp_path: Path) -> SingleRunRequest:
        bp = tmp_path / "llama-bench"
        bp.write_text("#!/bin/sh\necho '{\"samples_ts\":[10.0,11.0,12.0]}'")
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        from tests.conftest import _build_minimal_gguf
        _build_minimal_gguf(mp, {"general.architecture": (8, "llama")})
        return SingleRunRequest(
            binary_path=bp,
            model_path=mp,
            journal_path=tmp_path / "journal.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
        )

    def _toolchain_stdout(self) -> str:
        return (
            "version: test123\n"
            "backends: cpu\n"
            "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
            "supported_output_formats: json\n"
        )

    @dataclass
    class DynamicFakeAdapter:
        json_output: str = '{"samples_ts":[10.0,11.0,12.0]}'
        raise_on_launch: type[BaseException] | None = None

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
            if self.raise_on_launch is ProcessStartError:
                raise ProcessStartError("simulated ProcessStartError")
            if self.raise_on_launch is not None:
                raise RuntimeError("simulated launch failure")
            return ProcessResult(returncode=0, stdout=self.json_output, stderr="")

    def test_default_path_preserves_distinct_pp_tg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no evaluator injected, screen_candidates runs via
        the production evaluator which does PP and TG workloads separately.
        Since both workloads parse the same llama-bench JSON (which has
        'samples_ts': [10,11,12]) the PP and TG series will be equal for now,
        but the pipeline preserves them as separate tuples."""
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        adapter = self.DynamicFakeAdapter()
        app = Application(runner, process_adapter=adapter)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE)

        assert report.baseline_disposition is not None
        assert report.baseline_disposition.success
        assert report.baseline_disposition.observation is not None
        pp = report.baseline_disposition.observation.pp_samples
        tg = report.baseline_disposition.observation.tg_samples
        assert len(pp) == QUICK_PROFILE.min_repetitions
        assert len(tg) == QUICK_PROFILE.min_repetitions
        assert all(x > 0 for x in pp)
        assert all(x > 0 for x in tg)

    def test_measure_process_start_error(
        self, tmp_path: Path,
    ) -> None:
        from llama_parameter_scout.runner import FakeRunner, ProcessStartError

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        adapter = self.DynamicFakeAdapter(raise_on_launch=ProcessStartError)
        app = Application(runner, process_adapter=adapter)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE)

        assert not report.baseline_disposition.success
        assert report.baseline_disposition.disposition in ("unsupported",)

    def test_measure_generic_exception(
        self, tmp_path: Path,
    ) -> None:
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        adapter = self.DynamicFakeAdapter(raise_on_launch=RuntimeError)
        app = Application(runner, process_adapter=adapter)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE)

        assert not report.baseline_disposition.success
        assert report.baseline_disposition.disposition == "systemic_error"

    def test_budget_preflight_no_launches(
        self, tmp_path: Path,
    ) -> None:
        """A budget too small for the model should prevent any process launch."""
        from llama_parameter_scout.model_inspector import inspect_model
        from llama_parameter_scout.models import SystemTopology
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        model = inspect_model(req.model_path)
        topology = SystemTopology(
            cpu_count=4, system_memory_bytes=1_000_000,
            accelerators=(), memory_unified=True,
        )
        spec = SessionSpecification(
            topology=topology, cpu_only=True,
            selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(ram_bytes=1_000_000, vram_bytes=None, headroom_ratio=0.0),
            advisory=None,  # type: ignore[arg-type]
            model=model, acknowledged=True, fingerprint="x",
        )

        adapter = self.DynamicFakeAdapter()
        app = Application(runner, process_adapter=adapter, detect_topology_fn=lambda: topology)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE, session_spec=spec)

        assert not report.baseline_disposition.success
        assert report.baseline_disposition.disposition in ("budget_exhausted", "unsupported")

    def test_repetition_count_in_argv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the -r flag matches requested repetitions."""
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        captured_argv: list[list[str]] = []

        class CapturingAdapter:
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
                captured_argv.append(list(argv))
                return ProcessResult(0, '{"samples_ts":[1.0,2.0,3.0]}', "")

        app = Application(runner, process_adapter=CapturingAdapter())

        from llama_parameter_scout.search import QUICK_PROFILE
        app.screen_candidates(req, QUICK_PROFILE)

        # Check that measurement calls (not warmup) use the correct -r value.
        measurement_argv = [
            argv for argv in captured_argv
            if not any(
                a == "-r" and str(argv[idx + 1]) == "1"
                for idx, a in enumerate(argv) if a == "-r"
            )
        ]
        assert measurement_argv, "No measurement calls found (all captured were warmup?)"
        for argv in measurement_argv:
            pairs = dict(zip(argv[1::2], argv[2::2]))
            r_val = int(pairs["-r"])
            assert r_val == QUICK_PROFILE.min_repetitions, (
                f"Expected -r {QUICK_PROFILE.min_repetitions}, got -r {r_val}"
            )

    def test_nctx_matches_workload(
        self, tmp_path: Path,
    ) -> None:
        """Verify PP workload gets a PP-heavy n_ctx and TG workload TG-heavy."""
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        captured: list[list[str]] = []

        class NctxCapturingAdapter:
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
                captured.append(list(argv))
                return ProcessResult(0, '{"samples_ts":[1.0,2.0,3.0]}', "")

        app = Application(runner, process_adapter=NctxCapturingAdapter())

        from llama_parameter_scout.search import QUICK_PROFILE
        app.screen_candidates(req, QUICK_PROFILE)

        # Should have PP and TG workload calls with specific n_ctx patterns
        pp_call = None
        tg_call = None
        for argv in captured:
            pairs = dict(zip(argv[1::2], argv[2::2]))
            if "-p" in pairs and "-n" in pairs:
                p_val = int(pairs["-p"])
                n_val = int(pairs["-n"])
                if p_val > n_val:
                    pp_call = argv
                elif n_val > p_val:
                    tg_call = argv

        assert pp_call is not None, "Expected a PP-heavy call (prompt > gen)"
        assert tg_call is not None, "Expected a TG-heavy call (gen > prompt)"

    def test_non_none_session_spec_used(
        self, tmp_path: Path,
    ) -> None:
        """When a requested-context SessionSpecification is provided,
        the derived per-workload spec uses the workload n_ctx."""
        from llama_parameter_scout.model_inspector import inspect_model
        from llama_parameter_scout.models import SystemTopology
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        model = inspect_model(req.model_path)
        topology = SystemTopology(
            cpu_count=4, system_memory_bytes=32_000_000_000,
            accelerators=(), memory_unified=True,
        )
        spec = SessionSpecification(
            topology=topology, cpu_only=True, selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(
                ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1,
            ),
            advisory=None,  # type: ignore[arg-type]
            model=model, acknowledged=True, fingerprint="testfp",
        )

        captured_specs: list[str] = []

        class SpecRecordingAdapter:
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
                captured_specs.append(" ".join(str(a) for a in argv))
                return ProcessResult(0, '{"samples_ts":[1.0,2.0,3.0]}', "")

        app = Application(runner, process_adapter=SpecRecordingAdapter(),
                          detect_topology_fn=lambda: topology)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE, session_spec=spec)

        assert report is not None
