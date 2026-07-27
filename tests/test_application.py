from __future__ import annotations

import json
from pathlib import Path

import pytest

from llama_parameter_scout.application import (
    Application,
    _build_argv,
    _compute_stats,
    _parse_measurements,
    _validate_request,
)
from llama_parameter_scout.models import (
    BinaryNotFoundError,
    CandidateConfig,
    CandidateError,
    InvalidNctxError,
    JournalError,
    MalformedOutputError,
    ModelFingerprint,
    ModelNotFoundError,
    NctxComposition,
    SingleRunRequest,
    SubprocessError,
    ToolchainError,
    ToolchainFingerprint,
)
from llama_parameter_scout.runner import FakeRunner, ProcessResult


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
        assert journal["version"] == 2
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
