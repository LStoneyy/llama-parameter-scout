from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from llama_parameter_scout.application import Application
from llama_parameter_scout.journal import read_journal_events
from llama_parameter_scout.models import (
    CandidateConfig,
    IncompatibleDraftError,
    ModelFingerprint,
    MtpComparison,
    MtpMeasurement,
    MtpMeasurementError,
    NctxComposition,
    ServerCapabilityError,
)
from llama_parameter_scout.mtp import (
    FakeServerLifecycle,
    ServerStartError,
    _build_server_argv,
    _compute_comparison,
    _compute_stats,
    _find_free_port,
    _measure_mode,
    _parse_help_flags,
    _parse_supported_flags,
    _send_completion,
    _start_server_with_retry,
    _validate_server_capabilities,
    collect_mtp_evidence,
    compare_mtp,
    inspect_server_capabilities,
    validate_draft_compatibility,
    validate_mtp_inputs,
)
from llama_parameter_scout.runner import FakeRunner, ProcessResult

# ── GGUF builder ──────────────────────────────────────────────────


def _build_minimal_gguf(path: Path, metadata: dict[str, tuple[int, Any]], version: int = 3) -> None:
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", version))
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
    elif value_type == 4:
        buf.extend(struct.pack("<I", value))
    elif value_type == 10:
        buf.extend(struct.pack("<Q", value))
    elif value_type == 7:
        buf.extend(struct.pack("<B", 1 if value else 0))
    else:
        raise ValueError(f"Unsupported type: {value_type}")


# ── Helpers ──────────────────────────────────────────────────────


def _make_main_fingerprint(**overrides: object) -> ModelFingerprint:
    base = ModelFingerprint(
        path=Path("/models/main.gguf"),
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=4096,
        quantization="Q4_K_M",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )
    field_values = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    field_values.update(overrides)
    return ModelFingerprint(**field_values)


def _make_draft_fingerprint(**overrides: object) -> ModelFingerprint:
    base = ModelFingerprint(
        path=Path("/models/draft.gguf"),
        architecture="llama",
        parameter_count=500_000_000,
        context_length=4096,
        quantization="Q8_0",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )
    field_values = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    field_values.update(overrides)
    return ModelFingerprint(**field_values)


def _server_help(cpu: bool = True, moe: bool = False) -> str:
    lines = [
        "usage: llama-server [options]",
        "  -m FNAME            model path",
        "  -md FNAME           draft model path",
        "  --host HOST         host",
        "  --port PORT         port",
        "  -c N                context size",
        "  -t N                threads",
        "  -ngl N              GPU layers",
        "  -fa on|off          flash attention",
        "  -ctk FNAME          K cache {f16,f32}",
        "  -ctv FNAME          V cache {f16,f32}",
        "  -b N                batch size",
        "  --ubatch-size N     ubatch size",
    ]
    if not cpu:
        lines.append("  --split-mode MODE   split mode")
        lines.append("  --main-gpu N        main GPU")
    if moe:
        lines.append("  --n-cpu-moe N       MoE CPU offload")
    return "\n".join(lines)


def _server_version(extra_flags: str = "") -> str:
    base = "-m,--host,--port,-c,-t,-ngl,-fa,-ctk,-ctv,-b,--ubatch-size"
    if extra_flags:
        base += "," + extra_flags
    return (
        "version: b1234\n"
        f"backends: cpu\n"
        f"supported_flags: {base}\n"
        "supported_output_formats: json\n"
    )


def _enqueue_server(runner: FakeRunner, sp: Path, *, cpu: bool = True, moe: bool = False) -> None:
    runner.enqueue(
        (str(sp), "--version"),
        ProcessResult(returncode=0, stdout=_server_version(), stderr=""),
    )
    runner.enqueue(
        (str(sp), "--help"),
        ProcessResult(returncode=0, stdout=_server_help(cpu=cpu, moe=moe), stderr=""),
    )


def _big_candidate(**kw: Any) -> CandidateConfig:
    return CandidateConfig(
        threads=kw.get("threads", 4),
        batch_size=kw.get("batch_size", 512),
        ubatch_size=kw.get("ubatch_size", 512),
        n_gpu_layers=kw.get("n_gpu_layers", 0),
        flash_attn=kw.get("flash_attn", False),
        cache_type_k=kw.get("cache_type_k", "f16"),
        cache_type_v=kw.get("cache_type_v", "f16"),
        n_cpu_moe=kw.get("n_cpu_moe"),
    )


def _ctx2048() -> NctxComposition:
    return NctxComposition(total=2048, prompt=512, generation=1024, depth=512)


# ── Port selection tests ─────────────────────────────────────────


class TestFindFreePort:
    def test_returns_positive_port(self) -> None:
        port = _find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_loopback_only(self) -> None:
        import socket
        port = _find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            assert s.getsockname()[0] == "127.0.0.1"


# ── Input validation tests ───────────────────────────────────────


class TestValidateMtpInputs:
    def test_valid(self) -> None:
        validate_mtp_inputs(_big_candidate(), _ctx2048(), cpu_only=True, device_id=None)

    def test_device_required_when_not_cpu_only(self) -> None:
        with pytest.raises(ValueError, match="device_id is required"):
            validate_mtp_inputs(_big_candidate(), _ctx2048(), cpu_only=False, device_id=None)

    def test_device_id_conflict(self) -> None:
        with pytest.raises(ValueError, match="device_id must be None"):
            validate_mtp_inputs(_big_candidate(), _ctx2048(), cpu_only=True, device_id=0)

    def test_nctx_total_zero(self) -> None:
        ctx = NctxComposition(total=0, prompt=0, generation=0, depth=0)
        with pytest.raises(ValueError, match="n_ctx.total"):
            validate_mtp_inputs(_big_candidate(), ctx, cpu_only=True, device_id=None)

    def test_nctx_negative_component(self) -> None:
        ctx = NctxComposition(total=2048, prompt=-1, generation=1024, depth=1025)
        with pytest.raises(ValueError, match="n_ctx components"):
            validate_mtp_inputs(_big_candidate(), ctx, cpu_only=True, device_id=None)

    def test_nctx_no_sum(self) -> None:
        ctx = NctxComposition(total=2048, prompt=500, generation=1000, depth=500)
        with pytest.raises(ValueError, match="do not sum"):
            validate_mtp_inputs(_big_candidate(), ctx, cpu_only=True, device_id=None)

    def test_candidate_threads_zero(self) -> None:
        with pytest.raises(ValueError, match="threads"):
            validate_mtp_inputs(
                _big_candidate(threads=0), _ctx2048(), cpu_only=True, device_id=None,
            )


# ── Server argv tests ────────────────────────────────────────────


class TestBuildServerArgv:
    def test_loopback_host(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), None,
            8080, _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
        )
        assert "--host" in argv
        assert argv[argv.index("--host") + 1] == "127.0.0.1"

    def test_no_0_0_0_0(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), None,
            8080, _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
        )
        assert "0.0.0.0" not in argv

    def test_includes_draft_when_provided(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), Path("/draft.gguf"),
            8080, _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
        )
        assert "-md" in argv
        assert argv[argv.index("-md") + 1] == "/draft.gguf"

    def test_no_draft_when_none(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), None,
            8080, _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
        )
        assert "-md" not in argv

    def test_device_id_included_when_provided(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), None,
            8080, _ctx2048(), _big_candidate(n_gpu_layers=20),
            cpu_only=False, device_id=0,
        )
        assert "--split-mode" in argv
        assert "--main-gpu" in argv
        assert argv[argv.index("--main-gpu") + 1] == "0"

    def test_device_id_not_included_when_cpu_only(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), None,
            8080, _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
        )
        assert "--split-mode" not in argv
        assert "--main-gpu" not in argv

    def test_n_cpu_moe_included(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), None,
            8080, _ctx2048(), _big_candidate(n_cpu_moe=2),
            cpu_only=True, device_id=None,
        )
        assert "--n-cpu-moe" in argv

    def test_n_cpu_moe_omitted_when_none(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), None,
            8080, _ctx2048(), _big_candidate(n_cpu_moe=None),
            cpu_only=True, device_id=None,
        )
        assert "--n-cpu-moe" not in argv


# ── Capability validation tests ──────────────────────────────────


class TestParseHelpFlags:
    def test_parses_short_and_long(self) -> None:
        help_text = "  -m FNAME  model\n  --host HOST  host\n  -c N  context\n"
        flags = _parse_help_flags(help_text)
        assert "-m" in flags
        assert "--host" in flags
        assert "-c" in flags

    def test_no_duplicate_required(self) -> None:
        """Only flag forms emitted by argv need to exist in help."""
        help_text = "  --model FNAME  model\n  --host HOST  host\n"
        flags = _parse_help_flags(help_text)
        assert "--model" in flags
        assert "-m" not in flags  # only --model, not -m

    def test_version_flags_supplement_help(self) -> None:
        flags = _parse_supported_flags("supported_flags: -m,--host,-c\n")
        assert flags == {"-m", "--host", "-c"}

    def test_version_flags_empty_when_missing(self) -> None:
        flags = _parse_supported_flags("version: b1\n")
        assert flags == set()


class TestValidateServerCapabilities:
    def test_cpu_only_checks_correct_flags(self) -> None:
        version_output = _server_version()
        help_output = _server_help(cpu=True)
        version, flags = _validate_server_capabilities(
            version_output, help_output, Path("/server"),
            cpu_only=True, device_id=None, has_n_cpu_moe=False,
        )
        assert version == "b1234"

    def test_gpu_mode_checks_split_mode(self) -> None:
        version_output = _server_version("--split-mode,--main-gpu")
        help_output = _server_help(cpu=False)
        _validate_server_capabilities(
            version_output, help_output, Path("/server"),
            cpu_only=False, device_id=0, has_n_cpu_moe=False,
        )

    def test_gpu_missing_split_mode_rejected(self) -> None:
        version_output = _server_version()
        help_output = _server_help(cpu=True)
        with pytest.raises(ServerCapabilityError, match="--split-mode"):
            _validate_server_capabilities(
                version_output, help_output, Path("/server"),
                cpu_only=False, device_id=0, has_n_cpu_moe=False,
            )

    def test_moe_checks_n_cpu_moe_flag(self) -> None:
        version_output = _server_version("--n-cpu-moe")
        help_output = _server_help(moe=True)
        _validate_server_capabilities(
            version_output, help_output, Path("/server"),
            cpu_only=True, device_id=None, has_n_cpu_moe=True,
        )

    def test_moe_flag_missing_rejected(self) -> None:
        version_output = _server_version()
        help_output = _server_help(cpu=True)
        with pytest.raises(ServerCapabilityError, match="--n-cpu-moe"):
            _validate_server_capabilities(
                version_output, help_output, Path("/server"),
                cpu_only=True, device_id=None, has_n_cpu_moe=True,
            )

    def test_version_missing_rejected(self) -> None:
        with pytest.raises(ServerCapabilityError, match="version"):
            _validate_server_capabilities(
                "no version line", _server_help(), Path("/server"),
            )

    def test_help_as_authoritative_source(self) -> None:
        """Flags only in --help (not --version) are accepted."""
        version_output = "version: b1\nbackends: cpu\nsupported_output_formats: json\n"
        help_output = _server_help(cpu=True)
        _validate_server_capabilities(
            version_output, help_output, Path("/server"),
            cpu_only=True, device_id=None, has_n_cpu_moe=False,
        )


class TestInspectServerCapabilities:
    def test_success(self) -> None:
        runner = FakeRunner()
        sp = Path("/server")
        _enqueue_server(runner, sp)
        inspect_server_capabilities(runner, sp)

    def test_version_failure_rejected(self) -> None:
        runner = FakeRunner()
        sp = Path("/server")
        runner.enqueue(
            (str(sp), "--version"),
            ProcessResult(returncode=1, stdout="", stderr="error"),
        )
        with pytest.raises(ServerCapabilityError):
            inspect_server_capabilities(runner, sp)


# ── Draft compatibility tests ────────────────────────────────────


class TestValidateDraftCompatibility:
    def test_compatible(self) -> None:
        validate_draft_compatibility(_make_main_fingerprint(), _make_draft_fingerprint())

    def test_rejects_mismatched_architecture(self) -> None:
        with pytest.raises(IncompatibleDraftError, match="Architecture mismatch"):
            validate_draft_compatibility(
                _make_main_fingerprint(architecture="llama"),
                _make_draft_fingerprint(architecture="mistral"),
            )

    def test_rejects_draft_shorter_context(self) -> None:
        with pytest.raises(IncompatibleDraftError, match="context length"):
            validate_draft_compatibility(
                _make_main_fingerprint(context_length=8192),
                _make_draft_fingerprint(context_length=4096),
            )

    def test_accepts_missing_context_on_draft(self) -> None:
        validate_draft_compatibility(
            _make_main_fingerprint(context_length=4096),
            _make_draft_fingerprint(context_length=None),
        )

    def test_rejects_draft_not_smaller(self) -> None:
        with pytest.raises(IncompatibleDraftError, match="parameter count"):
            validate_draft_compatibility(
                _make_main_fingerprint(parameter_count=500_000_000),
                _make_draft_fingerprint(parameter_count=7_000_000_000),
            )


# ── Fake server lifecycle tests ──────────────────────────────────


class TestFakeServerLifecycle:
    def test_start_stop_sequence(self) -> None:
        lc = FakeServerLifecycle()
        lc.start(["./server"], port=1234)
        assert lc.port == 1234
        assert lc.wait_for_ready()
        lc.stop()
        assert lc.stop_called

    def test_start_oserror(self) -> None:
        lc = FakeServerLifecycle(_start_fail=True)
        with pytest.raises(OSError):
            lc.start(["./server"], port=1234)

    def test_start_serverstart_error(self) -> None:
        lc = FakeServerLifecycle(_start_fail_with_server_error=True)
        with pytest.raises(ServerStartError):
            lc.start(["./server"], port=1234)

    def test_not_ready(self) -> None:
        lc = FakeServerLifecycle(_ready=False)
        lc.start(["./server"], port=1234)
        assert not lc.wait_for_ready()

    def test_diagnostics_default(self) -> None:
        lc = FakeServerLifecycle()
        assert lc.diagnostics == ""


# ── Stats tests ──────────────────────────────────────────────────


class TestComputeStats:
    def test_median_odd(self) -> None:
        median, _ = _compute_stats([10.0, 20.0, 30.0])
        assert median == 20.0

    def test_median_even(self) -> None:
        median, _ = _compute_stats([10.0, 20.0, 30.0, 40.0])
        assert median == 25.0


# ── Measurement mode tests ───────────────────────────────────────


class TestMeasureMode:
    def test_warmup_then_3_measurements(self) -> None:
        warmup_log: list[int] = []
        measure_log: list[float] = []

        def _warm(p: int) -> None:
            warmup_log.append(p)

        def _meas(p: int) -> float:
            measure_log.append(p)
            return 50.0

        m = _measure_mode(9999, "off", _warmup_fn=_warm, _measure_fn=_meas)
        assert len(warmup_log) == 1
        assert len(measure_log) == 3
        assert m.mode == "off"
        assert m.median_tps == 50.0
        assert len(m.samples_tps) == 3

    def test_computes_correct_stats(self) -> None:
        values = [40.0, 50.0, 60.0]

        def _meas(p: int) -> float:
            return values.pop(0)

        m = _measure_mode(9999, "on", _warmup_fn=lambda p: None, _measure_fn=_meas)
        assert m.median_tps == 50.0

    def test_minimum_3_measurements(self) -> None:
        calls: list[int] = []

        def _meas(p: int) -> float:
            calls.append(p)
            return 50.0

        _measure_mode(9999, "off", _warmup_fn=lambda p: None, _measure_fn=_meas)
        assert len(calls) >= 3


# ── Comparison tests ─────────────────────────────────────────────


class TestComputeComparison:
    def test_clear_benefit(self) -> None:
        off = MtpMeasurement(
            mode="off", samples_tps=(40.0, 41.0, 42.0),
            median_tps=41.0, dispersion_tps=0.8,
        )
        on = MtpMeasurement(
            mode="on", samples_tps=(60.0, 62.0, 64.0),
            median_tps=62.0, dispersion_tps=1.6,
        )
        result = _compute_comparison(off, on)
        assert result.benefit == "established"
        assert result.improvement_pct is not None

    def test_tie_no_benefit(self) -> None:
        off = MtpMeasurement(
            mode="off", samples_tps=(50.0, 51.0, 52.0),
            median_tps=51.0, dispersion_tps=2.0,
        )
        on = MtpMeasurement(
            mode="on", samples_tps=(51.5, 52.0, 52.5),
            median_tps=52.0, dispersion_tps=1.0,
        )
        result = _compute_comparison(off, on)
        assert result.benefit == "not_established"
        assert result.improvement_pct is None

    def test_caveat_present(self) -> None:
        off = MtpMeasurement(
            mode="off", samples_tps=(50.0, 51.0, 52.0),
            median_tps=51.0, dispersion_tps=1.0,
        )
        on = MtpMeasurement(
            mode="on", samples_tps=(60.0, 62.0, 64.0),
            median_tps=62.0, dispersion_tps=1.0,
        )
        result = _compute_comparison(off, on)
        assert "synthetic generation throughput" in result.caveat
        assert "prompt processing" in result.caveat


# ── Server retry tests ───────────────────────────────────────────


class TestStartServerWithRetry:
    def test_success_first_attempt(self) -> None:
        lc = FakeServerLifecycle()
        port, argv = _start_server_with_retry(
            lc, Path("/server"), Path("/main.gguf"), None,
            _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
        )
        assert isinstance(port, int) and port > 0
        assert len(lc.start_calls) == 1
        assert isinstance(argv, tuple)
        assert len(argv) > 0

    def test_retry_on_oserror(self) -> None:
        lc = FakeServerLifecycle(_start_fail=True)
        with pytest.raises(OSError):
            _start_server_with_retry(
                lc, Path("/server"), Path("/main.gguf"), None,
                _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
            )

    def test_retry_on_server_start_error(self) -> None:
        """Production raises ServerStartError; retry must catch it."""
        lc = FakeServerLifecycle(_start_fail_with_server_error=True)
        with pytest.raises(ServerStartError):
            _start_server_with_retry(
                lc, Path("/server"), Path("/main.gguf"), None,
                _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
            )

    def test_retry_on_not_ready(self) -> None:
        lc = FakeServerLifecycle(_ready_timeout=True)
        with pytest.raises(ServerCapabilityError, match="did not become ready"):
            _start_server_with_retry(
                lc, Path("/server"), Path("/main.gguf"), None,
                _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
            )

    def test_draft_model_in_argv(self) -> None:
        lc = FakeServerLifecycle()
        _, argv = _start_server_with_retry(
            lc, Path("/server"), Path("/main.gguf"), Path("/draft.gguf"),
            _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
        )
        assert "-md" in argv
        assert argv[argv.index("-md") + 1] == "/draft.gguf"


# ── Send completion tests ────────────────────────────────────────


class TestSendCompletion:
    def test_rejects_malformed_json(self) -> None:
        """Network-level error is expected since no server is running."""
        with pytest.raises((MtpMeasurementError, OSError)):
            _send_completion(1, "test", 128)


# ── Full comparison orchestration tests ──────────────────────────


class TestCompareMtp:
    def test_clear_benefit(self) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        lc = FakeServerLifecycle()
        values = [30.0, 31.0, 32.0, 60.0, 62.0, 64.0]

        def _meas(p: int) -> float:
            return values.pop(0)

        result = compare_mtp(
            candidate=_big_candidate(),
            main_model_path=Path("/main.gguf"),
            draft_model_path=Path("/draft.gguf"),
            server_path=Path("/server"),
            n_ctx=_ctx2048(),
            cpu_only=True,
            runner=runner,
            server_lifecycle=lc,
            main_fingerprint=_make_main_fingerprint(),
            draft_fingerprint=_make_draft_fingerprint(),
            _warmup_fn=lambda p: None,
            _measure_fn=_meas,
        )
        assert result.benefit == "established"
        assert result.off.median_tps == 31.0
        assert result.on.median_tps == 62.0
        assert lc.stop_called
        assert len(lc.start_calls) == 2

    def test_tie_no_benefit(self) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        lc = FakeServerLifecycle()
        values = [51.0, 52.0, 53.0, 50.0, 51.0, 52.0]

        def _meas(p: int) -> float:
            return values.pop(0)

        result = compare_mtp(
            candidate=_big_candidate(),
            main_model_path=Path("/main.gguf"),
            draft_model_path=Path("/draft.gguf"),
            server_path=Path("/server"),
            n_ctx=_ctx2048(),
            cpu_only=True,
            runner=runner,
            server_lifecycle=lc,
            main_fingerprint=_make_main_fingerprint(),
            draft_fingerprint=_make_draft_fingerprint(),
            _warmup_fn=lambda p: None,
            _measure_fn=_meas,
        )
        assert result.benefit == "not_established"
        assert result.improvement_pct is None

    def test_cleanup_on_ready_timeout(self) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        lc = FakeServerLifecycle(_ready_timeout=True)

        with pytest.raises(ServerCapabilityError):
            compare_mtp(
                candidate=_big_candidate(),
                main_model_path=Path("/main.gguf"),
                draft_model_path=Path("/draft.gguf"),
                server_path=Path("/server"),
                n_ctx=_ctx2048(),
                cpu_only=True,
                runner=runner,
                server_lifecycle=lc,
                main_fingerprint=_make_main_fingerprint(),
                draft_fingerprint=_make_draft_fingerprint(),
                _warmup_fn=lambda p: None,
                _measure_fn=lambda p: 50.0,
            )
        assert lc.stop_called

    def test_cleanup_on_warmup_failure(self) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        lc = FakeServerLifecycle()

        def _warm(p: int) -> None:
            raise RuntimeError("warmup failed")

        with pytest.raises(RuntimeError, match="warmup failed"):
            compare_mtp(
                candidate=_big_candidate(),
                main_model_path=Path("/main.gguf"),
                draft_model_path=Path("/draft.gguf"),
                server_path=Path("/server"),
                n_ctx=_ctx2048(),
                cpu_only=True,
                runner=runner,
                server_lifecycle=lc,
                main_fingerprint=_make_main_fingerprint(),
                draft_fingerprint=_make_draft_fingerprint(),
                _warmup_fn=_warm,
                _measure_fn=lambda p: 50.0,
            )
        assert lc.stop_called

    def test_incompatible_draft_rejected_before_lifecycle(self) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        lc = FakeServerLifecycle()

        with pytest.raises(IncompatibleDraftError):
            compare_mtp(
                candidate=_big_candidate(),
                main_model_path=Path("/main.gguf"),
                draft_model_path=Path("/draft.gguf"),
                server_path=Path("/server"),
                n_ctx=_ctx2048(),
                cpu_only=True,
                runner=runner,
                server_lifecycle=lc,
                main_fingerprint=_make_main_fingerprint(architecture="llama"),
                draft_fingerprint=_make_draft_fingerprint(architecture="falcon"),
                _warmup_fn=lambda p: None,
                _measure_fn=lambda p: 50.0,
            )
        assert len(lc.start_calls) == 0
        assert not lc.stop_called

    def test_capability_rejected_before_lifecycle_start(self) -> None:
        runner = FakeRunner()
        sp = Path("/server")
        runner.enqueue(
            (str(sp), "--version"),
            ProcessResult(
                returncode=0,
                stdout="version: b1\nsupported_flags: -m\nsupported_output_formats: json\n",
                stderr="",
            ),
        )
        runner.enqueue(
            (str(sp), "--help"),
            ProcessResult(returncode=0, stdout="usage: server\n  -m FNAME  model\n", stderr=""),
        )
        lc = FakeServerLifecycle()

        with pytest.raises(ServerCapabilityError):
            compare_mtp(
                candidate=_big_candidate(),
                main_model_path=Path("/main.gguf"),
                draft_model_path=Path("/draft.gguf"),
                server_path=sp,
                n_ctx=_ctx2048(),
                cpu_only=True,
                runner=runner,
                server_lifecycle=lc,
                main_fingerprint=_make_main_fingerprint(),
                draft_fingerprint=_make_draft_fingerprint(),
                _warmup_fn=lambda p: None,
                _measure_fn=lambda p: 50.0,
            )
        assert len(lc.start_calls) == 0

    def test_model_validation_before_server_inspection(self) -> None:
        """Model fingerprint/compatibility is checked before any server subprocess."""
        runner = FakeRunner()
        sp = Path("/server")
        lc = FakeServerLifecycle()
        # Don't enqueue any server results — if server probe is attempted, it will error.

        with pytest.raises(IncompatibleDraftError):
            compare_mtp(
                candidate=_big_candidate(),
                main_model_path=Path("/main.gguf"),
                draft_model_path=Path("/draft.gguf"),
                server_path=sp,
                n_ctx=_ctx2048(),
                cpu_only=True,
                runner=runner,
                server_lifecycle=lc,
                main_fingerprint=_make_main_fingerprint(architecture="llama"),
                draft_fingerprint=_make_draft_fingerprint(architecture="falcon"),
                _warmup_fn=lambda p: None,
                _measure_fn=lambda p: 50.0,
            )
        # No server subprocess calls should have been attempted.
        assert runner.calls == []

    def test_both_modes_same_main_settings(self) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        lc = FakeServerLifecycle()

        def _meas(p: int) -> float:
            return 50.0

        _ = compare_mtp(
            candidate=_big_candidate(),
            main_model_path=Path("/main.gguf"),
            draft_model_path=Path("/draft.gguf"),
            server_path=Path("/server"),
            n_ctx=_ctx2048(),
            cpu_only=True,
            runner=runner,
            server_lifecycle=lc,
            main_fingerprint=_make_main_fingerprint(),
            draft_fingerprint=_make_draft_fingerprint(),
            _warmup_fn=lambda p: None,
            _measure_fn=_meas,
        )
        off_argv = lc.start_calls[0]
        on_argv = lc.start_calls[1]
        assert off_argv[off_argv.index("-m") + 1] == on_argv[on_argv.index("-m") + 1]
        assert "-md" not in off_argv
        assert "-md" in on_argv


# ── Application seam tests ───────────────────────────────────────


class TestApplicationCompareMtp:
    def test_delegates_with_pre_inspected_fingerprints(self) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        app = Application(runner)
        lc = FakeServerLifecycle()

        result = app.compare_mtp(
            candidate=_big_candidate(),
            main_model_path=Path("/main.gguf"),
            draft_model_path=Path("/draft.gguf"),
            server_path=Path("/server"),
            n_ctx=_ctx2048(),
            cpu_only=True,
            server_lifecycle=lc,
            main_fingerprint=_make_main_fingerprint(),
            draft_fingerprint=_make_draft_fingerprint(),
            _warmup_fn=lambda p: None,
            _measure_fn=lambda p: 50.0,
        )
        assert isinstance(result, MtpComparison)
        assert lc.stop_called
        assert len(lc.start_calls) == 2

    def test_with_real_gguf_files_and_journal(self, tmp_path: Path) -> None:
        main_path = tmp_path / "main.gguf"
        _build_minimal_gguf(main_path, {
            "general.architecture": (8, "llama"),
            "general.parameter_count": (10, 7_000_000_000),
            "llama.context_length": (4, 4096),
        })

        draft_dir = tmp_path / "draft_shards"
        draft_dir.mkdir()
        draft_01 = draft_dir / "draft-00001-of-00002.gguf"
        _build_minimal_gguf(draft_01, {
            "general.architecture": (8, "llama"),
            "general.parameter_count": (10, 500_000_000),
            "llama.context_length": (4, 4096),
            "split.no": (4, 0),
            "split.count": (4, 2),
        })
        draft_02 = draft_dir / "draft-00002-of-00002.gguf"
        _build_minimal_gguf(draft_02, {
            "general.architecture": (8, "llama"),
            "general.parameter_count": (10, 500_000_000),
            "llama.context_length": (4, 4096),
            "split.no": (4, 1),
            "split.count": (4, 2),
        })

        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        app = Application(runner)
        lc = FakeServerLifecycle()
        jp = tmp_path / "journal.json"

        result = app.compare_mtp(
            candidate=_big_candidate(),
            main_model_path=main_path,
            draft_model_path=draft_01,
            server_path=Path("/server"),
            n_ctx=_ctx2048(),
            cpu_only=True,
            journal_path=jp,
            server_lifecycle=lc,
            _warmup_fn=lambda p: None,
            _measure_fn=lambda p: 50.0,
        )
        assert isinstance(result, MtpComparison)

        events = read_journal_events(jp)
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "mtp_evidence"
        evidence = ev["mtp_evidence"]
        assert "main_fingerprint" in evidence
        assert "draft_fingerprint" in evidence
        assert evidence["candidate"]["threads"] == 4
        assert evidence["n_ctx"]["total"] == 2048

        draft_fp = evidence["draft_fingerprint"]
        assert len(draft_fp["shards"]) == 2
        assert draft_fp["shards"][0]["path"] == str(draft_01)
        assert draft_fp["shards"][1]["path"] == str(draft_02)

        assert "comparison" in evidence
        assert evidence["comparison"]["benefit"] in ("established", "not_established")

        # Assert rich provenance fields are non-empty — Application calls
        # collect_mtp_evidence which populates everything.
        assert evidence["server_version"] == "b1234"
        assert len(evidence["server_supported_flags"]) > 0
        assert len(evidence["off_argv"]) > 0
        assert len(evidence["on_argv"]) > 0
        assert evidence["off_argv"][0] == str(Path("/server"))
        assert evidence["on_argv"][0] == str(Path("/server"))

    def test_with_journal_and_parent_session_id(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        app = Application(runner)
        lc = FakeServerLifecycle()
        jp = tmp_path / "journal.json"

        app.compare_mtp(
            candidate=_big_candidate(),
            main_model_path=Path("/main.gguf"),
            draft_model_path=Path("/draft.gguf"),
            server_path=Path("/server"),
            n_ctx=_ctx2048(),
            cpu_only=True,
            journal_path=jp,
            session_id="child",
            parent_session_id="parent",
            server_lifecycle=lc,
            main_fingerprint=_make_main_fingerprint(),
            draft_fingerprint=_make_draft_fingerprint(),
            _warmup_fn=lambda p: None,
            _measure_fn=lambda p: 50.0,
        )

        events = read_journal_events(jp)
        assert len(events) == 1
        ev = events[0]
        assert ev["session_id"] == "child"
        assert ev["parent_session_id"] == "parent"


# ── Collect MtpEvidence tests ────────────────────────────────────


class TestCollectMtpEvidence:
    def test_rich_evidence_via_collect(self) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        lc = FakeServerLifecycle()

        def _meas(p: int) -> float:
            return 50.0

        evidence = collect_mtp_evidence(
            candidate=_big_candidate(),
            main_model_path=Path("/main.gguf"),
            draft_model_path=Path("/draft.gguf"),
            server_path=Path("/server"),
            n_ctx=_ctx2048(),
            cpu_only=True,
            runner=runner,
            server_lifecycle=lc,
            main_fingerprint=_make_main_fingerprint(),
            draft_fingerprint=_make_draft_fingerprint(),
            _warmup_fn=lambda p: None,
            _measure_fn=_meas,
        )
        assert evidence.server_version == "b1234"
        assert len(evidence.server_supported_flags) > 0
        assert evidence.cpu_only
        assert evidence.device_id is None
        assert evidence.comparison.benefit in ("established", "not_established")
        assert len(evidence.off_argv) > 0
        assert len(evidence.on_argv) > 0
        assert evidence.main_fingerprint.architecture == "llama"
        assert evidence.draft_fingerprint.architecture == "llama"
        assert evidence.candidate.threads == 4
        assert evidence.n_ctx.total == 2048


# ── Journal event tests ──────────────────────────────────────────


class TestJournalMtpEvidence:
    def test_event_structure(self, tmp_path: Path) -> None:
        from llama_parameter_scout.journal import write_mtp_evidence
        from llama_parameter_scout.models import MtpEvidence as MtpEvidenceCls

        jp = tmp_path / "journal.json"
        off = MtpMeasurement(
            mode="off", samples_tps=(40.0, 41.0, 42.0),
            median_tps=41.0, dispersion_tps=0.8,
        )
        on = MtpMeasurement(
            mode="on", samples_tps=(60.0, 62.0, 64.0),
            median_tps=62.0, dispersion_tps=1.6,
        )
        comparison = MtpComparison(
            off=off, on=on, benefit="established",
            improvement_pct=51.2, tie_bound_tps=1.6, caveat="test",
        )
        evidence = MtpEvidenceCls(
            main_fingerprint=_make_main_fingerprint(),
            draft_fingerprint=_make_draft_fingerprint(),
            server_path="/server",
            server_version="b1234",
            server_supported_flags=("-m", "--host"),
            candidate=_big_candidate(),
            n_ctx=_ctx2048(),
            cpu_only=True,
            device_id=None,
            off_argv=(),
            on_argv=(),
            comparison=comparison,
        )
        write_mtp_evidence(jp, evidence, session_id="sid123")

        events = read_journal_events(jp)
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "mtp_evidence"
        assert ev["session_id"] == "sid123"
        assert ev["event_version"] == 1
        e = ev["mtp_evidence"]
        assert e["server_path"] == "/server"
        assert e["server_version"] == "b1234"
        assert e["comparison"]["benefit"] == "established"


# ── Port-race tests ──────────────────────────────────────────────


class TestPortRace:
    def test_fake_retry_on_oserror(self) -> None:
        lc = FakeServerLifecycle(_start_fail=True)
        with pytest.raises(OSError):
            _start_server_with_retry(
                lc, Path("/server"), Path("/main.gguf"), None,
                _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
            )

    def test_fake_retry_on_serverstart_error(self) -> None:
        lc = FakeServerLifecycle(_start_fail_with_server_error=True)
        with pytest.raises(ServerStartError):
            _start_server_with_retry(
                lc, Path("/server"), Path("/main.gguf"), None,
                _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
            )


# ── Loopback-only tests ──────────────────────────────────────────


class TestLoopbackOnly:
    def test_argv_host_is_127_0_0_1(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), None,
            8080, _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
        )
        assert "--host" in argv
        assert argv[argv.index("--host") + 1] == "127.0.0.1"

    def test_no_0_0_0_0_in_argv(self) -> None:
        argv = _build_server_argv(
            Path("/server"), Path("/main.gguf"), None,
            8080, _ctx2048(), _big_candidate(), cpu_only=True, device_id=None,
        )
        assert all("0.0.0.0" not in a for a in argv)


# ── Cleanup tests ────────────────────────────────────────────────


class TestCleanup:
    def test_stop_called_on_success(self) -> None:
        runner = FakeRunner()
        _enqueue_server(runner, Path("/server"))
        lc = FakeServerLifecycle()

        def _meas(p: int) -> float:
            return 50.0

        compare_mtp(
            candidate=_big_candidate(),
            main_model_path=Path("/main.gguf"),
            draft_model_path=Path("/draft.gguf"),
            server_path=Path("/server"),
            n_ctx=_ctx2048(),
            cpu_only=True,
            runner=runner,
            server_lifecycle=lc,
            main_fingerprint=_make_main_fingerprint(),
            draft_fingerprint=_make_draft_fingerprint(),
            _warmup_fn=lambda p: None,
            _measure_fn=_meas,
        )
        assert lc.stop_called

    def test_no_lifecycle_on_incompatible_draft(self) -> None:
        lc = FakeServerLifecycle()
        with pytest.raises(IncompatibleDraftError):
            validate_draft_compatibility(
                _make_main_fingerprint(architecture="llama"),
                _make_draft_fingerprint(architecture="falcon"),
            )
        assert not lc.stop_called
        assert len(lc.start_calls) == 0
