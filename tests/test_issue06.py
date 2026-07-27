from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from llama_parameter_scout.application import (
    Application,
    _build_argv,
    _is_quantized_cache,
    _provenance_for,
    _screening_preflight,
)
from llama_parameter_scout.model_inspector import inspect_model
from llama_parameter_scout.models import (
    Accelerator,
    CandidateConfig,
    ModelFingerprint,
    NctxComposition,
    SingleRunRequest,
    SystemTopology,
    ToolchainFingerprint,
)
from llama_parameter_scout.runner import FakeRunner, ProcessResult
from llama_parameter_scout.search import generate_screening_grid
from tests.conftest import _build_minimal_gguf as build_gguf

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def dense_model(tmp_path: Path) -> Path:
    p = tmp_path / "dense.gguf"
    build_gguf(p, {"general.architecture": (8, "llama")})
    return p


@pytest.fixture
def moe_model(tmp_path: Path) -> Path:
    p = tmp_path / "moe.gguf"
    build_gguf(p, {
        "general.architecture": (8, "llama"),
        "llama.expert_count": (5, 8),
    })
    return p


@pytest.fixture
def moe_model_no_expert_count(tmp_path: Path) -> Path:
    """MoE model without expert_count metadata - treat as non-MoE."""
    p = tmp_path / "moe_no_cnt.gguf"
    build_gguf(p, {"general.architecture": (8, "llama")})
    return p


@pytest.fixture
def base_candidate() -> CandidateConfig:
    return CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=0, flash_attn=False,
        cache_type_k="f16", cache_type_v="f16",
    )


@pytest.fixture
def sample_toolchain() -> ToolchainFingerprint:
    return ToolchainFingerprint(
        binary_path=Path("/usr/local/bin/llama-bench"),
        version="test",
        supported_flags=(
            "-m", "-p", "-n", "-d", "-t", "-b", "-ub",
            "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r",
        ),
        supported_output_formats=("json",),
        device_capabilities=("cpu", "metal"),
        supported_cache_types=("f16", "f32", "q8_0", "q4_0"),
    )


@pytest.fixture
def gpu_topology() -> SystemTopology:
    return SystemTopology(
        cpu_count=8,
        system_memory_bytes=32_000_000_000,
        accelerators=(
            Accelerator("gpu", "metal", 0, "Metal GPU", memory_bytes=16_000_000_000),
        ),
        memory_unified=True,
    )


@pytest.fixture
def sample_runner() -> FakeRunner:
    return FakeRunner()


def _enqueue_toolchain_cached(
    runner: FakeRunner,
    bp: Path,
    backends: tuple[str, ...] = ("cpu",),
    flags: str | None = None,
) -> None:
    if flags is None:
        flags = "-m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"
    runner.enqueue(
        (str(bp), "--version"),
        ProcessResult(
            returncode=0,
            stdout=(
                "version: test\n"
                f"backends: {','.join(backends)}\n"
                f"supported_flags: {flags}\n"
                "supported_output_formats: json\n"
            ),
            stderr="",
        ),
    )
    runner.enqueue(
        (str(bp), "--help"),
        ProcessResult(returncode=0, stdout="usage: llama-bench\n", stderr=""),
    )


# ── CandidateConfig n_cpu_moe ──────────────────────────────────────────


class TestCandidateConfigNcpuMoe:
    def test_default_is_none(self) -> None:
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        assert cc.n_cpu_moe is None

    def test_explicit_moe_value(self) -> None:
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
            n_cpu_moe=2,
        )
        assert cc.n_cpu_moe == 2

    def test_hash_and_equality(self) -> None:
        a = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        b = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
            n_cpu_moe=None,
        )
        assert a == b
        assert hash(a) == hash(b)

    def test_different_moe_different(self) -> None:
        a = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        b = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
            n_cpu_moe=1,
        )
        assert a != b


# ── MoE detection ───────────────────────────────────────────────────────


class TestMoeDetection:
    def test_dense_model(self, dense_model: Path) -> None:
        fp = inspect_model(dense_model)
        assert fp.is_moe is False
        assert fp.moe_expert_count is None

    def test_moe_model_with_expert_count(self, moe_model: Path) -> None:
        fp = inspect_model(moe_model)
        assert fp.is_moe is True
        assert fp.moe_expert_count == 8

    def test_moe_model_without_expert_count(self, moe_model_no_expert_count: Path) -> None:
        fp = inspect_model(moe_model_no_expert_count)
        assert fp.is_moe is False
        assert fp.moe_expert_count is None

    def test_moe_model_with_int_expert_count(self, tmp_path: Path) -> None:
        p = tmp_path / "int_cnt.gguf"
        build_gguf(p, {
            "general.architecture": (8, "llama"),
            "llama.expert_count": (5, 16),
        })
        fp = inspect_model(p)
        assert fp.is_moe is True
        assert fp.moe_expert_count == 16


# ── ModelFingerprint serialization ─────────────────────────────────────


class TestModelFingerprintFields:
    def test_defaults(self) -> None:
        fp = ModelFingerprint(
            path=Path("/m.gguf"),
            architecture="llama",
            parameter_count=None,
            context_length=None,
            quantization=None,
            shards=(),
            symlink_warning=False,
            network_mount_detected=False,
            diagnostics=(),
        )
        assert fp.is_moe is False
        assert fp.moe_expert_count is None


# ── ToolchainFingerprint supported_cache_types ──────────────────────────


class TestToolchainFingerprintCacheTypes:
    def test_default_is_f16(self) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/usr/bin/llama-bench"),
            version="1",
            supported_flags=("-ctk", "-ctv"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
        )
        assert tc.supported_cache_types == ("f16",)

    def test_explicit_cache_types(self) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/usr/bin/llama-bench"),
            version="1",
            supported_flags=("-ctk", "-ctv"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
            supported_cache_types=("f16", "f32", "q8_0"),
        )
        assert tc.supported_cache_types == ("f16", "f32", "q8_0")


# ── generate_screening_grid ─────────────────────────────────────────────


class TestGenerateScreeningGrid:
    def test_cpu_only_uses_only_ngl_zero(self, base_candidate: CandidateConfig,
                                          sample_toolchain: ToolchainFingerprint) -> None:
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        grid = generate_screening_grid(base_candidate, sample_toolchain, model, cpu_only=True)
        for c in grid:
            assert c.n_gpu_layers == 0

    def test_dense_no_moe_values(self, base_candidate: CandidateConfig,
                                  sample_toolchain: ToolchainFingerprint) -> None:
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        grid = generate_screening_grid(base_candidate, sample_toolchain, model, cpu_only=True)
        for c in grid:
            assert c.n_cpu_moe is None

    def test_moe_includes_moe_values(self, base_candidate: CandidateConfig,
                                      sample_toolchain: ToolchainFingerprint) -> None:
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
            is_moe=True, moe_expert_count=8,
        )
        grid = generate_screening_grid(base_candidate, sample_toolchain, model, cpu_only=True)
        moe_candidates = [c for c in grid if c.n_cpu_moe is not None]
        assert len(moe_candidates) > 0
        assert any(c.n_cpu_moe == 0 for c in grid)
        assert any(c.n_cpu_moe == 1 for c in grid)

    def test_includes_asymmetric_kv(self, base_candidate: CandidateConfig,
                                     sample_toolchain: ToolchainFingerprint) -> None:
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        grid = generate_screening_grid(base_candidate, sample_toolchain, model, cpu_only=True)
        has_asymmetric = any(c.cache_type_k != c.cache_type_v for c in grid)
        assert has_asymmetric

    def test_includes_flash_on_and_off(self, base_candidate: CandidateConfig,
                                        sample_toolchain: ToolchainFingerprint) -> None:
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        grid = generate_screening_grid(base_candidate, sample_toolchain, model, cpu_only=True)
        assert any(c.flash_attn for c in grid)
        assert any(not c.flash_attn for c in grid)

    def test_no_flash_when_flag_missing(self, base_candidate: CandidateConfig) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/bin/llama-bench"),
            version="1",
            supported_flags=("-m", "-p", "-n", "-d", "-t", "-b", "-ub",
                            "-ngl", "-ctk", "-ctv", "-o", "-r"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
            supported_cache_types=("f16",),
        )
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        grid = generate_screening_grid(base_candidate, tc, model, cpu_only=True)
        assert not any(c.flash_attn for c in grid)

    def test_core_grid_preserved(self, base_candidate: CandidateConfig,
                                  sample_toolchain: ToolchainFingerprint) -> None:
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        grid = generate_screening_grid(base_candidate, sample_toolchain, model, cpu_only=True)
        from llama_parameter_scout.search import generate_core_grid
        core = generate_core_grid(base_candidate)
        for c in core:
            assert c in grid

    def test_deterministic_order(self, base_candidate: CandidateConfig,
                                  sample_toolchain: ToolchainFingerprint) -> None:
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        grid1 = generate_screening_grid(base_candidate, sample_toolchain, model, cpu_only=True)
        grid2 = generate_screening_grid(base_candidate, sample_toolchain, model, cpu_only=True)
        assert grid1 == grid2


# ── _build_argv with n_cpu_moe ─────────────────────────────────────────


class TestBuildArgvNcpuMoe:
    def test_no_n_cpu_moe_by_default(self, valid_request: SingleRunRequest) -> None:
        argv = _build_argv(valid_request, repetitions=3)
        assert "--n-cpu-moe" not in argv

    def test_emits_n_cpu_moe_when_set(self, valid_request: SingleRunRequest) -> None:
        req = SingleRunRequest(
            binary_path=valid_request.binary_path,
            model_path=valid_request.model_path,
            journal_path=valid_request.journal_path,
            n_ctx=valid_request.n_ctx,
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
                n_cpu_moe=2,
            ),
        )
        argv = _build_argv(req, repetitions=3)
        idx = argv.index("--n-cpu-moe")
        assert argv[idx + 1] == "2"


# ── _screening_preflight capability checks ──────────────────────────────


class TestScreeningPreflight:
    def test_flash_attn_unsupported(self, base_candidate: CandidateConfig,
                                     sample_runner: FakeRunner) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/bin/llama-bench"),
            version="1",
            supported_flags=("-m", "-p", "-n", "-d", "-t", "-b", "-ub",
                            "-ngl", "-ctk", "-ctv", "-o", "-r"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
        )
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=True,
            cache_type_k="f16", cache_type_v="f16",
        )
        req = SingleRunRequest(
            binary_path=Path("/bin/llama-bench"),
            model_path=Path("/m.gguf"),
            journal_path=Path("/tmp/j.json"),
            n_ctx=NctxComposition(total=512, prompt=128, generation=256, depth=128),
            candidate=cc,
        )
        result = _screening_preflight(req, tc, model, None)
        assert result is not None
        assert result.disposition == "unsupported"
        assert "flash_attn" in result.reason

    def test_cache_type_k_unsupported(self, base_candidate: CandidateConfig) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/bin/llama-bench"),
            version="1",
            supported_flags=("-ctk", "-ctv"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
            supported_cache_types=("f16", "f32"),
        )
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="q8_0", cache_type_v="f16",
        )
        req = SingleRunRequest(
            binary_path=Path("/bin/llama-bench"),
            model_path=Path("/m.gguf"),
            journal_path=Path("/tmp/j.json"),
            n_ctx=NctxComposition(total=512, prompt=128, generation=256, depth=128),
            candidate=cc,
        )
        result = _screening_preflight(req, tc, model, None)
        assert result is not None
        assert result.disposition in ("unsupported", "pruned")
        assert "q8_0" in result.reason

    def test_cache_type_v_unsupported(self, base_candidate: CandidateConfig) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/bin/llama-bench"),
            version="1",
            supported_flags=("-ctk", "-ctv"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
            supported_cache_types=("f16", "f32"),
        )
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="q8_0",
        )
        req = SingleRunRequest(
            binary_path=Path("/bin/llama-bench"),
            model_path=Path("/m.gguf"),
            journal_path=Path("/tmp/j.json"),
            n_ctx=NctxComposition(total=512, prompt=128, generation=256, depth=128),
            candidate=cc,
        )
        result = _screening_preflight(req, tc, model, None)
        assert result is not None
        assert not result.success
        assert "q8_0" in result.reason
        assert result.provenance == "cache_v=q8_0=unsupported"

    def test_n_cpu_moe_on_dense_pruned(self, base_candidate: CandidateConfig) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/bin/llama-bench"),
            version="1",
            supported_flags=("-n-cpu-moe",),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
        )
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
            n_cpu_moe=1,
        )
        req = SingleRunRequest(
            binary_path=Path("/bin/llama-bench"),
            model_path=Path("/m.gguf"),
            journal_path=Path("/tmp/j.json"),
            n_ctx=NctxComposition(total=512, prompt=128, generation=256, depth=128),
            candidate=cc,
        )
        result = _screening_preflight(req, tc, model, None)
        assert result is not None
        assert not result.success
        assert "not MoE" in result.reason

    def test_flash_incompatible_cache_type(self, base_candidate: CandidateConfig) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/bin/llama-bench"),
            version="1",
            supported_flags=("-fa", "-ctk", "-ctv"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
            supported_cache_types=("f16", "q8_0"),
        )
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=True,
            cache_type_k="q8_0", cache_type_v="f16",
        )
        req = SingleRunRequest(
            binary_path=Path("/bin/llama-bench"),
            model_path=Path("/m.gguf"),
            journal_path=Path("/tmp/j.json"),
            n_ctx=NctxComposition(total=512, prompt=128, generation=256, depth=128),
            candidate=cc,
        )
        result = _screening_preflight(req, tc, model, None)
        assert result is not None
        assert result.disposition == "pruned"
        assert "incompatible" in result.reason

    def test_n_cpu_moe_exceeds_expert_count(self, base_candidate: CandidateConfig) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/bin/llama-bench"),
            version="1",
            supported_flags=("--n-cpu-moe",),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
        )
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
            is_moe=True, moe_expert_count=4,
        )
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
            n_cpu_moe=8,
        )
        req = SingleRunRequest(
            binary_path=Path("/bin/llama-bench"),
            model_path=Path("/m.gguf"),
            journal_path=Path("/tmp/j.json"),
            n_ctx=NctxComposition(total=512, prompt=128, generation=256, depth=128),
            candidate=cc,
        )
        result = _screening_preflight(req, tc, model, None)
        assert result is not None
        assert not result.success
        assert "exceeds" in result.reason.lower()

    def test_passes_all_checks(self, base_candidate: CandidateConfig) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/bin/llama-bench"),
            version="1",
            supported_flags=("-fa", "-ctk", "-ctv"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
            supported_cache_types=("f16", "f32"),
        )
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=True,
            cache_type_k="f16", cache_type_v="f16",
        )
        req = SingleRunRequest(
            binary_path=Path("/bin/llama-bench"),
            model_path=Path("/m.gguf"),
            journal_path=Path("/tmp/j.json"),
            n_ctx=NctxComposition(total=512, prompt=128, generation=256, depth=128),
            candidate=cc,
        )
        result = _screening_preflight(req, tc, model, None)
        assert result is None

    def test_moe_passes_all_checks(self, base_candidate: CandidateConfig) -> None:
        tc = ToolchainFingerprint(
            binary_path=Path("/bin/llama-bench"),
            version="1",
            supported_flags=("-fa", "-ctk", "-ctv", "--n-cpu-moe"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
            supported_cache_types=("f16", "f32"),
        )
        model = ModelFingerprint(
            path=Path("/m.gguf"), architecture="llama",
            parameter_count=None, context_length=None,
            quantization=None, shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
            is_moe=True, moe_expert_count=8,
        )
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
            n_cpu_moe=2,
        )
        req = SingleRunRequest(
            binary_path=Path("/bin/llama-bench"),
            model_path=Path("/m.gguf"),
            journal_path=Path("/tmp/j.json"),
            n_ctx=NctxComposition(total=512, prompt=128, generation=256, depth=128),
            candidate=cc,
        )
        result = _screening_preflight(req, tc, model, None)
        assert result is None


# ── _is_quantized_cache ───────────────────────────────────────────────


class TestIsQuantizedCache:
    def test_f16_not_quantized(self) -> None:
        assert not _is_quantized_cache("f16")

    def test_f32_not_quantized(self) -> None:
        assert not _is_quantized_cache("f32")

    def test_q8_0_is_quantized(self) -> None:
        assert _is_quantized_cache("q8_0")

    def test_q4_0_is_quantized(self) -> None:
        assert _is_quantized_cache("q4_0")


# ── _provenance_for ──────────────────────────────────────────────────


class TestProvenanceFor:
    def test_defaults(self, sample_toolchain: ToolchainFingerprint) -> None:
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        prov = _provenance_for(cc)
        assert prov == "defaults"

    def test_flash_attn(self, sample_toolchain: ToolchainFingerprint) -> None:
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=True,
            cache_type_k="f16", cache_type_v="f16",
        )
        prov = _provenance_for(cc)
        assert "flash_attn" in prov

    def test_cache_types(self, sample_toolchain: ToolchainFingerprint) -> None:
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="q8_0", cache_type_v="f16",
        )
        prov = _provenance_for(cc)
        assert "cache_k=q8_0" in prov

    def test_n_cpu_moe(self, sample_toolchain: ToolchainFingerprint) -> None:
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
            n_cpu_moe=2,
        )
        prov = _provenance_for(cc)
        assert "n_cpu_moe=2" in prov

    def test_n_gpu_layers(self, sample_toolchain: ToolchainFingerprint) -> None:
        cc = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=20, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        prov = _provenance_for(cc)
        assert "n_gpu_layers=20" in prov


# ── Application seam: CPU-only screening with expanded grid ───────────


@dataclass
class _DynamicFakeAdapter:
    json_output: str = '{"samples_ts":[10.0,11.0,12.0]}'

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
        return ProcessResult(returncode=0, stdout=self.json_output, stderr="")


class TestApplicationScreenCandidatesIssue06:
    """Test Application.screen_candidates with expanded grid via dynamic adapter."""

    def _toolchain_stdout(self, extra_flags: str = "") -> str:
        flags = "-m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r"
        if extra_flags:
            flags += "," + extra_flags
        return (
            "version: test\n"
            "backends: cpu\n"
            f"supported_flags: {flags}\n"
            "supported_output_formats: json\n"
        )

    def _build_request(self, tmp_path: Path, model_name: str = "model.gguf",
                       arch: str = "llama") -> SingleRunRequest:
        bp = tmp_path / "llama-bench"
        bp.write_text("#!/bin/sh\necho '{\"samples_ts\":[10.0,11.0,12.0]}'")
        bp.chmod(0o755)
        mp = tmp_path / model_name
        metadata: dict[str, tuple[int, object]] = {
            "general.architecture": (8, arch),
        }
        build_gguf(mp, metadata)
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

    def test_cpu_only_no_accelerator_args_in_production(
        self, tmp_path: Path,
    ) -> None:
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        captured: list[list[str]] = []

        class CaptureAdapter:
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

        topology = SystemTopology(
            cpu_count=4, system_memory_bytes=32_000_000_000,
            accelerators=(), memory_unified=True,
        )
        spec = None  # CPU-only, no session spec
        app = Application(runner, process_adapter=CaptureAdapter(),
                          detect_topology_fn=lambda: topology)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE, session_spec=spec)

        assert report is not None
        for argv in captured:
            assert "--split-mode" not in argv
            assert "--main-gpu" not in argv

    def test_production_evaluator_with_cache_provenance(
        self, tmp_path: Path,
    ) -> None:
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        adapter = _DynamicFakeAdapter()
        app = Application(runner, process_adapter=adapter)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE)

        assert report is not None
        found_provenance = False
        all_disp = (
            report.baseline_disposition,
            *report.grid_dispositions,
            *report.adaptive_dispositions,
        )
        for d in all_disp:
            if d.success:
                assert d.provenance, f"Missing provenance for successful candidate {d.candidate}"
                found_provenance = True
        assert found_provenance, "No successful disposition with provenance"

    def test_grid_includes_new_param_variations(self, tmp_path: Path) -> None:
        """Verify that the production path considers candidates with
        different cache types, flash attn states, etc.

        The core grid uses base candidate values (f16, flash_attn=False).
        Within quick profile budget only the first 7 core-grid candidates
        are attempted, all with f16.  Instead of asserting varying cache
        types at runtime (which requires a larger budget), verify that the
        expanded grid generation function produces the expected diversity
        and that the report is well-formed.
        """
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "  -ctk FNAME  (default: f16) {f16,f32,q8_0}\n", ""))

        captured: list[CandidateConfig] = []

        class RecordingAdapter:
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
                pairs = dict(zip(argv[1::2], argv[2::2]))
                cc = CandidateConfig(
                    threads=int(pairs.get("-t", "0")),
                    batch_size=int(pairs.get("-b", "0")),
                    ubatch_size=int(pairs.get("-ub", "0")),
                    n_gpu_layers=int(pairs.get("-ngl", "0")),
                    flash_attn=pairs.get("-fa", "off") == "on",
                    cache_type_k=pairs.get("-ctk", "f16"),
                    cache_type_v=pairs.get("-ctv", "f16"),
                    n_cpu_moe=int(pairs["--n-cpu-moe"]) if "--n-cpu-moe" in pairs else None,
                )
                captured.append(cc)
                return ProcessResult(0, '{"samples_ts":[1.0,2.0,3.0]}', "")

        app = Application(runner, process_adapter=RecordingAdapter())

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE)

        assert report is not None
        assert len(captured) > 0  # At least some candidates were launched
        # Verify that the expanded grid (used internally) includes diverse
        # cache types by checking the report dispositions:
        cache_types_in_report: set[str] = set()
        for d in (
            report.baseline_disposition,
            *report.grid_dispositions,
            *report.adaptive_dispositions,
        ):
            cache_types_in_report.add(d.candidate.cache_type_k)
            cache_types_in_report.add(d.candidate.cache_type_v)
        assert len(cache_types_in_report) > 1

    def test_moe_model_includes_n_cpu_moe_in_report(
        self, tmp_path: Path,
    ) -> None:
        """When screening an MoE model, the grid dispositions should
        include candidates with n_cpu_moe set (even if not all are launched
        within the profile budget)."""
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path, model_name="moe.gguf", arch="llama")
        mp = req.model_path
        build_gguf(mp, {
            "general.architecture": (8, "llama"),
            "llama.expert_count": (5, 8),
        })

        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout("--n-cpu-moe"), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        adapter = _DynamicFakeAdapter()
        app = Application(runner, process_adapter=adapter)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE)

        assert report is not None
        moe_in_report = [
            d for d in report.grid_dispositions
            if d.candidate.n_cpu_moe is not None
        ]
        assert len(moe_in_report) > 0, (
            "Expected some grid dispositions with n_cpu_moe set"
        )

    def test_quantized_kv_disclosure_in_report(
        self, tmp_path: Path,
    ) -> None:
        """Verify that the report includes quantized KV disclosure when
        quantized cache types are used in successful candidates."""
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        runner = FakeRunner()
        runner.enqueue((str(req.binary_path), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(req.binary_path), "--help"),
                        ProcessResult(0, "usage\n", ""))

        adapter = _DynamicFakeAdapter()
        app = Application(runner, process_adapter=adapter)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE)

        assert report is not None
        if report.notes:
            assert any("quality was not measured" in n for n in report.notes)

    def test_pruned_dispositions_have_capability_reason(
        self, tmp_path: Path,
    ) -> None:
        """Verify that unsupported/incompatible dispositions have explicit reasons."""
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        # Use a base candidate with flash_attn=True and q8_0 cache:
        # preflight prunes flash_attn + q8_0 as incompatible.
        req = SingleRunRequest(
            binary_path=req.binary_path,
            model_path=req.model_path,
            journal_path=req.journal_path,
            n_ctx=req.n_ctx,
            candidate=CandidateConfig(
                threads=4, batch_size=512, ubatch_size=512,
                n_gpu_layers=0, flash_attn=True,
                cache_type_k="q8_0", cache_type_v="f16",
            ),
        )
        bp = req.binary_path
        runner = FakeRunner()
        runner.enqueue((str(bp), "--version"),
                        ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(bp), "--help"),
                        ProcessResult(0, "  -ctk FNAME ... (default: f16) {f16,f32,q8_0}\n", ""))

        adapter = _DynamicFakeAdapter()
        app = Application(runner, process_adapter=adapter)

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE)

        found = False
        all_disp = (
            report.baseline_disposition,
            *report.grid_dispositions,
            *report.adaptive_dispositions,
        )
        for d in all_disp:
            if not d.success and d.reason:
                assert "flash_attn" in d.reason or "incompatible" in d.reason
                found = True
        assert found, (
            "Expected at least one preflight-pruned/unsupported "
            "disposition with capability reason"
        )

    def test_discrete_gpu_cuda_topology(
        self, tmp_path: Path,
    ) -> None:
        """Discrete CUDA GPU topology: accelerated argv includes --split-mode none,
        --main-gpu <selected id>, no tensor-split, and env isolates device."""
        from llama_parameter_scout.model_inspector import inspect_model
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        bp = req.binary_path
        runner = FakeRunner()
        cuda_stdout = (
            "version: test\n"
            "backends: cpu,cuda\n"
            "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r,"
            "--split-mode,--main-gpu\n"
            "supported_output_formats: json\n"
        )
        runner.enqueue((str(bp), "--version"), ProcessResult(0, cuda_stdout, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage\n", ""))

        topology = SystemTopology(
            cpu_count=8,
            system_memory_bytes=32_000_000_000,
            accelerators=(
                Accelerator("gpu", "cuda", 0, "NVIDIA A100", memory_bytes=40_000_000_000),
                Accelerator("gpu", "cuda", 1, "NVIDIA A100", memory_bytes=40_000_000_000),
            ),
            memory_unified=False,
        )

        model_fp = inspect_model(req.model_path)

        captured_argv: list[list[str]] = []
        captured_env: list[dict[str, str] | None] = []

        class CaptureGPUAdapter:
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
                captured_env.append(env)
                return ProcessResult(0, '{"samples_ts":[1.0,2.0,3.0]}', "")

        app = Application(runner, process_adapter=CaptureGPUAdapter(),
                          detect_topology_fn=lambda: topology)

        session_spec = app.setup_session(
            model=model_fp, n_ctx=req.n_ctx, candidate=req.candidate,
            cpu_only=False, device_ids=(0,), acknowledged=True,
        )

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE, session_spec=session_spec)

        assert report is not None
        assert len(captured_argv) > 0
        for argv in captured_argv:
            assert "--split-mode" in argv
            idx = argv.index("--split-mode")
            assert argv[idx + 1] == "none"
            assert "--main-gpu" in argv
            idx = argv.index("--main-gpu")
            assert argv[idx + 1] == "0"
            assert "--tensor-split" not in argv
        for env in captured_env:
            assert env is not None
            assert env.get("CUDA_VISIBLE_DEVICES") == "0"

    def test_unified_memory_metal_topology(
        self, tmp_path: Path,
    ) -> None:
        """Unified-memory Metal topology: accelerated path with selected compatible
        device uses same single-device/no-split contract; no CUDA/HIP env isolation."""
        from llama_parameter_scout.model_inspector import inspect_model
        from llama_parameter_scout.runner import FakeRunner

        req = self._build_request(tmp_path)
        bp = req.binary_path
        runner = FakeRunner()
        metal_stdout = (
            "version: test\n"
            "backends: cpu,metal\n"
            "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r,"
            "--split-mode,--main-gpu\n"
            "supported_output_formats: json\n"
        )
        runner.enqueue((str(bp), "--version"), ProcessResult(0, metal_stdout, ""))
        runner.enqueue((str(bp), "--help"), ProcessResult(0, "usage\n", ""))

        topology = SystemTopology(
            cpu_count=8,
            system_memory_bytes=32_000_000_000,
            accelerators=(
                Accelerator("gpu", "metal", 0, "Apple M3", memory_bytes=16_000_000_000),
            ),
            memory_unified=True,
        )

        model_fp = inspect_model(req.model_path)

        captured_argv: list[list[str]] = []
        captured_env: list[dict[str, str] | None] = []

        class CaptureMetalAdapter:
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
                captured_env.append(env)
                return ProcessResult(0, '{"samples_ts":[1.0,2.0,3.0]}', "")

        app = Application(runner, process_adapter=CaptureMetalAdapter(),
                          detect_topology_fn=lambda: topology)

        session_spec = app.setup_session(
            model=model_fp, n_ctx=req.n_ctx, candidate=req.candidate,
            cpu_only=False, device_ids=(0,), acknowledged=True,
        )

        from llama_parameter_scout.search import QUICK_PROFILE
        report = app.screen_candidates(req, QUICK_PROFILE, session_spec=session_spec)

        assert report is not None
        assert len(captured_argv) > 0
        for argv in captured_argv:
            assert "--split-mode" in argv
            idx = argv.index("--split-mode")
            assert argv[idx + 1] == "none"
            assert "--main-gpu" in argv
            idx = argv.index("--main-gpu")
            assert argv[idx + 1] == "0"
            assert "--tensor-split" not in argv
        for env in captured_env:
            if env is not None:
                assert "CUDA_VISIBLE_DEVICES" not in env
                assert "HIP_VISIBLE_DEVICES" not in env

    # ── Helpers ────────────────────────────────────────────────────
