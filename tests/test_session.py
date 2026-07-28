from __future__ import annotations

from pathlib import Path

import pytest

from llama_parameter_scout.application import (
    Application,
    _check_resource_budget,
    _compute_model_weight_bytes,
    _estimate_advisory,
)
from llama_parameter_scout.models import (
    Accelerator,
    CandidateConfig,
    CandidateError,
    ContextAdvisory,
    ContextExceededWithoutAcknowledgementError,
    InconsistentNglError,
    ModelFingerprint,
    MultipleAcceleratorsSelectedError,
    NctxComposition,
    NoAcceleratorSelectedError,
    NonpositiveBudgetError,
    ResourceBudget,
    SessionSetupError,
    SessionSpecification,
    ShardFingerprint,
    SystemTopology,
    UnknownDeviceError,
)
from llama_parameter_scout.runner import FakeRunner


def _make_topology(
    cpu_count: int = 8,
    memory: int = 32_000_000_000,
    accels: tuple[Accelerator, ...] = (),
    unified: bool = False,
) -> SystemTopology:
    return SystemTopology(
        cpu_count=cpu_count,
        system_memory_bytes=memory,
        accelerators=accels,
        memory_unified=unified,
    )


def _make_model(context_length: int | None = 4096, path: Path | None = None) -> ModelFingerprint:
    return ModelFingerprint(
        path=path or Path("/fake/model.gguf"),
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=context_length,
        quantization="Q4_K_M",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )


SAMPLE_NCTX = NctxComposition(total=2048, prompt=512, generation=1024, depth=512)
SAMPLE_CANDIDATE = CandidateConfig(
    threads=4, batch_size=512, ubatch_size=512,
    n_gpu_layers=0, flash_attn=False,
    cache_type_k="f16", cache_type_v="f16",
)


def test_cpu_only_on_multi_gpu_host() -> None:
    topology = _make_topology(
        accels=(
            Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),
            Accelerator("gpu", "cuda", 1, "RTX 4080", memory_bytes=16_000_000_000),
        ),
    )
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec.cpu_only is True
    assert spec.selected_device_ids == ()


def test_cpu_only_no_accelerator() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec.cpu_only is True


def test_one_gpu_selected_from_multiple() -> None:
    topology = _make_topology(
        accels=(
            Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),
            Accelerator("gpu", "cuda", 1, "RTX 4080", memory_bytes=16_000_000_000),
        ),
    )
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=False,
        device_ids=(1,),
        acknowledged=True,
    )
    assert spec.cpu_only is False
    assert spec.selected_device_ids == (1,)


def test_one_gpu_without_selection_raises() -> None:
    topology = _make_topology(
        accels=(Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),),
    )
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(NoAcceleratorSelectedError):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=False,
            acknowledged=True,
        )


def test_no_accel_and_cpu_false_raises() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(NoAcceleratorSelectedError):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=False,
            acknowledged=True,
        )


def test_multiple_gpu_error() -> None:
    topology = _make_topology(
        accels=(
            Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),
            Accelerator("gpu", "cuda", 1, "RTX 4080", memory_bytes=16_000_000_000),
        ),
    )
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(NoAcceleratorSelectedError, match="no --device"):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=False,
        )


def test_multiple_device_ids_raises() -> None:
    topology = _make_topology(
        accels=(
            Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),
            Accelerator("gpu", "cuda", 1, "RTX 4080", memory_bytes=16_000_000_000),
        ),
    )
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(MultipleAcceleratorsSelectedError):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=False,
            device_ids=(0, 1),
            acknowledged=True,
        )


def test_unknown_device_id_error() -> None:
    topology = _make_topology(
        accels=(Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),),
    )
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(UnknownDeviceError, match="not found"):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=False,
            device_ids=(99,),
            acknowledged=True,
        )


def test_cpu_only_rejects_gpu_offload() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    candidate = CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=20, flash_attn=False,
        cache_type_k="f16", cache_type_v="f16",
    )
    with pytest.raises(InconsistentNglError, match="n_gpu_layers"):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=candidate,
            cpu_only=True,
            acknowledged=True,
        )


def test_non_cpu_allows_gpu_offload() -> None:
    topology = _make_topology(
        accels=(Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),),
    )
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    candidate = CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=20, flash_attn=False,
        cache_type_k="f16", cache_type_v="f16",
    )
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=candidate,
        cpu_only=False,
        device_ids=(0,),
        acknowledged=True,
    )
    assert spec.cpu_only is False


def test_returns_session_specification() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert isinstance(spec, SessionSpecification)
    assert spec.n_ctx.total == 2048


def test_session_has_fingerprint() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec.fingerprint != ""
    assert len(spec.fingerprint) == 64


def test_fingerprint_is_deterministic() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec1 = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    spec2 = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec1.fingerprint == spec2.fingerprint


def test_different_budget_gives_different_fingerprint() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec1 = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    spec2 = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
        budget_overrides=(8_000_000_000, None),
    )
    assert spec1.fingerprint != spec2.fingerprint


def test_fingerprint_changes_with_selected_device_memory() -> None:
    topology = _make_topology(
        accels=(
            Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),
        ),
    )
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=False,
        device_ids=(0,),
        acknowledged=True,
    )
    topology2 = _make_topology(
        accels=(
            Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=16_000_000_000),
        ),
    )
    app2 = Application(FakeRunner(), detect_topology_fn=lambda: topology2)
    spec2 = app2.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=False,
        device_ids=(0,),
        acknowledged=True,
    )
    assert spec.fingerprint != spec2.fingerprint


def test_fingerprint_changes_with_shard_digest() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    model1 = ModelFingerprint(
        path=Path("/fake/model.gguf"),
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=4096,
        quantization="Q4_K_M",
        shards=(ShardFingerprint(Path("/fake/shard1.gguf"), 100, 1.0, "a" * 64),),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )
    model2 = ModelFingerprint(
        path=Path("/fake/model.gguf"),
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=4096,
        quantization="Q4_K_M",
        shards=(ShardFingerprint(Path("/fake/shard1.gguf"), 100, 1.0, "b" * 64),),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )
    spec1 = app.setup_session(
        model1, SAMPLE_NCTX, SAMPLE_CANDIDATE, cpu_only=True, acknowledged=True)
    spec2 = app.setup_session(
        model2, SAMPLE_NCTX, SAMPLE_CANDIDATE, cpu_only=True, acknowledged=True)
    assert spec1.fingerprint != spec2.fingerprint


def test_fingerprint_changes_with_device_description() -> None:
    topology1 = _make_topology(
        accels=(Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),),
    )
    topology2 = _make_topology(
        accels=(Accelerator("gpu", "cuda", 0, "RTX 4090 Ti", memory_bytes=24_000_000_000),),
    )
    app1 = Application(FakeRunner(), detect_topology_fn=lambda: topology1)
    app2 = Application(FakeRunner(), detect_topology_fn=lambda: topology2)
    spec1 = app1.setup_session(
        _make_model(), SAMPLE_NCTX, SAMPLE_CANDIDATE,
        cpu_only=False, device_ids=(0,), acknowledged=True,
    )
    spec2 = app2.setup_session(
        _make_model(), SAMPLE_NCTX, SAMPLE_CANDIDATE,
        cpu_only=False, device_ids=(0,), acknowledged=True,
    )
    assert spec1.fingerprint != spec2.fingerprint


def test_advisory_included() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert isinstance(spec.advisory, ContextAdvisory)
    assert spec.advisory.likely_fit_lower > 0
    assert spec.advisory.likely_fit_upper >= spec.advisory.likely_fit_lower
    assert spec.advisory.uncertain_maximum >= spec.advisory.likely_fit_upper
    assert "known" in spec.advisory.note.lower()
    assert "headroom" in spec.advisory.note.lower()


def test_budget_included() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert isinstance(spec.budget, ResourceBudget)
    assert spec.budget.ram_bytes > 0
    assert spec.budget.headroom_ratio >= 0


def test_advisory_estimate_lower_and_upper_range() -> None:
    budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    advisory = _estimate_advisory(_make_model(), SAMPLE_NCTX, budget)
    assert advisory.likely_fit_lower < advisory.likely_fit_upper
    assert advisory.uncertain_maximum >= advisory.likely_fit_upper


def test_advisory_note_says_estimate() -> None:
    budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    advisory = _estimate_advisory(_make_model(), SAMPLE_NCTX, budget)
    assert "Likely-fit range" in advisory.note
    assert "not a tested guarantee" in advisory.note.lower()


def test_advisory_sensitive_to_model_footprint() -> None:
    budget = ResourceBudget(ram_bytes=128_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    small_model = _make_model(context_length=128000)
    large_model = ModelFingerprint(
        path=Path("/fake/large.gguf"),
        architecture="llama",
        parameter_count=70_000_000_000,
        context_length=128000,
        quantization="Q4_K_M",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )
    small_adv = _estimate_advisory(small_model, SAMPLE_NCTX, budget)
    large_adv = _estimate_advisory(large_model, SAMPLE_NCTX, budget)
    assert large_adv.likely_fit_upper < small_adv.likely_fit_upper


def test_advisory_sensitive_to_cache_type() -> None:
    budget = ResourceBudget(ram_bytes=256_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    model = _make_model(context_length=256000)
    f32_adv = _estimate_advisory(model, SAMPLE_NCTX, budget, cache_type_k="f32", cache_type_v="f32")
    f16_adv = _estimate_advisory(model, SAMPLE_NCTX, budget, cache_type_k="f16", cache_type_v="f16")
    assert f32_adv.likely_fit_upper < f16_adv.likely_fit_upper


def test_advisory_q4_0_permits_larger_context_than_f16() -> None:
    budget = ResourceBudget(ram_bytes=256_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    model = _make_model(context_length=None)
    q4_adv = _estimate_advisory(
        model, SAMPLE_NCTX, budget, cache_type_k="q4_0", cache_type_v="q4_0",
    )
    f16_adv = _estimate_advisory(model, SAMPLE_NCTX, budget, cache_type_k="f16", cache_type_v="f16")
    assert q4_adv.kv_cache_bytes_estimate < f16_adv.kv_cache_bytes_estimate
    assert q4_adv.likely_fit_upper > f16_adv.likely_fit_upper


def test_advisory_sensitive_to_cache_type_v_only() -> None:
    budget = ResourceBudget(ram_bytes=256_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    model = _make_model(context_length=None)
    f16_f16 = _estimate_advisory(model, SAMPLE_NCTX, budget, cache_type_k="f16", cache_type_v="f16")
    f16_q4 = _estimate_advisory(model, SAMPLE_NCTX, budget, cache_type_k="f16", cache_type_v="q4_0")
    assert f16_q4.kv_cache_bytes_estimate < f16_f16.kv_cache_bytes_estimate
    assert f16_q4.likely_fit_upper > f16_f16.likely_fit_upper


def test_advisory_sensitive_to_budget() -> None:
    small_budget = ResourceBudget(ram_bytes=8_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    large_budget = ResourceBudget(ram_bytes=64_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    model = _make_model(context_length=256000)
    small_adv = _estimate_advisory(model, SAMPLE_NCTX, small_budget)
    large_adv = _estimate_advisory(model, SAMPLE_NCTX, large_budget)
    assert small_adv.likely_fit_upper < large_adv.likely_fit_upper


def test_advisory_known_training_context_status() -> None:
    budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    model = _make_model(context_length=4096)
    advisory = _estimate_advisory(model, SAMPLE_NCTX, budget)
    assert advisory.training_context_status == "known"
    assert advisory.training_context_tokens == 4096


def test_advisory_unknown_training_context_status() -> None:
    budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    model = _make_model(context_length=None)
    advisory = _estimate_advisory(model, SAMPLE_NCTX, budget)
    assert advisory.training_context_status == "unknown"
    assert advisory.training_context_tokens is None


def test_advisory_conflicted_training_context_status() -> None:
    budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    model = ModelFingerprint(
        path=Path("/fake/model.gguf"),
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=4096,
        quantization="Q4_K_M",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=("context length conflict",),
    )
    advisory = _estimate_advisory(model, SAMPLE_NCTX, budget)
    assert advisory.training_context_status == "conflicted"
    assert advisory.training_context_tokens == 4096


def test_advisory_known_context_bounds_upper() -> None:
    budget = ResourceBudget(ram_bytes=256_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    known_model = _make_model(context_length=1024)
    known_adv = _estimate_advisory(known_model, SAMPLE_NCTX, budget)
    assert known_adv.likely_fit_upper <= 1024


def test_resource_gate_sensitive_to_cache_type_v() -> None:
    topology = SystemTopology(
        cpu_count=8, system_memory_bytes=1_000_000_000,
        accelerators=(), memory_unified=True,
    )
    model = _make_model(context_length=4096)
    budget = ResourceBudget(ram_bytes=900_000_000, vram_bytes=None, headroom_ratio=0.0)
    nctx = NctxComposition(total=2048, prompt=512, generation=1024, depth=512)
    spec = SessionSpecification(
        topology=topology, cpu_only=True, selected_device_ids=(),
        n_ctx=nctx, budget=budget, advisory=None,  # type: ignore[arg-type]
        model=model, acknowledged=True, fingerprint="x",
    )
    f16_both = CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=0, flash_attn=False,
        cache_type_k="f16", cache_type_v="f16",
    )
    f16_q4_v = CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=0, flash_attn=False,
        cache_type_k="f16", cache_type_v="q4_0",
    )
    outcome_f16 = _check_resource_budget(spec, nctx, f16_both)
    outcome_mixed = _check_resource_budget(spec, nctx, f16_q4_v)
    assert outcome_f16 is not None
    assert outcome_f16.disposition == "budget_exhausted"
    assert outcome_mixed is None


def test_fingerprint_deterministic() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec1 = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    spec2 = app.setup_session(
        model=_make_model(),
        n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec1.fingerprint != spec2.fingerprint


def test_fingerprint_changes_with_nctx() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec1 = app.setup_session(
        model=_make_model(),
        n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    spec2 = app.setup_session(
        model=_make_model(),
        n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec1.fingerprint != spec2.fingerprint


def test_fingerprint_includes_model() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=_make_model(),
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec.model == _make_model()


def test_context_above_training_without_acknowledgement_raises() -> None:
    topology = _make_topology()
    model = _make_model(context_length=1024)
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(ContextExceededWithoutAcknowledgementError):
        app.setup_session(
            model=model,
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=True,
            acknowledged=False,
        )


def test_context_above_training_with_acknowledgement_ok() -> None:
    topology = _make_topology()
    model = _make_model(context_length=1024)
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=model,
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec.acknowledged is True


def test_unknown_context_without_acknowledgement_raises() -> None:
    topology = _make_topology()
    model = _make_model(context_length=None)
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(ContextExceededWithoutAcknowledgementError):
        app.setup_session(
            model=model,
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=True,
            acknowledged=False,
        )


def test_unknown_context_with_acknowledgement_ok() -> None:
    topology = _make_topology()
    model = _make_model(context_length=None)
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=model,
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec.acknowledged is True


def test_nonpositive_budget_ram_raises() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(NonpositiveBudgetError):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=True,
            budget_overrides=(0, None),
            acknowledged=True,
        )


def test_negative_headroom_raises() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(SessionSetupError):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=True,
            headroom_ratio=-0.1,
            acknowledged=True,
        )


def test_session_spec_has_model() -> None:
    topology = _make_topology()
    model = _make_model()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    spec = app.setup_session(
        model=model,
        n_ctx=SAMPLE_NCTX,
        candidate=SAMPLE_CANDIDATE,
        cpu_only=True,
        acknowledged=True,
    )
    assert spec.model == model


def test_unknown_vram_on_selected_device_raises() -> None:
    topology = _make_topology(
        accels=(Accelerator("gpu", "cuda", 0, "GPU"),),
    )
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    with pytest.raises(SessionSetupError, match="Cannot determine VRAM"):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=SAMPLE_CANDIDATE,
            cpu_only=False,
            device_ids=(0,),
            acknowledged=True,
        )


def test_compute_model_weight_bytes_single_model(tmp_path: Path) -> None:
    p = tmp_path / "model.gguf"
    p.write_bytes(b"\x00" * 1000)
    model = ModelFingerprint(
        path=p, architecture="llama", parameter_count=None,
        context_length=None, quantization=None, shards=(),
        symlink_warning=False, network_mount_detected=False, diagnostics=(),
    )
    assert _compute_model_weight_bytes(model) == 1000


def test_compute_model_weight_bytes_shards_only(tmp_path: Path) -> None:
    p = tmp_path / "model-00001-of-00002.gguf"
    p.write_bytes(b"\x00" * 500)
    s2 = tmp_path / "model-00002-of-00002.gguf"
    s2.write_bytes(b"\x01" * 1500)
    model = ModelFingerprint(
        path=p, architecture="llama", parameter_count=None,
        context_length=None, quantization=None,
        shards=(
            ShardFingerprint(p, 500, 1.0, "a" * 64),
            ShardFingerprint(s2, 1500, 2.0, "b" * 64),
        ),
        symlink_warning=False, network_mount_detected=False, diagnostics=(),
    )
    assert _compute_model_weight_bytes(model) == 2000


def test_build_workload_session_spec_non_f16_cache_reflected_in_advisory() -> None:
    from llama_parameter_scout.application import _build_workload_session_spec

    topology = _make_topology()
    model = _make_model()
    base_spec = SessionSpecification(
        topology=topology, cpu_only=True, selected_device_ids=(),
        n_ctx=SAMPLE_NCTX,
        budget=ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1),
        advisory=None,  # type: ignore[arg-type]
        model=model, acknowledged=True, fingerprint="base",
    )
    workload_nctx = NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024)
    candidate = CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=0, flash_attn=False,
        cache_type_k="q8_0", cache_type_v="q4_0",
    )
    spec_f16 = _build_workload_session_spec(base_spec, workload_nctx)
    spec_non_f16 = _build_workload_session_spec(base_spec, workload_nctx, candidate=candidate)
    assert spec_non_f16.advisory.kv_cache_bytes_estimate < spec_f16.advisory.kv_cache_bytes_estimate
    assert spec_non_f16.advisory.uncertain_maximum > spec_f16.advisory.uncertain_maximum


def test_setup_session_rejects_n_gpu_layers_below_minus_one() -> None:
    topology = _make_topology()
    app = Application(FakeRunner(), detect_topology_fn=lambda: topology)
    candidate = CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=-2, flash_attn=False,
        cache_type_k="f16", cache_type_v="f16",
    )
    with pytest.raises(CandidateError, match="n_gpu_layers"):
        app.setup_session(
            model=_make_model(),
            n_ctx=SAMPLE_NCTX,
            candidate=candidate,
            cpu_only=True,
            acknowledged=True,
        )
