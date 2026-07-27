from __future__ import annotations

from pathlib import Path

from llama_parameter_scout.models import (
    Accelerator,
    CandidateConfig,
    ContextAdvisory,
    MemoryEvidence,
    ModelFingerprint,
    MtpComparison,
    MtpDraftSettings,
    MtpMeasurement,
    MtpProfileReport,
    MtpSearchProfile,
    NctxComposition,
    ResourceBudget,
    SessionSpecification,
    ShardFingerprint,
    SystemTopology,
    ToolchainFingerprint,
    ValidatedCandidate,
    ValidatedRecommendation,
)
from llama_parameter_scout.mtp import (
    FakeServerLifecycle,
    check_combined_memory_fit,
    check_mtp_memory_fit,
    collect_mtp_profile,
    generate_mtp_draft_grid,
    select_mtp_main_candidate,
)
from llama_parameter_scout.runner import ProcessResult


def _ctx2048() -> NctxComposition:
    return NctxComposition(total=2048, prompt=512, generation=1024, depth=512)


def _make_session_spec() -> SessionSpecification:
    topology = SystemTopology(
        cpu_count=8, system_memory_bytes=32_000_000_000,
        accelerators=(Accelerator(kind="cpu", backend="cpu", device_id=0, description="CPU"),),
        memory_unified=True,
    )
    budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    advisory = ContextAdvisory(
        likely_fit_lower=1000, likely_fit_upper=2000,
        uncertain_maximum=3000, model_weight_bytes=8_000_000_000,
        kv_cache_bytes_estimate=500_000_000,
    )
    model = ModelFingerprint(
        path=Path("/main.gguf"), architecture="llama",
        parameter_count=7_000_000_000, context_length=4096,
        quantization="Q4_K_M", shards=(),
        symlink_warning=False, network_mount_detected=False, diagnostics=(),
    )
    return SessionSpecification(
        topology=topology, cpu_only=True, selected_device_ids=(),
        n_ctx=_ctx2048(), budget=budget, advisory=advisory,
        model=model, acknowledged=True, fingerprint="test",
    )


def _make_toolchain_and_model() -> tuple[ToolchainFingerprint, ModelFingerprint]:
    tc = ToolchainFingerprint(
        binary_path=Path("/server"), version="b1",
        supported_flags=(
            "-m", "--host", "--port", "-c", "-t", "-ngl", "-fa",
            "-ctk", "-ctv", "-b", "--ubatch-size",
        ),
        supported_output_formats=("json",),
        device_capabilities=("cpu",),
    )
    model = ModelFingerprint(
        path=Path("/main.gguf"), architecture="llama",
        parameter_count=7_000_000_000, context_length=4096,
        quantization="Q4_K_M", shards=(),
        symlink_warning=False, network_mount_detected=False, diagnostics=(),
    )
    return tc, model


def _make_validated(
    candidate: CandidateConfig,
    balanced: float = 1.0,
    memory: MemoryEvidence | None = None,
) -> ValidatedCandidate:
    return ValidatedCandidate(
        candidate=candidate,
        pp_score=balanced,
        tg_score=balanced,
        balanced=balanced,
        pp_median=100.0,
        tg_median=100.0,
        pp_relative_dispersion=0.05,
        tg_relative_dispersion=0.05,
        memory_evidence=memory,
    )


def _candidate(**kw: object) -> CandidateConfig:
    return CandidateConfig(
        threads=kw.get("threads", 4),  # type: ignore[arg-type]
        batch_size=kw.get("batch_size", 512),  # type: ignore[arg-type]
        ubatch_size=kw.get("ubatch_size", 512),  # type: ignore[arg-type]
        n_gpu_layers=kw.get("n_gpu_layers", 0),  # type: ignore[arg-type]
        flash_attn=kw.get("flash_attn", False),  # type: ignore[arg-type]
        cache_type_k=kw.get("cache_type_k", "f16"),  # type: ignore[arg-type]
        cache_type_v=kw.get("cache_type_v", "f16"),  # type: ignore[arg-type]
        n_cpu_moe=kw.get("n_cpu_moe"),  # type: ignore[arg-type]
    )


def _draft_fingerprint() -> ModelFingerprint:
    return ModelFingerprint(
        path=Path("/draft.gguf"),
        architecture="llama",
        parameter_count=500_000_000,
        context_length=4096,
        quantization="Q8_0",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )


def _mk_comparison(benefit: str = "not_established") -> MtpComparison:
    off = MtpMeasurement(
        mode="off", samples_tps=(50.0, 51.0, 52.0),
        median_tps=51.0, dispersion_tps=1.0,
    )
    on = MtpMeasurement(
        mode="on", samples_tps=(51.0, 52.0, 53.0),
        median_tps=52.0, dispersion_tps=1.0,
    )
    pct: float | None = None
    if benefit == "established":
        pct = ((on.median_tps - off.median_tps) / off.median_tps) * 100
    return MtpComparison(
        off=off, on=on, benefit=benefit,  # type: ignore[arg-type]
        improvement_pct=pct, tie_bound_tps=1.0,
    )


_budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)


def _ev_established(
    c: CandidateConfig, d: MtpDraftSettings,
) -> tuple[MtpComparison, MemoryEvidence]:
    return _mk_comparison("established"), MemoryEvidence(
        ram_bytes=4_000_000_000, quality="complete",
    )


def _ev_not_established(
    c: CandidateConfig, d: MtpDraftSettings,
) -> tuple[MtpComparison, MemoryEvidence]:
    return _mk_comparison("not_established"), MemoryEvidence(quality="complete")


def _ev_partial(
    c: CandidateConfig, d: MtpDraftSettings,
) -> tuple[MtpComparison, MemoryEvidence]:
    return _mk_comparison("established"), MemoryEvidence(
        quality="partial", ram_bytes=4_000_000_000,
    )


# ── Test 1: exact preservation of normal winner ──────────────────


class TestNormalWinnerPreserved:
    def test_winner_unchanged(self) -> None:
        w_cand = _candidate(threads=4)
        winner = _make_validated(w_cand, memory=MemoryEvidence(
            ram_bytes=8_000_000_000, quality="complete",
        ))
        validation = ValidatedRecommendation(
            recommendation=winner, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=2, draft_gpu_layers_values=(0,)),
            evaluator=lambda c, d: (_mk_comparison(), MemoryEvidence(quality="incomparable")),
        )

        assert report.normal_winner.candidate == w_cand
        assert report.normal_winner.balanced == 1.0
        assert isinstance(report, MtpProfileReport)
        assert report.benefit == "not_established"


# ── Rank shift: best established MTP candidate wins ──────────────


class TestRankShift:
    def test_higher_on_median_wins(self) -> None:
        calls: list[tuple[CandidateConfig, MtpDraftSettings]] = []

        def _evaluator(
            c: CandidateConfig, d: MtpDraftSettings,
        ) -> tuple[MtpComparison, MemoryEvidence]:
            calls.append((c, d))
            if d.draft_max == 16:
                m = _mk_comparison("established")
                m2 = MtpComparison(
                    off=m.off, on=MtpMeasurement(
                        mode="on", samples_tps=(65.0, 66.0, 67.0),
                        median_tps=66.0, dispersion_tps=1.0,
                    ),
                    benefit="established", improvement_pct=29.4, tie_bound_tps=1.0,
                )
                return m2, MemoryEvidence(ram_bytes=4_000_000_000, quality="complete")
            else:
                m = _mk_comparison("established")
                m2 = MtpComparison(
                    off=m.off, on=MtpMeasurement(
                        mode="on", samples_tps=(55.0, 56.0, 57.0),
                        median_tps=56.0, dispersion_tps=1.0,
                    ),
                    benefit="established", improvement_pct=9.8, tie_bound_tps=1.0,
                )
                return m2, MemoryEvidence(ram_bytes=4_000_000_000, quality="complete")

        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(
                max_attempts=2, draft_gpu_layers_values=(0,),
                draft_n_max_values=(16, 32),
            ),
            evaluator=_evaluator,
        )

        assert report.benefit == "established"
        assert report.recommendation is not None
        assert report.recommendation.on.median_tps == 66.0
        assert len(report.observations) == 2


class TestTiePrefersOff:
    def test_tie_produces_no_recommendation(self) -> None:
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=2, draft_gpu_layers_values=(0,)),
            evaluator=_ev_not_established,
        )

        assert report.benefit == "not_established"
        assert report.recommendation is None
        for obs in report.observations:
            assert obs.disposition == "not_established"


# ── Partial metrics excluded ──────────────────────────────────────


class TestPartialMetricsExcluded:
    def test_incomplete_evidence_excluded(self) -> None:
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=2, draft_gpu_layers_values=(0,)),
            evaluator=_ev_partial,
        )

        assert report.benefit == "not_established"
        assert report.recommendation is None
        assert report.observations[0].disposition == "incomplete_evidence"


# ── Successful established benefit ────────────────────────────────


class TestSuccessfulBenefit:
    def test_established_benefit_returned(self) -> None:
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        off = MtpMeasurement(
            mode="off", samples_tps=(40.0, 41.0, 42.0),
            median_tps=41.0, dispersion_tps=0.8,
        )
        on = MtpMeasurement(
            mode="on", samples_tps=(60.0, 62.0, 64.0),
            median_tps=62.0, dispersion_tps=1.6,
        )
        comp = MtpComparison(
            off=off, on=on, benefit="established",
            improvement_pct=51.2, tie_bound_tps=1.6,
        )

        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=2, draft_gpu_layers_values=(0,)),
            evaluator=lambda c, d: (
                comp, MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
            ),
        )

        assert report.benefit == "established"
        assert report.recommendation is not None
        assert report.recommendation.benefit == "established"
        assert report.recommendation.on.median_tps == 62.0


# ── Memory fit tests ─────────────────────────────────────────────


class TestCheckMtpMemoryFit:
    def test_unified_fits(self) -> None:
        mem = MemoryEvidence(ram_bytes=8_000_000_000, quality="complete")
        fp = _draft_fingerprint()
        ds = MtpDraftSettings(n_gpu_layers_draft=0)
        budget = ResourceBudget(ram_bytes=10_000_000_000, vram_bytes=None, headroom_ratio=0.1)
        assert check_mtp_memory_fit(mem, fp, ds, budget, memory_unified=True)

    def test_discrete_draft_in_vram_fits(self) -> None:
        mem = MemoryEvidence(ram_bytes=2_000_000_000, vram_bytes=4_000_000_000, quality="complete")
        fp = _draft_fingerprint()
        ds = MtpDraftSettings(n_gpu_layers_draft=-1)
        budget = ResourceBudget(
            ram_bytes=4_000_000_000, vram_bytes=6_000_000_000, headroom_ratio=0.0,
        )
        assert check_mtp_memory_fit(mem, fp, ds, budget, memory_unified=False)

    def test_discrete_draft_in_vram_no_fit(self) -> None:
        mem = MemoryEvidence(ram_bytes=2_000_000_000, vram_bytes=6_000_000_000, quality="complete")
        fp = _draft_fingerprint()
        ds = MtpDraftSettings(n_gpu_layers_draft=-1)
        budget = ResourceBudget(
            ram_bytes=4_000_000_000, vram_bytes=4_000_000_000, headroom_ratio=0.0,
        )
        assert not check_mtp_memory_fit(mem, fp, ds, budget, memory_unified=False)

    def test_discrete_draft_in_ram_fits(self) -> None:
        mem = MemoryEvidence(ram_bytes=2_000_000_000, vram_bytes=4_000_000_000, quality="complete")
        fp = _draft_fingerprint()
        ds = MtpDraftSettings(n_gpu_layers_draft=0)
        budget = ResourceBudget(
            ram_bytes=8_000_000_000, vram_bytes=6_000_000_000, headroom_ratio=0.0,
        )
        assert check_mtp_memory_fit(mem, fp, ds, budget, memory_unified=False)

    def test_unified_no_fit(self) -> None:
        mem = MemoryEvidence(ram_bytes=10_000_000_000, quality="complete")
        fp = _draft_fingerprint()
        ds = MtpDraftSettings(n_gpu_layers_draft=0)
        budget = ResourceBudget(
            ram_bytes=5_000_000_000, vram_bytes=None, headroom_ratio=0.0,
        )
        assert not check_mtp_memory_fit(mem, fp, ds, budget, memory_unified=True)


# ── Select MTP main candidate ────────────────────────────────────


class TestSelectMtpMainCandidate:
    def test_winner_used_when_it_fits(self) -> None:
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        selected = select_mtp_main_candidate(
            validation, _draft_fingerprint(), _budget, memory_unified=True,
        )
        assert selected == w.candidate

    def test_switches_to_alternative_when_winner_does_not_fit(self) -> None:
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=30_000_000_000, quality="complete"),
        )
        alt = _make_validated(
            _candidate(threads=2),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(alt,),
            failures=(), attempts_used=3, budget=10,
        )
        selected = select_mtp_main_candidate(
            validation, _draft_fingerprint(), _budget, memory_unified=True,
        )
        assert selected == alt.candidate


# ── Draft grid generation ────────────────────────────────────────


class TestGenerateMtpDraftGrid:
    def test_non_moe_skips_cpu_moe(self) -> None:
        fp = _draft_fingerprint()
        profile = MtpSearchProfile(
            draft_gpu_layers_values=(0, -1),
            draft_n_max_values=(16,),
        )
        grid = generate_mtp_draft_grid(fp, profile)
        assert len(grid) == 2
        for ds in grid:
            assert ds.n_cpu_moe_draft is None

    def test_moe_includes_cpu_moe(self) -> None:
        fp = ModelFingerprint(
            path=Path("/draft.gguf"), architecture="llama",
            parameter_count=500_000_000, context_length=4096,
            quantization="Q8_0", shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(), is_moe=True, moe_expert_count=2,
        )
        profile = MtpSearchProfile(
            draft_gpu_layers_values=(0,),
            draft_n_max_values=(16,),
            draft_n_cpu_moe_values=(0, 1),
        )
        grid = generate_mtp_draft_grid(fp, profile)
        assert len(grid) == 2
        assert all(ds.n_cpu_moe_draft is not None for ds in grid)


# ── No-fit produces no evaluator calls ────────────────────────────


class TestNoFit:
    def test_no_fit_no_evaluator_calls(self) -> None:
        calls: list[object] = []
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=30_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        def _noop(
            c: CandidateConfig, d: MtpDraftSettings,
        ) -> tuple[MtpComparison, MemoryEvidence]:
            calls.append((c, d))
            return _mk_comparison(), MemoryEvidence()

        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=ResourceBudget(
                ram_bytes=10_000_000_000, vram_bytes=None, headroom_ratio=0.0,
            ),
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=2, draft_gpu_layers_values=(0,)),
            evaluator=_noop,
        )

        assert len(calls) == 0
        assert report.recommendation is None
        assert report.mtp_main_candidate is None
        assert report.benefit == "not_established"


# ── Reporter: server command with draft/MTP flags ───────────────


class TestBuildServerCommandDraftFlags:
    def test_draft_flags_included(self) -> None:
        from llama_parameter_scout.reporter import build_llama_server_command

        cmd = build_llama_server_command(
            _candidate(), _ctx2048(), "/main.gguf",
            draft_model_path="/draft.gguf",
            draft_settings=MtpDraftSettings(n_gpu_layers_draft=0, draft_max=16),
        )
        assert "-md" in cmd
        assert cmd[cmd.index("-md") + 1] == "/draft.gguf"
        assert "--gpu-layers-draft" in cmd
        assert "--draft-max" in cmd
        assert cmd[cmd.index("--draft-max") + 1] == "16"

    def test_old_call_preserved(self) -> None:
        from llama_parameter_scout.reporter import build_llama_server_command

        cmd = build_llama_server_command(
            _candidate(), _ctx2048(), "/main.gguf",
        )
        assert "-md" not in cmd
        assert "--gpu-layers-draft" not in cmd


class TestBuildPresetDraftKeys:
    def test_preset_includes_draft_keys(self) -> None:
        from llama_parameter_scout.reporter import build_preset_ini_section

        preset = build_preset_ini_section(
            _candidate(), _ctx2048(),
            draft_model_path="/draft.gguf",
            draft_settings=MtpDraftSettings(n_gpu_layers_draft=0, draft_max=16),
        )
        assert "model_draft = /draft.gguf" in preset
        assert "n_gpu_layers_draft = 0" in preset
        assert "draft_max = 16" in preset

    def test_preset_omitted_when_not_provided(self) -> None:
        from llama_parameter_scout.reporter import build_preset_ini_section

        preset = build_preset_ini_section(
            _candidate(), _ctx2048(),
        )
        assert "model_draft" not in preset
        assert "n_gpu_layers_draft" not in preset


# ── Reporter: JSON MTP profile section ──────────────────────────


class TestJsonMtpProfile:
    def test_json_has_mtp_profile_section(self) -> None:
        import json

        from llama_parameter_scout.reporter import render_export_json

        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=1, draft_gpu_layers_values=(0,)),
            evaluator=_ev_established,
        )

        session_spec = _make_session_spec()
        toolchain, model = _make_toolchain_and_model()
        j = render_export_json(
            session=session_spec, toolchain=toolchain, model=model,
            validation_result=validation, mtp_profile=report,
        )
        data = json.loads(j)
        assert "mtp_profile" in data
        mp = data["mtp_profile"]
        assert mp["benefit"] == "established"
        assert "normal_winner" in mp
        assert "draft_fingerprint" in mp
        assert "observations" in mp
        assert "recommendation" in mp
        assert mp["recommendation"]["benefit"] == "established"

    def test_json_draft_provenance(self) -> None:
        import json

        from llama_parameter_scout.reporter import render_export_json

        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=1, draft_gpu_layers_values=(0,)),
            evaluator=_ev_established,
        )

        session_spec = _make_session_spec()
        toolchain, model = _make_toolchain_and_model()
        j = render_export_json(
            session=session_spec, toolchain=toolchain, model=model,
            validation_result=validation, mtp_profile=report,
        )
        data = json.loads(j)
        mp = data["mtp_profile"]
        dfp = mp["draft_fingerprint"]
        assert dfp["architecture"] == "llama"
        assert dfp["parameter_count"] == 500_000_000


# ── Reporter: Markdown draft path redaction ─────────────────────


class TestMarkdownMtpProfile:
    def test_markdown_draft_path_rendered(self) -> None:
        from llama_parameter_scout.reporter import render_export_markdown

        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=1, draft_gpu_layers_values=(0,)),
            evaluator=_ev_established,
        )

        session_spec = _make_session_spec()
        toolchain, model = _make_toolchain_and_model()
        md = render_export_markdown(
            session=session_spec, toolchain=toolchain, model=model,
            validation_result=validation, mtp_profile=report,
        )
        assert "MTP Profile" in md
        assert "Benefit:" in md
        assert "established" in md
        assert "Draft model:" in md
        assert "draft.gguf" in md


# ── Test: max_attempt grid truncation ────────────────────────────


class TestMaxAttemptsTruncation:
    def test_grid_truncated_to_max_attempts(self) -> None:
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        profile = MtpSearchProfile(
            max_attempts=1, draft_gpu_layers_values=(0, -1),
            draft_n_max_values=(16, 32),
        )
        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=profile,
            evaluator=_ev_established,
        )
        assert len(report.observations) <= profile.max_attempts

    def test_no_truncation_when_under_max(self) -> None:
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        profile = MtpSearchProfile(
            max_attempts=10, draft_gpu_layers_values=(0,),
            draft_n_max_values=(16,),
        )
        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=profile,
            evaluator=_ev_established,
        )
        assert len(report.observations) == 1


# ── Test: incomplete discrete VRAM evidence rejection ────────────


class TestIncompleteVramEvidence:
    def test_discrete_vram_none_rejected(self) -> None:
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete", vram_bytes=None),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        eval_fn = _ev_established
        budget = ResourceBudget(
            ram_bytes=8_000_000_000, vram_bytes=6_000_000_000, headroom_ratio=0.0,
        )
        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=budget,
            memory_unified=False,
            profile=MtpSearchProfile(max_attempts=2, draft_gpu_layers_values=(-1,)),
            evaluator=eval_fn,
        )
        assert report.benefit == "not_established"
        assert report.recommendation is None


# ── End-to-end acceptance test through Application ────────────────


class TestApplicationCollectMtpProfile:
    def test_established_recommendation_returned(self) -> None:
        from llama_parameter_scout.application import Application
        from llama_parameter_scout.runner import FakeRunner

        runner = FakeRunner()
        app = Application(runner=runner)

        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        session_spec = _make_session_spec()
        dfp = _draft_fingerprint()

        report = app.collect_mtp_profile(
            validation_result=validation,
            draft_model_path=dfp.path,
            server_path=Path("/fake_server"),
            session_spec=session_spec,
            profile=MtpSearchProfile(max_attempts=1, draft_gpu_layers_values=(0,)),
            profile_evaluator=_ev_established,
            main_fingerprint=session_spec.model,
            draft_fingerprint=dfp,
        )

        assert report.benefit == "established"
        assert report.recommendation is not None
        assert report.mtp_main_candidate is not None
        assert report.winning_draft_settings is not None

    def test_no_benefit_when_not_established(self) -> None:
        from llama_parameter_scout.application import Application
        from llama_parameter_scout.runner import FakeRunner

        runner = FakeRunner()
        app = Application(runner=runner)

        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        session_spec = _make_session_spec()
        dfp = _draft_fingerprint()

        report = app.collect_mtp_profile(
            validation_result=validation,
            draft_model_path=dfp.path,
            server_path=Path("/fake_server"),
            session_spec=session_spec,
            profile=MtpSearchProfile(max_attempts=1, draft_gpu_layers_values=(0,)),
            profile_evaluator=_ev_not_established,
            main_fingerprint=session_spec.model,
            draft_fingerprint=dfp,
        )

        assert report.benefit == "not_established"
        assert report.recommendation is None

    def test_alternative_candidate_memory_used_when_winner_does_not_fit(self) -> None:
        from llama_parameter_scout.application import Application
        from llama_parameter_scout.runner import FakeRunner

        draft_path = Path("/draft.gguf")
        draft_shard = ShardFingerprint(
            path=draft_path, size_bytes=500_000_000,
            modified_time=0, digest_sha256="abc",
        )
        dfp = ModelFingerprint(
            path=draft_path, architecture="llama",
            parameter_count=500_000_000, context_length=4096,
            quantization="Q8_0", shards=(draft_shard,),
            symlink_warning=False, network_mount_detected=False, diagnostics=(),
        )

        help_flags = (
            "-m  --host  --port  -c  -t  -ngl  -fa  -ctk  -ctv  -b  "
            "--ubatch-size  -md  --gpu-layers-draft  --draft-max"
        )
        server_path = Path("/fake_server")
        runner = FakeRunner()
        runner.enqueue(
            (str(server_path), "--version"),
            ProcessResult(returncode=0, stdout="version: b1\n", stderr=""),
        )
        runner.enqueue(
            (str(server_path), "--help"),
            ProcessResult(returncode=0, stdout=help_flags, stderr=""),
        )

        app = Application(runner=runner)

        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=30_000_000_000, quality="complete"),
        )
        alt = _make_validated(
            _candidate(threads=2),
            memory=MemoryEvidence(ram_bytes=20_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(alt,),
            failures=(), attempts_used=3, budget=10,
        )

        session_spec = _make_session_spec()
        server_lifecycle = FakeServerLifecycle()
        _measure_results = iter([50.0, 50.0, 50.0, 65.0, 65.0, 65.0])

        def _warmup(port: int) -> None:
            return None

        def _measure(port: int) -> float:
            return next(_measure_results)

        report = app.collect_mtp_profile(
            validation_result=validation,
            draft_model_path=dfp.path,
            server_path=server_path,
            session_spec=session_spec,
            profile=MtpSearchProfile(
                max_attempts=2, draft_gpu_layers_values=(0,),
                draft_n_max_values=(16,),
            ),
            main_fingerprint=session_spec.model,
            draft_fingerprint=dfp,
            server_lifecycle=server_lifecycle,
            _warmup_fn=_warmup,
            _measure_fn=_measure,
        )

        assert report.mtp_main_candidate == alt.candidate
        assert report.benefit == "established"
        assert report.recommendation is not None
        assert len(report.observations) == 1
        obs = report.observations[0]
        assert obs.combined_memory is not None
        assert obs.combined_memory.ram_bytes == 20_000_000_000 + 500_000_000
        assert obs.combined_memory.quality == "complete"


# ── Test: MTP preset in rendered report ─────────────────────────


class TestMtpPresetRendering:
    def test_markdown_mtp_preset_includes_draft_keys(self) -> None:
        from llama_parameter_scout.reporter import render_export_markdown

        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=1, draft_gpu_layers_values=(0,)),
            evaluator=_ev_established,
        )

        session_spec = _make_session_spec()
        toolchain, model = _make_toolchain_and_model()
        md = render_export_markdown(
            session=session_spec, toolchain=toolchain, model=model,
            validation_result=validation, mtp_profile=report,
        )

        assert "Preset.ini (MTP recommendation)" in md
        assert "model_draft = /draft.gguf" in md
        assert "n_gpu_layers_draft = 0" in md
        assert "draft_max = 16" in md

    def test_json_mtp_preset_includes_draft_keys(self) -> None:
        import json

        from llama_parameter_scout.reporter import render_export_json

        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=_draft_fingerprint(),
            budget=_budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=1, draft_gpu_layers_values=(0,)),
            evaluator=_ev_established,
        )

        session_spec = _make_session_spec()
        toolchain, model = _make_toolchain_and_model()
        j = render_export_json(
            session=session_spec, toolchain=toolchain, model=model,
            validation_result=validation, mtp_profile=report,
        )
        data = json.loads(j)
        mp = data["mtp_profile"]
        assert "mtp_preset" in mp
        assert mp["mtp_preset"]["model_draft"] == "/draft.gguf"
        assert mp["mtp_preset"]["draft_settings"]["n_gpu_layers_draft"] == 0
        assert mp["mtp_preset"]["draft_settings"]["draft_max"] == 16


# ── Issue 12: double-counted draft reservation in combined memory fit ─


class TestIssue12CombinedMemoryFit:
    """RED/GREEN boundary: main-only (20 GB) + draft (5 GB) = combined 25 GB.

    Unified budget: 30 GB, headroom 0.1 → 27 GB effective cap.
    Old code (check_mtp_memory_fit) computed 25 + 5 = 30 GB > 27 → reject.
    Fixed code (check_combined_memory_fit) computes 25 <= 27 → accept.
    """

    def test_combined_fit_accepts_within_budget(self) -> None:
        draft_shard = ShardFingerprint(
            path=Path("/draft.gguf"), size_bytes=5_000_000_000,
            modified_time=0, digest_sha256="abc",
        )
        dfp = ModelFingerprint(
            path=Path("/draft.gguf"), architecture="llama",
            parameter_count=500_000_000, context_length=4096,
            quantization="Q8_0", shards=(draft_shard,),
            symlink_warning=False, network_mount_detected=False, diagnostics=(),
        )

        main_memory = MemoryEvidence(ram_bytes=20_000_000_000, quality="complete")
        combined_memory = MemoryEvidence(ram_bytes=25_000_000_000, quality="complete")

        budget = ResourceBudget(ram_bytes=30_000_000_000, vram_bytes=None, headroom_ratio=0.1)

        # The contract: check_combined_memory_fit accepts (25 GB <= 27 GB)
        assert check_combined_memory_fit(combined_memory, budget, memory_unified=True)

        # Old behavior would reject: check_mtp_memory_fit adds draft again
        ds = MtpDraftSettings(n_gpu_layers_draft=0)
        assert not check_mtp_memory_fit(
            combined_memory, dfp, ds, budget, memory_unified=True,
        )

        # Full loop: profile is established with fixed behavior
        w = _make_validated(
            _candidate(threads=4),
            memory=main_memory,
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        def _combined_evaluator(
            c: CandidateConfig, d: MtpDraftSettings,
        ) -> tuple[MtpComparison, MemoryEvidence]:
            return _mk_comparison("established"), combined_memory

        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=dfp,
            budget=budget,
            memory_unified=True,
            profile=MtpSearchProfile(max_attempts=2, draft_gpu_layers_values=(0,)),
            evaluator=_combined_evaluator,
        )

        assert report.benefit == "established"
        assert report.recommendation is not None
        assert report.mtp_main_candidate == w.candidate

    def test_combined_fit_rejects_when_overshoot(self) -> None:
        combined_memory = MemoryEvidence(ram_bytes=28_000_000_000, quality="complete")
        budget = ResourceBudget(ram_bytes=30_000_000_000, vram_bytes=None, headroom_ratio=0.1)
        assert not check_combined_memory_fit(combined_memory, budget, memory_unified=True)

    def test_discrete_rejects_without_vram(self) -> None:
        combined_memory = MemoryEvidence(ram_bytes=4_000_000_000, quality="complete")
        budget = ResourceBudget(
            ram_bytes=8_000_000_000, vram_bytes=6_000_000_000, headroom_ratio=0.0,
        )
        assert not check_combined_memory_fit(combined_memory, budget, memory_unified=False)

    def test_discrete_accepts_with_vram(self) -> None:
        combined_memory = MemoryEvidence(
            ram_bytes=4_000_000_000, vram_bytes=5_000_000_000, quality="complete",
        )
        budget = ResourceBudget(
            ram_bytes=8_000_000_000, vram_bytes=6_000_000_000, headroom_ratio=0.0,
        )
        assert check_combined_memory_fit(combined_memory, budget, memory_unified=False)

    def test_incomplete_quality_rejected(self) -> None:
        combined_memory = MemoryEvidence(ram_bytes=4_000_000_000, quality="partial")
        budget = ResourceBudget(ram_bytes=8_000_000_000, vram_bytes=None, headroom_ratio=0.0)
        assert not check_combined_memory_fit(combined_memory, budget, memory_unified=True)

    def test_no_ram_rejected(self) -> None:
        combined_memory = MemoryEvidence(quality="complete")
        budget = ResourceBudget(ram_bytes=8_000_000_000, vram_bytes=None, headroom_ratio=0.0)
        assert not check_combined_memory_fit(combined_memory, budget, memory_unified=True)


# ── Issue 12: main-candidate preselection uses finite draft grid ─

_draft_shard_issue12 = ShardFingerprint(
    path=Path("/draft.gguf"), size_bytes=4_000_000_000,
    modified_time=0, digest_sha256="abc",
)
_dfp_issue12 = ModelFingerprint(
    path=Path("/draft.gguf"), architecture="llama",
    parameter_count=500_000_000, context_length=4096,
    quantization="Q8_0", shards=(_draft_shard_issue12,),
    symlink_warning=False, network_mount_detected=False, diagnostics=(),
)
_budget_discrete_issue12 = ResourceBudget(
    ram_bytes=20_000_000_000, vram_bytes=12_000_000_000, headroom_ratio=0.0,
)
_main_evidence_discrete_issue12 = MemoryEvidence(
    ram_bytes=18_000_000_000, vram_bytes=6_000_000_000, quality="complete",
)


class TestIssue12PreselectionUsesDraftGrid:
    """RED/GREEN: preselection considers bounded draft grid, not only ngl=0."""

    def test_preselection_selects_candidate_when_draft_gpu_fits(self) -> None:
        main_evidence = _main_evidence_discrete_issue12
        dfp = _dfp_issue12
        budget = _budget_discrete_issue12

        w = _make_validated(_candidate(threads=4), memory=main_evidence)
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        profile = MtpSearchProfile(
            max_attempts=4, draft_gpu_layers_values=(0, -1),
            draft_n_max_values=(16,),
        )

        calls: list[MtpDraftSettings] = []

        def _evaluator(
            c: CandidateConfig, d: MtpDraftSettings,
        ) -> tuple[MtpComparison, MemoryEvidence]:
            calls.append(d)
            if d.n_gpu_layers_draft != 0:
                return _mk_comparison("established"), MemoryEvidence(
                    ram_bytes=18_000_000_000, vram_bytes=10_000_000_000,
                    quality="complete",
                )
            return _mk_comparison("established"), MemoryEvidence(
                ram_bytes=22_000_000_000, quality="complete",
            )

        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=dfp,
            budget=budget,
            memory_unified=False,
            profile=profile,
            evaluator=_evaluator,
        )

        assert report.benefit == "established"
        assert report.mtp_main_candidate == w.candidate
        assert report.recommendation is not None
        assert report.winning_draft_settings is not None
        assert report.winning_draft_settings.n_gpu_layers_draft != 0
        assert len(calls) == 2

    def test_truncation_before_gpu_setting_yields_honest_no_fit(self) -> None:
        main_evidence = _main_evidence_discrete_issue12
        dfp = _dfp_issue12
        budget = _budget_discrete_issue12

        w = _make_validated(_candidate(threads=4), memory=main_evidence)
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        profile = MtpSearchProfile(
            max_attempts=1, draft_gpu_layers_values=(0, -1),
            draft_n_max_values=(16,),
        )

        report = collect_mtp_profile(
            validation_result=validation,
            draft_fingerprint=dfp,
            budget=budget,
            memory_unified=False,
            profile=profile,
            evaluator=lambda c, d: (
                _mk_comparison("established"),
                MemoryEvidence(ram_bytes=22_000_000_000, quality="complete"),
            ),
        )

        assert report.mtp_main_candidate is None
        assert report.benefit == "not_established"
        assert report.recommendation is None
        assert len(report.observations) == 0

    def test_select_main_candidate_with_explicit_seq(self) -> None:
        main_evidence = _main_evidence_discrete_issue12
        dfp = _dfp_issue12
        budget = _budget_discrete_issue12

        w = _make_validated(_candidate(threads=4), memory=main_evidence)
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )

        grid = (
            MtpDraftSettings(n_gpu_layers_draft=0, draft_max=16),
            MtpDraftSettings(n_gpu_layers_draft=-1, draft_max=16),
        )

        selected = select_mtp_main_candidate(
            validation, dfp, budget, memory_unified=False,
            draft_settings_seq=grid,
        )
        assert selected == w.candidate

    def test_select_main_candidate_default_seq_preserved(self) -> None:
        w = _make_validated(
            _candidate(threads=4),
            memory=MemoryEvidence(ram_bytes=4_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=w, alternatives=(),
            failures=(), attempts_used=3, budget=10,
        )
        selected = select_mtp_main_candidate(
            validation, _draft_fingerprint(), _budget, memory_unified=True,
        )
        assert selected == w.candidate
