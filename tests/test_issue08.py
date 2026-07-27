from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from llama_parameter_scout.models import (
    CandidateConfig,
    MemoryEvidence,
    MemoryQuality,
    NctxComposition,
    ResourceBudget,
    ScreeningDisposition,
    ScreeningObservation,
    ScreeningResult,
    SearchProfile,
)
from llama_parameter_scout.search import SCREENING_N_CTX, screen, validate_shortlist

# ── Shared ─────────────────────────────────────────────────────────

BASE_CANDIDATE = CandidateConfig(
    threads=4, batch_size=512, ubatch_size=512,
    n_gpu_layers=0, flash_attn=False,
    cache_type_k="f16", cache_type_v="f16",
)

CANDIDATE_A = CandidateConfig(
    threads=2, batch_size=256, ubatch_size=256,
    n_gpu_layers=0, flash_attn=False,
    cache_type_k="f16", cache_type_v="f16",
)

CANDIDATE_B = CandidateConfig(
    threads=8, batch_size=1024, ubatch_size=1024,
    n_gpu_layers=0, flash_attn=False,
    cache_type_k="f16", cache_type_v="f16",
)


def _complete_memory(ram: int, vram: int | None = None) -> MemoryEvidence:
    return MemoryEvidence(ram_bytes=ram, vram_bytes=vram, quality="complete")


def _ok_obs(
    candidate: CandidateConfig,
    pp_samples: tuple[float, ...] = (100.0, 101.0, 99.0),
    tg_samples: tuple[float, ...] = (20.0, 21.0, 19.0),
    memory: MemoryEvidence | None = None,
) -> ScreeningObservation:
    return ScreeningObservation(
        candidate=candidate,
        pp_samples=pp_samples,
        tg_samples=tg_samples,
        memory=memory,
    )


def _ok_result(
    candidate: CandidateConfig,
    pp_samples: tuple[float, ...] | list[float] = (100.0, 101.0, 99.0),
    tg_samples: tuple[float, ...] | list[float] = (20.0, 21.0, 19.0),
    memory: MemoryEvidence | None = None,
) -> ScreeningResult:
    return ScreeningResult(
        candidate=candidate,
        success=True,
        disposition="survived",
        observation=_ok_obs(candidate, tuple(pp_samples), tuple(tg_samples), memory),
    )


def _fail_result(
    candidate: CandidateConfig,
    disposition: ScreeningDisposition = "crash",
    reason: str = "simulated failure",
) -> ScreeningResult:
    return ScreeningResult(
        candidate=candidate,
        success=False,
        disposition=disposition,
        reason=reason,
    )


@dataclass
class FakeScreeningEvaluator:
    _results: dict[tuple[int, int, int, int], ScreeningResult] = field(default_factory=dict)
    calls: list[tuple[CandidateConfig, NctxComposition, int]] = field(default_factory=list)

    def enqueue(
        self,
        candidate: CandidateConfig,
        repetitions: int,
        result: ScreeningResult,
    ) -> None:
        key = (candidate.threads, candidate.batch_size, candidate.ubatch_size, repetitions)
        self._results[key] = result

    def __call__(
        self,
        candidate: CandidateConfig,
        n_ctx: NctxComposition,
        repetitions: int,
    ) -> ScreeningResult:
        self.calls.append((candidate, n_ctx, repetitions))
        key = (candidate.threads, candidate.batch_size, candidate.ubatch_size, repetitions)
        result = self._results.get(key)
        if result is None:
            raise ValueError(
                f"No enqueued result for threads={candidate.threads} "
                f"batch={candidate.batch_size} ubatch={candidate.ubatch_size} "
                f"repetitions={repetitions}"
            )
        return result


def _stable_profile(reps: int = 3) -> SearchProfile:
    return SearchProfile(
        label="stable",
        max_attempts=20,
        min_repetitions=reps,
        max_repetitions_per_candidate=reps,
        tie_bound_coefficient=0.05,
        dispersion_threshold=0.5,
    )


# ── Test 1: Memory objective picks fastest stable fit ─────────────


class TestMemoryObjectiveFastestStableFit:
    def test_picks_fastest_stable_with_complete_fit(self) -> None:
        """When multiple candidates have complete memory evidence within budget,
        the memory objective selects the fastest (highest balanced score) stable one."""
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),  # baseline incomplete → cannot win memory
        )

        budget = ResourceBudget(ram_bytes=16_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        grid = (CANDIDATE_A, CANDIDATE_B)
        evaluator.enqueue(
            CANDIDATE_A, profile.min_repetitions,
            _ok_result(CANDIDATE_A, [400.0] * 3, [30.0] * 3,
                       memory=_complete_memory(ram=6_000_000_000)),
        )
        evaluator.enqueue(
            CANDIDATE_B, profile.min_repetitions,
            _ok_result(CANDIDATE_B, [200.0] * 3, [15.0] * 3,
                       memory=_complete_memory(ram=4_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=grid,
        )

        assert report.leader is not None
        assert report.leader.candidate == CANDIDATE_A
        assert report.objective == "memory"


class TestMemoryNoFit:
    def test_no_leader_when_all_exceed_budget(self) -> None:
        """When all candidates have complete memory evidence that exceeds budget,
        no memory leader is selected."""
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=_complete_memory(ram=8_000_000_000)),
        )

        budget = ResourceBudget(ram_bytes=4_000_000_000, vram_bytes=None, headroom_ratio=0.0)
        evaluator.enqueue(
            CANDIDATE_A, profile.min_repetitions,
            _ok_result(CANDIDATE_A, [400.0] * 3, [30.0] * 3,
                       memory=_complete_memory(ram=6_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(CANDIDATE_A,),
        )

        assert report.leader is None


class TestStabilityEnforcement:
    def test_faster_unstable_loses_to_stable(self) -> None:
        """When the fastest candidate has high dispersion and another candidate
        is stable and fits, the stable candidate wins."""
        profile = SearchProfile(
            label="strict", max_attempts=20, min_repetitions=3,
            max_repetitions_per_candidate=3, tie_bound_coefficient=0.05,
            dispersion_threshold=0.02,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),
        )

        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        fast_unstable = CANDIDATE_A
        evaluator.enqueue(
            fast_unstable, profile.min_repetitions,
            _ok_result(fast_unstable, [400.0, 500.0, 300.0], [30.0, 15.0, 45.0],
                       memory=_complete_memory(ram=6_000_000_000)),
        )
        stable = CANDIDATE_B
        evaluator.enqueue(
            stable, profile.min_repetitions,
            _ok_result(stable, [200.0, 201.0, 199.0], [10.0, 10.1, 9.9],
                       memory=_complete_memory(ram=4_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(fast_unstable, stable),
        )

        assert report.leader is not None
        assert report.leader.candidate == stable

    def test_validation_stable_over_fast_unstable(self) -> None:
        """Validation also enforces stability for memory objective."""
        from llama_parameter_scout.models import (
            NoWinnerReport,
            RankedCandidate,
            ScreeningReport,
        )
        from llama_parameter_scout.search import validate_shortlist

        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        a = CANDIDATE_A
        survivors = (
            RankedCandidate(
                candidate=a, pp_score=1.0, tg_score=1.0, balanced=1.0,
                pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )

        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        class UnstableValidator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return ScreeningResult(
                    candidate=candidate, success=True,
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(100.0, 500.0, 50.0),
                        tg_samples=(20.0, 60.0, 10.0),
                        memory=_complete_memory(ram=6_000_000_000),
                        observed_prompt=1024, observed_generation=2048,
                        observed_depth=1024, observed_total=4096,
                    ),
                )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=UnstableValidator(),
            resource_budget=budget,
            memory_unified=True,
        )

        assert isinstance(result, NoWinnerReport)


class TestMemoryObjectiveDominance:
    def test_dominated_candidate_retained_as_leader(self) -> None:
        """When a candidate is PP/TG dominated but is the only fitting one,
        it is retained and becomes memory leader (no PP/TG pruning for memory)."""
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),
        )

        budget = ResourceBudget(ram_bytes=8_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        fast_exceed = CANDIDATE_A
        evaluator.enqueue(
            fast_exceed, profile.min_repetitions,
            _ok_result(fast_exceed, [400.0] * 3, [30.0] * 3,
                       memory=_complete_memory(ram=12_000_000_000)),
        )
        slow_fit = CANDIDATE_B
        evaluator.enqueue(
            slow_fit, profile.min_repetitions,
            _ok_result(slow_fit, [200.0] * 3, [15.0] * 3,
                       memory=_complete_memory(ram=4_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(fast_exceed, slow_fit),
        )

        assert report.leader is not None
        assert report.leader.candidate == slow_fit
        assert any(s.candidate == slow_fit for s in report.survivors)
        for d in report.grid_dispositions:
            assert d.disposition != "dominance_pruned"

    def test_non_memory_still_prunes(self) -> None:
        """Non-memory objectives still use PP/TG dominance pruning."""
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3),
        )
        grid = (CANDIDATE_A, CANDIDATE_B)
        evaluator.enqueue(
            CANDIDATE_A, profile.min_repetitions,
            _ok_result(CANDIDATE_A, [400.0] * 3, [30.0] * 3),
        )
        evaluator.enqueue(
            CANDIDATE_B, profile.min_repetitions,
            _ok_result(CANDIDATE_B, [200.0] * 3, [15.0] * 3),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="balanced",
            grid=grid,
        )

        pruned = [d for d in report.grid_dispositions
                  if d.disposition == "dominance_pruned"]
        assert len(pruned) >= 1


class TestBaselineIncludedMemoryObjective:
    def test_baseline_in_survivors_and_frontier(self) -> None:
        """Baseline appears in survivors and frontier for memory objective."""
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=_complete_memory(ram=8_000_000_000)),
        )

        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)
        evaluator.enqueue(
            CANDIDATE_A, profile.min_repetitions,
            _ok_result(CANDIDATE_A, [400.0] * 3, [30.0] * 3,
                       memory=_complete_memory(ram=6_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(CANDIDATE_A,),
        )

        assert any(s.candidate == BASE_CANDIDATE for s in report.survivors)
        frontier_candidates = {e.candidate for e in report.pareto_frontier}
        assert BASE_CANDIDATE in frontier_candidates

    def test_baseline_not_duplicated(self) -> None:
        """Baseline appears once in survivors."""
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=_complete_memory(ram=8_000_000_000)),
        )

        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        from llama_parameter_scout.search import generate_core_grid
        grid = generate_core_grid(BASE_CANDIDATE)
        for c in grid:
            if c == BASE_CANDIDATE:
                continue
            evaluator.enqueue(
                c, profile.min_repetitions,
                _ok_result(c, [100.0] * 3, [20.0] * 3,
                           memory=_complete_memory(ram=6_000_000_000)),
            )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
        )

        baseline_count = sum(1 for s in report.survivors if s.candidate == BASE_CANDIDATE)
        assert baseline_count == 1

    def test_balanced_objective_baseline_not_in_survivors(self) -> None:
        """Non-memory objectives do NOT include baseline in survivors."""
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3),
        )
        from llama_parameter_scout.search import generate_core_grid
        grid = generate_core_grid(BASE_CANDIDATE)
        for c in grid:
            if c == BASE_CANDIDATE:
                continue
            evaluator.enqueue(
                c, profile.min_repetitions,
                _ok_result(c, [100.0] * 3, [20.0] * 3),
            )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="balanced",
        )

        baseline_survivors = [s for s in report.survivors if s.candidate == BASE_CANDIDATE]
        assert len(baseline_survivors) == 0


class TestMemoryObjectiveNoBudget:
    def test_screen_returns_report_without_evaluator(self) -> None:
        """screen() with objective='memory' and no ResourceBudget returns
        a report without calling evaluator."""
        profile = _stable_profile()
        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=FakeScreeningEvaluator(),
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
        )
        assert report.leader is None
        assert report.attempts_used == 0
        assert "requires a ResourceBudget" in report.disposition_summary

    def test_validate_returns_nowinner_without_budget(self) -> None:
        """validate_shortlist() returns NoWinnerReport when no budget."""
        from llama_parameter_scout.models import (
            NoWinnerReport,
            RankedCandidate,
            ScreeningReport,
        )
        from llama_parameter_scout.search import validate_shortlist

        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        survivors = (
            RankedCandidate(
                candidate=CANDIDATE_A, pp_score=1.0, tg_score=1.0, balanced=1.0,
                pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=FakeScreeningEvaluator(),  # type: ignore[arg-type]
        )

        assert isinstance(result, NoWinnerReport)
        assert result.attempts_used == 0
        assert any("requires a ResourceBudget" in f.reason for f in result.failures)


class TestValidationMemoryEvidenceAndFrontier:
    def test_validated_candidate_has_memory_evidence(self) -> None:
        """ValidatedCandidate carries memory_evidence from observation."""
        from llama_parameter_scout.models import (
            RankedCandidate,
            ScreeningReport,
            ValidatedRecommendation,
        )
        from llama_parameter_scout.search import validate_shortlist

        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        survivors = (
            RankedCandidate(
                candidate=CANDIDATE_A, pp_score=1.0, tg_score=1.0, balanced=1.0,
                pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )

        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        class MemValidator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return ScreeningResult(
                    candidate=candidate, success=True,
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(100.0, 101.0, 99.0),
                        tg_samples=(20.0, 21.0, 19.0),
                        memory=_complete_memory(ram=6_000_000_000),
                        observed_prompt=1024, observed_generation=2048,
                        observed_depth=1024, observed_total=4096,
                    ),
                )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=MemValidator(),
            resource_budget=budget,
            memory_unified=True,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert result.recommendation.memory_evidence is not None
        assert result.recommendation.memory_evidence.ram_bytes == 6_000_000_000
        assert result.recommendation.memory_evidence.quality == "complete"

    def test_validation_frontier_non_empty(self) -> None:
        """Validation populates a non-empty Pareto frontier."""
        from llama_parameter_scout.models import (
            RankedCandidate,
            ScreeningReport,
            ValidatedRecommendation,
        )
        from llama_parameter_scout.search import validate_shortlist

        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        survivors = (
            RankedCandidate(
                candidate=CANDIDATE_A, pp_score=1.0, tg_score=1.0, balanced=1.0,
                pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )

        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        class MemValidator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return ScreeningResult(
                    candidate=candidate, success=True,
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(100.0, 101.0, 99.0),
                        tg_samples=(20.0, 21.0, 19.0),
                        memory=_complete_memory(ram=6_000_000_000),
                        observed_prompt=1024, observed_generation=2048,
                        observed_depth=1024, observed_total=4096,
                    ),
                )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=MemValidator(),
            resource_budget=budget,
            memory_unified=True,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert len(result.pareto_frontier) > 0
        assert result.pareto_frontier[0].memory.quality == "complete"

    def test_validation_nowinner_frontier(self) -> None:
        """NoWinnerReport carries Pareto frontier from validated observations."""
        from llama_parameter_scout.models import (
            NoWinnerReport,
            RankedCandidate,
            ScreeningReport,
        )
        from llama_parameter_scout.search import validate_shortlist

        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        survivors = (
            RankedCandidate(
                candidate=CANDIDATE_A, pp_score=1.0, tg_score=1.0, balanced=1.0,
                pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )

        budget = ResourceBudget(ram_bytes=4_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        class MemValidator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return ScreeningResult(
                    candidate=candidate, success=True,
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(100.0, 101.0, 99.0),
                        tg_samples=(20.0, 21.0, 19.0),
                        memory=_complete_memory(ram=6_000_000_000),
                        observed_prompt=1024, observed_generation=2048,
                        observed_depth=1024, observed_total=4096,
                    ),
                )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=MemValidator(),
            resource_budget=budget,
            memory_unified=True,
        )

        assert isinstance(result, NoWinnerReport)
        assert len(result.pareto_frontier) > 0
        assert result.pareto_frontier[0].memory.ram_bytes == 6_000_000_000

    def test_validation_discrete_frontier(self) -> None:
        """Validation frontier for discrete GPU includes both RAM and VRAM."""
        from llama_parameter_scout.models import (
            RankedCandidate,
            ScreeningReport,
            ValidatedRecommendation,
        )
        from llama_parameter_scout.search import validate_shortlist

        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        survivors = (
            RankedCandidate(
                candidate=CANDIDATE_A, pp_score=1.0, tg_score=1.0, balanced=1.0,
                pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )

        budget = ResourceBudget(
            ram_bytes=32_000_000_000, vram_bytes=16_000_000_000,
            headroom_ratio=0.0,
        )

        class MemValidator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return ScreeningResult(
                    candidate=candidate, success=True,
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(100.0, 101.0, 99.0),
                        tg_samples=(20.0, 21.0, 19.0),
                        memory=_complete_memory(ram=8_000_000_000, vram=8_000_000_000),
                        observed_prompt=1024, observed_generation=2048,
                        observed_depth=1024, observed_total=4096,
                    ),
                )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=MemValidator(),
            resource_budget=budget,
            memory_unified=False,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert len(result.pareto_frontier) > 0
        assert result.pareto_frontier[0].memory.vram_bytes == 8_000_000_000

    def test_validation_incomplete_no_fit_distinction(self) -> None:
        """No-fit and incomplete produce distinct failure reasons."""
        from llama_parameter_scout.models import (
            NoWinnerReport,
            RankedCandidate,
            ScreeningReport,
        )
        from llama_parameter_scout.search import validate_shortlist

        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        a = CANDIDATE_A
        b = CANDIDATE_B
        survivors = (
            RankedCandidate(
                candidate=a, pp_score=1.0, tg_score=1.0, balanced=1.0,
                pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
            RankedCandidate(
                candidate=b, pp_score=0.8, tg_score=0.8, balanced=0.8,
                pp_median=80.0, tg_median=16.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )

        budget = ResourceBudget(ram_bytes=4_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        observed = iter([
            ScreeningResult(
                candidate=a, success=True,
                observation=ScreeningObservation(
                    candidate=a,
                    pp_samples=(100.0, 101.0, 99.0),
                    tg_samples=(20.0, 21.0, 19.0),
                    memory=None,
                    observed_prompt=1024, observed_generation=2048,
                    observed_depth=1024, observed_total=4096,
                ),
            ),
            ScreeningResult(
                candidate=b, success=True,
                observation=ScreeningObservation(
                    candidate=b,
                    pp_samples=(80.0, 81.0, 79.0),
                    tg_samples=(16.0, 16.1, 15.9),
                    memory=_complete_memory(ram=6_000_000_000),
                    observed_prompt=1024, observed_generation=2048,
                    observed_depth=1024, observed_total=4096,
                ),
            ),
        ])

        class SeqValidator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return next(observed)

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=SeqValidator(),
            resource_budget=budget,
            memory_unified=True,
        )

        assert isinstance(result, NoWinnerReport)
        assert len(result.failures) == 2
        inc_reason = result.failures[0].reason
        nofit_reason = result.failures[1].reason
        assert "Incomplete" in inc_reason
        assert "exceeds budget" in nofit_reason


class TestValidationMemoryObjective:
    def test_validation_no_winner_when_incomplete_memory(self) -> None:
        """Validation returns NoWinnerReport when validated candidates lack
        complete memory evidence for memory objective."""
        from llama_parameter_scout.models import (
            NoWinnerReport,
            RankedCandidate,
            ScreeningReport,
        )

        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        survivors = (
            RankedCandidate(
                candidate=CANDIDATE_A, pp_score=1.0, tg_score=1.0, balanced=1.0,
                pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )

        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        class Validator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return ScreeningResult(
                    candidate=candidate, success=True,
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(100.0, 101.0, 99.0),
                        tg_samples=(20.0, 21.0, 19.0),
                        memory=None,
                        observed_prompt=1024,
                        observed_generation=2048,
                        observed_depth=1024,
                        observed_total=4096,
                    ),
                )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=Validator(),
            resource_budget=budget,
            memory_unified=True,
        )

        assert isinstance(result, NoWinnerReport)
        assert len(result.failures) == 1
        assert "Incomplete memory metrics" in result.failures[0].reason


class TestApplicationSeamMemory:
    def _enqueue_all_grid(
        self,
        evaluator: FakeScreeningEvaluator,
        base: CandidateConfig,
        profile: SearchProfile,
        memory: MemoryEvidence | None = None,
    ) -> None:
        from llama_parameter_scout.search import generate_core_grid
        reps = profile.min_repetitions
        for c in generate_core_grid(base):
            if c == base:
                continue
            evaluator.enqueue(
                c, reps,
                _ok_result(c, [100.0] * reps, [20.0] * reps, memory=memory),
            )

    def test_screen_candidates_injected_evaluator_passes_budget(
        self, tmp_path: Path,
    ) -> None:
        """Application.screen_candidates passes session budget/topology
        to the injected evaluator path — observable via memory leader."""
        from llama_parameter_scout.application import Application
        from llama_parameter_scout.model_inspector import inspect_model
        from llama_parameter_scout.models import (
            ContextAdvisory,
            NctxComposition,
            ResourceBudget,
            SessionSpecification,
            SingleRunRequest,
            SystemTopology,
        )
        from llama_parameter_scout.runner import FakeRunner

        tmp = Path(tmp_path)

        bp = tmp / "llama-bench"
        bp.write_text("#!/bin/sh\necho '{\"samples_ts\":[100.0,101.0,99.0]}'")
        bp.chmod(0o755)
        mp = tmp / "model.gguf"
        from tests.conftest import _build_minimal_gguf
        _build_minimal_gguf(mp, {"general.architecture": (8, "llama")})

        runner = FakeRunner()
        app = Application(runner)
        model_fp = inspect_model(mp)

        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()

        # Base candidate must fit in budget
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),  # baseline incomplete → ineligible as leader
        )

        # Enqueue the full core grid with complete memory that fits 16 GB
        self._enqueue_all_grid(
            evaluator, BASE_CANDIDATE, profile,
            memory=_complete_memory(ram=6_000_000_000),
        )

        # Override CANDIDATE_A (threads=2, batch=256, ubatch=256) which is in
        # the core grid: make it the fastest (highest balanced) fit.
        cand_a_key = (CANDIDATE_A.threads, CANDIDATE_A.batch_size,
                      CANDIDATE_A.ubatch_size)
        evaluator._results[(cand_a_key[0], cand_a_key[1], cand_a_key[2],
                           profile.min_repetitions)] = _ok_result(
            CANDIDATE_A, [400.0] * 3, [30.0] * 3,
            memory=_complete_memory(ram=6_000_000_000),
        )

        spec = SessionSpecification(
            topology=SystemTopology(
                cpu_count=4, system_memory_bytes=32_000_000_000,
                accelerators=(), memory_unified=True,
            ),
            cpu_only=True, selected_device_ids=(),
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            budget=ResourceBudget(ram_bytes=16_000_000_000, vram_bytes=None, headroom_ratio=0.0),
            advisory=ContextAdvisory(
                likely_fit_lower=1000, likely_fit_upper=5000,
                uncertain_maximum=8000, model_weight_bytes=4_000_000_000,
                kv_cache_bytes_estimate=500_000_000,
            ),
            model=model_fp,
            acknowledged=True, fingerprint="test",
        )

        request = SingleRunRequest(
            binary_path=bp, model_path=mp,
            journal_path=tmp / "journal.json",
            n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            candidate=BASE_CANDIDATE, binary_source="explicit",
        )

        report = app.screen_candidates(
            request, profile,
            evaluator=evaluator,
            session_spec=spec,
            objective="memory",
        )

        assert report.objective == "memory"
        assert report.leader is not None
        assert report.leader.candidate == CANDIDATE_A
        assert report.pareto_frontier is not None


class TestDiscreteRamVramEnforcement:
    def test_both_ram_and_vram_must_fit_discrete(self) -> None:
        """On discrete GPU, both RAM and VRAM evidence must be complete and
        independently within their respective caps."""
        profile = _stable_profile()
        budget = ResourceBudget(
            ram_bytes=32_000_000_000, vram_bytes=16_000_000_000,
            headroom_ratio=0.0,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),  # baseline incomplete
        )

        a = CANDIDATE_A
        evaluator.enqueue(
            a, profile.min_repetitions,
            _ok_result(a, [400.0] * 3, [30.0] * 3,
                       memory=_complete_memory(ram=8_000_000_000, vram=20_000_000_000)),
        )
        b = CANDIDATE_B
        evaluator.enqueue(
            b, profile.min_repetitions,
            _ok_result(b, [300.0] * 3, [20.0] * 3,
                       memory=_complete_memory(ram=40_000_000_000, vram=8_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=False,
            grid=(a, b),
        )

        assert report.leader is None

    def test_both_fit_discrete_picks_fastest(self) -> None:
        """When both RAM and VRAM fit for discrete, pick fastest."""
        profile = _stable_profile()
        budget = ResourceBudget(
            ram_bytes=32_000_000_000, vram_bytes=16_000_000_000,
            headroom_ratio=0.0,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),  # baseline incomplete
        )

        a = CANDIDATE_A
        evaluator.enqueue(
            a, profile.min_repetitions,
            _ok_result(a, [400.0] * 3, [30.0] * 3,
                       memory=_complete_memory(ram=8_000_000_000, vram=8_000_000_000)),
        )
        b = CANDIDATE_B
        evaluator.enqueue(
            b, profile.min_repetitions,
            _ok_result(b, [200.0] * 3, [15.0] * 3,
                       memory=_complete_memory(ram=4_000_000_000, vram=4_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=False,
            grid=(a, b),
        )

        assert report.leader is not None
        assert report.leader.candidate == a


class TestParetoFrontier:
    def test_frontier_excludes_dominated_candidates_unified(self) -> None:
        """Pareto frontier excludes candidates dominated in speed and unified RAM."""
        profile = _stable_profile()
        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),  # baseline incomplete → not in frontier
        )

        a = CANDIDATE_A  # balanced ~0.98, ram=6GB
        evaluator.enqueue(
            a, profile.min_repetitions,
            _ok_result(a, [400.0] * 3, [30.0] * 3,
                       memory=_complete_memory(ram=6_000_000_000)),
        )
        b = CANDIDATE_B  # balanced ~0.49, ram=4GB (slower but less RAM)
        evaluator.enqueue(
            b, profile.min_repetitions,
            _ok_result(b, [200.0] * 3, [15.0] * 3,
                       memory=_complete_memory(ram=4_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(a, b),
        )

        frontier = report.pareto_frontier
        assert len(frontier) == 2
        frontier_candidates = {e.candidate for e in frontier}
        assert a in frontier_candidates
        assert b in frontier_candidates

    def test_frontier_excludes_incomplete_evidence(self) -> None:
        """Candidates without complete memory evidence are excluded from frontier."""
        profile = _stable_profile()
        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),  # baseline incomplete
        )

        a = CANDIDATE_A
        evaluator.enqueue(
            a, profile.min_repetitions,
            _ok_result(a, [400.0] * 3, [30.0] * 3,
                       memory=None),  # missing memory
        )
        b = CANDIDATE_B
        evaluator.enqueue(
            b, profile.min_repetitions,
            _ok_result(b, [200.0] * 3, [15.0] * 3,
                       memory=_complete_memory(ram=4_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(a, b),
        )

        frontier = report.pareto_frontier
        assert len(frontier) == 1
        assert frontier[0].candidate == b


class TestUnifiedEnforcement:
    def test_unified_single_ram_cap_enforced(self) -> None:
        """On unified/CPU topology, only the RAM cap is checked (shared memory)."""
        profile = _stable_profile()
        budget = ResourceBudget(ram_bytes=8_000_000_000, vram_bytes=None, headroom_ratio=0.0)
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),  # baseline incomplete
        )

        a = CANDIDATE_A
        evaluator.enqueue(
            a, profile.min_repetitions,
            _ok_result(a, [400.0] * 3, [30.0] * 3,
                       memory=_complete_memory(ram=6_000_000_000)),
        )
        b = CANDIDATE_B
        evaluator.enqueue(
            b, profile.min_repetitions,
            _ok_result(b, [200.0] * 3, [15.0] * 3,
                       memory=_complete_memory(ram=12_000_000_000)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(a, b),
        )

        # A fits, B doesn't. Leader should be A (fastest that fits)
        assert report.leader is not None
        assert report.leader.candidate == a


class TestMemoryIncompleteMetrics:
    def test_no_leader_when_memory_missing(self) -> None:
        """When memory evidence is None (missing), no memory leader can be selected."""
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),
        )

        budget = ResourceBudget(ram_bytes=16_000_000_000, vram_bytes=None, headroom_ratio=0.0)
        evaluator.enqueue(
            CANDIDATE_A, profile.min_repetitions,
            _ok_result(CANDIDATE_A, [400.0] * 3, [30.0] * 3,
                       memory=None),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(CANDIDATE_A,),
        )

        assert report.leader is None

    @pytest.mark.parametrize(
        "quality",
        (
            pytest.param("partial", id="partial"),
            pytest.param("estimated", id="estimated"),
            pytest.param("incomparable", id="incomparable"),
        ),
    )
    def test_no_leader_when_memory_not_complete(self, quality: MemoryQuality) -> None:
        """Partial, estimated, or incomparable evidence cannot select a leader."""
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=MemoryEvidence(ram_bytes=4_000_000_000, quality=quality)),
        )

        budget = ResourceBudget(ram_bytes=16_000_000_000, vram_bytes=None, headroom_ratio=0.0)
        evaluator.enqueue(
            CANDIDATE_A, profile.min_repetitions,
            _ok_result(CANDIDATE_A, [400.0] * 3, [30.0] * 3,
                       memory=MemoryEvidence(ram_bytes=4_000_000_000, quality=quality)),
        )

        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(CANDIDATE_A,),
        )

        assert report.leader is None


class TestHeadroomGap:
    def test_screening_headroom_gap_excludes_candidate(self) -> None:
        profile = _stable_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            BASE_CANDIDATE, profile.min_repetitions,
            _ok_result(BASE_CANDIDATE, [500.0] * 3, [25.0] * 3,
                       memory=None),
        )
        budget = ResourceBudget(ram_bytes=10_000, vram_bytes=None, headroom_ratio=0.10)
        a = CANDIDATE_A
        evaluator.enqueue(
            a, profile.min_repetitions,
            _ok_result(a, [400.0] * 3, [30.0] * 3,
                       memory=_complete_memory(ram=9_500)),
        )
        report = screen(
            base_candidate=BASE_CANDIDATE,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="memory",
            resource_budget=budget,
            memory_unified=True,
            grid=(a,),
        )
        assert report.leader is None


class TestValidationFrontierUnique:
    def test_validation_frontier_no_duplicates(self) -> None:
        from llama_parameter_scout.models import (
            RankedCandidate,
            ScreeningReport,
            ValidatedRecommendation,
        )
        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        survivors = (
            RankedCandidate(
                candidate=CANDIDATE_A, pp_score=1.0, tg_score=1.0,
                balanced=1.0, pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )
        budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.0)

        class MemValidator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return ScreeningResult(
                    candidate=candidate, success=True,
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(100.0, 101.0, 99.0),
                        tg_samples=(20.0, 21.0, 19.0),
                        memory=_complete_memory(ram=6_000_000_000),
                        observed_prompt=1024, observed_generation=2048,
                        observed_depth=1024, observed_total=4096,
                    ),
                )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=MemValidator(),
            resource_budget=budget,
            memory_unified=True,
        )
        assert isinstance(result, ValidatedRecommendation)
        frontier = result.pareto_frontier
        frontier_candidates = [e.candidate for e in frontier]
        assert len(frontier_candidates) == len(set(frontier_candidates))

    def test_validation_headroom_gap_returns_nowinner(self) -> None:
        from llama_parameter_scout.models import (
            NoWinnerReport,
            RankedCandidate,
            ScreeningReport,
        )
        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        survivors = (
            RankedCandidate(
                candidate=CANDIDATE_A, pp_score=1.0, tg_score=1.0,
                balanced=1.0, pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )
        budget = ResourceBudget(ram_bytes=10_000, vram_bytes=None, headroom_ratio=0.10)

        class HeadroomValidator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return ScreeningResult(
                    candidate=candidate, success=True,
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(100.0, 101.0, 99.0),
                        tg_samples=(20.0, 21.0, 19.0),
                        memory=_complete_memory(ram=9_500),
                        observed_prompt=1024, observed_generation=2048,
                        observed_depth=1024, observed_total=4096,
                    ),
                )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=HeadroomValidator(),
            resource_budget=budget,
            memory_unified=True,
        )
        assert isinstance(result, NoWinnerReport)
        assert any("exceeds budget" in f.reason for f in result.failures)

    def test_overbudget_complete_remains_in_validation_frontier(self) -> None:
        from llama_parameter_scout.models import (
            NoWinnerReport,
            RankedCandidate,
            ScreeningReport,
        )
        profile = _stable_profile()
        base_obs = ScreeningObservation(
            candidate=BASE_CANDIDATE,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
        )
        survivors = (
            RankedCandidate(
                candidate=CANDIDATE_A, pp_score=1.0, tg_score=1.0,
                balanced=1.0, pp_median=100.0, tg_median=20.0,
                pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            ),
        )
        screening_report = ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=BASE_CANDIDATE, success=True, observation=base_obs,
            ),
            grid_dispositions=(),
            survivors=survivors,
            objective="memory",
            disposition_summary="test",
        )
        budget = ResourceBudget(ram_bytes=4_000, vram_bytes=None, headroom_ratio=0.0)

        class OverBudgetValidator:
            def __call__(
                self, candidate: CandidateConfig, n_ctx: NctxComposition,
            ) -> ScreeningResult:
                return ScreeningResult(
                    candidate=candidate, success=True,
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(100.0, 101.0, 99.0),
                        tg_samples=(20.0, 21.0, 19.0),
                        memory=_complete_memory(ram=8_000),
                        observed_prompt=1024, observed_generation=2048,
                        observed_depth=1024, observed_total=4096,
                    ),
                )

        result = validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            objective="memory",
            evaluator=OverBudgetValidator(),
            resource_budget=budget,
            memory_unified=True,
        )
        assert isinstance(result, NoWinnerReport)
        assert len(result.pareto_frontier) == 1
        assert result.pareto_frontier[0].memory.ram_bytes == 8_000


