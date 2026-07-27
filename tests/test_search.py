from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from llama_parameter_scout.models import (
    CandidateConfig,
    NctxComposition,
    ScreeningObservation,
    ScreeningResult,
    SearchProfile,
)
from llama_parameter_scout.search import (
    QUICK_PROFILE,
    SCREENING_N_CTX,
    STANDARD_PROFILE,
    THOROUGH_PROFILE,
    generate_core_grid,
    get_profile,
    screen,
)

# ── Fake evaluator ─────────────────────────────────────────────────


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


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def base_candidate() -> CandidateConfig:
    return CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=0, flash_attn=False,
        cache_type_k="f16", cache_type_v="f16",
    )


def _make_obs(
    candidate: CandidateConfig,
    pp_samples: list[float],
    tg_samples: list[float],
    memory: int | None = None,
) -> ScreeningObservation:
    return ScreeningObservation(
        candidate=candidate,
        pp_samples=tuple(pp_samples),
        tg_samples=tuple(tg_samples),
        memory_bytes=memory,
    )


def _ok_result(
    candidate: CandidateConfig,
    pp_samples: list[float],
    tg_samples: list[float],
    memory: int | None = None,
) -> ScreeningResult:
    return ScreeningResult(
        candidate=candidate,
        success=True,
        disposition="survived",
        observation=_make_obs(candidate, pp_samples, tg_samples, memory),
    )


def _fail_result(
    candidate: CandidateConfig,
    disposition: str = "crash",
    reason: str = "simulated failure",
) -> ScreeningResult:
    return ScreeningResult(
        candidate=candidate,
        success=False,
        disposition=disposition,  # type: ignore[arg-type]
        reason=reason,
    )


def _enqueue_grid(
    evaluator: FakeScreeningEvaluator,
    base: CandidateConfig,
    reps: int,
    pp_val: float = 100.0,
    tg_val: float = 20.0,
) -> None:
    grid = generate_core_grid(base)
    for c in grid:
        if c == base:
            continue
        evaluator.enqueue(c, reps, _ok_result(c, [pp_val] * reps, [tg_val] * reps))


def _enqueue_profile(
    evaluator: FakeScreeningEvaluator,
    base: CandidateConfig,
    profile: SearchProfile,
    pp_val: float = 100.0,
    tg_val: float = 20.0,
) -> None:
    _enqueue_grid(evaluator, base, profile.min_repetitions, pp_val, tg_val)


def _enqueue_adaptive(
    evaluator: FakeScreeningEvaluator,
    base: CandidateConfig,
    profile: SearchProfile,
) -> None:
    grid = generate_core_grid(base)
    for c in grid:
        if c == base:
            continue
        evaluator.enqueue(c, 1, _ok_result(c, [100.0], [20.0]))


# ── Profile tests ─────────────────────────────────────────────────


class TestProfiles:
    def test_quick_min_reps_at_least_three(self) -> None:
        assert QUICK_PROFILE.min_repetitions >= 3

    def test_standard_min_reps_at_least_three(self) -> None:
        assert STANDARD_PROFILE.min_repetitions >= 3

    def test_thorough_min_reps_at_least_three(self) -> None:
        assert THOROUGH_PROFILE.min_repetitions >= 3

    def test_quick_monotonic_vs_standard(self) -> None:
        assert QUICK_PROFILE.max_attempts < STANDARD_PROFILE.max_attempts
        assert QUICK_PROFILE.min_repetitions < STANDARD_PROFILE.min_repetitions
        q_max = QUICK_PROFILE.max_repetitions_per_candidate
        s_max = STANDARD_PROFILE.max_repetitions_per_candidate
        assert q_max < s_max
        assert QUICK_PROFILE.tie_bound_coefficient > STANDARD_PROFILE.tie_bound_coefficient
        assert QUICK_PROFILE.dispersion_threshold > STANDARD_PROFILE.dispersion_threshold

    def test_standard_monotonic_vs_thorough(self) -> None:
        assert STANDARD_PROFILE.max_attempts < THOROUGH_PROFILE.max_attempts
        assert STANDARD_PROFILE.min_repetitions < THOROUGH_PROFILE.min_repetitions
        s_max = STANDARD_PROFILE.max_repetitions_per_candidate
        t_max = THOROUGH_PROFILE.max_repetitions_per_candidate
        assert s_max < t_max
        assert STANDARD_PROFILE.tie_bound_coefficient > THOROUGH_PROFILE.tie_bound_coefficient
        assert STANDARD_PROFILE.dispersion_threshold > THOROUGH_PROFILE.dispersion_threshold

    def test_get_profile_known(self) -> None:
        assert get_profile("quick") == QUICK_PROFILE
        assert get_profile("standard") == STANDARD_PROFILE
        assert get_profile("thorough") == THOROUGH_PROFILE

    def test_get_profile_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("turbo")

    def test_screening_n_ctx_is_fixed(self) -> None:
        assert SCREENING_N_CTX.total == 512
        assert SCREENING_N_CTX.prompt == 128
        assert SCREENING_N_CTX.generation == 256
        assert SCREENING_N_CTX.depth == 128


class TestProfileValidation:
    def test_zero_attempts_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            SearchProfile(label="x", max_attempts=0, min_repetitions=3,
                          max_repetitions_per_candidate=3,
                          tie_bound_coefficient=0.05, dispersion_threshold=0.1)

    def test_below_three_min_reps_raises(self) -> None:
        with pytest.raises(ValueError, match="user story 31"):
            SearchProfile(label="x", max_attempts=5, min_repetitions=2,
                          max_repetitions_per_candidate=3,
                          tie_bound_coefficient=0.05, dispersion_threshold=0.1)

    def test_max_reps_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_repetitions_per_candidate"):
            SearchProfile(label="x", max_attempts=5, min_repetitions=3,
                          max_repetitions_per_candidate=2,
                          tie_bound_coefficient=0.05, dispersion_threshold=0.1)

    def test_negative_tie_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="tie_bound_coefficient"):
            SearchProfile(label="x", max_attempts=5, min_repetitions=3,
                          max_repetitions_per_candidate=5,
                          tie_bound_coefficient=-0.1, dispersion_threshold=0.1)

    def test_negative_dispersion_raises(self) -> None:
        with pytest.raises(ValueError, match="dispersion_threshold"):
            SearchProfile(label="x", max_attempts=5, min_repetitions=3,
                          max_repetitions_per_candidate=5,
                          tie_bound_coefficient=0.05, dispersion_threshold=-0.1)


# ── Grid generation tests ─────────────────────────────────────────


class TestCoreCandidateGrid:
    def test_generates_variations(self, base_candidate: CandidateConfig) -> None:
        grid = generate_core_grid(base_candidate)
        assert len(grid) > 1

    def test_preserves_fixed_params(self, base_candidate: CandidateConfig) -> None:
        grid = generate_core_grid(base_candidate)
        for c in grid:
            assert c.n_gpu_layers == 0
            assert c.flash_attn is False
            assert c.cache_type_k == "f16"
            assert c.cache_type_v == "f16"

    def test_includes_base_candidate(self, base_candidate: CandidateConfig) -> None:
        grid = generate_core_grid(base_candidate)
        matching = [c for c in grid if c == base_candidate]
        assert len(matching) == 1


# ── Baseline evaluation ───────────────────────────────────────────


class TestBaseline:
    def test_baseline_evaluated_first(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)
        screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert len(evaluator.calls) >= 2
        assert evaluator.calls[0][0] == base_candidate

    def test_baseline_failure_returns_empty(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _fail_result(base_candidate, "crash", "binary not found"),
        )
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert len(report.survivors) == 0
        assert report.leader is None
        assert report.baseline_disposition.success is False

    def test_baseline_counts_toward_budget(self, base_candidate: CandidateConfig) -> None:
        tiny = SearchProfile(
            label="tiny", max_attempts=1, min_repetitions=3,
            max_repetitions_per_candidate=3, tie_bound_coefficient=0.05,
            dispersion_threshold=0.15,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, 3,
            _ok_result(base_candidate, [500.0] * 3, [25.0] * 3),
        )
        report = screen(
            base_candidate=base_candidate,
            profile=tiny,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.attempts_used == 1


# ── Disposition list tests ────────────────────────────────────────


class TestDispositionList:
    def test_all_candidates_have_dispositions(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.baseline_disposition is not None
        assert len(report.grid_dispositions) > 0

    def test_non_survivors_not_ranked(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        grid = generate_core_grid(base_candidate)
        for i, c in enumerate(grid):
            if c == base_candidate:
                continue
            if i % 2 == 0:
                evaluator.enqueue(c, QUICK_PROFILE.min_repetitions,
                                   _fail_result(c, "crash", "OOM"))
            else:
                evaluator.enqueue(c, QUICK_PROFILE.min_repetitions,
                                   _ok_result(c, [100.0] * QUICK_PROFILE.min_repetitions,
                                             [20.0] * QUICK_PROFILE.min_repetitions))
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        for d in report.grid_dispositions:
            if not d.success:
                assert not any(s.candidate == d.candidate for s in report.survivors)

    def test_budget_exhaustion_recorded(self, base_candidate: CandidateConfig) -> None:
        small = SearchProfile(
            label="small_budget",
            max_attempts=3, min_repetitions=3,
            max_repetitions_per_candidate=3, tie_bound_coefficient=0.05,
            dispersion_threshold=0.15,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, 3,
            _ok_result(base_candidate, [500.0] * 3, [25.0] * 3),
        )
        grid = generate_core_grid(base_candidate)
        enqueued = 0
        for c in grid:
            if c == base_candidate:
                continue
            if enqueued < 2:
                evaluator.enqueue(c, 3, _ok_result(c, [100.0] * 3, [20.0] * 3))
                enqueued += 1
        report = screen(
            base_candidate=base_candidate,
            profile=small,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        not_attempted = [d for d in report.grid_dispositions if d.disposition == "not_attempted"]
        assert len(not_attempted) > 0
        assert report.attempts_used == 3

    def test_matches_baseline_recorded(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        matches = [d for d in report.grid_dispositions if d.disposition == "matches_baseline"]
        assert len(matches) >= 1


# ── Scoring tests ─────────────────────────────────────────────────


class TestScoring:
    def test_pp_and_tg_scores(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE, pp_val=250.0, tg_val=12.5)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        for s in report.survivors:
            assert s.pp_score > 0
            assert s.tg_score > 0
            assert s.balanced > 0

    def test_balanced_is_normalized_geometric_mean(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE, pp_val=250.0, tg_val=12.5)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        for s in report.survivors:
            expected_pp = s.pp_median / 500.0
            expected_tg = s.tg_median / 25.0
            expected_balanced = math.sqrt(expected_pp * expected_tg)
            assert s.pp_score == pytest.approx(expected_pp)
            assert s.tg_score == pytest.approx(expected_tg)
            assert s.balanced == pytest.approx(expected_balanced)


# ── Ranking tests ─────────────────────────────────────────────────


class TestRanking:
    def test_rank_by_balanced_default(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        for i in range(len(report.survivors) - 1):
            assert report.survivors[i].balanced >= report.survivors[i + 1].balanced

    def test_rank_by_pp_objective(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, QUICK_PROFILE.min_repetitions,
                               _ok_result(c, [100.0] * QUICK_PROFILE.min_repetitions,
                                         [20.0] * QUICK_PROFILE.min_repetitions))
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="pp",
        )
        for i in range(len(report.survivors) - 1):
            assert report.survivors[i].pp_score >= report.survivors[i + 1].pp_score

    def test_rank_by_tg_objective(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, QUICK_PROFILE.min_repetitions,
                               _ok_result(c, [100.0] * QUICK_PROFILE.min_repetitions,
                                         [20.0] * QUICK_PROFILE.min_repetitions))
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="tg",
        )
        for i in range(len(report.survivors) - 1):
            assert report.survivors[i].tg_score >= report.survivors[i + 1].tg_score


# ── Objective-specific tie detection tests ────────────────────────


class TestObjectiveSpecificTies:
    def test_ties_use_pp_scores_when_pp_objective(self, base_candidate: CandidateConfig) -> None:
        wide = SearchProfile(
            label="wide_tie", max_attempts=27, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.5,
            dispersion_threshold=0.15,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, 3,
            _ok_result(base_candidate, [500.0] * 3, [25.0] * 3),
        )
        grid = generate_core_grid(base_candidate)
        targets = [c for c in grid if c != base_candidate][:3]
        a, b = targets[0], targets[1]
        # a has high pp_score, low tg_score; b has low pp_score, high tg_score
        evaluator.enqueue(a, 3, _ok_result(a, [300.0] * 3, [5.0] * 3))
        evaluator.enqueue(b, 3, _ok_result(b, [100.0] * 3, [15.0] * 3))
        for c in grid:
            if c == base_candidate or c == a or c == b:
                continue
            evaluator.enqueue(c, 3, _ok_result(c, [100.0] * 3, [15.0] * 3))

        report_pp = screen(
            base_candidate=base_candidate,
            profile=wide,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="pp",
        )
        if report_pp.survivors:
            top = report_pp.survivors[0]
            assert top.pp_score >= report_pp.survivors[-1].pp_score

    def test_ties_use_tg_scores_when_tg_objective(self, base_candidate: CandidateConfig) -> None:
        wide = SearchProfile(
            label="wide_tie", max_attempts=27, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.5,
            dispersion_threshold=0.15,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, 3,
            _ok_result(base_candidate, [500.0] * 3, [25.0] * 3),
        )
        grid = generate_core_grid(base_candidate)
        targets = [c for c in grid if c != base_candidate][:3]
        a, b = targets[0], targets[1]
        evaluator.enqueue(a, 3, _ok_result(a, [100.0] * 3, [22.0] * 3))
        evaluator.enqueue(b, 3, _ok_result(b, [200.0] * 3, [18.0] * 3))
        for c in grid:
            if c == base_candidate or c == a or c == b:
                continue
            evaluator.enqueue(c, 3, _ok_result(c, [100.0] * 3, [15.0] * 3))

        report_tg = screen(
            base_candidate=base_candidate,
            profile=wide,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="tg",
        )
        if report_tg.survivors:
            top = report_tg.survivors[0]
            assert top.tg_score >= report_tg.survivors[-1].tg_score


# ── Tie-breaking tests ─────────────────────────────────────────────


class TestTieBreaking:
    def _make_profile(self, label: str = "tight") -> SearchProfile:
        return SearchProfile(
            label=label, max_attempts=27, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.5,
            dispersion_threshold=0.15,
        )

    def _all_grid(
        self, base: CandidateConfig, profile: SearchProfile, evaluator: FakeScreeningEvaluator,
    ) -> None:
        evaluator.enqueue(base, profile.min_repetitions,
                           _ok_result(base, [500.0] * profile.min_repetitions,
                                     [25.0] * profile.min_repetitions))
        grid = generate_core_grid(base)
        for c in grid:
            if c == base:
                continue
            evaluator.enqueue(c, profile.min_repetitions,
                               _ok_result(c, [100.0] * profile.min_repetitions,
                                         [20.0] * profile.min_repetitions))

    def test_stability_breaks_tie(self, base_candidate: CandidateConfig) -> None:
        profile = self._make_profile()
        evaluator = FakeScreeningEvaluator()
        self._all_grid(base_candidate, profile, evaluator)
        grid = generate_core_grid(base_candidate)
        stable, unstable = [c for c in grid if c != base_candidate][:2]
        stable_reps = profile.min_repetitions
        evaluator.enqueue(stable, stable_reps,
                           _ok_result(stable, [300.0, 301.0, 302.0], [15.0, 15.1, 15.0]))
        evaluator.enqueue(unstable, stable_reps,
                           _ok_result(unstable, [300.0, 330.0, 280.0], [15.0, 20.0, 18.0]))
        report = screen(
            base_candidate=base_candidate,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        survivors = [s for s in report.survivors if s.candidate != base_candidate]
        assert len(survivors) >= 2
        stable_one = next(s for s in survivors if s.candidate == stable)
        unstable_one = next(s for s in survivors if s.candidate == unstable)
        stable_combined = stable_one.pp_relative_dispersion + stable_one.tg_relative_dispersion
        unstable_combined = unstable_one.pp_relative_dispersion + unstable_one.tg_relative_dispersion  # noqa: E501
        assert stable_combined <= unstable_combined

    def test_memory_breaks_tie(self, base_candidate: CandidateConfig) -> None:
        profile = self._make_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base_candidate, profile.min_repetitions,
                           _ok_result(base_candidate, [500.0] * profile.min_repetitions,
                                     [25.0] * profile.min_repetitions, memory=8000))
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, profile.min_repetitions,
                               _ok_result(c, [100.0] * profile.min_repetitions,
                                         [20.0] * profile.min_repetitions))
        targets = [c for c in grid if c != base_candidate][:3]
        low_mem, high_mem = targets[0], targets[1]
        low_samples = [300.0, 301.0, 302.0]
        tg_low = [15.0, 15.1, 15.0]
        evaluator.enqueue(low_mem, profile.min_repetitions,
                           _ok_result(low_mem, low_samples, tg_low, memory=4000))
        evaluator.enqueue(high_mem, profile.min_repetitions,
                           _ok_result(high_mem, low_samples, tg_low, memory=8000))
        report = screen(
            base_candidate=base_candidate,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        survivors = [s for s in report.survivors if s.candidate != base_candidate]
        assert len(survivors) >= 2
        low = next(s for s in survivors if s.candidate == low_mem)
        high = next(s for s in survivors if s.candidate == high_mem)
        assert survivors.index(low) < survivors.index(high)

    def test_unknown_memory_does_not_win(self, base_candidate: CandidateConfig) -> None:
        profile = self._make_profile()
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base_candidate, profile.min_repetitions,
                           _ok_result(base_candidate, [500.0] * profile.min_repetitions,
                                     [25.0] * profile.min_repetitions, memory=8000))
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, profile.min_repetitions,
                               _ok_result(c, [100.0] * profile.min_repetitions,
                                         [20.0] * profile.min_repetitions))
        targets = [c for c in grid if c != base_candidate][:3]
        known, unknown = targets[0], targets[1]
        kmem_row = [300.0, 301.0, 302.0]
        kmem_tg = [15.0, 15.1, 15.0]
        evaluator.enqueue(known, profile.min_repetitions,
                           _ok_result(known, kmem_row, kmem_tg, memory=8000))
        evaluator.enqueue(unknown, profile.min_repetitions,
                           _ok_result(unknown, kmem_row, kmem_tg))
        report = screen(
            base_candidate=base_candidate,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        survivors = [s for s in report.survivors if s.candidate != base_candidate]
        assert len(survivors) >= 2
        k = next(s for s in survivors if s.candidate == known)
        u = next(s for s in survivors if s.candidate == unknown)
        assert survivors.index(k) < survivors.index(u)

    def test_simpler_flags_breaks_tie(self, base_candidate: CandidateConfig) -> None:
        profile = self._make_profile()
        evaluator = FakeScreeningEvaluator()
        self._all_grid(base_candidate, profile, evaluator)
        simpler = CandidateConfig(
            threads=2, batch_size=256, ubatch_size=256,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        complex_cfg = CandidateConfig(
            threads=8, batch_size=1024, ubatch_size=1024,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        evaluator.enqueue(simpler, profile.min_repetitions,
                           _ok_result(simpler, [300.0, 301.0, 302.0], [15.0, 15.1, 15.0]))
        evaluator.enqueue(complex_cfg, profile.min_repetitions,
                           _ok_result(complex_cfg, [300.0, 301.0, 302.0], [15.0, 15.1, 15.0]))
        report = screen(
            base_candidate=base_candidate,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        sim_list = [s for s in report.survivors if s.candidate == simpler]
        cmp_list = [s for s in report.survivors if s.candidate == complex_cfg]
        assert sim_list, "Simpler candidate must be a survivor"
        assert cmp_list, "Complex candidate must be a survivor"
        assert report.survivors.index(sim_list[0]) < report.survivors.index(cmp_list[0])


# ── Dominance pruning tests ────────────────────────────────────────


class TestDominancePruning:
    def test_dominated_candidate_pruned(self, base_candidate: CandidateConfig) -> None:
        profile = SearchProfile(
            label="dom", max_attempts=27, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.05,
            dispersion_threshold=0.5,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base_candidate, 3,
                           _ok_result(base_candidate, [500.0] * 3, [25.0] * 3))
        grid = generate_core_grid(base_candidate)
        targets = [c for c in grid if c != base_candidate][:4]
        dom = targets[0]
        dominated = targets[1]
        evaluator.enqueue(dom, 3, _ok_result(dom, [300.0] * 3, [20.0] * 3))
        evaluator.enqueue(dominated, 3, _ok_result(dominated, [200.0] * 3, [10.0] * 3))
        for c in grid:
            if c == base_candidate or c == dom or c == dominated:
                continue
            evaluator.enqueue(c, 3, _ok_result(c, [100.0] * 3, [5.0] * 3))
        report = screen(
            base_candidate=base_candidate,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        pruned = [d for d in report.grid_dispositions if d.disposition == "dominance_pruned"]
        assert len(pruned) >= 1
        assert not any(s.candidate == dominated for s in report.survivors)

    def test_no_false_dominance(self, base_candidate: CandidateConfig) -> None:
        profile = SearchProfile(
            label="no_false", max_attempts=27, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.05,
            dispersion_threshold=0.5,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base_candidate, 3,
                           _ok_result(base_candidate, [500.0] * 3, [25.0] * 3))
        grid = generate_core_grid(base_candidate)
        targets = [c for c in grid if c != base_candidate][:3]
        a, b = targets[0], targets[1]
        # a has better PP but worse TG; neither strictly dominates
        evaluator.enqueue(a, 3, _ok_result(a, [300.0] * 3, [10.0] * 3))
        evaluator.enqueue(b, 3, _ok_result(b, [200.0] * 3, [15.0] * 3))
        for c in grid:
            if c == base_candidate or c == a or c == b:
                continue
            evaluator.enqueue(c, 3, _ok_result(c, [250.0] * 3, [12.5] * 3))
        report = screen(
            base_candidate=base_candidate,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        pruned = [d for d in report.grid_dispositions if d.disposition == "dominance_pruned"]
        assert len(pruned) == 0
        assert any(s.candidate == a for s in report.survivors)
        assert any(s.candidate == b for s in report.survivors)


# ── Provisional leader tests ──────────────────────────────────────


class TestProvisionalLeader:
    def test_leader_present_with_survivors(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.leader is not None
        assert report.leader.pp_score > 0

    def test_leader_has_correct_label(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.leader is not None
        assert "screening evidence only" in report.leader.label.lower()
        assert "not validated" in report.leader.label.lower()

    def test_leader_matches_objective(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)
        report_pp = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="pp",
        )
        if report_pp.leader is not None and report_pp.survivors:
            assert report_pp.leader.pp_score == report_pp.survivors[0].pp_score
            assert report_pp.leader.objective == "pp"


# ── Zero-survivor tests ────────────────────────────────────────────


class TestZeroSurvivors:
    def _all_fail_grid(
        self, base: CandidateConfig, evaluator: FakeScreeningEvaluator, reps: int,
    ) -> None:
        grid = generate_core_grid(base)
        for c in grid:
            if c == base:
                continue
            evaluator.enqueue(c, reps, _fail_result(c, "crash", "all failed"))

    def test_empty_shortlist_when_all_fail(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        self._all_fail_grid(base_candidate, evaluator, QUICK_PROFILE.min_repetitions)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert len(report.survivors) == 0
        assert report.leader is None

    def test_no_leader_with_no_survivors(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        self._all_fail_grid(base_candidate, evaluator, QUICK_PROFILE.min_repetitions)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.leader is None

    def test_all_failures_have_reasons(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        self._all_fail_grid(base_candidate, evaluator, QUICK_PROFILE.min_repetitions)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        for d in report.grid_dispositions:
            if not d.success:
                assert d.reason != ""

    def test_disposition_summary_on_zero_survivors(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        self._all_fail_grid(base_candidate, evaluator, QUICK_PROFILE.min_repetitions)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.disposition_summary
        assert "Baseline:" in report.disposition_summary
        assert "0 survived" in report.disposition_summary


# ── Adaptive repetitions tests ────────────────────────────────────


class TestAdaptiveRepetitions:
    def test_high_dispersion_triggers_adaptive(self, base_candidate: CandidateConfig) -> None:
        adaptive_profile = SearchProfile(
            label="adaptive",
            max_attempts=30, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.05,
            dispersion_threshold=0.01,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base_candidate, 3,
                           _ok_result(base_candidate, [500.0] * 3, [25.0] * 3))
        grid = generate_core_grid(base_candidate)
        targets: list[CandidateConfig] = []
        for c in grid:
            if c == base_candidate:
                continue
            if len(targets) < 2:
                targets.append(c)
                evaluator.enqueue(c, 3, _ok_result(c, [100.0, 200.0, 150.0], [10.0, 30.0, 20.0]))
                evaluator.enqueue(c, 1, _ok_result(c, [1500.0], [30.0]))
        for rc in grid:
            if rc == base_candidate or rc in targets:
                continue
            evaluator.enqueue(rc, 3, _ok_result(rc, [100.0] * 3, [20.0] * 3))
        report = screen(
            base_candidate=base_candidate,
            profile=adaptive_profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        for s in report.survivors:
            if s.candidate in targets:
                assert s.total_repetitions >= 4

    def test_budget_limits_adaptive(self, base_candidate: CandidateConfig) -> None:
        tight = SearchProfile(
            label="tight_adaptive",
            max_attempts=3, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.05,
            dispersion_threshold=0.01,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base_candidate, 3,
                           _ok_result(base_candidate, [500.0] * 3, [25.0] * 3))
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, 3, _ok_result(c, [100.0] * 3, [20.0] * 3))
        report = screen(
            base_candidate=base_candidate,
            profile=tight,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.attempts_used == 3

    def test_no_adaptive_when_dispersion_low(self, base_candidate: CandidateConfig) -> None:
        stable = SearchProfile(
            label="stable",
            max_attempts=28, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.0,
            dispersion_threshold=0.50,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base_candidate, 3,
                           _ok_result(base_candidate, [500.0, 501.0, 499.0], [25.0, 25.1, 24.9]))
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, 3, _ok_result(c, [100.0, 100.1, 99.9], [20.0, 20.1, 19.9]))
        report = screen(
            base_candidate=base_candidate,
            profile=stable,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        for s in report.survivors:
            assert s.total_repetitions == 3


# ── Overlap-driven adaptive tests ──────────────────────────────────


class TestOverlapAdaptive:
    def test_overlap_does_not_trigger_when_clearly_separated(
        self, base_candidate: CandidateConfig,
    ) -> None:
        profile = SearchProfile(
            label="sep", max_attempts=28, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.0,
            dispersion_threshold=0.50,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base_candidate, 3,
                           _ok_result(base_candidate, [500.0] * 3, [25.0] * 3))
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, 3, _ok_result(c, [100.0] * 3, [20.0] * 3))
        report = screen(
            base_candidate=base_candidate,
            profile=profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        for s in report.survivors:
            assert s.total_repetitions == 3


# ── Disposition summary tests ─────────────────────────────────────


class TestDispositionSummary:
    def test_summary_generated(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.disposition_summary
        assert "Baseline:" in report.disposition_summary


# ── Positive overlap adaptation test ──────────────────────────────


class TestPositiveOverlapAdaptation:
    def test_overlapping_leaders_trigger_extra_reps(self, base_candidate: CandidateConfig) -> None:
        overlap_profile = SearchProfile(
            label="overlap",
            max_attempts=30, min_repetitions=3,
            max_repetitions_per_candidate=6, tie_bound_coefficient=0.3,
            dispersion_threshold=0.5,
        )
        evaluator = FakeScreeningEvaluator()
        leader_pp = [300.0, 301.0, 302.0]
        leader_tg = [15.0, 15.1, 15.0]
        evaluator.enqueue(base_candidate, 3,
                           _ok_result(base_candidate, [500.0] * 3, [25.0] * 3))
        grid = generate_core_grid(base_candidate)
        targets: list[CandidateConfig] = []
        for c in grid:
            if c == base_candidate:
                continue
            if len(targets) < 2:
                targets.append(c)
                evaluator.enqueue(c, 3, _ok_result(c, leader_pp, leader_tg))
                evaluator.enqueue(c, 1, _ok_result(c, [305.0], [15.5]))
        for rc in grid:
            if rc == base_candidate or rc in targets:
                continue
            evaluator.enqueue(rc, 3, _ok_result(rc, [100.0] * 3, [5.0] * 3))
        report = screen(
            base_candidate=base_candidate,
            profile=overlap_profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        overlap_survivors = [s for s in report.survivors if s.candidate in targets]
        for s in overlap_survivors:
            assert s.total_repetitions >= 4


# ── Evaluator contract validation tests ──────────────────────────


class TestEvaluatorContractValidation:
    def test_wrong_candidate_rejected(self, base_candidate: CandidateConfig) -> None:
        wrong = CandidateConfig(
            threads=99, batch_size=99, ubatch_size=99,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )

        def bad_evaluator(c, n, r):  # type: ignore[no-untyped-def]
            return ScreeningResult(
                candidate=wrong if c == base_candidate else c,
                success=True,
                observation=ScreeningObservation(
                    candidate=wrong if c == base_candidate else c,
                    pp_samples=(500.0,) * r,
                    tg_samples=(25.0,) * r,
                ),
            )

        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=bad_evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.baseline_disposition.disposition == "systemic_error"
        assert "wrong candidate" in report.baseline_disposition.reason.lower()

    def test_wrong_observation_candidate_rejected(self, base_candidate: CandidateConfig) -> None:
        wrong = CandidateConfig(
            threads=99, batch_size=99, ubatch_size=99,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )

        def bad_obs_evaluator(c, n, r):  # type: ignore[no-untyped-def]
            return ScreeningResult(
                candidate=c,
                success=True,
                observation=ScreeningObservation(
                    candidate=(wrong if c == base_candidate else c),
                    pp_samples=(500.0,) * r,
                    tg_samples=(25.0,) * r,
                ),
            )

        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=bad_obs_evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.baseline_disposition.disposition == "systemic_error"
        assert "observation candidate" in report.baseline_disposition.reason.lower()

    def test_wrong_sample_count_rejected(self, base_candidate: CandidateConfig) -> None:
        def wrong_count_eval(c, n, r):  # type: ignore[no-untyped-def]
            return ScreeningResult(
                candidate=c,
                success=True,
                observation=ScreeningObservation(
                    candidate=c,
                    pp_samples=(500.0,) * (r + 1),
                    tg_samples=(25.0,) * r,
                ),
            )

        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=wrong_count_eval,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.baseline_disposition.disposition == "malformed_output"
        assert "PP samples" in report.baseline_disposition.reason

    def test_non_finite_samples_rejected(self, base_candidate: CandidateConfig) -> None:
        def nan_eval(c, n, r):  # type: ignore[no-untyped-def]
            return ScreeningResult(
                candidate=c,
                success=True,
                observation=ScreeningObservation(
                    candidate=c,
                    pp_samples=(float("nan"),) * r,
                    tg_samples=(25.0,) * r,
                ),
            )

        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=nan_eval,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.baseline_disposition.disposition == "malformed_output"
        reason = report.baseline_disposition.reason
        assert "Non-positive" in reason or "non-finite" in reason

    def test_adaptive_invalid_extra_ignored(self, base_candidate: CandidateConfig) -> None:
        adapt_profile = SearchProfile(
            label="adapt_val", max_attempts=30, min_repetitions=3,
            max_repetitions_per_candidate=6, tie_bound_coefficient=0.3,
            dispersion_threshold=0.5,
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base_candidate, 3,
                           _ok_result(base_candidate, [500.0] * 3, [25.0] * 3))
        grid = generate_core_grid(base_candidate)
        targets: list[CandidateConfig] = []
        for c in grid:
            if c == base_candidate:
                continue
            if len(targets) < 1:
                targets.append(c)
                evaluator.enqueue(c, 3, _ok_result(c, [300.0, 301.0, 302.0], [15.0, 15.1, 15.0]))
                # Adaptive call returns wrong candidate
                wrong = CandidateConfig(
                    threads=99, batch_size=99, ubatch_size=99,
                    n_gpu_layers=0, flash_attn=False,
                    cache_type_k="f16", cache_type_v="f16",
                )
                evaluator.enqueue(c, 1, ScreeningResult(
                    candidate=wrong, success=True,
                    observation=ScreeningObservation(
                        candidate=wrong, pp_samples=(150.0,), tg_samples=(10.0,),
                    ),
                ))
            else:
                evaluator.enqueue(c, 3, _ok_result(c, [100.0] * 3, [5.0] * 3))

        report = screen(
            base_candidate=base_candidate,
            profile=adapt_profile,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        # Despite invalid adaptive extra, the original observation (3 reps) survives.
        for s in report.survivors:
            if s.candidate in targets:
                assert s.total_repetitions == 3


# ── Objective-specific dispersion tests ──────────────────────────


class TestObjectiveSpecificDispersion:
    def test_pp_objective_uses_pp_dispersion(self) -> None:
        wide = SearchProfile(
            label="wide", max_attempts=10, min_repetitions=3,
            max_repetitions_per_candidate=5, tie_bound_coefficient=0.5,
            dispersion_threshold=0.5,
        )
        cand_a = CandidateConfig(
            threads=2, batch_size=256, ubatch_size=256,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        cand_b = CandidateConfig(
            threads=8, batch_size=1024, ubatch_size=1024,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        base = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(base, 3, _ok_result(base, [500.0] * 3, [25.0] * 3))
        evaluator.enqueue(cand_a, 3, _ok_result(cand_a, [300.0, 301.0, 302.0], [15.0, 5.0, 25.0]))
        evaluator.enqueue(cand_b, 3, _ok_result(cand_b, [200.0, 201.0, 202.0], [14.9, 15.0, 15.1]))
        grid = generate_core_grid(base)
        for c in grid:
            if c == base or c == cand_a or c == cand_b:
                continue
            evaluator.enqueue(c, 3, _ok_result(c, [100.0] * 3, [10.0] * 3))

        # PP objective: use PP dispersion for tie bound.
        # cand_a pp_rd = ~0.003, cand_b pp_rd = ~0.005
        # cand_a tg_rd = ~0.47, cand_b tg_rd = ~0.007
        # Under PP objective, only PP rd matters for tie bound.
        report_pp = screen(
            base_candidate=base, profile=wide,
            evaluator=evaluator, screening_n_ctx=SCREENING_N_CTX, objective="pp",
        )
        assert report_pp.survivors[0].pp_score >= report_pp.survivors[-1].pp_score


# ── Non-empty reasons for all dispositions test ───────────────────


class TestNonEmptyReasons:
    def test_survived_has_reason(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, QUICK_PROFILE.min_repetitions,
                               _ok_result(c, [100.0] * QUICK_PROFILE.min_repetitions,
                                         [20.0] * QUICK_PROFILE.min_repetitions))
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        for d in report.grid_dispositions:
            assert d.reason, (  # noqa: E501
                f"Missing reason for disposition {d.disposition} on {d.candidate}"
            )

    def test_baseline_has_reason(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, QUICK_PROFILE.min_repetitions,
                               _ok_result(c, [100.0] * QUICK_PROFILE.min_repetitions,
                                         [20.0] * QUICK_PROFILE.min_repetitions))
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.baseline_disposition.reason or report.baseline_disposition.disposition


# ── Disposition summary actionable guidance (zero-survivor) ───────


class TestDispositionSummaryGuidance:
    def test_zero_survivor_has_guidance(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        grid = generate_core_grid(base_candidate)
        for c in grid:
            if c == base_candidate:
                continue
            evaluator.enqueue(c, QUICK_PROFILE.min_repetitions,
                               _fail_result(c, "crash", "segfault"))
        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert "Guidance:" in report.disposition_summary
        assert "crash" in report.disposition_summary


# ── Unsupported / cancellation disposition tests ──────────────────


class TestScreeningDispositionCoverage:
    def test_unsupported_disposition_accepted(self) -> None:
        sr = ScreeningResult(
            candidate=CandidateConfig(
                threads=1, batch_size=1, ubatch_size=1,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
            success=False,
            disposition="unsupported",
            reason="test unsupported",
        )
        assert sr.disposition == "unsupported"

    def test_cancellation_disposition_accepted(self) -> None:
        sr = ScreeningResult(
            candidate=CandidateConfig(
                threads=1, batch_size=1, ubatch_size=1,
                n_gpu_layers=0, flash_attn=False,
                cache_type_k="f16", cache_type_v="f16",
            ),
            success=False,
            disposition="cancellation",
            reason="test cancellation",
        )
        assert sr.disposition == "cancellation"


# ── Application production seam test (with session_spec) ─────────


class TestApplicationSeam:
    def test_screen_candidates_external_evaluator(self, base_candidate: CandidateConfig) -> None:
        from llama_parameter_scout.search import screen as _screen_fn

        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)

        report = _screen_fn(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
        )
        assert report.baseline_disposition is not None
        assert len(report.grid_dispositions) > 0
        assert report.objective == "balanced"

    def test_screen_candidates_with_pp_objective(self, base_candidate: CandidateConfig) -> None:
        evaluator = FakeScreeningEvaluator()
        evaluator.enqueue(
            base_candidate, QUICK_PROFILE.min_repetitions,
            _ok_result(base_candidate, [500.0] * QUICK_PROFILE.min_repetitions,
                       [25.0] * QUICK_PROFILE.min_repetitions),
        )
        _enqueue_profile(evaluator, base_candidate, QUICK_PROFILE)

        report = screen(
            base_candidate=base_candidate,
            profile=QUICK_PROFILE,
            evaluator=evaluator,
            screening_n_ctx=SCREENING_N_CTX,
            objective="pp",
        )
        assert report.objective == "pp"
