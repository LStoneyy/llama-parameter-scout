from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from llama_parameter_scout.models import (
    CandidateConfig,
    NctxComposition,
    NoWinnerReport,
    ProvisionalLeader,
    RankedCandidate,
    ScreeningObservation,
    ScreeningReport,
    ScreeningResult,
    SingleRunRequest,
    ValidatedRecommendation,
)
from llama_parameter_scout.search import (
    QUICK_PROFILE,
    validate_shortlist,
)

# ── Shared test values ────────────────────────────────────────────

TOTAL_CTX = 4096
PROMPT_CTX = 1024
GEN_CTX = 2048
DEPTH_CTX = 1024
REQUESTED_NCTX = NctxComposition(
    total=TOTAL_CTX, prompt=PROMPT_CTX,
    generation=GEN_CTX, depth=DEPTH_CTX,
)

BASE_CANDIDATE = CandidateConfig(
    threads=4, batch_size=512, ubatch_size=512,
    n_gpu_layers=0, flash_attn=False,
    cache_type_k="f16", cache_type_v="f16",
)


# ── Helpers ───────────────────────────────────────────────────────


def _ok_obs(
    candidate: CandidateConfig,
    pp_samples: tuple[float, ...] = (100.0, 101.0, 99.0),
    tg_samples: tuple[float, ...] = (20.0, 21.0, 19.0),
) -> ScreeningObservation:
    return ScreeningObservation(
        candidate=candidate,
        pp_samples=pp_samples,
        tg_samples=tg_samples,
        observed_prompt=PROMPT_CTX,
        observed_generation=GEN_CTX,
        observed_depth=DEPTH_CTX,
        observed_total=TOTAL_CTX,
    )


def _ok_result(
    candidate: CandidateConfig,
    pp_samples: tuple[float, ...] = (100.0, 101.0, 99.0),
    tg_samples: tuple[float, ...] = (20.0, 21.0, 19.0),
) -> ScreeningResult:
    return ScreeningResult(
        candidate=candidate,
        success=True,
        disposition="survived",
        observation=_ok_obs(candidate, pp_samples, tg_samples),
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


def _ranked(
    candidate: CandidateConfig,
    pp_score: float = 1.0,
    tg_score: float = 1.0,
    balanced: float = 1.0,
    pp_median: float = 100.0,
    tg_median: float = 20.0,
) -> RankedCandidate:
    return RankedCandidate(
        candidate=candidate,
        pp_score=pp_score,
        tg_score=tg_score,
        balanced=balanced,
        pp_median=pp_median,
        tg_median=tg_median,
        pp_relative_dispersion=0.01,
        tg_relative_dispersion=0.02,
    )


def _screening_report(
    survivors: tuple[RankedCandidate, ...],
    leader_idx: int = 0,
    baseline_pp: tuple[float, ...] = (100.0, 101.0, 99.0),
    baseline_tg: tuple[float, ...] = (20.0, 21.0, 19.0),
) -> ScreeningReport:
    base_obs = ScreeningObservation(
        candidate=BASE_CANDIDATE,
        pp_samples=baseline_pp,
        tg_samples=baseline_tg,
    )
    leader = None
    if survivors:
        best = survivors[leader_idx]
        leader = ProvisionalLeader(
            candidate=best.candidate,
            objective="balanced",
            pp_score=best.pp_score,
            tg_score=best.tg_score,
            balanced=best.balanced,
            pp_median=best.pp_median,
            tg_median=best.tg_median,
        )
    return ScreeningReport(
        profile=QUICK_PROFILE,
        baseline_disposition=ScreeningResult(
            candidate=BASE_CANDIDATE,
            success=True,
            observation=base_obs,
        ),
        grid_dispositions=(),
        survivors=survivors,
        leader=leader,
        attempts_used=1,
        objective="balanced",
        disposition_summary="test report",
    )


# ── Fake evaluator ────────────────────────────────────────────────


@dataclass
class FakeValidationEvaluator:
    _results: dict[tuple[int, int, int], ScreeningResult] = field(default_factory=dict)
    calls: list[tuple[CandidateConfig, NctxComposition]] = field(default_factory=list)

    def enqueue(self, candidate: CandidateConfig, result: ScreeningResult) -> None:
        key = (candidate.threads, candidate.batch_size, candidate.ubatch_size)
        self._results[key] = result

    def __call__(
        self,
        candidate: CandidateConfig,
        n_ctx: NctxComposition,
    ) -> ScreeningResult:
        self.calls.append((candidate, n_ctx))
        key = (candidate.threads, candidate.batch_size, candidate.ubatch_size)
        result = self._results.get(key)
        if result is None:
            raise ValueError(
                f"No enqueued result for threads={candidate.threads} "
                f"batch={candidate.batch_size} ubatch={candidate.ubatch_size}"
            )
        return result


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def candidates() -> list[CandidateConfig]:
    return [
        CandidateConfig(
            threads=2, batch_size=256, ubatch_size=256,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        ),
        CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        ),
        CandidateConfig(
            threads=8, batch_size=1024, ubatch_size=1024,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        ),
        CandidateConfig(
            threads=12, batch_size=2048, ubatch_size=2048,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        ),
        CandidateConfig(
            threads=16, batch_size=4096, ubatch_size=4096,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        ),
    ]


# ── Tests ─────────────────────────────────────────────────────────


class TestTopFinalistOomAndReplacement:
    def test_oom_finalist_replaced(self, candidates: list[CandidateConfig]) -> None:
        a, b, c = candidates[:3]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5, tg_score=1.2),
                        _ranked(b, pp_score=1.3, tg_score=1.0),
                        _ranked(c, pp_score=1.1, tg_score=0.8)),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _fail_result(a, "oom", "Out of memory at full context"))
        evaluator.enqueue(b, _ok_result(b, (300.0, 301.0, 299.0), (60.0, 61.0, 59.0)))
        evaluator.enqueue(c, _ok_result(c, (100.0, 101.0, 99.0), (20.0, 21.0, 19.0)))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, ValidatedRecommendation), (
            f"Expected ValidatedRecommendation, got {type(result).__name__}"
        )
        assert result.recommendation.candidate == b, (
            f"Expected candidate b ({b}) after OOM replacement, "
            f"got {result.recommendation.candidate}"
        )
        assert len(result.failures) == 1
        assert result.failures[0].candidate == a
        assert result.failures[0].disposition == "oom"
        assert result.attempts_used == 3
        validated = (result.recommendation, *result.alternatives)
        assert len(validated) == 2  # b and c both validated

    def test_no_winner_when_all_fail(
        self, candidates: list[CandidateConfig]
    ) -> None:
        a, b = candidates[:2]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5), _ranked(b, pp_score=1.3)),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _fail_result(a, "crash", "segfault"))
        evaluator.enqueue(b, _fail_result(b, "oom", "OOM"))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport), (
            f"Expected NoWinnerReport, got {type(result).__name__}"
        )
        assert result.provisional_leader is not None
        assert result.provisional_leader.candidate == a
        assert len(result.failures) == 2
        assert result.failures[0].disposition == "crash"
        assert result.failures[1].disposition == "oom"
        assert result.attempts_used == 2

    def test_provisional_leader_label_preserved(self, candidates: list[CandidateConfig]) -> None:
        a = candidates[0]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _fail_result(a, "oom", "OOM"))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport)
        assert result.provisional_leader is not None
        assert "Screening evidence only" in result.provisional_leader.label

    def test_never_describe_screening_as_best_at_context(
        self, candidates: list[CandidateConfig]
    ) -> None:
        a = candidates[0]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _ok_result(a, (150.0, 151.0, 149.0), (25.0, 26.0, 24.0)))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert result.recommendation.label != (
            "Screening evidence only - not validated at requested context"
        )
        assert "Full-context validation" in result.recommendation.label
        assert "verified at requested context" in result.recommendation.label


class TestRankReversalAfterReplacement:
    def test_rank_reversal(self, candidates: list[CandidateConfig]) -> None:
        a, b, c = candidates[:3]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5, tg_score=1.2),
                        _ranked(b, pp_score=1.3, tg_score=1.0),
                        _ranked(c, pp_score=1.1, tg_score=0.8)),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _fail_result(a, "oom", "OOM at full context"))
        evaluator.enqueue(b, _ok_result(b, (200.0, 201.0, 199.0), (30.0, 31.0, 29.0)))
        evaluator.enqueue(c, _ok_result(c, (300.0, 301.0, 299.0), (40.0, 41.0, 39.0)))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert result.recommendation.candidate == c, (
            f"Expected rank reversal: candidate c ({c}) should be top, "
            f"got {result.recommendation.candidate}"
        )
        assert len(result.alternatives) == 1
        assert result.alternatives[0].candidate == b
        assert len(result.failures) == 1
        assert result.failures[0].candidate == a
        assert result.attempts_used == 3


class TestBudgetExhaustion:
    def test_budget_exhausted(self, candidates: list[CandidateConfig]) -> None:
        a, b, c = candidates[:3]
        report = _screening_report(
            survivors=(_ranked(a), _ranked(b), _ranked(c)),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        for cand in (a, b, c):
            evaluator.enqueue(cand, _fail_result(cand, "crash", "failure"))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            budget=2,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport)
        assert result.attempts_used == 2
        assert len(result.failures) == 3  # 2 actual + 1 budget exhaustion
        assert result.failures[2].disposition == "validation_budget_exhausted"
        assert "budget of 2" in result.failures[2].reason


class TestContextAllFieldsMissing:
    def test_context_all_fields_missing(self, candidates: list[CandidateConfig]) -> None:
        a = candidates[0]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        no_ctx_obs = ScreeningObservation(
            candidate=a,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
            observed_prompt=None,
            observed_generation=None,
            observed_depth=None,
            observed_total=None,
        )
        evaluator.enqueue(a, ScreeningResult(
            candidate=a, success=True, observation=no_ctx_obs,
        ))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport)
        assert len(result.failures) == 1
        assert result.failures[0].disposition == "context_mismatch"
        assert "missing" in result.failures[0].reason

    def test_context_partial_field_missing(self, candidates: list[CandidateConfig]) -> None:
        a = candidates[0]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        partial_obs = ScreeningObservation(
            candidate=a,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
            observed_prompt=1024,
            observed_generation=2048,
            observed_depth=None,
            observed_total=4096,
        )
        evaluator.enqueue(a, ScreeningResult(
            candidate=a, success=True, observation=partial_obs,
        ))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport)
        assert len(result.failures) == 1
        assert result.failures[0].disposition == "context_mismatch"
        assert "depth=missing" in result.failures[0].reason


class TestExactContextMismatch:
    def test_context_mismatch_detected(self, candidates: list[CandidateConfig]) -> None:
        a, b = candidates[:2]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5), _ranked(b, pp_score=1.3)),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        mismatched_obs = ScreeningObservation(
            candidate=a,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
            observed_prompt=512,
            observed_generation=1024,
            observed_depth=512,
            observed_total=2048,
        )
        mismatched_result = ScreeningResult(
            candidate=a,
            success=True,
            observation=mismatched_obs,
        )
        evaluator.enqueue(a, mismatched_result)
        evaluator.enqueue(b, _ok_result(b, (150.0, 151.0, 149.0), (25.0, 26.0, 24.0)))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert result.recommendation.candidate == b
        assert len(result.failures) == 1
        assert result.failures[0].disposition == "context_mismatch"
        assert "Effective context does not match" in result.failures[0].reason

    def test_context_sum_validation(self, candidates: list[CandidateConfig]) -> None:
        a = candidates[0]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        bad_sum_obs = ScreeningObservation(
            candidate=a,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
            observed_prompt=1024,
            observed_generation=2048,
            observed_depth=1024,
            observed_total=5000,
        )
        bad_sum_result = ScreeningResult(
            candidate=a,
            success=True,
            observation=bad_sum_obs,
        )
        evaluator.enqueue(a, bad_sum_result)

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport)
        assert len(result.failures) == 1
        assert result.failures[0].disposition == "context_mismatch"
        assert "Effective context does not match" in result.failures[0].reason

    def test_component_mismatch_on_prompt(self, candidates: list[CandidateConfig]) -> None:
        a = candidates[0]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        bad_prompt_obs = ScreeningObservation(
            candidate=a,
            pp_samples=(100.0, 101.0, 99.0),
            tg_samples=(20.0, 21.0, 19.0),
            observed_prompt=999,
            observed_generation=2048,
            observed_depth=1024,
            observed_total=4096,
        )
        evaluator.enqueue(a, ScreeningResult(
            candidate=a,
            success=True,
            observation=bad_prompt_obs,
        ))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport)
        assert result.failures[0].disposition == "context_mismatch"


class TestThreeSuccessfulValidations:
    def test_stops_at_three(self, candidates: list[CandidateConfig]) -> None:
        a, b, c, d, e = candidates
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),
                        _ranked(b, pp_score=1.4),
                        _ranked(c, pp_score=1.3),
                        _ranked(d, pp_score=1.2),
                        _ranked(e, pp_score=1.1)),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        for cand in (a, b, c, d, e):
            evaluator.enqueue(cand, _ok_result(
                cand, (100.0, 101.0, 99.0), (20.0, 21.0, 19.0),
            ))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert len(evaluator.calls) == 3
        validated_candidates = [result.recommendation, *result.alternatives]
        assert len(validated_candidates) == 3
        assert result.attempts_used == 3
        assert result.failures == ()

    def test_shortlist_natural_exhaustion_returns_real_outcomes(
        self, candidates: list[CandidateConfig]
    ) -> None:
        a = candidates[0]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _fail_result(a, "crash", "crash"))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport)
        assert result.attempts_used == 1
        assert len(result.failures) == 1
        assert result.failures[0].disposition == "crash"
        assert result.failures[0].candidate == a


class TestObjectiveRanking:
    def test_pp_objective(self, candidates: list[CandidateConfig]) -> None:
        a, b = candidates[:2]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5, tg_score=1.0),
                        _ranked(b, pp_score=1.3, tg_score=1.2)),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _ok_result(a, (100.0, 101.0, 99.0), (20.0, 21.0, 19.0)))
        evaluator.enqueue(b, _ok_result(b, (80.0, 81.0, 79.0), (30.0, 31.0, 29.0)))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            objective="pp",
            evaluator=evaluator,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert result.recommendation.candidate == a
        assert result.recommendation.pp_score >= result.alternatives[0].pp_score

    def test_tg_objective(self, candidates: list[CandidateConfig]) -> None:
        a, b = candidates[:2]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5, tg_score=1.0),
                        _ranked(b, pp_score=1.3, tg_score=1.2)),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _ok_result(a, (100.0, 101.0, 99.0), (20.0, 21.0, 19.0)))
        evaluator.enqueue(b, _ok_result(b, (80.0, 81.0, 79.0), (30.0, 31.0, 29.0)))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            objective="tg",
            evaluator=evaluator,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert result.recommendation.candidate == b
        assert result.recommendation.tg_score >= result.alternatives[0].tg_score

    def test_balanced_objective(self, candidates: list[CandidateConfig]) -> None:
        a, b = candidates[:2]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5, tg_score=1.0),
                        _ranked(b, pp_score=1.3, tg_score=1.2)),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _ok_result(a, (100.0, 101.0, 99.0), (20.0, 21.0, 19.0)))
        evaluator.enqueue(b, _ok_result(b, (80.0, 81.0, 79.0), (30.0, 31.0, 29.0)))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            objective="balanced",
            evaluator=evaluator,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert result.recommendation.candidate == b


class TestEdgeCases:
    def test_empty_survivors(self) -> None:
        report = _screening_report(survivors=())
        evaluator = FakeValidationEvaluator()

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport)
        assert result.provisional_leader is None
        assert result.failures == ()
        assert result.attempts_used == 0

    def test_evaluator_success_no_observation(self, candidates: list[CandidateConfig]) -> None:
        a = candidates[0]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, ScreeningResult(
            candidate=a, success=True, observation=None,
        ))

        result = validate_shortlist(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, NoWinnerReport)
        assert len(result.failures) == 1
        assert result.failures[0].disposition == "systemic_error"
        assert "no observation" in result.failures[0].reason

    def test_invalid_objective_raises(self, candidates: list[CandidateConfig]) -> None:
        a = candidates[0]
        report = _screening_report(survivors=(_ranked(a),))
        evaluator = FakeValidationEvaluator()

        with pytest.raises(ValueError, match="Objective must be"):
            validate_shortlist(
                screening_report=report,
                requested_n_ctx=REQUESTED_NCTX,
                objective="invalid",
                evaluator=evaluator,
            )


class TestApplicationSeam:
    def test_validate_candidates_injected_evaluator(
        self, candidates: list[CandidateConfig]
    ) -> None:
        from llama_parameter_scout.search import validate_shortlist as _validate_fn

        a = candidates[0]
        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )
        evaluator = FakeValidationEvaluator()
        evaluator.enqueue(a, _ok_result(a, (150.0, 151.0, 149.0), (25.0, 26.0, 24.0)))

        result = _validate_fn(
            screening_report=report,
            requested_n_ctx=REQUESTED_NCTX,
            evaluator=evaluator,
        )

        assert isinstance(result, ValidatedRecommendation)
        assert result.recommendation.candidate == a
        assert result.attempts_used == 1


class TestApplicationValidateCandidates:
    def _toolchain_stdout(self) -> str:
        return (
            "version: test123\n"
            "backends: cpu\n"
            "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
            "supported_output_formats: json\n"
        )

    def test_production_path_validates_survivor(
        self, tmp_path: Path, candidates: list[CandidateConfig],
    ) -> None:
        from llama_parameter_scout.application import Application
        from llama_parameter_scout.runner import FakeRunner, ProcessResult

        a = candidates[0]
        bp = tmp_path / "llama-bench"
        bp.write_text("#!/bin/sh\necho dummy")
        bp.chmod(0o755)
        mp = tmp_path / "model.gguf"
        from tests.conftest import _build_minimal_gguf
        _build_minimal_gguf(mp, {"general.architecture": (8, "llama")})

        runner = FakeRunner()
        runner.enqueue((str(bp), "--version"),
                       ProcessResult(0, self._toolchain_stdout(), ""))
        runner.enqueue((str(bp), "--help"),
                       ProcessResult(0, "usage\n", ""))

        from collections.abc import Callable, Sequence
        from threading import Event

        class ValidationAdapter:
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
                return ProcessResult(0, '{"samples_ts":[100.0,101.0,99.0]}', "")

        app = Application(runner, process_adapter=ValidationAdapter())

        report = _screening_report(
            survivors=(_ranked(a, pp_score=1.5),),
            leader_idx=0,
        )

        req = SingleRunRequest(
            binary_path=bp,
            model_path=mp,
            journal_path=tmp_path / "journal.json",
            n_ctx=REQUESTED_NCTX,
            candidate=BASE_CANDIDATE,
            binary_source="explicit",
        )

        result = app.validate_candidates(req, report)

        assert isinstance(result, ValidatedRecommendation), (
            f"Expected ValidatedRecommendation, got {type(result).__name__}"
        )
        assert result.recommendation.candidate == a
        assert result.failures == ()
        assert result.attempts_used == 1
        assert "Full-context validation" in result.recommendation.label
