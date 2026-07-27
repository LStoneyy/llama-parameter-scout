from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace as dc_replace
from statistics import median as _median
from typing import Protocol

from .models import (
    CandidateConfig,
    NctxComposition,
    ProvisionalLeader,
    RankedCandidate,
    ScreeningObservation,
    ScreeningReport,
    ScreeningResult,
    SearchProfile,
)

_SortKeyFn = Callable[[RankedCandidate], tuple[float, ...]]

# ── Built-in profiles ────────────────────────────────────────────

SCREENING_N_CTX = NctxComposition(total=512, prompt=128, generation=256, depth=128)

QUICK_PROFILE = SearchProfile(
    label="quick", max_attempts=8, min_repetitions=3,
    max_repetitions_per_candidate=5, tie_bound_coefficient=0.05,
    dispersion_threshold=0.15,
)

STANDARD_PROFILE = SearchProfile(
    label="standard", max_attempts=20, min_repetitions=4,
    max_repetitions_per_candidate=6, tie_bound_coefficient=0.03,
    dispersion_threshold=0.12,
)

THOROUGH_PROFILE = SearchProfile(
    label="thorough", max_attempts=40, min_repetitions=5,
    max_repetitions_per_candidate=8, tie_bound_coefficient=0.02,
    dispersion_threshold=0.10,
)

_SEARCH_PROFILES: dict[str, SearchProfile] = {
    "quick": QUICK_PROFILE,
    "standard": STANDARD_PROFILE,
    "thorough": THOROUGH_PROFILE,
}


def get_profile(label: str) -> SearchProfile:
    profile = _SEARCH_PROFILES.get(label)
    if profile is None:
        raise ValueError(
            f"Unknown profile {label!r}; choose from {', '.join(sorted(_SEARCH_PROFILES))}",
        )
    return profile


# ── Evaluator protocol ───────────────────────────────────────────


class ScreeningEvaluator(Protocol):
    def __call__(
        self,
        candidate: CandidateConfig,
        n_ctx: NctxComposition,
        repetitions: int,
    ) -> ScreeningResult: ...


# ── Grid generation ──────────────────────────────────────────────


def _unique_sorted(values: list[int]) -> list[int]:
    seen: list[int] = []
    for v in sorted(values):
        if not seen or seen[-1] != v:
            seen.append(v)
    return seen


def generate_core_grid(base: CandidateConfig) -> tuple[CandidateConfig, ...]:
    threads_vals = _unique_sorted([max(1, base.threads // 2), base.threads, base.threads * 2])
    batch_vals = _unique_sorted([
        max(1, base.batch_size // 2), base.batch_size, base.batch_size * 2,
    ])
    ubatch_vals = _unique_sorted([
        max(1, base.ubatch_size // 2), base.ubatch_size, base.ubatch_size * 2,
    ])
    grid: list[CandidateConfig] = []
    for t in threads_vals:
        for b in batch_vals:
            for ub in ubatch_vals:
                grid.append(CandidateConfig(
                    threads=t, batch_size=b, ubatch_size=ub,
                    n_gpu_layers=base.n_gpu_layers,
                    flash_attn=base.flash_attn,
                    cache_type_k=base.cache_type_k,
                    cache_type_v=base.cache_type_v,
                ))
    return tuple(grid)


# ── Statistics helpers ───────────────────────────────────────────


def _pop_stddev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


def _pool_observations(
    observations: Sequence[ScreeningObservation],
) -> tuple[list[float], list[float]]:
    pp_all: list[float] = []
    tg_all: list[float] = []
    for obs in observations:
        pp_all.extend(obs.pp_samples)
        tg_all.extend(obs.tg_samples)
    return pp_all, tg_all


def _compute_aggregate(
    pp_samples: Sequence[float],
    tg_samples: Sequence[float],
) -> tuple[float, float, float, float]:
    pp_med = _median(pp_samples)
    tg_med = _median(tg_samples)
    pp_disp = _pop_stddev(pp_samples)
    tg_disp = _pop_stddev(tg_samples)
    return pp_med, tg_med, pp_disp, tg_disp


# ── Scoring ──────────────────────────────────────────────────────


def _balanced_score(pp_score: float, tg_score: float) -> float:
    return math.sqrt(pp_score * tg_score)


# ── Relative dispersion ──────────────────────────────────────────


def _relative_dispersion(median: float, dispersion: float) -> float:
    if median <= 0:
        return float("inf")
    return dispersion / median


# ── Objective-specific helpers ───────────────────────────────────


def _objective_score(candidate: RankedCandidate, objective: str) -> float:
    if objective == "pp":
        return candidate.pp_score
    if objective == "tg":
        return candidate.tg_score
    return candidate.balanced


def _objective_dispersion(candidate: RankedCandidate, objective: str) -> float:
    if objective == "pp":
        return candidate.pp_relative_dispersion
    if objective == "tg":
        return candidate.tg_relative_dispersion
    return (candidate.pp_relative_dispersion + candidate.tg_relative_dispersion) / 2.0


# ── Evaluator contract validation ────────────────────────────────


def _validate_evaluator_result(
    result: ScreeningResult,
    requested_candidate: CandidateConfig,
    requested_repetitions: int,
) -> ScreeningResult | None:
    if result.candidate != requested_candidate:
        return ScreeningResult(
            candidate=requested_candidate, success=False,
            disposition="systemic_error",
            reason=f"Evaluator returned result for wrong candidate "
                   f"({result.candidate} vs requested {requested_candidate})",
        )
    if result.success:
        obs = result.observation
        if obs is None:
            return ScreeningResult(
                candidate=requested_candidate, success=False,
                disposition="systemic_error",
                reason="Evaluator returned success with no observation",
            )
        if obs.candidate != requested_candidate:
            return ScreeningResult(
                candidate=requested_candidate, success=False,
                disposition="systemic_error",
                reason=f"Observation candidate ({obs.candidate}) does not match "
                       f"requested candidate ({requested_candidate})",
            )
        for name, samples, expected_len in [
            ("PP", obs.pp_samples, requested_repetitions),
            ("TG", obs.tg_samples, requested_repetitions),
        ]:
            if len(samples) != expected_len:
                return ScreeningResult(
                    candidate=requested_candidate, success=False,
                    disposition="malformed_output",
                    reason=f"Expected {expected_len} {name} samples, got {len(samples)}",
                )
            for x in samples:
                if not (x > 0 and math.isfinite(x)):
                    return ScreeningResult(
                        candidate=requested_candidate, success=False,
                        disposition="malformed_output",
                        reason=f"Non-positive or non-finite {name} sample: {x}",
                    )
    return None


# ── Dominance pruning (Pareto) ───────────────────────────────────


def _prune_dominated(
    scored: list[RankedCandidate],
) -> tuple[list[RankedCandidate], set[int]]:
    kept_indices: set[int] = set()
    pruned_indices: set[int] = set()

    for i, a in enumerate(scored):
        dominated = False
        for j, b in enumerate(scored):
            if i == j:
                continue
            if b.pp_score >= a.pp_score and b.tg_score >= a.tg_score:
                if b.pp_score > a.pp_score or b.tg_score > a.tg_score:
                    dominated = True
                    break
        if dominated:
            pruned_indices.add(i)
        else:
            kept_indices.add(i)

    kept = [scored[i] for i in sorted(kept_indices)]
    return kept, pruned_indices


# ── Adaptive target selection (dispersion + overlap) ─────────────


def _find_adaptive_target(
    survivors: list[tuple[CandidateConfig, list[ScreeningObservation]]],
    profile: SearchProfile,
    objective: str,
    survivors_ranked: list[RankedCandidate] | None = None,
    blocked: set[CandidateConfig] | None = None,
) -> CandidateConfig | None:
    blocked_set = blocked or set()
    for candidate, obs_list in survivors:
        if candidate in blocked_set:
            continue
        total_reps = sum(len(o.pp_samples) for o in obs_list)
        if total_reps >= profile.max_repetitions_per_candidate:
            continue
        pp_all, tg_all = _pool_observations(obs_list)
        pp_med, tg_med, pp_disp, tg_disp = _compute_aggregate(pp_all, tg_all)
        rd = max(
            _relative_dispersion(pp_med, pp_disp),
            _relative_dispersion(tg_med, tg_disp),
        )
        if rd > profile.dispersion_threshold:
            return candidate

    if survivors_ranked and len(survivors_ranked) >= 2:
        tie_bound = _compute_tie_bound(survivors_ranked, profile, objective)
        if tie_bound > 0:
            leader_score = _objective_score(survivors_ranked[0], objective)
            for idx in range(1, len(survivors_ranked)):
                other_score = _objective_score(survivors_ranked[idx], objective)
                max_s = max(leader_score, other_score)
                if max_s <= 0:
                    continue
                if abs(leader_score - other_score) / max_s <= tie_bound:
                    for pos in (0, idx):
                        c = survivors_ranked[pos].candidate
                        if c in blocked_set:
                            continue
                        obs_a = _find_observations(survivors, c)
                        if obs_a is not None and _reps_remaining(obs_a, profile):
                            return c
    return None


def _reps_remaining(
    obs_list: list[ScreeningObservation], profile: SearchProfile,
) -> bool:
    total_reps = sum(len(o.pp_samples) for o in obs_list)
    return total_reps < profile.max_repetitions_per_candidate


def _find_observations(
    survivors: list[tuple[CandidateConfig, list[ScreeningObservation]]],
    candidate: CandidateConfig,
) -> list[ScreeningObservation] | None:
    for c, obs_list in survivors:
        if c == candidate:
            return obs_list
    return None


# ── Ranking and tie-breaking ─────────────────────────────────────


def _compute_tie_bound(
    scored_survivors: Sequence[RankedCandidate],
    profile: SearchProfile,
    objective: str,
) -> float:
    if len(scored_survivors) < 2:
        return 0.0
    avg_rd = sum(
        _objective_dispersion(c, objective) for c in scored_survivors
    ) / len(scored_survivors)
    return avg_rd * profile.tie_bound_coefficient


def _memory_key(candidate: RankedCandidate) -> tuple[int, ...]:
    if candidate.memory_bytes is None:
        return (1,)
    return (0, candidate.memory_bytes)


def _simplicity_key(candidate: RankedCandidate) -> tuple[int, int, int]:
    return (
        candidate.candidate.threads,
        candidate.candidate.batch_size,
        candidate.candidate.ubatch_size,
    )


def _stability_key(candidate: RankedCandidate) -> float:
    return candidate.pp_relative_dispersion + candidate.tg_relative_dispersion


def _is_tied(
    a: RankedCandidate, b: RankedCandidate, tie_bound: float, objective: str,
) -> bool:
    score_a = _objective_score(a, objective)
    score_b = _objective_score(b, objective)
    max_score = max(score_a, score_b)
    if max_score <= 0:
        return True
    return abs(score_a - score_b) / max_score <= tie_bound


def _rank_and_tiebreak(
    scored: list[RankedCandidate],
    profile: SearchProfile,
    objective: str,
) -> list[RankedCandidate]:
    if len(scored) <= 1:
        return scored

    tie_bound = _compute_tie_bound(scored, profile, objective)
    if tie_bound <= 0:
        return scored

    i = 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and _is_tied(scored[j], scored[j + 1], tie_bound, objective):
            j += 1
        if j > i:
            tie_group = scored[i:j + 1]
            tie_group.sort(key=lambda c: (
                _stability_key(c),
                _memory_key(c),
                _simplicity_key(c),
            ))
            for idx, rc in enumerate(tie_group):
                parts: list[str] = []
                if idx > 0 and _stability_key(tie_group[idx - 1]) < _stability_key(rc):
                    parts.append("stability")
                if idx > 0 and _memory_key(tie_group[idx - 1]) < _memory_key(rc):
                    parts.append("memory")
                if idx > 0 and _simplicity_key(tie_group[idx - 1]) < _simplicity_key(rc):
                    parts.append("simpler")
                if parts:
                    tie_group[idx] = dc_replace(rc, tie_break="+".join(parts))
            scored[i:j + 1] = tie_group
        i = j + 1

    return scored


def _sort_key(objective: str) -> _SortKeyFn:
    def key(c: RankedCandidate) -> tuple[float, ...]:
        if objective == "pp":
            return (-c.pp_score, _stability_key(c))
        elif objective == "tg":
            return (-c.tg_score, _stability_key(c))
        else:
            return (-c.balanced, _stability_key(c))
    return key


# ── Main screen function ─────────────────────────────────────────


def _build_result(
    candidate: CandidateConfig,
    observations: list[ScreeningObservation],
    base_pp: float,
    base_tg: float,
) -> RankedCandidate:
    pp_all, tg_all = _pool_observations(observations)
    pp_med, tg_med, pp_disp, tg_disp = _compute_aggregate(pp_all, tg_all)

    pp_score = pp_med / base_pp if base_pp > 0 else 0.0
    tg_score = tg_med / base_tg if base_tg > 0 else 0.0

    pp_rd = _relative_dispersion(pp_med, pp_disp)
    tg_rd = _relative_dispersion(tg_med, tg_disp)

    total_reps = sum(len(o.pp_samples) for o in observations)
    memory = observations[-1].memory_bytes if observations else None

    return RankedCandidate(
        candidate=candidate,
        pp_score=pp_score,
        tg_score=tg_score,
        balanced=_balanced_score(pp_score, tg_score),
        pp_median=pp_med,
        tg_median=tg_med,
        pp_relative_dispersion=pp_rd,
        tg_relative_dispersion=tg_rd,
        memory_bytes=memory,
        total_repetitions=total_reps,
    )


_ACTIONABLE_GUIDANCE: dict[str, str] = {
    "pruned": "Adjust candidate parameter bounds and retry.",
    "dominance_pruned": "Remove dominated configurations from the candidate list.",
    "budget_exhausted": "Increase resource budget or reduce context/model size.",
    "oom": "Reduce context size, model size, or increase available memory.",
    "timeout": "Increase candidate timeout or reduce workload.",
    "crash": "Check binary compatibility, model integrity, and system resources.",
    "malformed_output": "Verify llama-bench version and output format compatibility.",
    "systemic_error": "Investigate system state; possible driver or hardware issue.",
    "unsupported": "Check binary, model path, and configuration validity.",
    "cancellation": "The run was cancelled by the user or scheduler.",
    "not_attempted": "Increase profile budget or reduce candidate count.",
}


def _build_disposition_summary(
    baseline_disposition: ScreeningResult,
    grid_dispositions: tuple[ScreeningResult, ...],
    adaptive_dispositions: tuple[ScreeningResult, ...],
    survivors: tuple[RankedCandidate, ...],
) -> str:
    grid_total = len(grid_dispositions)
    survived = len(survivors)
    failed_types: dict[str, int] = {}
    action_suggestions: set[str] = set()
    for d in list(grid_dispositions) + list(adaptive_dispositions):
        if not d.success:
            failed_types[d.disposition] = failed_types.get(d.disposition, 0) + 1
            guidance = _ACTIONABLE_GUIDANCE.get(d.disposition)
            if guidance:
                action_suggestions.add(guidance)

    parts: list[str] = [
        f"Baseline: {baseline_disposition.disposition}",
        f"Grid: {grid_total} total, {survived} survived",
    ]
    if adaptive_dispositions:
        parts.append(f"Adaptive: {len(adaptive_dispositions)} extra attempts")
    if failed_types:
        f_str = ", ".join(f"{k}={v}" for k, v in sorted(failed_types.items()))
        parts.append(f"Failures: {f_str}")
    if not survived:
        guidance = "; ".join(sorted(action_suggestions)) if action_suggestions else (
            "No candidates survived screening. Review candidate bounds and runtime configuration."
        )
        parts.append(f"Guidance: {guidance}")
    return "; ".join(parts)


def screen(
    *,
    base_candidate: CandidateConfig,
    profile: SearchProfile,
    evaluator: ScreeningEvaluator,
    screening_n_ctx: NctxComposition,
    objective: str = "balanced",
) -> ScreeningReport:
    if objective not in ("pp", "tg", "balanced"):
        raise ValueError(f"Objective must be 'pp', 'tg', or 'balanced', got {objective!r}")

    remaining = profile.max_attempts

    baseline_result = evaluator(base_candidate, screening_n_ctx, profile.min_repetitions)
    remaining -= 1

    baseline_error = _validate_evaluator_result(
        baseline_result, base_candidate, profile.min_repetitions,
    )
    if baseline_error is not None:
        return ScreeningReport(
            profile=profile, baseline_disposition=baseline_error,
            grid_dispositions=(), survivors=(), leader=None,
            attempts_used=1, objective=objective,
            disposition_summary=f"Baseline failed: {baseline_error.reason}",
        )

    if not baseline_result.success or baseline_result.observation is None:
        return ScreeningReport(
            profile=profile, baseline_disposition=baseline_result,
            grid_dispositions=(), survivors=(), leader=None,
            attempts_used=1, objective=objective,
            disposition_summary=f"Baseline failed: {baseline_result.disposition}",
        )

    base_obs = baseline_result.observation
    base_pp_all, base_tg_all = _pool_observations([base_obs])
    base_pp_med, base_tg_med, _b_pp_d, _b_tg_d = _compute_aggregate(base_pp_all, base_tg_all)

    if base_pp_med <= 0 or base_tg_med <= 0:
        return ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=base_candidate, success=False,
                disposition="systemic_error",
                reason="Baseline produced non-positive PP or TG median",
            ),
            grid_dispositions=(), survivors=(), leader=None,
            attempts_used=1, objective=objective,
            disposition_summary="Baseline produced non-positive PP or TG",
        )

    grid = generate_core_grid(base_candidate)

    survivor_observations: list[tuple[CandidateConfig, list[ScreeningObservation]]] = []
    grid_dispositions: list[ScreeningResult] = []
    adaptive_dispositions: list[ScreeningResult] = []

    for candidate in grid:
        if candidate == base_candidate:
            grid_dispositions.append(ScreeningResult(
                candidate=candidate, success=True,
                disposition="matches_baseline",
                reason="Baseline candidate; not re-evaluated",
            ))
            continue

        if remaining <= 0:
            grid_dispositions.append(ScreeningResult(
                candidate=candidate, success=False,
                disposition="not_attempted",
                reason=f"Profile budget ({profile.max_attempts} attempts) exhausted",
            ))
            continue

        result = evaluator(candidate, screening_n_ctx, profile.min_repetitions)
        remaining -= 1

        validation_error = _validate_evaluator_result(
            result, candidate, profile.min_repetitions,
        )
        if validation_error is not None:
            grid_dispositions.append(validation_error)
            continue

        if result.success and result.observation is not None:
            survivor_observations.append((candidate, [result.observation]))
            grid_dispositions.append(ScreeningResult(
                candidate=candidate, success=True,
                disposition="survived", observation=result.observation,
                reason="Completed screening run",
            ))
        else:
            grid_dispositions.append(result)

    # Adaptive phase: dispersion-driven + overlap-driven
    adapt_blocked: set[CandidateConfig] = set()
    while remaining > 0:
        scored_temp: list[RankedCandidate] = []
        for candidate, obs_list in survivor_observations:
            rc = _build_result(candidate, obs_list, base_pp_med, base_tg_med)
            scored_temp.append(rc)
        scored_temp.sort(key=_sort_key(objective))

        target = _find_adaptive_target(
            survivor_observations, profile, objective, scored_temp, adapt_blocked,
        )
        if target is None:
            break
        extra = evaluator(target, screening_n_ctx, 1)
        remaining -= 1

        extra_error = _validate_evaluator_result(extra, target, 1)
        if extra_error is not None:
            adaptive_dispositions.append(extra_error)
            adapt_blocked.add(target)
            continue

        if extra.success and extra.observation is not None:
            adaptive_dispositions.append(extra)
            for idx, (c, obs_list) in enumerate(survivor_observations):
                if c == target:
                    obs_list.append(extra.observation)
                    break
        else:
            adaptive_dispositions.append(extra)
            adapt_blocked.add(target)

    # Score all survivors
    scored: list[RankedCandidate] = []
    for candidate, obs_list in survivor_observations:
        rc = _build_result(candidate, obs_list, base_pp_med, base_tg_med)
        scored.append(rc)

    # Pareto dominance pruning - preserve observation for pruned candidates
    kept_scored, pruned_indices = _prune_dominated(scored)
    for idx in pruned_indices:
        rc = scored[idx]
        for gidx, d in enumerate(grid_dispositions):
            if d.candidate == rc.candidate and d.success and d.observation is not None:
                grid_dispositions[gidx] = ScreeningResult(
                    candidate=rc.candidate, success=False,
                    disposition="dominance_pruned", observation=d.observation,
                    reason=f"Dominated by another survivor in both PP and TG; "
                           f"PP={rc.pp_score:.4f} TG={rc.tg_score:.4f}",
                )
                break

    kept_scored.sort(key=_sort_key(objective))
    ranked = _rank_and_tiebreak(kept_scored, profile, objective)

    leader: ProvisionalLeader | None = None
    if ranked:
        best = ranked[0]
        leader = ProvisionalLeader(
            candidate=best.candidate, objective=objective,
            pp_score=best.pp_score, tg_score=best.tg_score,
            balanced=best.balanced, pp_median=best.pp_median,
            tg_median=best.tg_median,
        )

    summary = _build_disposition_summary(
        baseline_result, tuple(grid_dispositions),
        tuple(adaptive_dispositions), tuple(ranked),
    )

    return ScreeningReport(
        profile=profile,
        baseline_disposition=baseline_result,
        grid_dispositions=tuple(grid_dispositions),
        adaptive_dispositions=tuple(adaptive_dispositions),
        survivors=tuple(ranked),
        leader=leader,
        attempts_used=profile.max_attempts - remaining,
        objective=objective,
        disposition_summary=summary,
    )
