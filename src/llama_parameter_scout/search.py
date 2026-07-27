from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace as dc_replace
from statistics import median as _median
from typing import Protocol

from .models import (
    CandidateConfig,
    FailedValidation,
    ModelFingerprint,
    NctxComposition,
    NoWinnerReport,
    ParetoEntry,
    ProvisionalLeader,
    RankedCandidate,
    ResourceBudget,
    ScreeningObservation,
    ScreeningReport,
    ScreeningResult,
    SearchProfile,
    ToolchainFingerprint,
    ValidatedCandidate,
    ValidatedRecommendation,
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


class ValidationEvaluator(Protocol):
    def __call__(
        self,
        candidate: CandidateConfig,
        n_ctx: NctxComposition,
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


# ── Expanded screening grid ──────────────────────────────────────


def _unique_sorted_vals(values: list[int]) -> list[int]:
    seen: list[int] = []
    for v in sorted(values):
        if not seen or seen[-1] != v:
            seen.append(v)
    return seen


def _n_gpu_layers_for_grid(base: CandidateConfig, cpu_only: bool) -> tuple[int, ...]:
    if cpu_only:
        return (0,)
    vals = _unique_sorted_vals([0, base.n_gpu_layers])
    return tuple(vals)


def generate_screening_grid(
    base: CandidateConfig,
    toolchain: ToolchainFingerprint,
    model: ModelFingerprint,
    cpu_only: bool,
) -> tuple[CandidateConfig, ...]:
    ngl_values = _n_gpu_layers_for_grid(base, cpu_only)

    fa_values: tuple[bool, ...]
    if "-fa" in toolchain.supported_flags:
        fa_values = (False, True)
    else:
        fa_values = (False,)

    cache_types = toolchain.supported_cache_types or ("f16",)

    if model.is_moe:
        moe_values: tuple[int | None, ...] = (None, 0, 1)
    else:
        moe_values = (None,)

    grid: list[CandidateConfig] = []
    seen: set[CandidateConfig] = set()

    core = generate_core_grid(base)
    for c in core:
        if c not in seen:
            seen.add(c)
            grid.append(c)

    for ngl in ngl_values:
        for fa in fa_values:
            for ck in cache_types:
                for cv in cache_types:
                    for moe in moe_values:
                        c = CandidateConfig(
                            threads=base.threads,
                            batch_size=base.batch_size,
                            ubatch_size=base.ubatch_size,
                            n_gpu_layers=ngl,
                            flash_attn=fa,
                            cache_type_k=ck,
                            cache_type_v=cv,
                            n_cpu_moe=moe,
                        )
                        if c not in seen:
                            seen.add(c)
                            grid.append(c)

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


def _memory_key(candidate: RankedCandidate) -> tuple[int | float, ...]:
    me = candidate.memory_evidence
    if me is not None and me.ram_bytes is not None:
        if me.vram_bytes is not None:
            return (0, me.ram_bytes, me.vram_bytes)
        return (0, me.ram_bytes)
    if candidate.memory_bytes is not None:
        return (0, candidate.memory_bytes)
    return (1,)


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
        elif objective == "memory":
            return (-c.balanced, _stability_key(c))
        else:
            return (-c.balanced, _stability_key(c))
    return key


def _validation_sort_key(objective: str) -> Callable[[ValidatedCandidate], tuple[float, ...]]:
    def key(vc: ValidatedCandidate) -> tuple[float, ...]:
        stability = vc.pp_relative_dispersion + vc.tg_relative_dispersion
        if objective == "pp":
            return (-vc.pp_score, stability)
        elif objective == "tg":
            return (-vc.tg_score, stability)
        elif objective == "memory":
            return (-vc.balanced, stability)
        else:
            return (-vc.balanced, stability)
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
    mem_evidence = observations[-1].memory if observations else None

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
        memory_evidence=mem_evidence,
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


def _memory_leader(
    ranked: list[RankedCandidate],
    resource_budget: ResourceBudget | None,
    memory_unified: bool,
    dispersion_threshold: float = 1.0,
) -> int | None:
    """Return the index of the fastest stable candidate with complete
    comparable memory evidence fitting within the budget, or None.

    Stable means both PP and TG relative dispersion <= dispersion_threshold.
    """
    if resource_budget is None:
        return None

    headroom = resource_budget.headroom_ratio

    eligible: list[int] = []
    for i, rc in enumerate(ranked):
        # Stability check
        if (rc.pp_relative_dispersion > dispersion_threshold
                or rc.tg_relative_dispersion > dispersion_threshold):
            continue
        me = rc.memory_evidence
        if me is None or me.quality != "complete":
            continue
        if me.ram_bytes is None:
            continue
        if memory_unified or resource_budget.vram_bytes is None:
            cap = int(resource_budget.ram_bytes * (1 - headroom))
            if me.ram_bytes <= cap:
                eligible.append(i)
        else:
            ram_cap = int(resource_budget.ram_bytes * (1 - headroom))
            vram_cap = int(resource_budget.vram_bytes * (1 - headroom))
            if me.ram_bytes is not None and me.vram_bytes is not None:
                if me.ram_bytes <= ram_cap and me.vram_bytes <= vram_cap:
                    eligible.append(i)

    if not eligible:
        return None
    # Among eligible, pick fastest (highest balanced)
    best_idx = eligible[0]
    for idx in eligible[1:]:
        if ranked[idx].balanced > ranked[best_idx].balanced:
            best_idx = idx
    return best_idx


def _compute_pareto_frontier(
    scored: list[RankedCandidate],
    memory_unified: bool,
) -> list[ParetoEntry]:
    """Compute speed-vs-memory Pareto frontier.
    Includes all candidates with complete comparable metrics.
    A candidate is dominated when another has >= balanced speed and
    <= every constrained memory resource, with at least one strict improvement."""
    entries: list[ParetoEntry] = []
    for rc in scored:
        me = rc.memory_evidence
        if me is None or me.quality != "complete":
            continue
        entries.append(ParetoEntry(
            candidate=rc.candidate,
            balanced=rc.balanced,
            memory=me,
        ))

    frontier: list[ParetoEntry] = []
    for i, a in enumerate(entries):
        dominated = False
        for j, b in enumerate(entries):
            if i == j:
                continue
            speed_ge = b.balanced >= a.balanced
            if memory_unified or (a.memory.vram_bytes is None and b.memory.vram_bytes is None):
                mem_le = (
                    (b.memory.ram_bytes is not None and a.memory.ram_bytes is not None)
                    and b.memory.ram_bytes <= a.memory.ram_bytes
                )
            else:
                mem_le = (
                    b.memory.ram_bytes is not None and a.memory.ram_bytes is not None
                    and b.memory.vram_bytes is not None and a.memory.vram_bytes is not None
                    and b.memory.ram_bytes <= a.memory.ram_bytes
                    and b.memory.vram_bytes <= a.memory.vram_bytes
                )
            if speed_ge and mem_le:
                strict = (b.balanced > a.balanced)
                if not strict and memory_unified:
                    strict = (b.memory.ram_bytes is not None and a.memory.ram_bytes is not None
                              and b.memory.ram_bytes < a.memory.ram_bytes)
                elif not strict:
                    strict = (
                        (b.memory.ram_bytes is not None and a.memory.ram_bytes is not None
                         and b.memory.ram_bytes < a.memory.ram_bytes)
                        or (b.memory.vram_bytes is not None and a.memory.vram_bytes is not None
                            and b.memory.vram_bytes < a.memory.vram_bytes)
                    )
                if strict:
                    dominated = True
                    break
        if not dominated:
            frontier.append(a)

    return frontier


def screen(
    *,
    base_candidate: CandidateConfig,
    profile: SearchProfile,
    evaluator: ScreeningEvaluator,
    screening_n_ctx: NctxComposition,
    objective: str = "balanced",
    grid: tuple[CandidateConfig, ...] | None = None,
    resource_budget: ResourceBudget | None = None,
    memory_unified: bool = False,
) -> ScreeningReport:
    if objective not in ("pp", "tg", "balanced", "memory"):
        raise ValueError(
            f"Objective must be 'pp', 'tg', 'balanced', or 'memory', "
            f"got {objective!r}",
        )

    if objective == "memory" and resource_budget is None:
        return ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=base_candidate, success=False,
                disposition="not_attempted",
                reason="Memory objective requires a ResourceBudget but none was provided",
            ),
            grid_dispositions=(), survivors=(), leader=None,
            attempts_used=0, objective=objective,
            disposition_summary=(
                "Memory objective requires a ResourceBudget but none was provided"
            ),
        )

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

    grid = grid if grid is not None else generate_core_grid(base_candidate)

    survivor_observations: list[tuple[CandidateConfig, list[ScreeningObservation]]] = []
    grid_dispositions: list[ScreeningResult] = []
    adaptive_dispositions: list[ScreeningResult] = []

    # For memory objective, seed survivors with the baseline observation
    if (objective == "memory" and baseline_result.success
            and baseline_result.observation is not None):
        survivor_observations.append((base_candidate, [baseline_result.observation]))

    for candidate in grid:
        if candidate == base_candidate:
            grid_dispositions.append(ScreeningResult(
                candidate=candidate, success=True,
                disposition="matches_baseline",
                reason="Baseline candidate; not re-evaluated",
                provenance=baseline_result.provenance,
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
                provenance=result.provenance,
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

    # Pareto dominance pruning — skip for memory objective (retain all scored)
    if objective == "memory":
        kept_scored = list(scored)
        pruned_indices: set[int] = set()
    else:
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
                        provenance=d.provenance,
                    )
                    break

    kept_scored.sort(key=_sort_key(objective))
    ranked = _rank_and_tiebreak(kept_scored, profile, objective)

    # Pareto frontier (speed vs memory) — includes ALL scored candidates
    # with complete metrics, not just PP/TG dominance survivors
    frontier = _compute_pareto_frontier(scored, memory_unified)

    leader: ProvisionalLeader | None = None
    if objective == "memory":
        mem_idx = _memory_leader(
            ranked, resource_budget, memory_unified,
            dispersion_threshold=profile.dispersion_threshold,
        )
        if mem_idx is not None:
            best = ranked[mem_idx]
            leader = ProvisionalLeader(
                candidate=best.candidate, objective=objective,
                pp_score=best.pp_score, tg_score=best.tg_score,
                balanced=best.balanced, pp_median=best.pp_median,
                tg_median=best.tg_median,
            )
    elif ranked:
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
        pareto_frontier=tuple(frontier),
    )


# ── Validation ───────────────────────────────────────────────────


def _baseline_medians(
    report: ScreeningReport,
) -> tuple[float, float]:
    obs = report.baseline_disposition.observation
    if obs is None or not obs.pp_samples:
        return 0.0, 0.0
    return _median(obs.pp_samples), _median(obs.tg_samples)


def validate_shortlist(
    *,
    screening_report: ScreeningReport,
    requested_n_ctx: NctxComposition,
    objective: str = "balanced",
    budget: int = 10,
    evaluator: ValidationEvaluator,
    resource_budget: ResourceBudget | None = None,
    memory_unified: bool = False,
) -> ValidatedRecommendation | NoWinnerReport:
    if objective not in ("pp", "tg", "balanced", "memory"):
        raise ValueError(
            f"Objective must be 'pp', 'tg', 'balanced', or 'memory', "
            f"got {objective!r}",
        )

    if objective == "memory" and resource_budget is None:
        return NoWinnerReport(
            provisional_leader=screening_report.leader,
            failures=(
                FailedValidation(
                    candidate=screening_report.baseline_disposition.candidate,
                    disposition="systemic_error",
                    reason="Memory objective requires a ResourceBudget but none was provided",
                ),
            ),
            attempts_used=0,
            budget=budget,
        )

    base_pp, base_tg = _baseline_medians(screening_report)

    successes: list[ValidatedCandidate] = []
    failures: list[FailedValidation] = []
    attempts = 0
    # Collect validated observations for frontier computation
    validated_observations: list[ScreeningObservation] = []

    for survivor in screening_report.survivors:
        if attempts >= budget:
            failures.append(FailedValidation(
                candidate=survivor.candidate,
                disposition="validation_budget_exhausted",
                reason=f"Validation budget of {budget} attempts exhausted",
            ))
            break

        if len(successes) >= 3:
            break

        attempts += 1
        result = evaluator(survivor.candidate, requested_n_ctx)

        if not result.success:
            failures.append(FailedValidation(
                candidate=survivor.candidate,
                disposition=result.disposition,
                reason=result.reason or f"Validation failed: {result.disposition}",
            ))
            continue

        obs = result.observation
        if obs is None:
            failures.append(FailedValidation(
                candidate=survivor.candidate,
                disposition="systemic_error",
                reason="Evaluator returned success with no observation",
            ))
            continue

        mismatch_reason = _context_mismatch(obs, requested_n_ctx)
        if mismatch_reason is not None:
            failures.append(FailedValidation(
                candidate=survivor.candidate,
                disposition="context_mismatch",
                reason=mismatch_reason,
            ))
            continue

        # Collect for frontier computation (complete metrics only filtered later)
        validated_observations.append(obs)

        # Memory objective: stability check (PP and TG both within threshold)
        if objective == "memory":
            pp_rd = _relative_dispersion(
                _median(obs.pp_samples),
                _pop_stddev(obs.pp_samples),
            )
            tg_rd = _relative_dispersion(
                _median(obs.tg_samples),
                _pop_stddev(obs.tg_samples),
            )
            threshold = screening_report.profile.dispersion_threshold
            if pp_rd > threshold or tg_rd > threshold:
                failures.append(FailedValidation(
                    candidate=survivor.candidate,
                    disposition="systemic_error",
                    reason=f"Unstable: PP dispersion {pp_rd:.4f} > {threshold} "
                           f"or TG dispersion {tg_rd:.4f} > {threshold}",
                ))
                continue

        # Memory objective: check complete in-budget evidence
        if objective == "memory" and resource_budget is not None:
            me = obs.memory
            if me is None or me.quality != "complete":
                failures.append(FailedValidation(
                    candidate=survivor.candidate,
                    disposition="systemic_error",
                    reason="Incomplete memory metrics: cannot validate memory fit",
                ))
                continue
            headroom = resource_budget.headroom_ratio
            if memory_unified or resource_budget.vram_bytes is None:
                cap = int(resource_budget.ram_bytes * (1 - headroom))
                if me.ram_bytes is None or me.ram_bytes > cap:
                    failures.append(FailedValidation(
                        candidate=survivor.candidate,
                        disposition="systemic_error",
                        reason=f"Memory exceeds budget: RAM {me.ram_bytes} > {cap}",
                    ))
                    continue
            else:
                ram_cap = int(resource_budget.ram_bytes * (1 - headroom))
                vram_cap = int(resource_budget.vram_bytes * (1 - headroom))
                if me.ram_bytes is None or me.vram_bytes is None:
                    failures.append(FailedValidation(
                        candidate=survivor.candidate,
                        disposition="systemic_error",
                        reason="Incomplete discrete memory metrics",
                    ))
                    continue
                over_parts: list[str] = []
                if me.ram_bytes > ram_cap:
                    over_parts.append(f"RAM {me.ram_bytes} > {ram_cap}")
                if me.vram_bytes > vram_cap:
                    over_parts.append(f"VRAM {me.vram_bytes} > {vram_cap}")
                if over_parts:
                    failures.append(FailedValidation(
                        candidate=survivor.candidate,
                        disposition="systemic_error",
                        reason="Memory exceeds budget: " + "; ".join(over_parts),
                    ))
                    continue

        ranked = _build_result(survivor.candidate, [obs], base_pp, base_tg)
        successes.append(ValidatedCandidate(
            candidate=ranked.candidate,
            pp_score=ranked.pp_score,
            tg_score=ranked.tg_score,
            balanced=ranked.balanced,
            pp_median=ranked.pp_median,
            tg_median=ranked.tg_median,
            pp_relative_dispersion=ranked.pp_relative_dispersion,
            tg_relative_dispersion=ranked.tg_relative_dispersion,
            memory_evidence=obs.memory,
        ))
        successes.sort(key=_validation_sort_key(objective))

    # Build validation-context Pareto frontier from all validated observations
    # with complete comparable metrics (including those exceeding budget)
    v_frontier_raw: list[RankedCandidate] = []
    for sobs in validated_observations:
        me = sobs.memory
        if me is not None and me.quality == "complete":
            rc = _build_result(sobs.candidate, [sobs], base_pp, base_tg)
            v_frontier_raw.append(rc)

    v_frontier = _compute_pareto_frontier(v_frontier_raw, memory_unified)

    if successes:
        return ValidatedRecommendation(
            recommendation=successes[0],
            alternatives=tuple(successes[1:]),
            failures=tuple(failures),
            attempts_used=attempts,
            budget=budget,
            pareto_frontier=tuple(v_frontier),
        )

    return NoWinnerReport(
        provisional_leader=screening_report.leader,
        failures=tuple(failures),
        attempts_used=attempts,
        budget=budget,
        pareto_frontier=tuple(v_frontier),
    )


def _context_mismatch(
    obs: ScreeningObservation,
    requested: NctxComposition,
) -> str | None:
    mismatches: list[str] = []

    fields = [
        ("total", obs.observed_total, requested.total),
        ("prompt", obs.observed_prompt, requested.prompt),
        ("generation", obs.observed_generation, requested.generation),
        ("depth", obs.observed_depth, requested.depth),
    ]
    for name, observed, expected in fields:
        if observed is None:
            mismatches.append(f"{name}=missing")
        elif observed != expected:
            mismatches.append(f"{name}={observed} (expected {expected})")

    if obs.observed_prompt is not None and obs.observed_generation is not None \
            and obs.observed_depth is not None and obs.observed_total is not None:
        comp_sum = obs.observed_prompt + obs.observed_generation + obs.observed_depth
        if comp_sum != obs.observed_total:
            mismatches.append(
                f"prompt+generation+depth={comp_sum} != total={obs.observed_total}"
            )

    if mismatches:
        return "Effective context does not match requested: " + "; ".join(mismatches)
    return None
