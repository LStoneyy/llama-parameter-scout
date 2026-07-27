from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from uuid import uuid4 as _uuid4

from .journal import (
    list_sessions as _journal_list_sessions,
)
from .journal import (
    read_journal_events,
    write_measurement_completed,
    write_mtp_evidence,
    write_observation,
    write_outcome,
    write_search_decision,
    write_session_start,
)
from .model_inspector import inspect_model
from .models import (
    BenchmarkObservation,
    BinaryNotFoundError,
    CandidateConfig,
    CandidateDisposition,
    CandidateError,
    CandidateOutcome,
    ContextAdvisory,
    ContextExceededWithoutAcknowledgementError,
    ExecutionPolicy,
    FailedValidation,
    InconsistentNglError,
    InvalidHeadroomRatioError,
    InvalidNctxError,
    JournalError,
    LaunchProvenance,
    MalformedOutputError,
    MemoryEvidence,
    ModelFingerprint,
    ModelFingerprintMismatchError,
    ModelNotFoundError,
    MtpComparison,
    MultipleAcceleratorsSelectedError,
    NctxComposition,
    NoAcceleratorSelectedError,
    NonpositiveBudgetError,
    NoWinnerReport,
    ResourceBudget,
    ScoutError,
    ScreeningDisposition,
    ScreeningObservation,
    ScreeningReport,
    ScreeningResult,
    SearchProfile,
    SessionResumeResult,
    SessionSetupError,
    SessionSpecification,
    ShardFingerprint,
    SingleRunRequest,
    SubprocessError,
    SystemTopology,
    ToolchainFingerprint,
    TrainingContextStatus,
    UnknownDeviceError,
    ValidatedRecommendation,
)
from .mtp import ServerLifecycle
from .runner import (
    ProcessAdapter,
    ProcessResult,
    ProcessRunner,
    ProcessStartError,
    ProductionProcessAdapter,
)
from .search import (
    SCREENING_N_CTX,
    ScreeningEvaluator,
    ValidationEvaluator,
    generate_screening_grid,
)
from .search import screen as _search_screen
from .search import validate_shortlist as _validate_shortlist
from .toolchain import inspect_toolchain
from .topology import DetectTopologyFn, detect_topology

# ── Defaults ─────────────────────────────────────────────────────

DEFAULT_HEADROOM_RATIO = 0.1
_ESTIMATED_BYTES_PER_TOKEN_PER_PARAM = 0.000075

# ── OOM heuristics ───────────────────────────────────────────────

_OOM_EXIT_CODES = frozenset({-9, 137})
_OOM_STDERR_TOKENS = frozenset({
    "killed",
    "out of memory",
    "oom-killer",
    "cuda out of memory",
    "cuda error: out of memory",
    "allocator ran out of memory",
    "cannot allocate memory",
    "bad_alloc",
})


def _is_oom(returncode: int, stderr_text: str) -> bool:
    if returncode in _OOM_EXIT_CODES:
        return True
    stderr_lower = stderr_text.lower()
    for token in _OOM_STDERR_TOKENS:
        if token in stderr_lower:
            return True
    return False


# ── Safe next actions ────────────────────────────────────────────

_NEXT_ACTION: dict[CandidateDisposition, str] = {
    "unsupported": "Check binary, model path, and configuration validity.",
    "pruned": "Adjust candidate parameter bounds and retry.",
    "not_attempted": "Ensure session is properly set up before launching.",
    "budget_exhausted": "Increase resource budget or reduce context/model size.",
    "oom": "Reduce context size, model size, or increase available memory.",
    "timeout": "Increase candidate timeout or reduce workload.",
    "crash": "Check binary compatibility, model integrity, and system resources.",
    "malformed_output": "Verify llama-bench version and output format compatibility.",
    "cancellation": "The run was cancelled by the user or scheduler.",
    "systemic_error": "Investigate system state; possible driver or hardware issue.",
    "completed": "",
}


def _env_for_device(spec: SessionSpecification) -> dict[str, str] | None:
    if spec.cpu_only or not spec.selected_device_ids:
        return None
    device_id = spec.selected_device_ids[0]
    device = next(
        (a for a in spec.topology.accelerators if a.device_id == device_id),
        None,
    )
    if device is None:
        return None
    if device.backend == "cuda":
        return {"CUDA_VISIBLE_DEVICES": str(device_id)}
    if device.backend == "hip":
        return {"HIP_VISIBLE_DEVICES": str(device_id)}
    if device.backend == "metal":
        return None
    return None


def _cache_element_size(cache_type: str) -> int:
    mapping = {"f16": 2, "f32": 4, "q8_0": 1, "q4_0": 1}
    return mapping.get(cache_type, 2)


def _build_argv(
    request: SingleRunRequest,
    *,
    repetitions: int,
    session_spec: SessionSpecification | None = None,
) -> list[str]:
    argv = [
        str(request.binary_path),
        "-m", str(request.model_path),
        "-p", str(request.n_ctx.prompt),
        "-n", str(request.n_ctx.generation),
        "-d", str(request.n_ctx.depth),
        "-t", str(request.candidate.threads),
        "-b", str(request.candidate.batch_size),
        "-ub", str(request.candidate.ubatch_size),
        "-ngl", str(request.candidate.n_gpu_layers),
        "-fa", "on" if request.candidate.flash_attn else "off",
        "-ctk", request.candidate.cache_type_k,
        "-ctv", request.candidate.cache_type_v,
        "-o", "json",
        "-r", str(repetitions),
    ]
    if request.candidate.n_cpu_moe is not None:
        argv.extend(["--n-cpu-moe", str(request.candidate.n_cpu_moe)])
    if session_spec is not None and not session_spec.cpu_only:
        if session_spec.selected_device_ids:
            argv.extend(["--split-mode", "none"])
            argv.extend(["--main-gpu", str(session_spec.selected_device_ids[0])])
    return argv


_QUANTIZED_CACHE_TYPES = frozenset({"q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q6_k", "q8_k"})

_FLASH_INCOMPATIBLE_CACHE_TYPES = frozenset({"q8_0", "q4_0", "q4_1", "q5_0", "q5_1"})


def _is_quantized_cache(ctype: str) -> bool:
    return ctype.lower() in _QUANTIZED_CACHE_TYPES


def _provenance_for(candidate: CandidateConfig) -> str:
    parts: list[str] = []
    if candidate.flash_attn:
        parts.append("flash_attn=enabled")
    if candidate.cache_type_k != "f16" or candidate.cache_type_v != "f16":
        parts.append(f"cache_k={candidate.cache_type_k},cache_v={candidate.cache_type_v}")
    if candidate.n_cpu_moe is not None:
        parts.append(f"n_cpu_moe={candidate.n_cpu_moe}")
    if candidate.n_gpu_layers != 0:
        parts.append(f"n_gpu_layers={candidate.n_gpu_layers}")
    return "; ".join(parts) if parts else "defaults"


def _validate_request(request: SingleRunRequest) -> None:
    bp = request.binary_path
    if not bp.exists():
        raise BinaryNotFoundError(f"Binary not found: {bp}")
    if not bp.is_file():
        raise BinaryNotFoundError(f"Binary is not a file: {bp}")
    if not os.access(str(bp), os.X_OK):
        raise BinaryNotFoundError(f"Binary is not executable: {bp}")

    mp = request.model_path
    if not mp.exists():
        raise ModelNotFoundError(f"Model not found: {mp}")
    if not mp.is_file():
        raise ModelNotFoundError(f"Model is not a file: {mp}")

    nc = request.n_ctx
    if nc.total <= 0:
        raise InvalidNctxError("n_ctx.total must be > 0")
    if nc.prompt < 0:
        raise InvalidNctxError("n_ctx.prompt must be >= 0")
    if nc.generation < 0:
        raise InvalidNctxError("n_ctx.generation must be >= 0")
    if nc.depth < 0:
        raise InvalidNctxError("n_ctx.depth must be >= 0")
    if nc.prompt + nc.generation + nc.depth != nc.total:
        raise InvalidNctxError(
            f"n_ctx components ({nc.prompt} + {nc.generation} + {nc.depth}) "
            f"do not sum to total ({nc.total})",
        )

    cc = request.candidate
    if cc.threads <= 0:
        raise CandidateError("threads must be > 0")
    if cc.batch_size <= 0:
        raise CandidateError("batch_size must be > 0")
    if cc.ubatch_size <= 0:
        raise CandidateError("ubatch_size must be > 0")
    if cc.n_gpu_layers < -1:
        raise CandidateError("n_gpu_layers must be >= -1")


def _parse_measurements(stdout: str) -> list[float]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        data = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                pass
        if not data:
            raise MalformedOutputError(
                f"Cannot parse llama-bench output as JSON or JSONL.\n"
                f"Raw output:\n{stdout}",
            )

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise MalformedOutputError(
            f"Top-level JSON is not an array or object.\n"
            f"Raw output:\n{stdout}",
        )

    samples: list[float] | None = None
    for record in data:
        if not isinstance(record, dict):
            continue
        if "samples_ts" not in record:
            continue
        s = record["samples_ts"]
        if not isinstance(s, list) or len(s) == 0:
            continue
        converted: list[float] = []
        for x in s:
            if isinstance(x, bool):
                raise MalformedOutputError(
                    f"Samples contain boolean values (bool must not become number).\n"
                    f"Raw output:\n{stdout}",
                )
            if not isinstance(x, (int, float)):
                raise MalformedOutputError(
                    f"Samples contain non-numeric value: {type(x).__name__}.\n"
                    f"Raw output:\n{stdout}",
                )
            val = float(x)
            if not math.isfinite(val) or val <= 0:
                raise MalformedOutputError(
                    f"samples_ts contains non-finite or non-positive values: {s}\n"
                    f"Raw output:\n{stdout}",
                )
            converted.append(val)
        if len(converted) >= 3:
            samples = converted
            break

    if samples is None or len(samples) < 3:
        raise MalformedOutputError(
            f"Could not find samples_ts with >= 3 entries in llama-bench output.\n"
            f"Raw output:\n{stdout}",
        )

    return samples


def _validate_warmup_output(stdout: str) -> None:
    try:
        json.loads(stdout)
        return
    except json.JSONDecodeError:
        pass
    for line in stdout.strip().splitlines():
        try:
            json.loads(line)
            return
        except json.JSONDecodeError:
            continue
    raise MalformedOutputError(
        f"Warmup output is not valid JSON or JSONL:\n{stdout}",
    )


def _compute_stats(samples: list[float]) -> tuple[float, float]:
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    if n % 2 == 1:
        median = sorted_samples[n // 2]
    else:
        median = (sorted_samples[n // 2 - 1] + sorted_samples[n // 2]) / 2.0

    mean = sum(sorted_samples) / n
    variance = sum((x - mean) ** 2 for x in sorted_samples) / n
    dispersion = math.sqrt(variance)

    return median, dispersion


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _compute_session_fingerprint(spec: SessionSpecification) -> str:
    model = spec.model
    canon = _canonical_json({
        "cpu_only": spec.cpu_only,
        "selected_device_ids": list(spec.selected_device_ids),
        "n_ctx": {
            "total": spec.n_ctx.total,
            "prompt": spec.n_ctx.prompt,
            "generation": spec.n_ctx.generation,
            "depth": spec.n_ctx.depth,
        },
        "budget": {
            "ram_bytes": spec.budget.ram_bytes,
            "vram_bytes": spec.budget.vram_bytes,
            "headroom_ratio": spec.budget.headroom_ratio,
        },
        "advisory": {
            "training_context_status": spec.advisory.training_context_status,
            "training_context_tokens": spec.advisory.training_context_tokens,
            "model_weight_bytes": spec.advisory.model_weight_bytes,
            "kv_cache_bytes_estimate": spec.advisory.kv_cache_bytes_estimate,
        },
        "topology": {
            "cpu_count": spec.topology.cpu_count,
            "system_memory_bytes": spec.topology.system_memory_bytes,
            "memory_unified": spec.topology.memory_unified,
            "accelerators": [
                {
                    "kind": a.kind,
                    "backend": a.backend,
                    "device_id": a.device_id,
                    "description": a.description,
                    "memory_bytes": a.memory_bytes,
                }
                for a in spec.topology.accelerators
            ],
        },
        "model": {
            "path": str(model.path),
            "architecture": model.architecture,
            "parameter_count": model.parameter_count,
            "context_length": model.context_length,
            "quantization": model.quantization,
            "diagnostics": list(model.diagnostics),
            "shards": [
                {
                    "path": str(s.path),
                    "size_bytes": s.size_bytes,
                    "modified_time": s.modified_time,
                    "digest_sha256": s.digest_sha256,
                }
                for s in model.shards
            ],
        },
        "acknowledged": spec.acknowledged,
    })
    return hashlib.sha256(canon.encode()).hexdigest()


def _validate_device_selection(
    topology: SystemTopology,
    cpu_only: bool,
    device_ids: tuple[int, ...],
) -> None:
    if cpu_only:
        return

    if len(device_ids) == 0:
        raise NoAcceleratorSelectedError(
            "Acceleration requested (cpu_only=False) but no --device specified. "
            "Use --device <id> to select exactly one accelerator, "
            "or use --cpu-only for CPU-only operation."
            if topology.accelerators
            else "Acceleration requested (cpu_only=False) but no accelerators detected. "
            "Use --cpu-only for CPU-only operation.",
        )

    if len(device_ids) > 1:
        raise MultipleAcceleratorsSelectedError(
            f"{len(device_ids)} device IDs provided. "
            "Exactly one accelerator device may be selected at a time.",
        )

    device_id = device_ids[0]
    if not any(a.device_id == device_id for a in topology.accelerators):
        known = ", ".join(str(a.device_id) for a in topology.accelerators)
        raise UnknownDeviceError(
            f"Device {device_id} not found among detected accelerators "
            f"(IDs: {known}).",
        )


def _validate_ngl_consistency(
    cpu_only: bool,
    n_gpu_layers: int,
) -> None:
    if cpu_only and n_gpu_layers > 0:
        raise InconsistentNglError(
            "CPU-only mode selected but n_gpu_layers > 0. "
            "Set --n-gpu-layers 0 for CPU-only operation.",
        )
    if cpu_only and n_gpu_layers == -1:
        raise InconsistentNglError(
            "CPU-only mode selected but n_gpu_layers == -1 (all layers). "
            "Set --n-gpu-layers 0 for CPU-only operation.",
        )


def _compute_model_weight_bytes(model: ModelFingerprint) -> int:
    if model.shards:
        return sum(s.size_bytes for s in model.shards)
    return model.path.stat().st_size if model.path.exists() else 0


def _estimate_advisory(
    model: ModelFingerprint,
    n_ctx: NctxComposition,
    budget: ResourceBudget,
    cache_type: str = "f16",
) -> ContextAdvisory:
    model_weight = _compute_model_weight_bytes(model)
    element_size = _cache_element_size(cache_type)
    param_count = model.parameter_count or 7_000_000_000
    bytes_per_token = int(param_count * _ESTIMATED_BYTES_PER_TOKEN_PER_PARAM * element_size // 2)
    kv_cache_bytes_estimate = bytes_per_token * n_ctx.total

    ram_after_headroom = int(budget.ram_bytes * (1 - budget.headroom_ratio))
    if budget.vram_bytes is not None and budget.vram_bytes > 0:
        vram_after_headroom = int(budget.vram_bytes * (1 - budget.headroom_ratio))
    else:
        vram_after_headroom = ram_after_headroom

    available_for_kv = max(0, min(ram_after_headroom, vram_after_headroom) - model_weight)

    ctx = model.context_length
    has_conflict = any("conflict" in d.lower() for d in model.diagnostics)
    if ctx is not None and not has_conflict:
        training_context_status: TrainingContextStatus = "known"
        training_context_tokens: int | None = ctx
    elif has_conflict:
        training_context_status = "conflicted"
        training_context_tokens = ctx
    else:
        training_context_status = "unknown"
        training_context_tokens = None

    if bytes_per_token > 0:
        max_tokens_raw = available_for_kv // bytes_per_token
        uncertain_maximum = max(1, int(max_tokens_raw * 0.95))

        if training_context_status == "known" and training_context_tokens is not None:
            upper = min(max_tokens_raw, training_context_tokens)
        else:
            upper = max_tokens_raw
        upper = max(1, int(upper * 0.8))
        likely_fit_lower = max(1, int(upper * 0.5))
    else:
        uncertain_maximum = available_for_kv
        upper = available_for_kv
        likely_fit_lower = 1

    status_label = {
        "known": "training context is known",
        "unknown": "training context is unknown",
        "conflicted": "training context is conflicted",
    }[training_context_status]

    return ContextAdvisory(
        likely_fit_lower=likely_fit_lower,
        likely_fit_upper=upper,
        uncertain_maximum=uncertain_maximum,
        model_weight_bytes=model_weight,
        kv_cache_bytes_estimate=kv_cache_bytes_estimate,
        training_context_status=training_context_status,
        training_context_tokens=training_context_tokens,
        note=(
            f"Status: {status_label}. "
            f"Likely-fit range based on model size ({model_weight} bytes model weight), "
            f"KV cache heuristics ({kv_cache_bytes_estimate} bytes at {n_ctx.total} tokens), "
            f"resource budget (RAM={budget.ram_bytes}, VRAM={budget.vram_bytes}), "
            f"and headroom ({budget.headroom_ratio * 100:.0f}%). "
            "It is not a tested guarantee. Actual fit depends on runtime overhead, "
            "memory fragmentation, and the specific build of llama.cpp."
        ),
    )


def _validate_budget(
    budget: ResourceBudget,
) -> None:
    if budget.ram_bytes <= 0:
        raise NonpositiveBudgetError(
            f"RAM budget must be positive, got {budget.ram_bytes}",
        )
    if budget.vram_bytes is not None and budget.vram_bytes <= 0:
        raise NonpositiveBudgetError(
            f"VRAM budget must be positive, got {budget.vram_bytes}",
        )
    if budget.headroom_ratio < 0.0 or budget.headroom_ratio >= 1.0:
        raise InvalidHeadroomRatioError(
            f"Headroom ratio must be in [0.0, 1.0), got {budget.headroom_ratio}",
        )


def _check_context_acknowledgement(
    model: ModelFingerprint,
    n_ctx: NctxComposition,
    acknowledged: bool,
) -> None:
    ctx = model.context_length
    has_conflict = any("conflict" in d.lower() for d in model.diagnostics)
    needs_ack = False

    if ctx is not None and n_ctx.total > ctx:
        needs_ack = True
    elif ctx is None:
        needs_ack = True

    if has_conflict:
        needs_ack = True

    if needs_ack and not acknowledged:
        if ctx is not None and n_ctx.total > ctx:
            msg = (
                f"Requested context ({n_ctx.total}) exceeds the model's known "
                f"training context ({ctx}). Use --yes to acknowledge."
            )
        elif has_conflict:
            msg = (
                "Model context length metadata is internally conflicted. "
                "Use --yes to acknowledge."
            )
        else:
            msg = (
                "Model training context length is unknown. "
                "Use --yes to acknowledge."
            )
        raise ContextExceededWithoutAcknowledgementError(msg)


def _make_observation(
    request: SingleRunRequest,
    samples: list[float],
    median: float,
    dispersion: float,
    toolchain: ToolchainFingerprint,
    model: ModelFingerprint,
    session_spec: SessionSpecification | None = None,
) -> BenchmarkObservation:
    return BenchmarkObservation(
        candidate=request.candidate,
        total_ctx=request.n_ctx.total,
        prompt_ctx=request.n_ctx.prompt,
        generation_ctx=request.n_ctx.generation,
        depth_ctx=request.n_ctx.depth,
        samples_tps=tuple(samples),
        median_tps=median,
        dispersion_tps=dispersion,
        binary_path=request.binary_path,
        binary_source=request.binary_source,
        model_path=request.model_path,
        toolchain=toolchain,
        model=model,
        session_spec=session_spec,
    )


def _build_launch_env(
    policy: ExecutionPolicy | None,
    session_spec: SessionSpecification | None,
) -> dict[str, str] | None:
    """Build the subprocess environment.

    Session-selected device visibility variables (e.g. ``CUDA_VISIBLE_DEVICES``)
    override any conflicting ``ExecutionPolicy.env_overrides`` because device
    isolation is the higher-priority concern — once a session pins a specific
    device, that selection must not be overridden by policy-level overrides.
    """
    env = None
    if policy is not None and policy.env_overrides:
        env = dict(policy.env_overrides)
    if session_spec is not None:
        device_env = _env_for_device(session_spec)
        if device_env:
            if env is None:
                env = {}
            env.update(device_env)
    return env


def _classify_launch(
    result: ProcessResult,
    request: SingleRunRequest,
    toolchain: ToolchainFingerprint,
    model: ModelFingerprint,
    session_spec: SessionSpecification | None,
    policy: ExecutionPolicy | None,
) -> CandidateOutcome:
    if result.termination == "timeout":
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="timeout",
            process_launched=True,
            diagnostic=(
                f"Candidate timed out after {policy.candidate_timeout_seconds}s"
                if policy and policy.candidate_timeout_seconds
                else "Candidate timed out"
            ),
            safe_next_action=_NEXT_ACTION["timeout"],
        )

    if result.termination == "cancellation":
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="cancellation",
            process_launched=True,
            diagnostic="Candidate execution was cancelled.",
            safe_next_action=_NEXT_ACTION["cancellation"],
        )

    if result.termination == "oom":
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="oom",
            process_launched=True,
            diagnostic=result.stderr[:500] if result.stderr
            else "Process terminated due to memory pressure.",
            safe_next_action=_NEXT_ACTION["oom"],
        )

    if result.returncode != 0:
        if _is_oom(result.returncode, result.stderr):
            disp: CandidateDisposition = "oom"
        else:
            disp = "crash"
        return CandidateOutcome(
            candidate=request.candidate,
            disposition=disp,
            process_launched=True,
            diagnostic=(
                f"Process exited with code {result.returncode}.\n"
                f"{result.stderr[:500]}"
            ),
            safe_next_action=_NEXT_ACTION[disp],
        )

    try:
        samples = _parse_measurements(result.stdout)
    except MalformedOutputError:
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="malformed_output",
            process_launched=True,
            diagnostic=(
                "Process output could not be parsed as valid benchmark data.\n"
                f"Raw output:\n{result.stdout[:500]}"
            ),
            safe_next_action=_NEXT_ACTION["malformed_output"],
        )

    median, dispersion = _compute_stats(samples)
    observation = _make_observation(
        request, samples, median, dispersion, toolchain, model, session_spec,
    )

    return CandidateOutcome(
        candidate=request.candidate,
        disposition="completed",
        process_launched=True,
        diagnostic="",
        safe_next_action=_NEXT_ACTION["completed"],
        observation=observation,
    )


def _warmup_and_classify(
    adapter: ProcessAdapter,
    argv: list[str],
    request: SingleRunRequest,
    policy: ExecutionPolicy | None,
    session_spec: SessionSpecification | None = None,
) -> CandidateOutcome:
    env = _build_launch_env(policy, session_spec)

    try:
        result = adapter.launch(
            argv,
            env=env,
            timeout_seconds=policy.candidate_timeout_seconds if policy else None,
            cancellation=policy.cancellation if policy else None,
            memory_pressure_fn=policy.memory_pressure_fn if policy else None,
            memory_pressure_backoff_seconds=(
                policy.memory_pressure_backoff_seconds if policy else 1.0
            ),
            memory_pressure_max_backoffs=(
                policy.memory_pressure_max_backoffs if policy else 3
            ),
        )
    except ProcessStartError as e:
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="unsupported",
            process_launched=False,
            diagnostic=f"Warmup process could not be started: {e}",
            safe_next_action=_NEXT_ACTION["unsupported"],
        )
    except Exception as e:
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="systemic_error",
            process_launched=True,
            diagnostic=f"Warmup process launch failed: {e}",
            safe_next_action=_NEXT_ACTION["systemic_error"],
        )

    if result.termination == "timeout":
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="timeout",
            process_launched=True,
            diagnostic=f"Warmup timed out after {policy.candidate_timeout_seconds}s"
            if policy and policy.candidate_timeout_seconds else "Warmup timed out",
            safe_next_action=_NEXT_ACTION["timeout"],
        )

    if result.termination == "cancellation":
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="cancellation",
            process_launched=True,
            diagnostic="Warmup was cancelled.",
            safe_next_action=_NEXT_ACTION["cancellation"],
        )

    if result.termination == "oom":
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="oom",
            process_launched=True,
            diagnostic=result.stderr[:500] if result.stderr
            else "Warmup terminated due to memory pressure.",
            safe_next_action=_NEXT_ACTION["oom"],
        )

    if result.returncode != 0:
        if _is_oom(result.returncode, result.stderr):
            disp: CandidateDisposition = "oom"
        else:
            disp = "crash"
        return CandidateOutcome(
            candidate=request.candidate,
            disposition=disp,
            process_launched=True,
            diagnostic=f"Warmup exited with code {result.returncode}.\n{result.stderr[:500]}",
            safe_next_action=_NEXT_ACTION[disp],
        )

    try:
        _validate_warmup_output(result.stdout)
    except MalformedOutputError:
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="malformed_output",
            process_launched=True,
            diagnostic="Warmup produced unparseable output.",
            safe_next_action=_NEXT_ACTION["malformed_output"],
        )

    return CandidateOutcome(
        candidate=request.candidate,
        disposition="completed",
        process_launched=True,
        diagnostic="",
        safe_next_action=_NEXT_ACTION["completed"],
    )


def _launch_and_classify(
    adapter: ProcessAdapter,
    argv: list[str],
    request: SingleRunRequest,
    toolchain: ToolchainFingerprint,
    model: ModelFingerprint,
    session_spec: SessionSpecification | None,
    policy: ExecutionPolicy | None,
) -> CandidateOutcome:
    env = _build_launch_env(policy, session_spec)

    try:
        result = adapter.launch(
            argv,
            env=env,
            timeout_seconds=policy.candidate_timeout_seconds if policy else None,
            cancellation=policy.cancellation if policy else None,
            memory_pressure_fn=policy.memory_pressure_fn if policy else None,
            memory_pressure_backoff_seconds=(
                policy.memory_pressure_backoff_seconds if policy else 1.0
            ),
            memory_pressure_max_backoffs=(
                policy.memory_pressure_max_backoffs if policy else 3
            ),
        )
    except ProcessStartError as e:
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="unsupported",
            process_launched=False,
            diagnostic=f"Process could not be started: {e}",
            safe_next_action=_NEXT_ACTION["unsupported"],
        )
    except Exception as e:
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="systemic_error",
            process_launched=True,
            diagnostic=f"Process launch failed: {e}",
            safe_next_action=_NEXT_ACTION["systemic_error"],
        )

    return _classify_launch(result, request, toolchain, model, session_spec, policy)


def _build_workload_session_spec(
    base_spec: SessionSpecification,
    workload_nctx: NctxComposition,
) -> SessionSpecification:
    budget = ResourceBudget(
        ram_bytes=base_spec.budget.ram_bytes,
        vram_bytes=base_spec.budget.vram_bytes,
        headroom_ratio=base_spec.budget.headroom_ratio,
    )
    advisory = _estimate_advisory(base_spec.model, workload_nctx, budget)
    spec = SessionSpecification(
        topology=base_spec.topology,
        cpu_only=base_spec.cpu_only,
        selected_device_ids=base_spec.selected_device_ids,
        n_ctx=workload_nctx,
        budget=budget,
        advisory=advisory,
        model=base_spec.model,
        acknowledged=base_spec.acknowledged,
        fingerprint="",
    )
    fp = _compute_session_fingerprint(spec)
    return SessionSpecification(
        topology=base_spec.topology,
        cpu_only=base_spec.cpu_only,
        selected_device_ids=base_spec.selected_device_ids,
        n_ctx=workload_nctx,
        budget=budget,
        advisory=advisory,
        model=base_spec.model,
        acknowledged=base_spec.acknowledged,
        fingerprint=fp,
    )


def _screening_preflight(
    request: SingleRunRequest,
    toolchain: ToolchainFingerprint,
    model: ModelFingerprint,
    session_spec: SessionSpecification | None,
) -> ScreeningResult | None:
    cc = request.candidate

    if cc.flash_attn and "-fa" not in toolchain.supported_flags:
        return ScreeningResult(
            candidate=cc, success=False,
            disposition="unsupported",
            reason="flash_attn requested but toolchain does not support '-fa' flag.",
            provenance="flash_attn=unsupported",
        )

    if cc.flash_attn:
        if cc.cache_type_k.lower() in _FLASH_INCOMPATIBLE_CACHE_TYPES:
            return ScreeningResult(
                candidate=cc, success=False,
                disposition="pruned",
                reason=f"flash_attn is incompatible with cache_type_k='{cc.cache_type_k}'.",
                provenance=f"flash_attn + cache_k={cc.cache_type_k}=incompatible",
            )
        if cc.cache_type_v.lower() in _FLASH_INCOMPATIBLE_CACHE_TYPES:
            return ScreeningResult(
                candidate=cc, success=False,
                disposition="pruned",
                reason=f"flash_attn is incompatible with cache_type_v='{cc.cache_type_v}'.",
                provenance=f"flash_attn + cache_v={cc.cache_type_v}=incompatible",
            )

    if toolchain.supported_cache_types:
        if cc.cache_type_k not in toolchain.supported_cache_types:
            return ScreeningResult(
                candidate=cc, success=False,
                disposition="unsupported",
                reason=f"cache_type_k '{cc.cache_type_k}' not in toolchain supported types: "
                       f"{', '.join(toolchain.supported_cache_types)}",
                provenance=f"cache_k={cc.cache_type_k}=unsupported",
            )
        if cc.cache_type_v not in toolchain.supported_cache_types:
            return ScreeningResult(
                candidate=cc, success=False,
                disposition="unsupported",
                reason=f"cache_type_v '{cc.cache_type_v}' not in toolchain supported types: "
                       f"{', '.join(toolchain.supported_cache_types)}",
                provenance=f"cache_v={cc.cache_type_v}=unsupported",
            )

    if cc.n_cpu_moe is not None:
        if not model.is_moe:
            return ScreeningResult(
                candidate=cc, success=False,
                disposition="pruned",
                reason="n_cpu_moe specified but model is not MoE.",
                provenance="n_cpu_moe=not-moe",
            )
        if model.moe_expert_count is not None and cc.n_cpu_moe > model.moe_expert_count:
            return ScreeningResult(
                candidate=cc, success=False,
                disposition="pruned",
                reason=f"n_cpu_moe={cc.n_cpu_moe} exceeds model expert count "
                       f"({model.moe_expert_count}).",
                provenance=f"n_cpu_moe={cc.n_cpu_moe}>experts={model.moe_expert_count}",
            )
        if "--n-cpu-moe" not in toolchain.supported_flags:
            return ScreeningResult(
                candidate=cc, success=False,
                disposition="unsupported",
                reason="n_cpu_moe requested but toolchain does not support '--n-cpu-moe' flag.",
                provenance="n_cpu_moe=unsupported",
            )

    if session_spec is None:
        return None

    if session_spec.model != model:
        return ScreeningResult(
            candidate=request.candidate, success=False,
            disposition="unsupported",
            reason=(
                "Session spec model fingerprint does not match "
                "freshly inspected model."
            ),
        )

    try:
        _validate_ngl_consistency(session_spec.cpu_only, request.candidate.n_gpu_layers)
    except ScoutError as e:
        return ScreeningResult(
            candidate=request.candidate, success=False,
            disposition="pruned", reason=str(e),
        )

    if not session_spec.cpu_only:
        if ("--split-mode" not in toolchain.supported_flags
                or "--main-gpu" not in toolchain.supported_flags):
            return ScreeningResult(
                candidate=request.candidate, success=False,
                disposition="unsupported",
                reason="Accelerated launch requested but toolchain does not "
                       "support --split-mode or --main-gpu.",
            )
        device_id = session_spec.selected_device_ids[0]
        selected_accelerator = next(
            (a for a in session_spec.topology.accelerators
             if a.device_id == device_id),
            None,
        )
        if selected_accelerator is not None:
            if selected_accelerator.backend not in toolchain.device_capabilities:
                return ScreeningResult(
                    candidate=request.candidate, success=False,
                    disposition="unsupported",
                    reason=f"Selected device {device_id} requires backend "
                           f"'{selected_accelerator.backend}' but toolchain "
                           f"supports: {', '.join(toolchain.device_capabilities)}",
                )

    pre_outcome = pre_execution_disposition(request, session_spec)
    if pre_outcome is not None:
        return ScreeningResult(
            candidate=request.candidate, success=False,
            disposition=cast("ScreeningDisposition", pre_outcome.disposition),
            reason=pre_outcome.diagnostic,
        )

    return None


def _run_screening_workload(
    app: Application,
    request: SingleRunRequest,
    candidate: CandidateConfig,
    workload_nctx: NctxComposition,
    repetitions: int,
    toolchain: ToolchainFingerprint,
    model: ModelFingerprint,
    base_session_spec: SessionSpecification | None,
    policy: ExecutionPolicy | None,
) -> ScreeningResult:
    wr = SingleRunRequest(
        binary_path=request.binary_path,
        model_path=request.model_path,
        journal_path=request.journal_path,
        n_ctx=workload_nctx,
        candidate=candidate,
        binary_source=request.binary_source,
    )

    workload_spec: SessionSpecification | None = None
    if base_session_spec is not None:
        workload_spec = _build_workload_session_spec(base_session_spec, workload_nctx)

    preflight = _screening_preflight(wr, toolchain, model, workload_spec)
    if preflight is not None:
        return preflight

    env = _build_launch_env(policy, workload_spec)

    warmup_argv = _build_argv(wr, repetitions=1, session_spec=workload_spec)
    warmup_outcome = _warmup_and_classify(
        app._adapter, warmup_argv, wr, policy, session_spec=workload_spec,
    )
    if warmup_outcome.disposition != "completed":
        return ScreeningResult(
            candidate=candidate, success=False,
            disposition=warmup_outcome.disposition,
            reason=warmup_outcome.diagnostic or "Warmup failed",
        )

    measure_argv = _build_argv(wr, repetitions=repetitions, session_spec=workload_spec)
    try:
        result = app._adapter.launch(
            measure_argv,
            env=env,
            timeout_seconds=policy.candidate_timeout_seconds if policy else None,
            cancellation=policy.cancellation if policy else None,
            memory_pressure_fn=policy.memory_pressure_fn if policy else None,
            memory_pressure_backoff_seconds=(
                policy.memory_pressure_backoff_seconds if policy else 1.0
            ),
            memory_pressure_max_backoffs=(
                policy.memory_pressure_max_backoffs if policy else 3
            ),
        )
    except ProcessStartError as e:
        return ScreeningResult(
            candidate=candidate, success=False,
            disposition="unsupported",
            reason=f"Process could not be started: {e}",
        )
    except Exception as e:
        return ScreeningResult(
            candidate=candidate, success=False,
            disposition="systemic_error",
            reason=f"Process launch failed: {e}",
        )

    outcome = _classify_launch(
        result, wr, toolchain, model, workload_spec, policy,
    )
    if outcome.disposition != "completed" or outcome.observation is None:
        return ScreeningResult(
            candidate=candidate, success=False,
            disposition=cast("ScreeningDisposition", outcome.disposition),
            reason=outcome.diagnostic or f"Screening run failed: {outcome.disposition}",
        )
    obs = outcome.observation
    return ScreeningResult(
        candidate=candidate,
        success=True,
        observation=ScreeningObservation(
            candidate=candidate,
            pp_samples=tuple(obs.samples_tps),
            tg_samples=tuple(obs.samples_tps),
            observed_prompt=obs.prompt_ctx,
            observed_generation=obs.generation_ctx,
            observed_depth=obs.depth_ctx,
            observed_total=obs.total_ctx,
        ),
    )


class Application:
    def __init__(
        self,
        runner: ProcessRunner,
        detect_topology_fn: DetectTopologyFn | None = None,
        process_adapter: ProcessAdapter | None = None,
    ) -> None:
        self._runner = runner
        self._detect_topology = detect_topology_fn or detect_topology
        self._adapter = process_adapter or ProductionProcessAdapter()

    def setup_session(
        self,
        model: ModelFingerprint,
        n_ctx: NctxComposition,
        candidate: CandidateConfig,
        cpu_only: bool,
        device_ids: tuple[int, ...] = (),
        budget_overrides: tuple[int | None, int | None] | None = None,
        headroom_ratio: float = DEFAULT_HEADROOM_RATIO,
        acknowledged: bool = False,
    ) -> SessionSpecification:
        topology = self._detect_topology()

        _validate_device_selection(topology, cpu_only, device_ids)
        _validate_ngl_consistency(cpu_only, candidate.n_gpu_layers)

        if candidate.n_gpu_layers < -1:
            raise CandidateError("n_gpu_layers must be >= -1")

        if budget_overrides is not None and budget_overrides[0] is not None:
            ram = budget_overrides[0]
        else:
            ram = topology.system_memory_bytes

        if cpu_only or topology.memory_unified or not topology.accelerators:
            vram: int | None = None
        else:
            if budget_overrides is not None and budget_overrides[1] is not None:
                vram = budget_overrides[1]
            else:
                selected_device = None
                if device_ids:
                    device_id = device_ids[0]
                    for a in topology.accelerators:
                        if a.device_id == device_id:
                            selected_device = a
                            break
                if selected_device is not None and selected_device.memory_bytes is not None:
                    vram = selected_device.memory_bytes
                elif not topology.accelerators:
                    vram = None
                elif device_ids:
                    raise SessionSetupError(
                        f"Cannot determine VRAM for device {device_ids[0]}. "
                        "Specify --budget-vram to provide a VRAM budget."
                    )
                else:
                    vram = None

        budget = ResourceBudget(
            ram_bytes=ram, vram_bytes=vram,
            headroom_ratio=headroom_ratio,
        )
        _validate_budget(budget)
        _validate_nctx_at_setup(n_ctx)

        _check_context_acknowledgement(model, n_ctx, acknowledged)

        advisory = _estimate_advisory(model, n_ctx, budget, candidate.cache_type_k)

        spec = SessionSpecification(
            topology=topology,
            cpu_only=cpu_only,
            selected_device_ids=device_ids,
            n_ctx=n_ctx,
            budget=budget,
            advisory=advisory,
            model=model,
            acknowledged=acknowledged,
            fingerprint="",
        )
        spec_fingerprint = _compute_session_fingerprint(spec)
        spec = SessionSpecification(
            topology=topology,
            cpu_only=cpu_only,
            selected_device_ids=device_ids,
            n_ctx=n_ctx,
            budget=budget,
            advisory=advisory,
            model=model,
            acknowledged=acknowledged,
            fingerprint=spec_fingerprint,
        )
        return spec

    def start_or_resume_session(
        self,
        journal_path: Path,
        spec: SessionSpecification,
        model: ModelFingerprint,
        candidate: CandidateConfig | None = None,
        profile: SearchProfile | None = None,
        objective: str = "balanced",
        toolchain: ToolchainFingerprint | None = None,
    ) -> SessionResumeResult:
        components, digest = _compute_full_fingerprint_components(
            spec, model, candidate=candidate, profile=profile,
            objective=objective, toolchain=toolchain,
        )
        events = read_journal_events(journal_path)
        session_id = _uuid4().hex
        parent_session_id: str | None = None
        mismatch_reason: str | None = None

        if not events:
            write_session_start(journal_path, spec, session_id, None, components, digest)
            return SessionResumeResult(action="new", session_id=session_id)

        parent_session_id = None
        observation: BenchmarkObservation | None = None

        # Check if any v4 legacy or incompatible schema event exists
        for ev in events:
            if ev.get("session_id") == "v4-legacy":
                parent_session_id = "v4-legacy"
                mismatch_reason = "Incompatible legacy schema version"
            if ev.get("event_type") == "incompatible_schema":
                sid = ev.get("session_id", "")
                if sid:
                    parent_session_id = sid
                    mismatch_reason = str(ev.get("reason", "Incompatible schema version"))

        for ev in reversed(events):
            if ev.get("event_type") != "session_start":
                continue
            stored_sid = ev.get("session_id", "")
            stored_components = ev.get("fingerprint_components") or {}

            if stored_components.get("schema_version") != CURRENT_SCHEMA_VERSION:
                parent_session_id = stored_sid or "v4-legacy"
                mismatch_reason = "Incompatible legacy schema version"
                break

            match_reason = _compare_fingerprint_components(stored_components, components)
            if match_reason is not None:
                parent_session_id = stored_sid
                mismatch_reason = match_reason
                break

            stored_sid_match = stored_sid
            for rev_ev in reversed(events):
                if (rev_ev.get("event_type") == "observation_completed"
                        and rev_ev.get("session_id") == stored_sid_match):
                    obs_data = rev_ev.get("observation")
                    if obs_data is not None and isinstance(obs_data, dict):
                        obs_version = obs_data.get("version", 4)
                        if obs_version == 4:
                            observation = _dict_to_observation(obs_data)
                    break

            search_results: list[ScreeningResult] = []
            for sr_ev in events:
                if (sr_ev.get("event_type") in ("measurement_completed",)
                        and sr_ev.get("session_id") == stored_sid_match):
                    sr_data = sr_ev.get("screening_result")
                    if sr_data is not None and isinstance(sr_data, dict):
                        try:
                            sr = _dict_to_screening_result(sr_data)
                            if sr is not None:
                                search_results.append(sr)
                        except Exception:
                            pass

            return SessionResumeResult(
                action="resumed",
                session_id=stored_sid_match,
                observation=observation,
                completed_search_results=tuple(search_results),
            )

        write_session_start(
            journal_path, spec, session_id, parent_session_id, components, digest,
            reason=mismatch_reason,
        )
        return SessionResumeResult(
            action="derived",
            session_id=session_id,
            parent_session_id=parent_session_id,
            mismatch_reason=mismatch_reason or "unknown",
        )

    def evaluate_candidate(
        self,
        request: SingleRunRequest,
        *,
        on_ready: Callable[[ToolchainFingerprint, ModelFingerprint], Any] | None = None,
        session_spec: SessionSpecification | None = None,
        policy: ExecutionPolicy | None = None,
        session_id: str | None = None,
    ) -> CandidateOutcome:
        try:
            _validate_request(request)
        except ScoutError as e:
            outcome = CandidateOutcome(
                candidate=request.candidate,
                disposition="unsupported",
                process_launched=False,
                diagnostic=str(e),
                safe_next_action=_NEXT_ACTION["unsupported"],
            )
            _journal_outcome(
                request.journal_path, outcome,
                session_spec=session_spec, session_id=session_id,
            )
            return outcome

        try:
            toolchain = inspect_toolchain(self._runner, request.binary_path)
        except ScoutError as e:
            outcome = CandidateOutcome(
                candidate=request.candidate,
                disposition="unsupported",
                process_launched=False,
                diagnostic=str(e),
                safe_next_action=_NEXT_ACTION["unsupported"],
            )
            _journal_outcome(
                request.journal_path, outcome,
                session_spec=session_spec, session_id=session_id,
            )
            return outcome

        try:
            model = inspect_model(request.model_path)
        except ScoutError as e:
            outcome = CandidateOutcome(
                candidate=request.candidate,
                disposition="unsupported",
                process_launched=False,
                diagnostic=str(e),
                safe_next_action=_NEXT_ACTION["unsupported"],
            )
            _journal_outcome(
                request.journal_path, outcome,
                session_spec=session_spec, session_id=session_id,
            )
            return outcome

        if on_ready is not None:
            on_ready(toolchain, model)

        if session_spec is not None:
            if session_spec.n_ctx != request.n_ctx:
                outcome = CandidateOutcome(
                    candidate=request.candidate,
                    disposition="unsupported",
                    process_launched=False,
                    diagnostic=(
                        f"Session spec n_ctx ({session_spec.n_ctx}) does not match "
                        f"request n_ctx ({request.n_ctx})"
                    ),
                    safe_next_action=_NEXT_ACTION["unsupported"],
                )
                _journal_outcome(
                    request.journal_path, outcome,
                    session_spec=session_spec, session_id=session_id,
                )
                return outcome
            if session_spec.model != model:
                outcome = CandidateOutcome(
                    candidate=request.candidate,
                    disposition="unsupported",
                    process_launched=False,
                    diagnostic=(
                        "Session spec model fingerprint does not match "
                        "freshly inspected model."
                    ),
                    safe_next_action=_NEXT_ACTION["unsupported"],
                )
                _journal_outcome(
                    request.journal_path, outcome,
                    session_spec=session_spec, session_id=session_id,
                )
                return outcome
            try:
                _validate_ngl_consistency(
                    session_spec.cpu_only,
                    request.candidate.n_gpu_layers,
                )
            except ScoutError as e:
                outcome = CandidateOutcome(
                    candidate=request.candidate,
                    disposition="pruned",
                    process_launched=False,
                    diagnostic=str(e),
                    safe_next_action=_NEXT_ACTION["pruned"],
                )
                _journal_outcome(
                    request.journal_path, outcome,
                    session_spec=session_spec, session_id=session_id,
                )
                return outcome

            if not session_spec.cpu_only:
                if ("--split-mode" not in toolchain.supported_flags
                        or "--main-gpu" not in toolchain.supported_flags):
                    outcome = CandidateOutcome(
                        candidate=request.candidate,
                        disposition="unsupported",
                        process_launched=False,
                        diagnostic=(
                            "Accelerated launch requested but toolchain does not "
                            "support --split-mode or --main-gpu."
                        ),
                        safe_next_action=_NEXT_ACTION["unsupported"],
                    )
                    _journal_outcome(
                        request.journal_path, outcome,
                        session_spec=session_spec, session_id=session_id,
                    )
                    return outcome

                device_id = session_spec.selected_device_ids[0]
                selected_accelerator = next(
                    (a for a in session_spec.topology.accelerators
                     if a.device_id == device_id),
                    None,
                )
                if selected_accelerator is not None:
                    if selected_accelerator.backend not in toolchain.device_capabilities:
                        outcome = CandidateOutcome(
                            candidate=request.candidate,
                            disposition="unsupported",
                            process_launched=False,
                            diagnostic=(
                                f"Selected device {device_id} requires backend "
                                f"'{selected_accelerator.backend}' but toolchain "
                                f"supports: {', '.join(toolchain.device_capabilities)}"
                            ),
                            safe_next_action=_NEXT_ACTION["unsupported"],
                        )
                        _journal_outcome(
                            request.journal_path, outcome,
                            session_spec=session_spec, session_id=session_id,
                        )
                        return outcome

        pre_outcome = pre_execution_disposition(request, session_spec)
        if pre_outcome is not None:
            _journal_outcome(
                request.journal_path, pre_outcome,
                session_spec=session_spec, session_id=session_id,
            )
            return pre_outcome

        warmup_argv = _build_argv(request, repetitions=1, session_spec=session_spec)
        warmup_outcome = _warmup_and_classify(
            self._adapter, warmup_argv, request, policy, session_spec=session_spec,
        )
        if warmup_outcome.disposition != "completed":
            try:
                _journal_outcome(
                    request.journal_path, warmup_outcome,
                    session_spec=session_spec, session_id=session_id,
                )
            except JournalError as e:
                return CandidateOutcome(
                    candidate=request.candidate,
                    disposition="systemic_error",
                    process_launched=True,
                    diagnostic=f"Warmup outcome journal write failed: {e}",
                    safe_next_action=_NEXT_ACTION["systemic_error"],
                )
            return warmup_outcome

        measure_argv = _build_argv(request, repetitions=3, session_spec=session_spec)
        measure_outcome = _launch_and_classify(
            self._adapter, measure_argv, request, toolchain, model,
            session_spec, policy,
        )

        if measure_outcome.disposition == "completed":
            assert measure_outcome.observation is not None
            try:
                write_observation(
                    request.journal_path,
                    measure_outcome.observation,
                    session_spec=session_spec,
                    session_id=session_id,
                )
            except OSError as e:
                outcome = CandidateOutcome(
                    candidate=request.candidate,
                    disposition="systemic_error",
                    process_launched=True,
                    diagnostic=f"Failed to write journal: {e}",
                    safe_next_action=_NEXT_ACTION["systemic_error"],
                )
                return outcome
        else:
            try:
                write_outcome(
                    request.journal_path, measure_outcome,
                    session_spec=session_spec, session_id=session_id,
                )
            except OSError as e:
                return CandidateOutcome(
                    candidate=request.candidate,
                    disposition="systemic_error",
                    process_launched=measure_outcome.process_launched,
                    diagnostic=f"Failed to write outcome journal: {e}",
                    safe_next_action=_NEXT_ACTION["systemic_error"],
                )

        return measure_outcome

    def screen_candidates(
        self,
        request: SingleRunRequest,
        profile: SearchProfile,
        *,
        evaluator: ScreeningEvaluator | None = None,
        session_spec: SessionSpecification | None = None,
        policy: ExecutionPolicy | None = None,
        objective: str = "balanced",
        session_id: str | None = None,
        parent_session_id: str | None = None,
        completed_search_results: tuple[ScreeningResult, ...] = (),
    ) -> ScreeningReport:
        _validate_request(request)

        prior_by_candidate: dict[CandidateConfig, deque[ScreeningResult]] = {}
        for r in completed_search_results:
            if r.candidate not in prior_by_candidate:
                prior_by_candidate[r.candidate] = deque()
            prior_by_candidate[r.candidate].append(r)

        def _wrap_for_prior_and_journal(
            inner: ScreeningEvaluator,
        ) -> ScreeningEvaluator:
            def wrapped(
                candidate: CandidateConfig,
                n_ctx: NctxComposition,
                repetitions: int,
            ) -> ScreeningResult:
                if candidate in prior_by_candidate and prior_by_candidate[candidate]:
                    return prior_by_candidate[candidate].popleft()
                result = inner(candidate, n_ctx, repetitions)
                if session_id is not None:
                    try:
                        write_measurement_completed(
                            request.journal_path, result, session_id,
                            parent_session_id=parent_session_id,
                        )
                    except OSError as e:
                        raise JournalError(
                            f"Failed to write measurement_completed journal: {e}",
                        ) from e
                return result
            return wrapped

        if evaluator is not None:
            wrapped = _wrap_for_prior_and_journal(evaluator)
            inj_budget = session_spec.budget if session_spec is not None else None
            inj_memory_unified = (
                session_spec.topology.memory_unified
                if session_spec is not None else False
            )
            report = _search_screen(
                base_candidate=request.candidate,
                profile=profile,
                evaluator=wrapped,
                screening_n_ctx=SCREENING_N_CTX,
                objective=objective,
                resource_budget=inj_budget,
                memory_unified=inj_memory_unified,
            )
            if session_id is not None:
                _journal_search_decisions(
                    request.journal_path, report, session_id,
                    parent_session_id=parent_session_id,
                )
            return report

        # Inspect toolchain and model once, before the search.
        try:
            toolchain = inspect_toolchain(self._runner, request.binary_path)
        except ScoutError as e:
            baseline = ScreeningResult(
                candidate=request.candidate, success=False,
                disposition="unsupported", reason=str(e),
            )
            if session_id is not None:
                write_search_decision(
                    request.journal_path,
                    candidate=asdict(baseline.candidate),
                    disposition=baseline.disposition,
                    reason=baseline.reason,
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                )
            return ScreeningReport(
                profile=profile,
                baseline_disposition=baseline,
                grid_dispositions=(),
                survivors=(),
                leader=None,
                attempts_used=0,
                objective=objective,
                disposition_summary="Toolchain inspection failed; no evaluations attempted",
            )

        try:
            model = inspect_model(request.model_path)
        except ScoutError as e:
            baseline = ScreeningResult(
                candidate=request.candidate, success=False,
                disposition="unsupported", reason=str(e),
            )
            if session_id is not None:
                write_search_decision(
                    request.journal_path,
                    candidate=asdict(baseline.candidate),
                    disposition=baseline.disposition,
                    reason=baseline.reason,
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                )
            return ScreeningReport(
                profile=profile,
                baseline_disposition=baseline,
                grid_dispositions=(),
                survivors=(),
                leader=None,
                attempts_used=0,
                objective=objective,
                disposition_summary="Model inspection failed; no evaluations attempted",
            )

        if session_spec is not None and session_spec.model != model:
            baseline = ScreeningResult(
                candidate=request.candidate, success=False,
                disposition="unsupported",
                reason="Session spec model fingerprint does not match freshly inspected model.",
            )
            if session_id is not None:
                write_search_decision(
                    request.journal_path,
                    candidate=asdict(baseline.candidate),
                    disposition=baseline.disposition,
                    reason=baseline.reason,
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                )
            return ScreeningReport(
                profile=profile,
                baseline_disposition=baseline,
                grid_dispositions=(),
                survivors=(),
                leader=None,
                attempts_used=0,
                objective=objective,
                disposition_summary="Model fingerprint mismatch; no evaluations attempted",
            )

        cpu_only = session_spec.cpu_only if session_spec is not None else False
        expanded_grid = generate_screening_grid(
            request.candidate, toolchain, model, cpu_only,
        )

        def _production_evaluator(
            candidate: CandidateConfig,
            n_ctx: NctxComposition,
            repetitions: int,
        ) -> ScreeningResult:
            preflight = _screening_preflight(
                SingleRunRequest(
                    binary_path=request.binary_path,
                    model_path=request.model_path,
                    journal_path=request.journal_path,
                    n_ctx=request.n_ctx,
                    candidate=candidate,
                    binary_source=request.binary_source,
                ),
                toolchain,
                model,
                session_spec,
            )
            if preflight is not None:
                return preflight

            pp_nctx = NctxComposition(
                total=n_ctx.prompt + 2, prompt=n_ctx.prompt, generation=1, depth=1,
            )
            tg_nctx = NctxComposition(
                total=n_ctx.generation + 2, prompt=1, generation=n_ctx.generation, depth=1,
            )

            pp_result = _run_screening_workload(
                self, request, candidate, pp_nctx, repetitions,
                toolchain, model, session_spec, policy,
            )
            if not pp_result.success or pp_result.observation is None:
                return pp_result

            tg_result = _run_screening_workload(
                self, request, candidate, tg_nctx, repetitions,
                toolchain, model, session_spec, policy,
            )
            if not tg_result.success or tg_result.observation is None:
                return tg_result

            return ScreeningResult(
                candidate=candidate,
                success=True,
                observation=ScreeningObservation(
                    candidate=candidate,
                    pp_samples=tuple(pp_result.observation.pp_samples),
                    tg_samples=tuple(tg_result.observation.tg_samples),
                ),
                provenance=_provenance_for(candidate),
            )

        wrapped_production = _wrap_for_prior_and_journal(_production_evaluator)
        budget = session_spec.budget if session_spec is not None else None
        memory_unified = (
            session_spec.topology.memory_unified
            if session_spec is not None else False
        )

        report = _search_screen(
            base_candidate=request.candidate,
            profile=profile,
            evaluator=wrapped_production,
            screening_n_ctx=SCREENING_N_CTX,
            objective=objective,
            grid=expanded_grid,
            resource_budget=budget,
            memory_unified=memory_unified,
        )

        notes: list[str] = []
        for d in (
            report.baseline_disposition,
            *report.grid_dispositions,
            *report.adaptive_dispositions,
        ):
            if d.success and d.observation is not None:
                cc = d.candidate
                if _is_quantized_cache(cc.cache_type_k) or _is_quantized_cache(cc.cache_type_v):
                    msg = (
                        f"Quantized KV cache type quality was not measured; "
                        f"only throughput was assessed (K={cc.cache_type_k}, V={cc.cache_type_v})."
                    )
                    if msg not in notes:
                        notes.append(msg)

        if notes:
            report = ScreeningReport(
                profile=report.profile,
                baseline_disposition=report.baseline_disposition,
                grid_dispositions=report.grid_dispositions,
                adaptive_dispositions=report.adaptive_dispositions,
                survivors=report.survivors,
                leader=report.leader,
                attempts_used=report.attempts_used,
                objective=report.objective,
                disposition_summary=report.disposition_summary,
                notes=tuple(notes),
                pareto_frontier=report.pareto_frontier,
            )

        if session_id is not None:
            _journal_search_decisions(
                request.journal_path, report, session_id,
                parent_session_id=parent_session_id,
            )
        return report

    def validate_candidates(
        self,
        request: SingleRunRequest,
        screening_report: ScreeningReport,
        *,
        budget: int = 10,
        objective: str = "balanced",
        validation_evaluator: ValidationEvaluator | None = None,
        session_spec: SessionSpecification | None = None,
        policy: ExecutionPolicy | None = None,
    ) -> ValidatedRecommendation | NoWinnerReport:
        _validate_request(request)

        if validation_evaluator is not None:
            ve_budget = session_spec.budget if session_spec is not None else None
            ve_memory_unified = (
                session_spec.topology.memory_unified
                if session_spec is not None else False
            )
            return _validate_shortlist(
                screening_report=screening_report,
                requested_n_ctx=request.n_ctx,
                objective=objective,
                budget=budget,
                evaluator=validation_evaluator,
                resource_budget=ve_budget,
                memory_unified=ve_memory_unified,
            )

        try:
            toolchain = inspect_toolchain(self._runner, request.binary_path)
        except ScoutError as e:
            return NoWinnerReport(
                provisional_leader=screening_report.leader,
                failures=(
                    FailedValidation(
                        candidate=request.candidate,
                        disposition="systemic_error",
                        reason=f"Toolchain inspection failed: {e}",
                    ),
                ),
                attempts_used=0,
                budget=budget,
                pareto_frontier=screening_report.pareto_frontier,
            )

        try:
            model = inspect_model(request.model_path)
        except ScoutError as e:
            return NoWinnerReport(
                provisional_leader=screening_report.leader,
                failures=(
                    FailedValidation(
                        candidate=request.candidate,
                        disposition="systemic_error",
                        reason=f"Model inspection failed: {e}",
                    ),
                ),
                attempts_used=0,
                budget=budget,
                pareto_frontier=screening_report.pareto_frontier,
            )

        def _production_validator(
            candidate: CandidateConfig,
            n_ctx: NctxComposition,
        ) -> ScreeningResult:
            return _run_screening_workload(
                self, request, candidate, n_ctx, 3,
                toolchain, model, session_spec, policy,
            )

        res_budget = session_spec.budget if session_spec is not None else None
        mem_unified = (
            session_spec.topology.memory_unified
            if session_spec is not None else False
        )

        return _validate_shortlist(
            screening_report=screening_report,
            requested_n_ctx=request.n_ctx,
            objective=objective,
            budget=budget,
            evaluator=_production_validator,
            resource_budget=res_budget,
            memory_unified=mem_unified,
        )

    def run(
        self,
        request: SingleRunRequest,
        *,
        on_ready: Callable[[ToolchainFingerprint, ModelFingerprint], Any] | None = None,
        session_spec: SessionSpecification | None = None,
        session_id: str | None = None,
    ) -> BenchmarkObservation:
        _validate_request(request)

        toolchain = inspect_toolchain(self._runner, request.binary_path)
        model = inspect_model(request.model_path)

        if session_spec is not None:
            if session_spec.n_ctx != request.n_ctx:
                raise SessionSetupError(
                    f"Session spec n_ctx ({session_spec.n_ctx}) does not match "
                    f"request n_ctx ({request.n_ctx})",
                )
            if session_spec.model != model:
                raise ModelFingerprintMismatchError(
                    f"Session spec model fingerprint does not match freshly inspected model.\n"
                    f"Spec model: path={session_spec.model.path} "
                    f"arch={session_spec.model.architecture} "
                    f"shards={len(session_spec.model.shards)}\n"
                    f"Fresh model: path={model.path} "
                    f"arch={model.architecture} "
                    f"shards={len(model.shards)}",
                )

            _validate_ngl_consistency(
                session_spec.cpu_only,
                request.candidate.n_gpu_layers,
            )

            if not session_spec.cpu_only:
                if "--split-mode" not in toolchain.supported_flags:
                    raise SessionSetupError(
                        "Accelerated launch requested but llama-bench does not support "
                        "--split-mode. Cannot guarantee device isolation.",
                    )
                if "--main-gpu" not in toolchain.supported_flags:
                    raise SessionSetupError(
                        "Accelerated launch requested but llama-bench does not support "
                        "--main-gpu. Cannot guarantee device isolation.",
                    )

                device_id = session_spec.selected_device_ids[0]
                selected_accelerator = next(
                    (a for a in session_spec.topology.accelerators
                     if a.device_id == device_id),
                    None,
                )
                if selected_accelerator is not None:
                    if selected_accelerator.backend not in toolchain.device_capabilities:
                        raise SessionSetupError(
                            f"Selected device {device_id} requires backend "
                            f"'{selected_accelerator.backend}' but toolchain supports: "
                            f"{', '.join(toolchain.device_capabilities)}",
                        )

        if on_ready is not None:
            on_ready(toolchain, model)

        warmup_argv = _build_argv(request, repetitions=1, session_spec=session_spec)
        warmup_result = self._runner.run(warmup_argv)
        if warmup_result.returncode != 0:
            raise SubprocessError(
                f"Warmup failed (exit {warmup_result.returncode}):\n"
                f"{warmup_result.stderr}",
            )
        _validate_warmup_output(warmup_result.stdout)

        measure_argv = _build_argv(request, repetitions=3, session_spec=session_spec)
        measure_result = self._runner.run(measure_argv)
        if measure_result.returncode != 0:
            raise SubprocessError(
                f"Measurement failed (exit {measure_result.returncode}):\n"
                f"{measure_result.stderr}",
            )

        samples = _parse_measurements(measure_result.stdout)
        median, dispersion = _compute_stats(samples)

        observation = BenchmarkObservation(
            candidate=request.candidate,
            total_ctx=request.n_ctx.total,
            prompt_ctx=request.n_ctx.prompt,
            generation_ctx=request.n_ctx.generation,
            depth_ctx=request.n_ctx.depth,
            samples_tps=tuple(samples),
            median_tps=median,
            dispersion_tps=dispersion,
            binary_path=request.binary_path,
            binary_source=request.binary_source,
            model_path=request.model_path,
            toolchain=toolchain,
            model=model,
            session_spec=session_spec,
        )

        try:
            write_observation(
                request.journal_path, observation,
                session_spec=session_spec, session_id=session_id,
            )
        except OSError as e:
            raise JournalError(f"Failed to write journal: {e}") from e

        return observation

    # ── MTP comparison seam ───────────────────────────────────────

    def compare_mtp(
        self,
        *,
        candidate: CandidateConfig,
        main_model_path: Path,
        draft_model_path: Path,
        server_path: Path,
        n_ctx: NctxComposition,
        cpu_only: bool = True,
        device_id: int | None = None,
        session_spec: SessionSpecification | None = None,
        journal_path: Path | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        server_lifecycle: ServerLifecycle | None = None,
        main_fingerprint: ModelFingerprint | None = None,
        draft_fingerprint: ModelFingerprint | None = None,
        _warmup_fn: Callable[[int], None] | None = None,
        _measure_fn: Callable[[int], float] | None = None,
    ) -> MtpComparison:
        from .mtp import collect_mtp_evidence as _collect_mtp_evidence

        if session_spec is not None:
            cpu_only = session_spec.cpu_only
            device_id = (
                session_spec.selected_device_ids[0]
                if session_spec.selected_device_ids
                else None
            )

        _validate_nctx_at_setup(n_ctx)
        _validate_ngl_consistency(cpu_only, candidate.n_gpu_layers)
        if not cpu_only and device_id is not None:
            _validate_device_selection(
                session_spec.topology if session_spec else self._detect_topology(),
                cpu_only,
                (device_id,) if device_id is not None else (),
            )

        main_fp = main_fingerprint or inspect_model(main_model_path)
        draft_fp = draft_fingerprint or inspect_model(draft_model_path)

        evidence = _collect_mtp_evidence(
            candidate=candidate,
            main_model_path=main_model_path,
            draft_model_path=draft_model_path,
            server_path=server_path,
            n_ctx=n_ctx,
            cpu_only=cpu_only,
            device_id=device_id,
            runner=self._runner,
            server_lifecycle=server_lifecycle,
            main_fingerprint=main_fp,
            draft_fingerprint=draft_fp,
            _warmup_fn=_warmup_fn,
            _measure_fn=_measure_fn,
        )

        if journal_path is not None:
            write_mtp_evidence(
                journal_path, evidence,
                session_id=session_id, parent_session_id=parent_session_id,
            )

        return evidence.comparison

    # ── Export seam ───────────────────────────────────────────────

    def export_report_json(
        self,
        *,
        session: SessionSpecification,
        toolchain: ToolchainFingerprint,
        model: ModelFingerprint,
        screening_report: ScreeningReport | None = None,
        validation_result: ValidatedRecommendation | NoWinnerReport | None = None,
        provenance: LaunchProvenance | None = None,
        include_hashes: bool = False,
    ) -> str:
        from .reporter import render_export_json as _render

        return _render(
            session=session,
            toolchain=toolchain,
            model=model,
            screening_report=screening_report,
            validation_result=validation_result,
            provenance=provenance,
            include_hashes=include_hashes,
        )

    def export_report_markdown(
        self,
        *,
        session: SessionSpecification,
        toolchain: ToolchainFingerprint,
        model: ModelFingerprint,
        screening_report: ScreeningReport | None = None,
        validation_result: ValidatedRecommendation | NoWinnerReport | None = None,
        provenance: LaunchProvenance | None = None,
        include_hashes: bool = False,
    ) -> str:
        from .reporter import render_export_markdown as _render

        return _render(
            session=session,
            toolchain=toolchain,
            model=model,
            screening_report=screening_report,
            validation_result=validation_result,
            provenance=provenance,
            include_hashes=include_hashes,
        )


# ── Resource budget gate ─────────────────────────────────────────


def _check_resource_budget(
    session_spec: SessionSpecification,
    n_ctx: NctxComposition,
    candidate: CandidateConfig,
    cache_type_k: str = "f16",
) -> CandidateOutcome | None:
    budget = session_spec.budget
    headroom = budget.headroom_ratio
    model = session_spec.model

    model_weight = 0
    if model.shards:
        model_weight = sum(s.size_bytes for s in model.shards)
    elif model.path.exists():
        model_weight = model.path.stat().st_size

    param_count = model.parameter_count or 7_000_000_000
    element_size = _cache_element_size(cache_type_k)
    bytes_per_token = int(param_count * _ESTIMATED_BYTES_PER_TOKEN_PER_PARAM * element_size // 2)
    kv_estimate = bytes_per_token * n_ctx.total

    ram_available = int(budget.ram_bytes * (1 - headroom))
    unified = session_spec.topology.memory_unified

    if unified or budget.vram_bytes is None:
        needed = model_weight + kv_estimate
        if needed > ram_available:
            return CandidateOutcome(
                candidate=candidate,
                disposition="budget_exhausted",
                process_launched=False,
                diagnostic=(
                    f"Estimated resource need ({needed} bytes) exceeds "
                    f"available RAM ({ram_available} bytes) after headroom."
                ),
                safe_next_action=_NEXT_ACTION["budget_exhausted"],
            )
    else:
        vram_available = int(budget.vram_bytes * (1 - headroom))
        model_in_vram = min(model_weight, vram_available)
        remaining_weight = model_weight - model_in_vram
        vram_after_model = vram_available - model_in_vram
        kv_in_vram = min(kv_estimate, vram_after_model)
        kv_in_ram = kv_estimate - kv_in_vram
        ram_needed = remaining_weight + kv_in_ram
        if ram_needed > ram_available:
            return CandidateOutcome(
                candidate=candidate,
                disposition="budget_exhausted",
                process_launched=False,
                diagnostic=(
                    f"Estimated resource need (model={model_weight}, "
                    f"KV cache={kv_estimate}) exceeds combined "
                    f"RAM ({ram_available}) and VRAM ({vram_available}) budget."
                ),
                safe_next_action=_NEXT_ACTION["budget_exhausted"],
            )

    return None


def pre_execution_disposition(
    request: SingleRunRequest,
    session_spec: SessionSpecification | None = None,
    skip_candidate: bool = False,
) -> CandidateOutcome | None:
    if session_spec is None:
        return None

    if skip_candidate:
        return CandidateOutcome(
            candidate=request.candidate,
            disposition="not_attempted",
            process_launched=False,
            diagnostic="Candidate skipped by session logic.",
            safe_next_action=_NEXT_ACTION["not_attempted"],
        )

    budget_outcome = _check_resource_budget(
        session_spec, request.n_ctx, request.candidate,
        cache_type_k=request.candidate.cache_type_k,
    )
    if budget_outcome is not None:
        assert budget_outcome.candidate == request.candidate
        return budget_outcome

    return None


# ── Memory pressure detector (stdlib, best-effort) ───────────────


def _detect_available_memory_bytes() -> int | None:
    if sys.platform == "linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        return int(parts[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    elif sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True, text=True, timeout=5,
            )
            pages_free = 0
            pages_inactive = 0
            for line in result.stdout.splitlines():
                s = line.strip()
                if s.startswith("Pages free:"):
                    pages_free = int(s.split()[2].rstrip("."))
                elif s.startswith("Pages inactive:"):
                    pages_inactive = int(s.split()[2].rstrip("."))
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages_free + pages_inactive) * page_size
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass
    elif sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes  # type: ignore[attr-defined]

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = _MEMORYSTATUSEX()
            memory_status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):  # type: ignore[attr-defined]
                return memory_status.ullAvailPhys
        except (OSError, AttributeError, ImportError):
            pass
    return None


def make_memory_pressure_fn(
    session_spec: SessionSpecification | None,
) -> Callable[[], bool] | None:
    if session_spec is None:
        return None
    budget = session_spec.budget
    headroom = budget.headroom_ratio
    threshold_ram = int(budget.ram_bytes * headroom)
    threshold_ram = min(threshold_ram, session_spec.topology.system_memory_bytes)

    def _is_pressured() -> bool:
        available = _detect_available_memory_bytes()
        if available is None:
            return False
        if available < threshold_ram:
            return True
        return False

    return _is_pressured


# ── Journal helper for outcomes ──────────────────────────────────


def _journal_outcome(
    path: Path,
    outcome: CandidateOutcome,
    session_spec: SessionSpecification | None = None,
    session_id: str | None = None,
) -> None:
    try:
        write_outcome(path, outcome, session_spec=session_spec, session_id=session_id)
    except OSError as e:
        raise JournalError(f"Failed to write journal: {e}") from e


def _dict_to_observation(d: dict[str, Any]) -> BenchmarkObservation:
    obs_version = d.get("version", 4)
    if obs_version != 4:
        raise JournalError(f"Unsupported observation version: {obs_version}")
    cand = CandidateConfig(**d.get("candidate", {}))
    tc_data = d.get("toolchain", {})
    tc = ToolchainFingerprint(
        binary_path=Path(tc_data["binary_path"]) if tc_data.get("binary_path") else Path(),
        version=tc_data.get("version", ""),
        supported_flags=tuple(tc_data.get("supported_flags", [])),
        supported_output_formats=tuple(tc_data.get("supported_output_formats", [])),
        device_capabilities=tuple(tc_data.get("device_capabilities", [])),
        supported_cache_types=tuple(tc_data.get("supported_cache_types", ("f16",))),
    )
    m_data = d.get("model", {})
    shards = tuple(
        ShardFingerprint(
            path=Path(s["path"]),
            size_bytes=s["size_bytes"],
            modified_time=s["modified_time"],
            digest_sha256=s["digest_sha256"],
        )
        for s in m_data.get("shards") or []
    )
    model = ModelFingerprint(
        path=Path(m_data.get("path", "")),
        architecture=m_data.get("architecture", ""),
        parameter_count=m_data.get("parameter_count"),
        context_length=m_data.get("context_length"),
        quantization=m_data.get("quantization"),
        shards=shards,
        symlink_warning=m_data.get("symlink_warning", False),
        network_mount_detected=m_data.get("network_mount_detected", False),
        diagnostics=tuple(m_data.get("diagnostics") or []),
        is_moe=m_data.get("is_moe", False),
        moe_expert_count=m_data.get("moe_expert_count"),
    )
    return BenchmarkObservation(
        candidate=cand,
        total_ctx=d.get("total_ctx", 0),
        prompt_ctx=d.get("prompt_ctx", 0),
        generation_ctx=d.get("generation_ctx", 0),
        depth_ctx=d.get("depth_ctx", 0),
        samples_tps=tuple(d.get("samples_tps") or []),
        median_tps=d.get("median_tps", 0.0),
        dispersion_tps=d.get("dispersion_tps", 0.0),
        binary_path=Path(d.get("binary_path", "")),
        binary_source=d.get("binary_source", "explicit"),
        model_path=Path(d.get("model_path", "")),
        toolchain=tc,
        model=model,
    )


def _dict_to_screening_result(d: dict[str, Any]) -> ScreeningResult | None:
    version = d.get("version")
    if version != 1:
        return None
    cand = CandidateConfig(**d.get("candidate", {}))
    obs_data = d.get("observation")
    obs: ScreeningObservation | None = None
    if obs_data is not None and isinstance(obs_data, dict):
        mem_data = obs_data.get("memory")
        mem: MemoryEvidence | None = None
        if mem_data is not None and isinstance(mem_data, dict):
            mem = MemoryEvidence(
                ram_bytes=mem_data.get("ram_bytes"),
                vram_bytes=mem_data.get("vram_bytes"),
                quality=mem_data.get("quality"),
            )
        obs = ScreeningObservation(
            candidate=CandidateConfig(**obs_data.get("candidate", {})),
            pp_samples=tuple(obs_data.get("pp_samples") or []),
            tg_samples=tuple(obs_data.get("tg_samples") or []),
            memory_bytes=obs_data.get("memory_bytes"),
            memory=mem,
            observed_prompt=obs_data.get("observed_prompt"),
            observed_generation=obs_data.get("observed_generation"),
            observed_depth=obs_data.get("observed_depth"),
            observed_total=obs_data.get("observed_total"),
        )
    return ScreeningResult(
        candidate=cand,
        success=bool(d.get("success", False)),
        disposition=d.get("disposition", "survived"),
        reason=d.get("reason", ""),
        observation=obs,
        provenance=d.get("provenance", ""),
    )


def _validate_nctx_at_setup(n_ctx: NctxComposition) -> None:
    if n_ctx.total <= 0:
        raise InvalidNctxError("n_ctx.total must be > 0")
    if n_ctx.prompt < 0:
        raise InvalidNctxError("n_ctx.prompt must be >= 0")
    if n_ctx.generation < 0:
        raise InvalidNctxError("n_ctx.generation must be >= 0")
    if n_ctx.depth < 0:
        raise InvalidNctxError("n_ctx.depth must be >= 0")
    if n_ctx.prompt + n_ctx.generation + n_ctx.depth != n_ctx.total:
        raise InvalidNctxError(
            f"n_ctx components ({n_ctx.prompt} + {n_ctx.generation} + {n_ctx.depth}) "
            f"do not sum to total ({n_ctx.total})",
        )


# ── Fingerprint v5: component families and comparison ────────────

CURRENT_SCHEMA_VERSION = 5


def _capture_environment() -> dict[str, str]:
    relevant_keys = [
        "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
    ]
    env = {}
    for key in relevant_keys:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    return env


def _compute_full_fingerprint_components(
    spec: SessionSpecification,
    model: ModelFingerprint,
    candidate: CandidateConfig | None = None,
    profile: SearchProfile | None = None,
    objective: str = "balanced",
    toolchain: ToolchainFingerprint | None = None,
) -> tuple[dict[str, Any], str]:
    components: dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "main_model": {
            "path": str(model.path),
            "architecture": model.architecture,
            "parameter_count": model.parameter_count,
            "context_length": model.context_length,
            "quantization": model.quantization,
            "shards": [
                {
                    "path": str(s.path),
                    "size_bytes": s.size_bytes,
                    "modified_time": s.modified_time,
                    "digest_sha256": s.digest_sha256,
                }
                for s in model.shards
            ],
        },
        "draft_model": None,
        "topology": {
            "cpu_count": spec.topology.cpu_count,
            "system_memory_bytes": spec.topology.system_memory_bytes,
            "memory_unified": spec.topology.memory_unified,
            "accelerators": [
                {
                    "kind": a.kind,
                    "backend": a.backend,
                    "device_id": a.device_id,
                    "description": a.description,
                    "memory_bytes": a.memory_bytes,
                }
                for a in spec.topology.accelerators
            ],
        },
        "device_selection": {
            "cpu_only": spec.cpu_only,
            "selected_device_ids": list(spec.selected_device_ids),
        },
        "context": {
            "total": spec.n_ctx.total,
            "prompt": spec.n_ctx.prompt,
            "generation": spec.n_ctx.generation,
            "depth": spec.n_ctx.depth,
        },
        "objective": objective,
        "environment": _capture_environment(),
        "budget": {
            "ram_bytes": spec.budget.ram_bytes,
            "vram_bytes": spec.budget.vram_bytes,
            "headroom_ratio": spec.budget.headroom_ratio,
        },
    }
    if toolchain is not None:
        components["toolchain"] = {
            "binary_path": str(toolchain.binary_path),
            "version": toolchain.version,
            "supported_flags": list(toolchain.supported_flags),
            "supported_output_formats": list(toolchain.supported_output_formats),
            "device_capabilities": list(toolchain.device_capabilities),
            "supported_cache_types": list(toolchain.supported_cache_types),
        }
    if candidate is not None:
        components["candidate"] = asdict(candidate)
    if profile is not None:
        components["profile"] = {
            "label": profile.label,
            "max_attempts": profile.max_attempts,
            "min_repetitions": profile.min_repetitions,
            "max_repetitions_per_candidate": profile.max_repetitions_per_candidate,
            "tie_bound_coefficient": profile.tie_bound_coefficient,
            "dispersion_threshold": profile.dispersion_threshold,
        }
    digest = hashlib.sha256(
        _canonical_json(components).encode(),
    ).hexdigest()
    return components, digest


_FINGERPRINT_FAMILIES: list[tuple[str, str]] = [
    ("schema_version", "schema version"),
    ("main_model", "model/shards"),
    ("draft_model", "draft model"),
    ("toolchain", "toolchain"),
    ("topology", "topology/backend/device"),
    ("device_selection", "device selection"),
    ("context", "context"),
    ("profile", "search profile"),
    ("objective", "objective"),
    ("environment", "environment"),
    ("budget", "budget"),
    ("candidate", "candidate"),
]


def _compare_fingerprint_components(stored: dict[str, Any], current: dict[str, Any]) -> str | None:
    differences: list[str] = []
    for key, label in _FINGERPRINT_FAMILIES:
        if stored.get(key) != current.get(key):
            differences.append(label)
    if not differences:
        return None
    return f"Mismatch in: {', '.join(differences)}"


def _journal_search_decisions(
    journal_path: Path,
    report: ScreeningReport,
    session_id: str,
    parent_session_id: str | None = None,
) -> None:
    all_results: list[ScreeningResult] = [
        report.baseline_disposition,
        *report.grid_dispositions,
        *report.adaptive_dispositions,
    ]
    for r in all_results:
        write_search_decision(
            journal_path,
            candidate=asdict(r.candidate),
            disposition=r.disposition,
            reason=r.reason,
            session_id=session_id,
            parent_session_id=parent_session_id,
        )


# ── Application session seam ─────────────────────────────────────


def list_session_summaries(journal_path: Path) -> list[dict[str, Any]]:
    return _journal_list_sessions(journal_path)
