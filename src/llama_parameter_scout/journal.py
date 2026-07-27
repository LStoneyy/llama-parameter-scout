from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .models import (
    Accelerator,
    BenchmarkObservation,
    CandidateOutcome,
    ContextAdvisory,
    ModelFingerprint,
    ResourceBudget,
    SessionSpecification,
    ShardFingerprint,
    SystemTopology,
    ToolchainFingerprint,
)


def write_observation(
    path: Path,
    observation: BenchmarkObservation,
    session_spec: SessionSpecification | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _observation_to_dict(observation)
    if session_spec is not None:
        data["session_spec"] = _session_spec_to_dict(session_spec)
    _atomic_write_json(path, data)


def write_outcome(
    path: Path,
    outcome: CandidateOutcome,
    session_spec: SessionSpecification | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _outcome_to_dict(outcome)
    if session_spec is not None:
        data["session_spec"] = _session_spec_to_dict(session_spec)
    _atomic_write_json(path, data)


def _atomic_write_json(path: Path, data: dict[str, object]) -> None:
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _observation_to_dict(obs: BenchmarkObservation) -> dict[str, object]:
    return {
        "version": 4,
        "disposition": "completed",
        "process_launched": True,
        "candidate": asdict(obs.candidate),
        "total_ctx": obs.total_ctx,
        "prompt_ctx": obs.prompt_ctx,
        "generation_ctx": obs.generation_ctx,
        "depth_ctx": obs.depth_ctx,
        "samples_tps": list(obs.samples_tps),
        "median_tps": obs.median_tps,
        "dispersion_tps": obs.dispersion_tps,
        "n_measurements": len(obs.samples_tps),
        "binary_path": str(obs.binary_path),
        "binary_source": obs.binary_source,
        "model_path": str(obs.model_path),
        "toolchain": _toolchain_to_dict(obs.toolchain),
        "model": _model_to_dict(obs.model),
    }


def _outcome_to_dict(outcome: CandidateOutcome) -> dict[str, object]:
    d: dict[str, object] = {
        "version": 4,
        "disposition": outcome.disposition,
        "process_launched": outcome.process_launched,
        "diagnostic": outcome.diagnostic,
        "safe_next_action": outcome.safe_next_action,
        "candidate": asdict(outcome.candidate),
    }
    if outcome.observation is not None:
        obs = outcome.observation
        d["total_ctx"] = obs.total_ctx
        d["prompt_ctx"] = obs.prompt_ctx
        d["generation_ctx"] = obs.generation_ctx
        d["depth_ctx"] = obs.depth_ctx
        if obs.samples_tps:
            d["samples_tps"] = list(obs.samples_tps)
            d["median_tps"] = obs.median_tps
            d["dispersion_tps"] = obs.dispersion_tps
            d["n_measurements"] = len(obs.samples_tps)
        d["binary_path"] = str(obs.binary_path)
        d["binary_source"] = obs.binary_source
        d["model_path"] = str(obs.model_path)
        d["toolchain"] = _toolchain_to_dict(obs.toolchain)
        d["model"] = _model_to_dict(obs.model)
    return d


def _session_spec_to_dict(spec: SessionSpecification) -> dict[str, object]:
    return {
        "cpu_only": spec.cpu_only,
        "selected_device_ids": list(spec.selected_device_ids),
        "n_ctx": asdict(spec.n_ctx),
        "acknowledged": spec.acknowledged,
        "fingerprint": spec.fingerprint,
        "topology": _topology_to_dict(spec.topology),
        "budget": _budget_to_dict(spec.budget),
        "advisory": _advisory_to_dict(spec.advisory),
        "model": _model_to_dict(spec.model),
    }


def _topology_to_dict(t: SystemTopology) -> dict[str, object]:
    return {
        "cpu_count": t.cpu_count,
        "system_memory_bytes": t.system_memory_bytes,
        "accelerators": [_accelerator_to_dict(a) for a in t.accelerators],
        "memory_unified": t.memory_unified,
    }


def _accelerator_to_dict(a: Accelerator) -> dict[str, object]:
    return {
        "kind": a.kind,
        "backend": a.backend,
        "device_id": a.device_id,
        "description": a.description,
        "memory_bytes": a.memory_bytes,
    }


def _budget_to_dict(b: ResourceBudget) -> dict[str, object]:
    return {
        "ram_bytes": b.ram_bytes,
        "vram_bytes": b.vram_bytes,
        "headroom_ratio": b.headroom_ratio,
    }


def _advisory_to_dict(a: ContextAdvisory) -> dict[str, object]:
    return {
        "likely_fit_lower": a.likely_fit_lower,
        "likely_fit_upper": a.likely_fit_upper,
        "uncertain_maximum": a.uncertain_maximum,
        "model_weight_bytes": a.model_weight_bytes,
        "kv_cache_bytes_estimate": a.kv_cache_bytes_estimate,
        "training_context_status": a.training_context_status,
        "training_context_tokens": a.training_context_tokens,
        "note": a.note,
    }


def _toolchain_to_dict(tc: ToolchainFingerprint) -> dict[str, object]:
    return {
        "binary_path": str(tc.binary_path),
        "version": tc.version,
        "supported_flags": list(tc.supported_flags),
        "supported_output_formats": list(tc.supported_output_formats),
        "device_capabilities": list(tc.device_capabilities),
        "supported_cache_types": list(tc.supported_cache_types),
    }


def _shard_to_dict(s: ShardFingerprint) -> dict[str, object]:
    return {
        "path": str(s.path),
        "size_bytes": s.size_bytes,
        "modified_time": s.modified_time,
        "digest_sha256": s.digest_sha256,
    }


def _model_to_dict(m: ModelFingerprint) -> dict[str, object]:
    d: dict[str, object] = {
        "path": str(m.path),
        "architecture": m.architecture,
        "parameter_count": m.parameter_count,
        "context_length": m.context_length,
        "quantization": m.quantization,
        "shards": [_shard_to_dict(s) for s in m.shards],
        "symlink_warning": m.symlink_warning,
        "network_mount_detected": m.network_mount_detected,
        "diagnostics": list(m.diagnostics),
    }
    if m.is_moe:
        d["is_moe"] = True
        if m.moe_expert_count is not None:
            d["moe_expert_count"] = m.moe_expert_count
    return d
