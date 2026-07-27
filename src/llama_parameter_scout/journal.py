from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4 as _uuid4

from .models import (
    Accelerator,
    BenchmarkObservation,
    CandidateOutcome,
    ContextAdvisory,
    JournalError,
    MemoryEvidence,
    ModelFingerprint,
    MtpComparison,
    MtpEvidence,
    ResourceBudget,
    ScreeningObservation,
    ScreeningResult,
    SessionSpecification,
    ShardFingerprint,
    SystemTopology,
    ToolchainFingerprint,
)

_SCHEMA_VERSION = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_event(
    event_type: str,
    session_id: str,
    payload: dict[str, object],
    parent_session_id: str | None = None,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "event_version": 1,
        "session_id": session_id,
        "parent_session_id": parent_session_id,
        "timestamp": _now(),
        **payload,
    }


# ── Journal I/O ─────────────────────────────────────────────────


def _read_journal(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text()
    except OSError as e:
        raise JournalError(f"Cannot read journal at {path}: {e}") from e
    if not raw.strip():
        raise JournalError(f"Journal at {path} is empty")
    try:
        raw_data: Any = json.loads(raw)
    except json.JSONDecodeError as e:
        raise JournalError(
            f"Journal at {path} contains invalid JSON: {e}",
        ) from e
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    if data.get("version") == 4:
        sid = "v4-legacy"
        event_type = "observation_completed" if "samples_tps" in data else "outcome_recorded"
        event: dict[str, object] = {
            "event_type": event_type,
            "event_version": 1,
            "session_id": sid,
            "parent_session_id": None,
            "timestamp": "unknown",
            "data": data,
        }
        return {"schema_version": _SCHEMA_VERSION, "events": [event]}
    if data.get("schema_version") == _SCHEMA_VERSION:
        return data
    sv = data.get("schema_version")
    if sv is not None:
        sid = f"v{sv}-incompatible"
        event = {
            "event_type": "incompatible_schema",
            "event_version": 1,
            "session_id": sid,
            "parent_session_id": None,
            "timestamp": "unknown",
            "data": data,
            "reason": f"Incompatible schema version v{sv} — cannot migrate",
        }
        return {"schema_version": _SCHEMA_VERSION, "events": [event]}
    return {"schema_version": _SCHEMA_VERSION, "events": []}


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _append_events(path: Path, new_events: list[dict[str, Any]]) -> None:
    journal = _read_journal(path) or {"schema_version": _SCHEMA_VERSION, "events": []}
    journal["events"].extend(new_events)
    _atomic_write(path, journal)


# ── Event factories ─────────────────────────────────────────────


def _make_session_start_event(
    spec: SessionSpecification,
    session_id: str,
    parent_session_id: str | None,
    fingerprint_components: dict[str, object],
    fingerprint_digest: str,
    reason: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "fingerprint_digest": fingerprint_digest,
        "fingerprint_components": fingerprint_components,
        "session_spec": _session_spec_to_dict(spec),
    }
    if reason:
        payload["derivation_reason"] = reason
    return _make_event("session_start", session_id, payload, parent_session_id=parent_session_id)


def _make_observation_event(
    observation: BenchmarkObservation,
    session_id: str,
    parent_session_id: str | None = None,
) -> dict[str, object]:
    payload = _observation_to_dict(observation)
    return _make_event("observation_completed", session_id, {
        "observation": payload,
    }, parent_session_id=parent_session_id)


def _make_outcome_event(
    outcome: CandidateOutcome,
    session_id: str,
    parent_session_id: str | None = None,
) -> dict[str, object]:
    payload = _outcome_to_dict(outcome)
    payload.pop("version", None)
    return _make_event("outcome_recorded", session_id, {
        "outcome": payload,
    }, parent_session_id=parent_session_id)


def _make_search_decision_event(
    candidate: dict[str, object],
    disposition: str,
    reason: str,
    session_id: str,
    parent_session_id: str | None = None,
) -> dict[str, object]:
    return _make_event("search_decision", session_id, {
        "candidate": candidate,
        "disposition": disposition,
        "reason": reason,
    }, parent_session_id=parent_session_id)


def _screening_observation_to_dict(obs: ScreeningObservation) -> dict[str, object]:
    return {
        "candidate": asdict(obs.candidate),
        "pp_samples": list(obs.pp_samples),
        "tg_samples": list(obs.tg_samples),
        "memory_bytes": obs.memory_bytes,
        "memory": _memory_evidence_to_dict(obs.memory) if obs.memory else None,
        "observed_prompt": obs.observed_prompt,
        "observed_generation": obs.observed_generation,
        "observed_depth": obs.observed_depth,
        "observed_total": obs.observed_total,
    }


def _screening_result_to_dict(result: ScreeningResult) -> dict[str, object]:
    d: dict[str, object] = {
        "version": 1,
        "candidate": asdict(result.candidate),
        "success": result.success,
        "disposition": result.disposition,
        "reason": result.reason,
        "provenance": result.provenance,
    }
    if result.observation is not None:
        d["observation"] = _screening_observation_to_dict(result.observation)
    return d


def _memory_evidence_to_dict(me: MemoryEvidence) -> dict[str, object]:
    return {
        "ram_bytes": me.ram_bytes,
        "vram_bytes": me.vram_bytes,
        "quality": me.quality,
    }


def _make_measurement_completed_event(
    result: ScreeningResult,
    session_id: str,
    parent_session_id: str | None = None,
) -> dict[str, object]:
    return _make_event("measurement_completed", session_id, {
        "screening_result": _screening_result_to_dict(result),
    }, parent_session_id=parent_session_id)


def _make_mtp_evidence_event(
    evidence: MtpEvidence,
    session_id: str,
    parent_session_id: str | None = None,
) -> dict[str, object]:
    return _make_event("mtp_evidence", session_id, {
        "mtp_evidence": _mtp_evidence_to_dict(evidence),
    }, parent_session_id=parent_session_id)


def _mtp_comparison_to_dict(c: MtpComparison) -> dict[str, object]:
    return {
        "benefit": c.benefit,
        "improvement_pct": c.improvement_pct,
        "tie_bound_tps": c.tie_bound_tps,
        "caveat": c.caveat,
        "off": {
            "mode": c.off.mode,
            "samples_tps": list(c.off.samples_tps),
            "median_tps": c.off.median_tps,
            "dispersion_tps": c.off.dispersion_tps,
        },
        "on": {
            "mode": c.on.mode,
            "samples_tps": list(c.on.samples_tps),
            "median_tps": c.on.median_tps,
            "dispersion_tps": c.on.dispersion_tps,
        },
    }


def _mtp_evidence_to_dict(ev: MtpEvidence) -> dict[str, object]:
    return {
        "main_fingerprint": _model_to_dict(ev.main_fingerprint),
        "draft_fingerprint": _model_to_dict(ev.draft_fingerprint),
        "server_path": ev.server_path,
        "server_version": ev.server_version,
        "server_supported_flags": list(ev.server_supported_flags),
        "candidate": asdict(ev.candidate),
        "n_ctx": asdict(ev.n_ctx),
        "cpu_only": ev.cpu_only,
        "device_id": ev.device_id,
        "off_argv": list(ev.off_argv),
        "on_argv": list(ev.on_argv),
        "comparison": _mtp_comparison_to_dict(ev.comparison),
    }


# ── Public API ──────────────────────────────────────────────────


def write_observation(
    path: Path,
    observation: BenchmarkObservation,
    session_spec: SessionSpecification | None = None,
    session_id: str | None = None,
    parent_session_id: str | None = None,
) -> None:
    sid = session_id or _uuid4().hex
    event = _make_observation_event(observation, sid, parent_session_id=parent_session_id)
    if session_spec is not None:
        event["session_spec"] = _session_spec_to_dict(session_spec)
    _append_events(path, [event])


def write_outcome(
    path: Path,
    outcome: CandidateOutcome,
    session_spec: SessionSpecification | None = None,
    session_id: str | None = None,
    parent_session_id: str | None = None,
) -> None:
    sid = session_id or _uuid4().hex
    event = _make_outcome_event(outcome, sid, parent_session_id=parent_session_id)
    if session_spec is not None:
        event["session_spec"] = _session_spec_to_dict(session_spec)
    _append_events(path, [event])


def write_session_start(
    path: Path,
    spec: SessionSpecification,
    session_id: str,
    parent_session_id: str | None,
    fingerprint_components: dict[str, object],
    fingerprint_digest: str,
    reason: str | None = None,
) -> None:
    event = _make_session_start_event(
        spec, session_id, parent_session_id,
        fingerprint_components, fingerprint_digest,
        reason=reason,
    )
    _append_events(path, [event])


def write_search_decision(
    path: Path,
    candidate: dict[str, object],
    disposition: str,
    reason: str,
    session_id: str,
    parent_session_id: str | None = None,
) -> None:
    event = _make_search_decision_event(
        candidate, disposition, reason, session_id,
        parent_session_id=parent_session_id,
    )
    _append_events(path, [event])


def write_measurement_completed(
    path: Path,
    result: ScreeningResult,
    session_id: str,
    parent_session_id: str | None = None,
) -> None:
    event = _make_measurement_completed_event(
        result, session_id, parent_session_id=parent_session_id,
    )
    _append_events(path, [event])


def write_mtp_evidence(
    path: Path,
    evidence: MtpEvidence,
    session_id: str | None = None,
    parent_session_id: str | None = None,
) -> None:
    sid = session_id or _uuid4().hex
    event = _make_mtp_evidence_event(evidence, sid, parent_session_id=parent_session_id)
    _append_events(path, [event])


def read_journal_events(path: Path) -> list[dict[str, Any]]:
    journal = _read_journal(path)
    if journal is None:
        return []
    return list(journal.get("events", []))


def list_sessions(path: Path) -> list[dict[str, object]]:
    events = read_journal_events(path)
    seen: dict[str, dict[str, object]] = {}
    for ev in events:
        sid = ev.get("session_id")
        if not sid or not isinstance(sid, str):
            continue
        if sid not in seen:
            seen[sid] = {
                "session_id": sid,
                "parent_session_id": ev.get("parent_session_id"),
                "status": "historical",
                "reason": "",
                "created_at": ev.get("timestamp", "unknown"),
            }
        if ev.get("event_type") == "observation_completed":
            seen[sid]["has_observation"] = True
        if ev.get("derivation_reason"):
            seen[sid]["status"] = "derived"
            seen[sid]["reason"] = str(ev["derivation_reason"])
        if ev.get("fingerprint_components"):
            seen[sid]["fingerprint_components"] = ev["fingerprint_components"]
    for sid, info in seen.items():
        if sid == "v4-legacy":
            info["status"] = "historical"
            if not info["reason"]:
                info["reason"] = "Legacy v4 journal format — incompatible schema"
        elif info.get("status") == "derived":
            pass
        elif info.get("has_observation"):
            info["status"] = "resumable"
        else:
            info["status"] = "historical"
    return list(seen.values())


# ── Serialisation helpers (unchanged structure, reused) ─────────


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
