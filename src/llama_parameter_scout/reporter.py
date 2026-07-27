from __future__ import annotations

import hashlib
import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    CandidateConfig,
    FailedValidation,
    LaunchProvenance,
    MemoryEvidence,
    ModelFingerprint,
    NctxComposition,
    NoWinnerReport,
    ProvisionalLeader,
    RankedCandidate,
    ScreeningReport,
    ScreeningResult,
    SessionSpecification,
    ToolchainFingerprint,
    ValidatedCandidate,
    ValidatedRecommendation,
)

REPORT_SCHEMA_VERSION = "scout-export-v1"

# ── Redaction ─────────────────────────────────────────────────────

_HOME: str = str(Path.home())


def redact_path(value: str) -> str:
    if _HOME:
        return value.replace(_HOME, "[HOME]")
    return value


def redact_all_paths(obj: object) -> object:
    if isinstance(obj, str):
        return redact_path(obj)
    if isinstance(obj, dict):
        return {k: redact_all_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_all_paths(v) for v in obj]
    return obj


_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def redact_device_id(value: str) -> str:
    return _UUID_RE.sub("[DEVICE-ID]", value)


def redact_all_device_ids(obj: object) -> object:
    if isinstance(obj, str):
        return redact_device_id(obj)
    if isinstance(obj, dict):
        return {k: redact_all_device_ids(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_all_device_ids(v) for v in obj]
    return obj


# ── Shell quoting ─────────────────────────────────────────────────


def quote_for_shell(value: str, dialect: str) -> str:
    if dialect == "posix":
        return shlex.quote(value)
    if dialect == "powershell":
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if dialect == "cmd":
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    raise ValueError(f"Unknown shell dialect: {dialect!r}")


def join_argv(argv: list[str], dialect: str) -> str:
    return " ".join(quote_for_shell(a, dialect) for a in argv)


# ── Reproduction argv builder ─────────────────────────────────────


def build_reproduction_argv(
    candidate: CandidateConfig,
    n_ctx: NctxComposition,
    *,
    binary_path: str = "llama-bench",
    model_path: str = "<MODEL_PATH>",
    repetitions: int | None = None,
    cpu_only: bool = True,
    selected_device_ids: tuple[int, ...] = (),
) -> list[str]:
    argv: list[str] = [
        binary_path,
        "-m", model_path,
        "-p", str(n_ctx.prompt),
        "-n", str(n_ctx.generation),
        "-d", str(n_ctx.depth),
        "-t", str(candidate.threads),
        "-b", str(candidate.batch_size),
        "-ub", str(candidate.ubatch_size),
        "-ngl", str(candidate.n_gpu_layers),
        "-fa", "on" if candidate.flash_attn else "off",
        "-ctk", candidate.cache_type_k,
        "-ctv", candidate.cache_type_v,
    ]
    if candidate.n_cpu_moe is not None:
        argv.extend(["--n-cpu-moe", str(candidate.n_cpu_moe)])
    if not cpu_only and selected_device_ids:
        argv.extend(["--split-mode", "none", "--main-gpu", str(selected_device_ids[0])])
    argv.extend(["-o", "json"])
    if repetitions is not None:
        argv.extend(["-r", str(repetitions)])
    return argv


def _collect_reproduction_candidates(
    screening_report: ScreeningReport | None,
    validation_result: ValidatedRecommendation | NoWinnerReport | None,
) -> tuple[list[tuple[CandidateConfig, int | None]], list[str]]:
    seen: dict[CandidateConfig, int | None] = {}
    notes: list[str] = []

    if screening_report is not None:
        for sr in [
            screening_report.baseline_disposition,
            *screening_report.grid_dispositions,
            *screening_report.adaptive_dispositions,
        ]:
            if sr.observation is not None:
                reps = max(len(sr.observation.pp_samples), len(sr.observation.tg_samples))
                if sr.candidate not in seen or reps > (seen[sr.candidate] or 0):
                    seen[sr.candidate] = reps

        for rc in screening_report.survivors:
            if rc.candidate in seen and rc.total_repetitions > 0:
                seen[rc.candidate] = max(seen[rc.candidate] or 0, rc.total_repetitions)

    if validation_result is not None and isinstance(validation_result, ValidatedRecommendation):
        for vc in [validation_result.recommendation, *validation_result.alternatives]:
            if vc.candidate not in seen:
                seen[vc.candidate] = None

    if not seen:
        return [], notes

    return list(seen.items()), notes


def _flash_attn_flag(candidate: CandidateConfig) -> str:
    if candidate.flash_attn:
        return "--flash-attn"
    return "--no-flash-attn"


def build_llama_cli_command(
    candidate: CandidateConfig,
    n_ctx: NctxComposition,
    model_path: str,
    *,
    cpu_only: bool = True,
    selected_device_ids: tuple[int, ...] = (),
) -> list[str]:
    cmd: list[str] = [
        "llama-cli",
        "-m", model_path,
        "-c", str(n_ctx.total),
        "-t", str(candidate.threads),
        "-b", str(candidate.batch_size),
        "-ub", str(candidate.ubatch_size),
        "-ngl", str(candidate.n_gpu_layers),
    ]
    cmd.append(_flash_attn_flag(candidate))
    if candidate.cache_type_k != "f16":
        cmd.extend(["-ctk", candidate.cache_type_k])
    if candidate.cache_type_v != "f16":
        cmd.extend(["-ctv", candidate.cache_type_v])
    if candidate.n_cpu_moe is not None:
        cmd.extend(["--n-cpu-moe", str(candidate.n_cpu_moe)])
    if not cpu_only and selected_device_ids:
        cmd.extend(["--main-gpu", str(selected_device_ids[0])])
    return cmd


def build_llama_server_command(
    candidate: CandidateConfig,
    n_ctx: NctxComposition,
    model_path: str,
    *,
    cpu_only: bool = True,
    selected_device_ids: tuple[int, ...] = (),
) -> list[str]:
    cmd: list[str] = [
        "llama-server",
        "-m", model_path,
        "-c", str(n_ctx.total),
        "-t", str(candidate.threads),
        "-b", str(candidate.batch_size),
        "-ub", str(candidate.ubatch_size),
        "-ngl", str(candidate.n_gpu_layers),
    ]
    cmd.append(_flash_attn_flag(candidate))
    if candidate.cache_type_k != "f16":
        cmd.extend(["-ctk", candidate.cache_type_k])
    if candidate.cache_type_v != "f16":
        cmd.extend(["-ctv", candidate.cache_type_v])
    if candidate.n_cpu_moe is not None:
        cmd.extend(["--n-cpu-moe", str(candidate.n_cpu_moe)])
    if not cpu_only and selected_device_ids:
        cmd.extend(["--main-gpu", str(selected_device_ids[0])])
    return cmd


# ── Preset .ini section ───────────────────────────────────────────


def build_preset_ini_section(
    candidate: CandidateConfig,
    n_ctx: NctxComposition,
    *,
    cpu_only: bool = True,
    selected_device_ids: tuple[int, ...] = (),
) -> str:
    lines: list[str] = ["[llama]"]
    lines.append(f"n_ctx = {n_ctx.total}")
    lines.append(f"n_batch = {candidate.batch_size}")
    lines.append(f"n_ubatch = {candidate.ubatch_size}")
    lines.append(f"n_threads = {candidate.threads}")
    lines.append(f"n_gpu_layers = {candidate.n_gpu_layers}")
    lines.append(f"flash_attn = {'true' if candidate.flash_attn else 'false'}")
    lines.append(f"cache_type_k = {candidate.cache_type_k}")
    lines.append(f"cache_type_v = {candidate.cache_type_v}")
    if candidate.n_cpu_moe is not None:
        lines.append(f"n_cpu_moe = {candidate.n_cpu_moe}")
    if not cpu_only and selected_device_ids:
        lines.append(f"main_gpu = {selected_device_ids[0]}")
    lines.append("")
    return "\n".join(lines)


# ── Export renderers ──────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_label(
    validation: ValidatedRecommendation | NoWinnerReport | None,
    screening_report: ScreeningReport | None,
) -> str:
    if validation is not None:
        if isinstance(validation, ValidatedRecommendation):
            return "validated_recommendation"
        return "no_winner"
    if screening_report is not None and screening_report.leader is not None:
        return "provisional"
    return "incomplete"


def _candidate_to_dict(c: CandidateConfig) -> dict[str, object]:
    return {
        "threads": c.threads,
        "batch_size": c.batch_size,
        "ubatch_size": c.ubatch_size,
        "n_gpu_layers": c.n_gpu_layers,
        "flash_attn": c.flash_attn,
        "cache_type_k": c.cache_type_k,
        "cache_type_v": c.cache_type_v,
        "n_cpu_moe": c.n_cpu_moe,
    }


def _memory_evidence_to_dict(me: MemoryEvidence | None) -> dict[str, object] | None:
    if me is None:
        return None
    return {"ram_bytes": me.ram_bytes, "vram_bytes": me.vram_bytes, "quality": me.quality}


def _toolchain_to_dict(tc: ToolchainFingerprint) -> dict[str, object]:
    return {
        "binary_path": str(tc.binary_path),
        "version": tc.version,
        "supported_flags": list(tc.supported_flags),
        "supported_output_formats": list(tc.supported_output_formats),
        "device_capabilities": list(tc.device_capabilities),
        "supported_cache_types": list(tc.supported_cache_types),
    }


def _model_to_dict(m: ModelFingerprint) -> dict[str, object]:
    return {
        "path": str(m.path),
        "architecture": m.architecture,
        "parameter_count": m.parameter_count,
        "context_length": m.context_length,
        "quantization": m.quantization,
        "shards": [
            {
                "path": str(s.path),
                "size_bytes": s.size_bytes,
                "modified_time": s.modified_time,
                "digest_sha256": s.digest_sha256,
            }
            for s in m.shards
        ],
        "symlink_warning": m.symlink_warning,
        "network_mount_detected": m.network_mount_detected,
        "diagnostics": list(m.diagnostics),
        "is_moe": m.is_moe,
        "moe_expert_count": m.moe_expert_count,
    }


def _provenance_to_dict(p: LaunchProvenance | None) -> dict[str, object] | None:
    if p is None:
        return None
    return {
        "cwd": p.cwd,
        "shell_dialect": p.shell_dialect,
        "executable_path": p.executable_path,
        "argv": list(p.argv),
        "environment": dict(p.environment),
    }


def _screening_report_to_dict(report: ScreeningReport) -> dict[str, object]:
    return {
        "profile": {
            "label": report.profile.label,
            "max_attempts": report.profile.max_attempts,
            "min_repetitions": report.profile.min_repetitions,
            "max_repetitions_per_candidate": report.profile.max_repetitions_per_candidate,
            "tie_bound_coefficient": report.profile.tie_bound_coefficient,
            "dispersion_threshold": report.profile.dispersion_threshold,
        },
        "objective": report.objective,
        "baseline": _screening_result_to_dict(report.baseline_disposition),
        "grid_dispositions": [_screening_result_to_dict(d) for d in report.grid_dispositions],
        "adaptive_dispositions": [
            _screening_result_to_dict(d) for d in report.adaptive_dispositions
        ],
        "survivors": [_ranked_candidate_to_dict(s) for s in report.survivors],
        "leader": _provisional_leader_to_dict(report.leader),
        "attempts_used": report.attempts_used,
        "disposition_summary": report.disposition_summary,
        "notes": list(report.notes),
        "pareto_frontier": [
            {
                "candidate": _candidate_to_dict(e.candidate),
                "balanced": e.balanced,
                "memory": _memory_evidence_to_dict(e.memory),
            }
            for e in report.pareto_frontier
        ],
    }


def _screening_result_to_dict(sr: ScreeningResult) -> dict[str, object]:
    d: dict[str, object] = {
        "candidate": _candidate_to_dict(sr.candidate),
        "success": sr.success,
        "disposition": sr.disposition,
        "reason": sr.reason,
        "provenance": sr.provenance,
    }
    if sr.observation is not None:
        obs = sr.observation
        obs_dict: dict[str, object] = {
            "candidate": _candidate_to_dict(obs.candidate),
            "pp_samples": list(obs.pp_samples),
            "tg_samples": list(obs.tg_samples),
            "memory_bytes": obs.memory_bytes,
            "memory": _memory_evidence_to_dict(obs.memory),
            "observed_prompt": obs.observed_prompt,
            "observed_generation": obs.observed_generation,
            "observed_depth": obs.observed_depth,
            "observed_total": obs.observed_total,
        }
        d["observation"] = obs_dict
    return d


def _ranked_candidate_to_dict(rc: RankedCandidate) -> dict[str, object]:
    return {
        "candidate": _candidate_to_dict(rc.candidate),
        "pp_score": rc.pp_score,
        "tg_score": rc.tg_score,
        "balanced": rc.balanced,
        "pp_median": rc.pp_median,
        "tg_median": rc.tg_median,
        "pp_relative_dispersion": rc.pp_relative_dispersion,
        "tg_relative_dispersion": rc.tg_relative_dispersion,
        "memory_bytes": rc.memory_bytes,
        "memory_evidence": _memory_evidence_to_dict(rc.memory_evidence),
        "total_repetitions": rc.total_repetitions,
        "tie_break": rc.tie_break,
    }


def _provisional_leader_to_dict(pl: ProvisionalLeader | None) -> dict[str, object] | None:
    if pl is None:
        return None
    return {
        "candidate": _candidate_to_dict(pl.candidate),
        "objective": pl.objective,
        "pp_score": pl.pp_score,
        "tg_score": pl.tg_score,
        "balanced": pl.balanced,
        "pp_median": pl.pp_median,
        "tg_median": pl.tg_median,
        "label": pl.label,
    }


def _validation_to_dict(
    validation: ValidatedRecommendation | NoWinnerReport | None,
) -> dict[str, object] | None:
    if validation is None:
        return None
    if isinstance(validation, ValidatedRecommendation):
        return {
            "type": "validated_recommendation",
            "recommendation": _validated_candidate_to_dict(validation.recommendation),
            "alternatives": [_validated_candidate_to_dict(a) for a in validation.alternatives],
            "failures": [_failed_validation_to_dict(f) for f in validation.failures],
            "attempts_used": validation.attempts_used,
            "budget": validation.budget,
            "pareto_frontier": [
                {
                    "candidate": _candidate_to_dict(e.candidate),
                    "balanced": e.balanced,
                    "memory": _memory_evidence_to_dict(e.memory),
                }
                for e in validation.pareto_frontier
            ],
        }
    return {
        "type": "no_winner",
        "provisional_leader": _provisional_leader_to_dict(validation.provisional_leader),
        "failures": [_failed_validation_to_dict(f) for f in validation.failures],
        "attempts_used": validation.attempts_used,
        "budget": validation.budget,
        "pareto_frontier": [
            {
                "candidate": _candidate_to_dict(e.candidate),
                "balanced": e.balanced,
                "memory": _memory_evidence_to_dict(e.memory),
            }
            for e in validation.pareto_frontier
        ],
    }


def _validated_candidate_to_dict(vc: ValidatedCandidate) -> dict[str, object]:
    return {
        "candidate": _candidate_to_dict(vc.candidate),
        "pp_score": vc.pp_score,
        "tg_score": vc.tg_score,
        "balanced": vc.balanced,
        "pp_median": vc.pp_median,
        "tg_median": vc.tg_median,
        "pp_relative_dispersion": vc.pp_relative_dispersion,
        "tg_relative_dispersion": vc.tg_relative_dispersion,
        "memory_evidence": _memory_evidence_to_dict(vc.memory_evidence),
        "label": vc.label,
    }


def _failed_validation_to_dict(fv: FailedValidation) -> dict[str, object]:
    return {
        "candidate": _candidate_to_dict(fv.candidate),
        "disposition": fv.disposition,
        "reason": fv.reason,
    }


def _session_to_dict(spec: SessionSpecification) -> dict[str, object]:
    return {
        "cpu_only": spec.cpu_only,
        "selected_device_ids": list(spec.selected_device_ids),
        "n_ctx": {
            "total": spec.n_ctx.total,
            "prompt": spec.n_ctx.prompt,
            "generation": spec.n_ctx.generation,
            "depth": spec.n_ctx.depth,
        },
        "fingerprint": spec.fingerprint,
        "acknowledged": spec.acknowledged,
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
        "budget": {
            "ram_bytes": spec.budget.ram_bytes,
            "vram_bytes": spec.budget.vram_bytes,
            "headroom_ratio": spec.budget.headroom_ratio,
        },
        "advisory": {
            "likely_fit_lower": spec.advisory.likely_fit_lower,
            "likely_fit_upper": spec.advisory.likely_fit_upper,
            "uncertain_maximum": spec.advisory.uncertain_maximum,
            "model_weight_bytes": spec.advisory.model_weight_bytes,
            "kv_cache_bytes_estimate": spec.advisory.kv_cache_bytes_estimate,
            "training_context_status": spec.advisory.training_context_status,
            "training_context_tokens": spec.advisory.training_context_tokens,
            "note": spec.advisory.note,
        },
        "model": _model_to_dict(spec.model),
    }


def _compute_hashes(
    session: SessionSpecification,
    toolchain: ToolchainFingerprint,
    model: ModelFingerprint,
    screening_report: ScreeningReport | None,
    validation_result: ValidatedRecommendation | NoWinnerReport | None,
) -> dict[str, str]:
    digests: dict[str, str] = {}
    digests["session_reference"] = session.fingerprint
    raw = str(session.fingerprint) + toolchain.version + str(model.path)
    digests["export_digest"] = hashlib.sha256(raw.encode()).hexdigest()
    return digests


_HASHES_CAVEAT = (
    "Content hashes provide integrity verification for the exported artifact. "
    "Performance reproducibility is never guaranteed — it depends on the exact "
    "binary build, system state, driver version, thermal conditions, and "
    "hardware behavior at the time of measurement."
)


def render_export_json(
    *,
    session: SessionSpecification,
    toolchain: ToolchainFingerprint,
    model: ModelFingerprint,
    screening_report: ScreeningReport | None = None,
    validation_result: ValidatedRecommendation | NoWinnerReport | None = None,
    provenance: LaunchProvenance | None = None,
    include_hashes: bool = False,
) -> str:
    root: dict[str, object] = {
        "schema": REPORT_SCHEMA_VERSION,
        "exported_at": _now_iso(),
        "artifact_type": "private_reproducibility_artifact",
        "artifact_label": (
            "Private reproducibility artifact — contains full system topology and "
            "paths. Do not share directly."
        ),
        "result": _result_label(validation_result, screening_report),
        "provenance": _provenance_to_dict(provenance),
        "session": _session_to_dict(session),
        "toolchain": _toolchain_to_dict(toolchain),
        "model": _model_to_dict(model),
        "hashes_include": include_hashes,
    }
    if screening_report is not None:
        root["screening"] = _screening_report_to_dict(screening_report)
    root["validation"] = _validation_to_dict(validation_result)

    candidates_repro, repro_notes = _collect_reproduction_candidates(
        screening_report, validation_result,
    )
    reproduction_argv: list[dict[str, object]] = []
    rep_unknown: list[int] = []
    for idx, (cand, reps) in enumerate(candidates_repro):
        if reps is None:
            rep_unknown.append(idx)
        argv = build_reproduction_argv(
            cand, session.n_ctx,
            binary_path=str(toolchain.binary_path),
            model_path=str(model.path),
            repetitions=reps,
            cpu_only=session.cpu_only,
            selected_device_ids=session.selected_device_ids,
        )
        entry: dict[str, object] = {
            "candidate": _candidate_to_dict(cand),
            "argv": argv,
            "repetition_count": reps,
        }
        reproduction_argv.append(entry)
    root["reproduction_argv"] = reproduction_argv
    root["benchmark_only_flags_note"] = (
        "The reproduction_argv entries are exact llama-bench invocations "
        "including `-o json` (machine-readable output) and `-r` (repetition "
        "count). These flags are retained in the reproduction argv for "
        "exactness but are benchmark-only — they are omitted from the "
        "compatible production llama-cli and llama-server commands below, "
        "which perform a single inference pass with default output formatting. "
        "No claim of 1:1 mapping to production usage is made."
    )
    if rep_unknown:
        root["reproduction_notes"] = (
            f"{len(rep_unknown)} candidate(s) (indices {rep_unknown}) lack "
            "recorded repetition count evidence from the available data model. "
            "The `-r` flag is omitted for those entries."
        )

    root["reproduction_commands"] = _build_reproduction_commands(
        session, screening_report, validation_result,
    )

    if include_hashes:
        root["hashes"] = _compute_hashes(
            session, toolchain, model, screening_report, validation_result,
        )
        root["caveat"] = _HASHES_CAVEAT

    return json.dumps(root, indent=2, ensure_ascii=False, default=str)


def _build_reproduction_commands(
    session: SessionSpecification,
    screening_report: ScreeningReport | None,
    validation_result: ValidatedRecommendation | NoWinnerReport | None,
    *,
    model_path_override: str | None = None,
) -> dict[str, object]:
    commands: dict[str, object] = {}
    target: CandidateConfig | None = None
    n_ctx = session.n_ctx

    if validation_result is not None and isinstance(validation_result, ValidatedRecommendation):
        target = validation_result.recommendation.candidate
    elif screening_report is not None and screening_report.leader is not None:
        target = screening_report.leader.candidate

    if target is not None:
        mp = model_path_override if model_path_override is not None else str(session.model.path)
        commands["llama_cli"] = build_llama_cli_command(
            target, n_ctx, mp,
            cpu_only=session.cpu_only,
            selected_device_ids=session.selected_device_ids,
        )
        commands["llama_server"] = build_llama_server_command(
            target, n_ctx, mp,
            cpu_only=session.cpu_only,
            selected_device_ids=session.selected_device_ids,
        )
        commands["preset_ini"] = build_preset_ini_section(
            target, n_ctx,
            cpu_only=session.cpu_only,
            selected_device_ids=session.selected_device_ids,
        )
    return commands


# ── Markdown export ───────────────────────────────────────────────


def _redact_for_share(obj: object) -> object:
    obj = redact_all_paths(obj)
    obj = redact_all_device_ids(obj)
    return obj


def _markdown_escape(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("_", "\\_").replace("*", "\\*")
    return text


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
    sep = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    divider = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data = "\n".join(
        "| " + " | ".join(
            (row[i] if i < len(row) else "").ljust(col_widths[i])
            for i in range(len(headers))
        ) + " |"
        for row in rows
    )
    return f"{sep}\n{divider}\n{data}"


def render_export_markdown(
    *,
    session: SessionSpecification,
    toolchain: ToolchainFingerprint,
    model: ModelFingerprint,
    screening_report: ScreeningReport | None = None,
    validation_result: ValidatedRecommendation | NoWinnerReport | None = None,
    provenance: LaunchProvenance | None = None,
    include_hashes: bool = False,
) -> str:
    lines: list[str] = []
    lines.append("# llama-parameter-scout Report")
    lines.append("")
    lines.append(
        "**Share-safe report** — paths are redacted and unique device "
        "identifiers are replaced."
    )
    lines.append("")
    lines.append(f"**Schema:** `{REPORT_SCHEMA_VERSION}`")
    lines.append(f"**Exported at:** `{_now_iso()}`")
    lines.append(f"**Result:** {_result_label(validation_result, screening_report)}")
    lines.append("")

    # Session overview
    lines.append("## Session")
    lines.append("")
    lines.append(f"- **Device:** {'CPU' if session.cpu_only else 'GPU'}")
    if not session.cpu_only and session.selected_device_ids:
        device_str = ", ".join(str(d) for d in session.selected_device_ids)
        lines.append(f"- **Selected device IDs:** {device_str}")
    lines.append(f"- **Context:** {session.n_ctx.total} tokens "
                 f"(prompt={session.n_ctx.prompt}, gen={session.n_ctx.generation}, "
                 f"depth={session.n_ctx.depth})")
    lines.append(f"- **Session fingerprint:** `{session.fingerprint}`")
    lines.append("")

    # Model info
    lines.append("## Model")
    lines.append("")
    m_path = redact_path(str(model.path))
    lines.append(f"- **Path:** `{m_path}`")
    lines.append(f"- **Architecture:** {model.architecture}")
    if model.parameter_count is not None:
        lines.append(f"- **Parameters:** {model.parameter_count:,}")
    if model.quantization is not None:
        lines.append(f"- **Quantization:** {model.quantization}")
    lines.append("")

    # Toolchain
    lines.append("## Toolchain")
    lines.append("")
    tc_path = redact_path(str(toolchain.binary_path))
    lines.append(f"- **Binary:** `{tc_path}`")
    lines.append(f"- **Version:** {toolchain.version}")
    lines.append(f"- **Supported flags:** `{' '.join(toolchain.supported_flags)}`")
    lines.append(f"- **Device capabilities:** `{' '.join(toolchain.device_capabilities)}`")
    lines.append("")

    # Provenance
    if provenance is not None:
        lines.append("## Launch Provenance")
        lines.append("")
        p_cwd = redact_path(provenance.cwd)
        lines.append(f"- **Working directory:** `{p_cwd}`")
        lines.append(f"- **Shell dialect:** {provenance.shell_dialect}")
        p_exe = redact_path(provenance.executable_path)
        lines.append(f"- **Executable:** `{p_exe}`")
        lines.append(f"- **Argv:** `{join_argv(list(provenance.argv), provenance.shell_dialect)}`")
        if provenance.environment:
            lines.append("- **Environment:**")
            for k, v in sorted(provenance.environment.items()):
                v_redacted = redact_device_id(redact_path(v))
                lines.append(f"  - `{k}={v_redacted}`")
        lines.append("")

    # Screening results
    if screening_report is not None:
        lines.append("## Screening Results")
        lines.append("")
        lines.append(f"- **Profile:** {screening_report.profile.label}")
        lines.append(f"- **Attempts used:** {screening_report.attempts_used}")
        lines.append(f"- **Objective:** {screening_report.objective}")
        disp_str = _markdown_escape(screening_report.disposition_summary)
        lines.append(f"- **Disposition summary:** {disp_str}")
        lines.append("")

        if screening_report.notes:
            lines.append("### Notes")
            for note in screening_report.notes:
                note_redacted = redact_device_id(redact_path(note))
                lines.append(f"- {_markdown_escape(note_redacted)}")
            lines.append("")

        if screening_report.survivors:
            lines.append("### Survivors (ranked)")
            lines.append("")
            headers = ["Rank", "Threads", "Batch", "UBatch", "nGL", "FA", "Cache K", "Cache V",
                        "PP Score", "TG Score", "Balanced"]
            rows: list[list[str]] = []
            for idx, rc in enumerate(screening_report.survivors, 1):
                c = rc.candidate
                rows.append([
                    str(idx),
                    str(c.threads),
                    str(c.batch_size),
                    str(c.ubatch_size),
                    str(c.n_gpu_layers),
                    "on" if c.flash_attn else "off",
                    c.cache_type_k,
                    c.cache_type_v,
                    f"{rc.pp_score:.4f}",
                    f"{rc.tg_score:.4f}",
                    f"{rc.balanced:.4f}",
                ])
            lines.append(_format_table(rows, headers))
            lines.append("")

        # Provisional leader
        if screening_report.leader is not None:
            pl = screening_report.leader
            lines.append("### Provisional Leader")
            lines.append("")
            lines.append(f"- **Label:** {pl.label}")
            lines.append(f"- **Candidate:** threads={pl.candidate.threads}, "
                         f"batch={pl.candidate.batch_size}, ubatch={pl.candidate.ubatch_size}, "
                         f"ngl={pl.candidate.n_gpu_layers}")
            lines.append(f"- **PP score:** {pl.pp_score:.4f}")
            lines.append(f"- **TG score:** {pl.tg_score:.4f}")
            lines.append(f"- **Balanced:** {pl.balanced:.4f}")
            lines.append("")

        # Reproduction commands — every observed candidate
        candidates_repro, repro_notes = _collect_reproduction_candidates(
            screening_report, validation_result,
        )
        if candidates_repro:
            dialect = provenance.shell_dialect if provenance is not None else "posix"
            m_binary_path = redact_path(str(toolchain.binary_path))
            m_model_path = redact_path(str(model.path))
            lines.append("## Reproduction Commands")
            lines.append("")
            for idx, (cand, reps) in enumerate(candidates_repro, 1):
                argv = build_reproduction_argv(
                    cand, session.n_ctx,
                    binary_path=m_binary_path,
                    model_path=m_model_path,
                    repetitions=reps,
                    cpu_only=session.cpu_only,
                    selected_device_ids=session.selected_device_ids,
                )
                lines.append(f"### Candidate {idx}")
                lines.append("")
                lines.append("**llama-bench:**")
                lines.append("")
                lines.append(f"```{dialect}")
                lines.append(join_argv(argv, dialect))
                lines.append("```")
                if reps is not None:
                    lines.append("")
                    lines.append(f"_Repetition count: {reps}_")
                else:
                    lines.append("")
                    lines.append("_Repetition count not available from the "
                                 "data model; `-r` omitted._")
                lines.append("")

            lines.append("> **Note:** The llama-bench commands above include "
                         "`-o json` (machine-readable output) and `-r` "
                         "(repetition count). These flags are benchmark-only "
                         "and are omitted from the compatible production "
                         "llama-cli and llama-server commands below.")
            lines.append("")

            # Compatible production commands only for leader/recommendation
            prod_cmds = _build_reproduction_commands(
                session, screening_report, validation_result,
                model_path_override=redact_path(str(session.model.path)),
            )
            cli_cmd = prod_cmds.get("llama_cli")
            if isinstance(cli_cmd, list):
                lines.append("**llama-cli (compatible):**")
                lines.append("")
                lines.append(f"```{dialect}")
                lines.append(join_argv(cli_cmd, dialect))
                lines.append("```")
                lines.append("")

            srv_cmd = prod_cmds.get("llama_server")
            if isinstance(srv_cmd, list):
                lines.append("**llama-server (compatible):**")
                lines.append("")
                lines.append(f"```{dialect}")
                lines.append(join_argv(srv_cmd, dialect))
                lines.append("```")
                lines.append("")

    # Validation results
    if validation_result is not None:
        lines.append("## Validation Results")
        lines.append("")
        if isinstance(validation_result, ValidatedRecommendation):
            vr = validation_result
            lines.append("- **Recommendation:** validated at requested context")
            vc = vr.recommendation
            lines.append(f"  - Candidate: threads={vc.candidate.threads}, "
                         f"batch={vc.candidate.batch_size}, ubatch={vc.candidate.ubatch_size}")
            lines.append(f"  - PP score: {vc.pp_score:.4f}, TG score: {vc.tg_score:.4f}")
            lines.append(f"  - Label: {vc.label}")
            lines.append("")
            if vr.alternatives:
                lines.append("### Alternatives")
                lines.append("")
                for a in vr.alternatives:
                    ac = a.candidate
                    lines.append(
                        f"- threads={ac.threads}, batch={ac.batch_size}, "
                        f"ubatch={ac.ubatch_size}"
                    )
                lines.append("")
        else:
            nw = validation_result
            lines.append("**No winner** — all candidates failed validation.")
            lines.append("")
            if nw.provisional_leader is not None:
                pl = nw.provisional_leader
                lines.append(f"- **Screening leader** (not validated): {pl.label}")
                lines.append(f"  - Candidate: threads={pl.candidate.threads}, "
                             f"batch={pl.candidate.batch_size}")
                lines.append("")
            lines.append("### Validation Failures")
            lines.append("")
            for fv in nw.failures:
                lines.append(f"- Candidate threads={fv.candidate.threads}, "
                             f"batch={fv.candidate.batch_size}: "
                             f"[{fv.disposition}] {fv.reason}")
            lines.append("")

    # Preset.ini (recommended candidate)
    preset_target: CandidateConfig | None = None
    if validation_result is not None and isinstance(validation_result, ValidatedRecommendation):
        preset_target = validation_result.recommendation.candidate
    elif screening_report is not None and screening_report.leader is not None:
        preset_target = screening_report.leader.candidate
    if preset_target is not None:
        preset = build_preset_ini_section(
            preset_target, session.n_ctx,
            cpu_only=session.cpu_only,
            selected_device_ids=session.selected_device_ids,
        )
        lines.append("## Preset.ini (recommended)")
        lines.append("")
        lines.append("```ini")
        lines.append(preset)
        lines.append("```")
        lines.append("")

    # Hashes
    if include_hashes:
        hashes = _compute_hashes(
            session, toolchain, model, screening_report, validation_result,
        )
        lines.append("## Content Hashes")
        lines.append("")
        for key, val in hashes.items():
            lines.append(f"- **{key}:** `{val}`")
        lines.append("")
        lines.append(f"> **Caveat:** {_HASHES_CAVEAT}")
        lines.append("")

    return "\n".join(lines)
