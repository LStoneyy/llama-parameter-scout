from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from llama_parameter_scout.application import (
    CURRENT_SCHEMA_VERSION,
    Application,
    _compare_fingerprint_components,
    _compute_full_fingerprint_components,
    list_session_summaries,
)
from llama_parameter_scout.journal import (
    _atomic_write,
    _read_journal,
    list_sessions,
    read_journal_events,
    write_measurement_completed,
    write_observation,
    write_search_decision,
    write_session_start,
)
from llama_parameter_scout.models import (
    Accelerator,
    BenchmarkObservation,
    CandidateConfig,
    JournalError,
    MemoryEvidence,
    ModelFingerprint,
    NctxComposition,
    ResourceBudget,
    ScreeningObservation,
    ScreeningResult,
    SearchProfile,
    SessionSpecification,
    ShardFingerprint,
    SystemTopology,
    ToolchainFingerprint,
)
from llama_parameter_scout.runner import FakeRunner, ProcessResult

# ── Helpers ──────────────────────────────────────────────────────

_SAMPLE_TOPOLOGY = SystemTopology(
    cpu_count=8, system_memory_bytes=32_000_000_000,
    accelerators=(), memory_unified=True,
)
_SAMPLE_NCTX = NctxComposition(total=2048, prompt=512, generation=1024, depth=512)
_SAMPLE_CANDIDATE = CandidateConfig(
    threads=4, batch_size=512, ubatch_size=512,
    n_gpu_layers=0, flash_attn=False,
    cache_type_k="f16", cache_type_v="f16",
)


def _make_model(path: str | None = None) -> ModelFingerprint:
    return ModelFingerprint(
        path=Path(path or "/tmp/model.gguf"),
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=4096,
        quantization="Q4_K_M",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )


def _make_toolchain() -> ToolchainFingerprint:
    return ToolchainFingerprint(
        binary_path=Path("/usr/local/bin/llama-bench"),
        version="1234",
        supported_flags=("-m", "-p", "-n"),
        supported_output_formats=("json",),
        device_capabilities=("cpu",),
    )


def _make_advisory() -> Any:
    from llama_parameter_scout.models import ContextAdvisory
    return ContextAdvisory(
        likely_fit_lower=1000, likely_fit_upper=2000,
        uncertain_maximum=3000, model_weight_bytes=4_000_000_000,
        kv_cache_bytes_estimate=500_000_000,
        training_context_status="known", training_context_tokens=4096,
        note="Test advisory",
    )


def _make_spec(model: ModelFingerprint | None = None) -> SessionSpecification:
    m = model or _make_model()
    budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)
    return SessionSpecification(
        topology=_SAMPLE_TOPOLOGY, cpu_only=True,
        selected_device_ids=(),
        n_ctx=_SAMPLE_NCTX,
        budget=budget,
        advisory=_make_advisory(),
        model=m, acknowledged=True, fingerprint="x",
    )


def _make_observation(path: Path | None = None) -> BenchmarkObservation:
    return BenchmarkObservation(
        candidate=_SAMPLE_CANDIDATE,
        total_ctx=2048,
        prompt_ctx=512,
        generation_ctx=1024,
        depth_ctx=512,
        samples_tps=(45.5, 44.8, 45.3),
        median_tps=45.3,
        dispersion_tps=0.286,
        binary_path=Path("/usr/local/bin/llama-bench"),
        binary_source="explicit",
        model_path=Path(str(path) if path else "/tmp/model.gguf"),
        toolchain=_make_toolchain(),
        model=_make_model(),
    )


def _enqueue_toolchain_std(runner: FakeRunner, bp: Path) -> None:
    runner.enqueue(
        (str(bp), "--version"),
        ProcessResult(returncode=0, stdout=(
            "version: test123\nbackends: cpu\n"
            "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
            "supported_output_formats: json\n"
        ), stderr=""),
    )
    runner.enqueue(
        (str(bp), "--help"),
        ProcessResult(returncode=0, stdout="usage: llama-bench\n", stderr=""),
    )


# ── Cycle 1: v5 journal envelope ────────────────────────────────


class TestV5JournalEnvelope:
    def test_writes_v5_envelope(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation()
        write_observation(path, obs)
        data = json.loads(path.read_text())
        assert data["schema_version"] == 5
        assert "events" in data
        assert isinstance(data["events"], list)
        assert len(data["events"]) == 1

    def test_event_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        write_observation(path, _make_observation())
        ev = json.loads(path.read_text())["events"][0]
        assert ev["event_type"] == "observation_completed"
        assert ev["event_version"] == 1
        assert ev["session_id"] != ""
        assert ev["timestamp"] != ""

    def test_append_preserves_prior_events(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs1 = _make_observation()
        obs2 = _make_observation()
        write_observation(path, obs1)
        write_observation(path, obs2)
        events = read_journal_events(path)
        assert len(events) == 2
        assert events[0]["event_type"] == "observation_completed"
        assert events[1]["event_type"] == "observation_completed"

    def test_interrupted_append_leaves_old_journal(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        write_observation(path, _make_observation())
        old_content = path.read_text()
        import tempfile
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
        )
        tmp_p = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w") as f:
                f.write("{corrupted")
        finally:
            if tmp_p.exists():
                tmp_p.unlink()
        assert path.exists()
        assert path.read_text() == old_content

    def test_no_temp_residue(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        write_observation(path, _make_observation())
        temps = list(tmp_path.glob(".*.json.*.tmp"))
        assert len(temps) == 0


# ── Cycle 2: session_start events and fingerprint components ────


class TestSessionStartEvents:
    def test_session_start_has_components(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-1", None, components, digest)
        events = read_journal_events(path)
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "session_start"
        assert ev["fingerprint_digest"] == digest
        assert ev["fingerprint_components"]["main_model"]["architecture"] == "llama"
        assert ev["fingerprint_components"]["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_fingerprint_components_cover_families(self, tmp_path: Path) -> None:
        spec = _make_spec()
        tc = _make_toolchain()
        components, digest = _compute_full_fingerprint_components(
            spec, spec.model, candidate=_SAMPLE_CANDIDATE,
            profile=None, objective="balanced", toolchain=tc,
        )
        families = [
            "schema_version", "main_model", "draft_model", "toolchain",
            "topology", "device_selection", "context", "objective",
            "environment", "budget", "candidate",
        ]
        for f in families:
            assert f in components, f"Missing fingerprint family: {f}"


# ── Cycle 3: exact match resume ─────────────────────────────────


class TestExactResume:
    def test_resume_returns_observation(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation(path=tmp_path / "model.gguf")
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-1", None, components, digest)
        write_observation(path, obs, session_id="session-1")

        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model)
        assert result.action == "resumed"
        assert result.observation is not None
        assert result.observation.median_tps == pytest.approx(45.3, rel=1e-6)

    def test_resume_skips_measurement(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation(path=tmp_path / "model.gguf")
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-1", None, components, digest)
        write_observation(path, obs, session_id="session-1")

        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model)
        assert result.action == "resumed"

    def test_exact_resume_does_not_add_event(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = _make_observation(path=tmp_path / "model.gguf")
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-1", None, components, digest)
        write_observation(path, obs, session_id="session-1")
        prior_count = len(read_journal_events(path))

        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model)
        assert result.action == "resumed"
        assert result.session_id == "session-1"
        assert len(read_journal_events(path)) == prior_count

    def test_no_journal_creates_new(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model)
        assert result.action == "new"
        assert result.session_id != ""
        assert result.observation is None
        events = read_journal_events(path)
        assert len(events) == 1
        assert events[0]["event_type"] == "session_start"


# ── Cycle 4: derivation on mismatch ─────────────────────────────


class TestDerivation:
    def test_derive_on_toolchain_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        tc1 = _make_toolchain()
        tc2 = ToolchainFingerprint(
            binary_path=Path("/usr/local/bin/llama-bench"),
            version="9999",
            supported_flags=("-m",),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
        )
        components1, _ = _compute_full_fingerprint_components(
            spec, spec.model, toolchain=tc1,
        )
        write_session_start(path, spec, "session-1", None, components1, "digest1")
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-1",
        )

        app = Application(FakeRunner())
        result = app.start_or_resume_session(
            path, spec, spec.model, toolchain=tc2,
        )
        assert result.action == "derived"
        assert "toolchain" in (result.mismatch_reason or "")

    def test_derive_on_model_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec1 = _make_spec(_make_model())
        m2 = _make_model()
        m2 = ModelFingerprint(
            path=m2.path, architecture="deepseek",
            parameter_count=m2.parameter_count,
            context_length=m2.context_length,
            quantization=m2.quantization,
            shards=m2.shards,
            symlink_warning=m2.symlink_warning,
            network_mount_detected=m2.network_mount_detected,
            diagnostics=m2.diagnostics,
        )
        components1, _ = _compute_full_fingerprint_components(spec1, spec1.model)
        write_session_start(path, spec1, "session-1", None, components1, "digest1")
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-1",
        )

        spec2 = _make_spec(m2)
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec2, m2)
        assert result.action == "derived"
        assert "model" in (result.mismatch_reason or "")

    def test_derive_on_topology_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec1 = _make_spec()
        spec2 = SessionSpecification(
            topology=SystemTopology(
                cpu_count=16, system_memory_bytes=64_000_000_000,
                accelerators=(), memory_unified=True,
            ),
            cpu_only=True, selected_device_ids=(),
            n_ctx=_SAMPLE_NCTX,
            budget=ResourceBudget(ram_bytes=64_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=_make_advisory(),
            model=_make_model(), acknowledged=True, fingerprint="y",
        )
        components1, _ = _compute_full_fingerprint_components(spec1, spec1.model)
        write_session_start(path, spec1, "session-1", None, components1, "digest1")
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-1",
        )

        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec2, spec2.model)
        assert result.action == "derived"
        assert "topology" in (result.mismatch_reason or "")

    def test_derive_on_budget_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec1 = _make_spec()
        spec2 = SessionSpecification(
            topology=_SAMPLE_TOPOLOGY, cpu_only=True, selected_device_ids=(),
            n_ctx=_SAMPLE_NCTX,
            budget=ResourceBudget(ram_bytes=8_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=_make_advisory(),
            model=_make_model(), acknowledged=True, fingerprint="z",
        )
        components1, _ = _compute_full_fingerprint_components(spec1, spec1.model)
        write_session_start(path, spec1, "session-1", None, components1, "digest1")
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-1",
        )

        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec2, spec2.model)
        assert result.action == "derived"
        assert "budget" in (result.mismatch_reason or "")

    def test_derive_on_context_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec1 = _make_spec()
        spec2 = SessionSpecification(
            topology=_SAMPLE_TOPOLOGY, cpu_only=True, selected_device_ids=(),
            n_ctx=NctxComposition(total=4096, prompt=1024, generation=2048, depth=1024),
            budget=ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=_make_advisory(), model=_make_model(), acknowledged=True, fingerprint="z",
        )
        components1, _ = _compute_full_fingerprint_components(spec1, spec1.model)
        write_session_start(path, spec1, "session-1", None, components1, "digest1")
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-1",
        )

        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec2, spec2.model)
        assert result.action == "derived"
        assert "context" in (result.mismatch_reason or "")

    def test_derive_on_objective_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        components1, _ = _compute_full_fingerprint_components(
            spec, spec.model, objective="balanced",
        )
        write_session_start(path, spec, "session-1", None, components1, "digest1")
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-1",
        )

        app = Application(FakeRunner())
        result = app.start_or_resume_session(
            path, spec, spec.model, objective="memory",
        )
        assert result.action == "derived"
        assert "objective" in (result.mismatch_reason or "")

    def test_derive_on_candidate_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        c1 = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        c2 = CandidateConfig(
            threads=8, batch_size=1024, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
        )
        components1, _ = _compute_full_fingerprint_components(
            spec, spec.model, candidate=c1,
        )
        write_session_start(path, spec, "session-1", None, components1, "digest1")
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-1",
        )

        app = Application(FakeRunner())
        result = app.start_or_resume_session(
            path, spec, spec.model, candidate=c2,
        )
        assert result.action == "derived"
        assert "candidate" in (result.mismatch_reason or "")

    def test_derive_on_device_selection_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        topology = SystemTopology(
            cpu_count=8, system_memory_bytes=32_000_000_000,
            accelerators=(
                Accelerator("gpu", "cuda", 0, "RTX 4090", memory_bytes=24_000_000_000),
            ),
            memory_unified=False,
        )
        spec1 = SessionSpecification(
            topology=topology, cpu_only=True, selected_device_ids=(),
            n_ctx=_SAMPLE_NCTX,
            budget=ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=_make_advisory(), model=_make_model(), acknowledged=True, fingerprint="x",
        )
        components1, _ = _compute_full_fingerprint_components(spec1, spec1.model)
        write_session_start(path, spec1, "session-1", None, components1, "digest1")

        spec2 = SessionSpecification(
            topology=topology, cpu_only=False, selected_device_ids=(0,),
            n_ctx=_SAMPLE_NCTX,
            budget=ResourceBudget(
                ram_bytes=32_000_000_000, vram_bytes=24_000_000_000,
                headroom_ratio=0.1,
            ),
            advisory=_make_advisory(), model=_make_model(), acknowledged=True, fingerprint="y",
        )
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec2, spec2.model)
        assert result.action == "derived"
        assert "device" in (result.mismatch_reason or "")

    def test_derive_fresh_evidence(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        components1, _ = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-1", None, components1, "digest1")
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-1",
        )

        m2 = _make_model()
        m2 = ModelFingerprint(
            path=m2.path, architecture="deepseek",
            parameter_count=m2.parameter_count,
            context_length=m2.context_length,
            quantization=m2.quantization,
            shards=m2.shards,
            symlink_warning=m2.symlink_warning,
            network_mount_detected=m2.network_mount_detected,
            diagnostics=m2.diagnostics,
        )
        spec2 = _make_spec(m2)
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec2, m2)
        assert result.action == "derived"
        assert result.observation is None


# ── Cycle 5: v4 legacy migration ────────────────────────────────


class TestV4Migration:
    def test_v4_read_as_historical(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        v4 = {"version": 4, "disposition": "completed", "candidate": {"threads": 4}}
        path.write_text(json.dumps(v4))
        journal = _read_journal(path)
        assert journal is not None
        assert journal["schema_version"] == 5
        assert len(journal["events"]) == 1
        assert journal["events"][0]["session_id"] == "v4-legacy"

    def test_v4_derive_creates_new_session(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        v4 = {"version": 4, "disposition": "completed", "candidate": {"threads": 4}}
        path.write_text(json.dumps(v4))

        spec = _make_spec()
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model)
        assert result.action == "derived"
        assert result.parent_session_id == "v4-legacy"

    def test_v4_content_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        v4 = {"version": 4, "disposition": "completed", "candidate": {"threads": 4}}
        path.write_text(json.dumps(v4))

        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "s2", "v4-legacy", components, digest)

        journal = _read_journal(path)
        assert journal is not None
        events = journal["events"]
        assert len(events) == 2
        assert events[0]["session_id"] == "v4-legacy"

    def test_v4_listed_as_historical(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        v4 = {"version": 4, "disposition": "completed", "candidate": {"threads": 4}}
        path.write_text(json.dumps(v4))
        summaries = list_sessions(path)
        v4_entries = [s for s in summaries if s.get("session_id") == "v4-legacy"]
        assert len(v4_entries) == 1
        assert v4_entries[0]["status"] == "historical"
        reason: str = str(v4_entries[0].get("reason", ""))
        assert "legacy" in reason.lower()


# ── Cycle 6: search decision journaling ─────────────────────────


class TestSearchDecisionJournaling:
    def test_search_decision_written(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        write_search_decision(
            path, candidate={"threads": 4}, disposition="pruned",
            reason="incompatible cache", session_id="session-1",
        )
        events = read_journal_events(path)
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "search_decision"
        assert ev["disposition"] == "pruned"
        assert ev["reason"] == "incompatible cache"

    def test_search_decisions_from_screening(self, tmp_path: Path) -> None:
        from llama_parameter_scout.application import _journal_search_decisions
        from llama_parameter_scout.models import ScreeningReport, ScreeningResult
        from llama_parameter_scout.search import QUICK_PROFILE

        profile = QUICK_PROFILE
        baseline = ScreeningResult(
            candidate=_SAMPLE_CANDIDATE, success=True, disposition="survived",
        )
        report = ScreeningReport(
            profile=profile,
            baseline_disposition=baseline,
            grid_dispositions=(),
        )
        _journal_search_decisions(tmp_path / "j.json", report, "session-z")
        events = read_journal_events(tmp_path / "j.json")
        assert len(events) == 1
        assert events[0]["event_type"] == "search_decision"


# ── Cycle 7: prior history retention ────────────────────────────


class TestHistoryRetention:
    def test_multiple_sessions_retained(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()

        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "s1", None, components, digest)
        write_observation(path, _make_observation(path=tmp_path / "model.gguf"), session_id="s1")

        m2 = _make_model()
        m2 = ModelFingerprint(
            path=m2.path, architecture="deepseek",
            parameter_count=m2.parameter_count,
            context_length=m2.context_length,
            quantization=m2.quantization,
            shards=m2.shards,
            symlink_warning=m2.symlink_warning,
            network_mount_detected=m2.network_mount_detected,
            diagnostics=m2.diagnostics,
        )
        spec2 = _make_spec(m2)
        components2, _ = _compute_full_fingerprint_components(spec2, m2)
        write_session_start(path, spec2, "s2", "s1", components2, digest)

        events = read_journal_events(path)
        assert len(events) == 3
        summaries = list_session_summaries(path)
        sids = [s["session_id"] for s in summaries]
        assert "s1" in sids
        assert "s2" in sids


# ── Cycle 8: CLI session listing ────────────────────────────────


class TestCliListSessions:
    def test_empty_journal_shows_none(self, tmp_path: Path, capsys: Any) -> None:
        from llama_parameter_scout.cli import main as cli_main
        path = tmp_path / "journal.json"
        path.write_text("{}")
        with pytest.raises(SystemExit):
            cli_main(["--list-sessions", "--journal-path", str(path)])
        captured = capsys.readouterr()
        assert "No sessions found" in captured.out

    def test_lists_sessions_with_status(self, tmp_path: Path, capsys: Any) -> None:
        from llama_parameter_scout.cli import main as cli_main
        path = tmp_path / "journal.json"
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-X", None, components, digest)
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-X",
        )
        with pytest.raises(SystemExit):
            cli_main(["--list-sessions", "--journal-path", str(path)])
        captured = capsys.readouterr()
        assert "session-X" in captured.out
        assert "resumable" in captured.out

    def test_reports_historical_v4(self, tmp_path: Path, capsys: Any) -> None:
        from llama_parameter_scout.cli import main as cli_main
        path = tmp_path / "journal.json"
        v4 = {"version": 4, "disposition": "completed"}
        path.write_text(json.dumps(v4))
        with pytest.raises(SystemExit):
            cli_main(["--list-sessions", "--journal-path", str(path)])
        captured = capsys.readouterr()
        assert "v4-legacy" in captured.out
        assert "historical" in captured.out

    def test_list_sessions_no_journal_path_fails(self, capsys: Any) -> None:
        from llama_parameter_scout.cli import main as cli_main
        with pytest.raises(SystemExit):
            cli_main(["--list-sessions"])


# ── Cycle 9: interrupted/cancelled atomic append ────────────────


class TestAtomicAppendFault:
    def test_fault_during_append_preserves_original(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        write_observation(path, _make_observation(path=tmp_path / "model.gguf"))
        original = path.read_text()


        class FaultyWriter:
            def __init__(self) -> None:
                self.called = False
            def __call__(self, *args: object, **kwargs: object) -> None:
                self.called = True
                raise OSError("write failure")

        original_aw = _atomic_write
        try:
            import llama_parameter_scout.journal as jmod
            jmod._atomic_write = FaultyWriter()
            with pytest.raises(OSError):
                write_observation(path, _make_observation(path=tmp_path / "model.gguf"))
        finally:
            jmod._atomic_write = original_aw

        assert path.read_text() == original

    def test_append_does_not_leave_temp_files(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        write_observation(path, _make_observation(path=tmp_path / "model.gguf"))
        temps = list(tmp_path.glob(".*.json.*.tmp"))
        assert len(temps) == 0


# ── Cycle 10: fingerprint comparison ────────────────────────────


class TestFingerprintComparison:
    def test_compare_identical_returns_none(self) -> None:
        obj = {"a": 1, "b": 2}
        assert _compare_fingerprint_components(obj, obj) is None

    def test_compare_different_returns_reason(self) -> None:
        stored = {"schema_version": 5, "main_model": {"architecture": "llama"}}
        current = {"schema_version": 5, "main_model": {"architecture": "deepseek"}}
        reason = _compare_fingerprint_components(stored, current)
        assert reason is not None
        assert "model" in reason

    def test_compare_schema_version_mismatch(self) -> None:
        stored = {"schema_version": 4}
        current = {"schema_version": 5}
        reason = _compare_fingerprint_components(stored, current)
        assert reason is not None
        assert "schema" in reason


# ── Cycle 11: toolchain supported_output_formats in fingerprint ──


class TestToolchainFingerprintOutputFormats:
    def test_toolchain_components_include_output_formats(self) -> None:
        tc = _make_toolchain()
        spec = _make_spec()
        components, _ = _compute_full_fingerprint_components(
            spec, spec.model, toolchain=tc,
        )
        tcc = components.get("toolchain", {})
        assert "supported_output_formats" in tcc
        assert tcc["supported_output_formats"] == ["json"]


# ── Cycle 12: derivation reason persistence and listing ──────────


class TestDerivationReasonPersistence:
    def test_derived_session_start_contains_reason(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        tc1 = _make_toolchain()
        tc2 = ToolchainFingerprint(
            binary_path=Path("/usr/local/bin/llama-bench"),
            version="9999",
            supported_flags=("-m",),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
        )
        components1, _ = _compute_full_fingerprint_components(
            spec, spec.model, toolchain=tc1,
        )
        write_session_start(path, spec, "s1", None, components1, "d1")
        write_observation(path, _make_observation(path=tmp_path / "model.gguf"), session_id="s1")
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model, toolchain=tc2)
        assert result.action == "derived"
        assert result.mismatch_reason is not None
        events = read_journal_events(path)
        session_start = [
            e for e in events
            if e.get("event_type") == "session_start"
            and e.get("session_id") == result.session_id
        ]
        assert len(session_start) == 1
        assert session_start[0].get("derivation_reason") == result.mismatch_reason

    def test_list_sessions_shows_derived_with_reason(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        tc1 = _make_toolchain()
        tc2 = ToolchainFingerprint(
            binary_path=Path("/usr/local/bin/llama-bench"),
            version="9999",
            supported_flags=("-m",),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
        )
        components1, _ = _compute_full_fingerprint_components(spec, spec.model, toolchain=tc1)
        write_session_start(path, spec, "s1", None, components1, "d1")
        write_observation(path, _make_observation(path=tmp_path / "model.gguf"), session_id="s1")
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model, toolchain=tc2)
        assert result.action == "derived"
        sessions = list_sessions(path)
        derived = [s for s in sessions if s.get("session_id") == result.session_id]
        assert len(derived) == 1
        assert derived[0]["status"] == "derived"
        assert derived[0]["reason"]

    def test_list_sessions_shows_resumable_for_exact_match(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "s1", None, components, digest)
        write_observation(path, _make_observation(path=tmp_path / "model.gguf"), session_id="s1")
        sessions = list_sessions(path)
        s1 = [s for s in sessions if s.get("session_id") == "s1"]
        assert len(s1) == 1
        assert s1[0]["status"] == "resumable"


# ── Cycle 13: unknown schema version preservation ────────────────


class TestUnknownSchemaPreservation:
    def test_v6_journal_preserved_as_incompatible(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        v6 = {"schema_version": 6, "custom_data": "test_value"}
        path.write_text(json.dumps(v6))
        journal = _read_journal(path)
        assert journal is not None
        assert journal["schema_version"] == 5
        assert len(journal["events"]) == 1
        ev = journal["events"][0]
        assert ev["session_id"] == "v6-incompatible"
        assert ev["event_type"] == "incompatible_schema"
        assert "reason" in ev
        assert "Incompatible schema" in str(ev["reason"])
        assert ev["data"]["custom_data"] == "test_value"

    def test_v6_derive_creates_new_session(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        v6 = {"schema_version": 6, "custom_data": "test_value"}
        path.write_text(json.dumps(v6))
        spec = _make_spec()
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model)
        assert result.action == "derived"
        assert result.parent_session_id == "v6-incompatible"

    def test_v6_original_payload_embedded_after_write(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        v6 = {"schema_version": 6, "custom_data": "test_value"}
        path.write_text(json.dumps(v6))
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "s2", "v6-incompatible", components, digest)
        journal = _read_journal(path)
        assert journal is not None
        events = journal["events"]
        assert len(events) == 2
        v6_event = events[0]
        assert v6_event["session_id"] == "v6-incompatible"
        assert v6_event["data"]["custom_data"] == "test_value"

    def test_v6_data_not_reused_as_resume(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        v6 = {"schema_version": 6, "custom_data": "test_value"}
        path.write_text(json.dumps(v6))
        spec = _make_spec()
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model)
        assert result.action == "derived"
        assert result.observation is None


# ── Cycle 14: corrupt/empty journal handling ─────────────────────


class TestCorruptJournalHandling:
    def test_empty_file_raises_journal_error(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        path.write_text("")
        with pytest.raises(JournalError, match="empty"):
            _read_journal(path)

    def test_corrupt_json_raises_journal_error(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        path.write_text("{corrupt json")
        with pytest.raises(JournalError, match="invalid JSON"):
            _read_journal(path)

    def test_corrupt_file_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        path.write_text("{corrupt json")
        try:
            _read_journal(path)
        except JournalError:
            pass
        assert path.read_text() == "{corrupt json"

    def test_cli_list_sessions_reports_corrupt(self, tmp_path: Path, capsys: Any) -> None:
        from llama_parameter_scout.cli import main as cli_main
        path = tmp_path / "journal.json"
        path.write_text("{corrupt json")
        with pytest.raises(SystemExit):
            cli_main(["--list-sessions", "--journal-path", str(path)])
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_empty_json_object_is_ok(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        path.write_text("{}")
        journal = _read_journal(path)
        assert journal is not None
        assert journal["schema_version"] == 5
        assert len(journal["events"]) == 0


# ── Cycle 15: toolchain change produces derivation + fresh exec ──


class TestToolchainChangeProducesDerivation:
    def test_changed_toolchain_no_stale_resume(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        tc1 = _make_toolchain()
        tc2 = ToolchainFingerprint(
            binary_path=Path("/usr/local/bin/llama-bench"),
            version="9999",
            supported_flags=("-m",),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
        )
        components1, _ = _compute_full_fingerprint_components(
            spec, spec.model, toolchain=tc1,
        )
        write_session_start(path, spec, "s1", None, components1, "d1")
        write_observation(path, _make_observation(path=tmp_path / "model.gguf"), session_id="s1")
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model, toolchain=tc2)
        assert result.action == "derived"
        assert result.session_id != "s1"
        assert result.observation is None
        sessions = list_sessions(path)
        s1_entry = [s for s in sessions if s.get("session_id") == "s1"]
        assert len(s1_entry) == 1
        assert s1_entry[0]["status"] == "resumable"


# ── Cycle 16: incomplete session resume (Gap 1) ─────────────────

SCREENING_PROFILE = SearchProfile(
    label="test", max_attempts=8, min_repetitions=3,
    max_repetitions_per_candidate=5, tie_bound_coefficient=0.05,
    dispersion_threshold=0.15,
)


class TestIncompleteSessionResume:
    def test_resume_incomplete_executes_process(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-1", None, components, digest)

        prior_count = len(read_journal_events(path))
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec, spec.model)
        assert result.action == "resumed"
        assert result.session_id == "session-1"
        assert result.observation is None

        assert len(read_journal_events(path)) == prior_count


# ── Cycle 17: immediate per-decision journal (Gap 2) ────────────


class TestImmediatePerDecisionJournal:
    def test_measurement_completed_written(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        result = ScreeningResult(
            candidate=_SAMPLE_CANDIDATE, success=True, disposition="survived",
        )
        write_measurement_completed(path, result, "session-x")
        events = read_journal_events(path)
        assert len(events) == 1
        ev = events[0]
        assert ev["event_type"] == "measurement_completed"
        assert ev["screening_result"]["disposition"] == "survived"

    def test_immediate_journal_survives_evaluator_raise(self, tmp_path: Path) -> None:
        binary = tmp_path / "llama-bench"
        binary.write_bytes(b"#!/bin/sh\necho ok\n")
        binary.chmod(0o755)
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF")
        jpath = tmp_path / "journal.json"
        app = Application(FakeRunner())

        call_count: list[int] = [0]

        def failing_evaluator(
            candidate: CandidateConfig,
            n_ctx: NctxComposition,
            reps: int,
        ) -> ScreeningResult:
            call_count[0] += 1
            if call_count[0] == 1:
                return ScreeningResult(
                    candidate=candidate, success=True, disposition="survived",
                    observation=ScreeningObservation(
                        candidate=candidate,
                        pp_samples=(45.0, 44.0, 43.0),
                        tg_samples=(46.0, 45.0, 44.0),
                    ),
                )
            raise RuntimeError("injected evaluator failure after first call")

        request = type("Req", (), {
            "binary_path": binary,
            "model_path": model_file,
            "journal_path": jpath,
            "n_ctx": NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            "candidate": _SAMPLE_CANDIDATE,
            "binary_source": "explicit",
        })()

        with pytest.raises(RuntimeError):
            app.screen_candidates(
                request, SCREENING_PROFILE,
                evaluator=failing_evaluator,  # type: ignore[arg-type]
                session_id="session-s",
            )

        events = read_journal_events(jpath)
        measurement_events = [
            e for e in events
            if e.get("event_type") == "measurement_completed"
        ]
        assert len(measurement_events) >= 1


# ── Cycle 18: ScreeningResult observation round-trip (Gap 3) ────


class TestScreeningResultRoundTrip:
    def test_measurement_completed_contains_screening_result(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        obs = ScreeningObservation(
            candidate=_SAMPLE_CANDIDATE,
            pp_samples=(45.0, 44.0, 43.0),
            tg_samples=(46.0, 45.0, 44.0),
            memory_bytes=8_000_000_000,
            memory=MemoryEvidence(ram_bytes=8_000_000_000, quality="complete"),
            observed_prompt=128, observed_generation=256,
            observed_depth=128, observed_total=512,
        )
        result = ScreeningResult(
            candidate=_SAMPLE_CANDIDATE, success=True, disposition="survived",
            observation=obs, provenance="test",
        )
        write_measurement_completed(path, result, "session-x")
        events = read_journal_events(path)
        sr_data = events[0].get("screening_result", {})
        assert sr_data.get("version") == 1
        assert sr_data.get("success") is True
        assert sr_data.get("disposition") == "survived"
        assert sr_data.get("provenance") == "test"
        obs_data = sr_data.get("observation", {})
        assert obs_data.get("pp_samples") == [45.0, 44.0, 43.0]
        assert obs_data.get("observed_prompt") == 128
        mem_data = obs_data.get("memory", {})
        assert mem_data.get("ram_bytes") == 8_000_000_000
        assert mem_data.get("quality") == "complete"

    def test_exact_resume_exposes_completed_search_results(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-1", None, components, digest)

        result = ScreeningResult(
            candidate=_SAMPLE_CANDIDATE, success=True, disposition="survived",
            observation=ScreeningObservation(
                candidate=_SAMPLE_CANDIDATE,
                pp_samples=(45.0, 44.0, 43.0),
                tg_samples=(46.0, 45.0, 44.0),
            ),
        )
        write_measurement_completed(path, result, "session-1")

        app = Application(FakeRunner())
        res = app.start_or_resume_session(path, spec, spec.model)
        assert res.action == "resumed"
        assert len(res.completed_search_results) == 1
        sr = res.completed_search_results[0]
        assert sr.candidate == _SAMPLE_CANDIDATE
        assert sr.success is True
        assert sr.observation is not None
        assert sr.observation.pp_samples == (45.0, 44.0, 43.0)

    def test_derived_session_no_completed_search_results(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec1 = _make_spec()
        components1, _ = _compute_full_fingerprint_components(spec1, spec1.model)
        write_session_start(path, spec1, "session-1", None, components1, "d1")
        write_observation(
            path, _make_observation(path=tmp_path / "model.gguf"),
            session_id="session-1",
        )

        m2 = ModelFingerprint(
            path=Path("/tmp/model2.gguf"), architecture="deepseek",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        spec2 = _make_spec(m2)
        app = Application(FakeRunner())
        result = app.start_or_resume_session(path, spec2, m2)
        assert result.action == "derived"
        assert result.completed_search_results == ()


# ── Cycle 19: search result prior reuse (Gap 4) ─────────────────


class TestSearchResultPriorReuse:
    def _make_request(self, tmp_path: Path, suffix: str = "") -> Any:
        binary = tmp_path / f"llama-bench{suffix}"
        binary.write_bytes(b"#!/bin/sh\necho ok\n")
        binary.chmod(0o755)
        model_file = tmp_path / f"model{suffix}.gguf"
        model_file.write_bytes(b"GGUF")
        jpath = tmp_path / f"journal{suffix}.json"
        return type("Req", (), {
            "binary_path": binary,
            "model_path": model_file,
            "journal_path": jpath,
            "n_ctx": NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            "candidate": _SAMPLE_CANDIDATE,
            "binary_source": "explicit",
        })()

    def test_exact_resume_reuses_completed_results(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-1", None, components, digest)

        prior_result = ScreeningResult(
            candidate=_SAMPLE_CANDIDATE, success=True, disposition="survived",
            observation=ScreeningObservation(
                candidate=_SAMPLE_CANDIDATE,
                pp_samples=(45.0, 44.0, 43.0),
                tg_samples=(46.0, 45.0, 44.0),
            ),
        )
        write_measurement_completed(path, prior_result, "session-1")

        app = Application(FakeRunner())
        resume = app.start_or_resume_session(path, spec, spec.model)
        assert resume.action == "resumed"
        assert len(resume.completed_search_results) == 1

        call_count: list[int] = [0]

        def counting_evaluator(
            candidate: CandidateConfig,
            n_ctx: NctxComposition,
            reps: int,
        ) -> ScreeningResult:
            call_count[0] += 1
            return ScreeningResult(
                candidate=candidate, success=True, disposition="survived",
                observation=ScreeningObservation(
                    candidate=candidate,
                    pp_samples=(50.0, 49.0, 48.0),
                    tg_samples=(51.0, 50.0, 49.0),
                ),
            )

        request = self._make_request(tmp_path, "_reuse")
        report = app.screen_candidates(
            request, SCREENING_PROFILE,
            evaluator=counting_evaluator,  # type: ignore[arg-type]
            completed_search_results=resume.completed_search_results,
        )
        without_prior = 8
        with_prior = without_prior - 1
        assert call_count[0] == with_prior, (
            f"Expected {with_prior} calls (baseline reused from prior), "
            f"got {call_count[0]}"
        )
        assert report.baseline_disposition.success is True
        assert report.baseline_disposition.candidate == _SAMPLE_CANDIDATE

    def test_no_prior_results_invoke_evaluator(self, tmp_path: Path) -> None:
        path = tmp_path / "journal.json"
        spec = _make_spec()
        app = Application(FakeRunner())
        components, digest = _compute_full_fingerprint_components(spec, spec.model)
        write_session_start(path, spec, "session-new", None, components, digest)
        resume = app.start_or_resume_session(path, spec, spec.model)
        assert resume.action == "resumed"
        assert len(resume.completed_search_results) == 0

        call_count: list[int] = [0]

        def counting_evaluator(
            candidate: CandidateConfig,
            n_ctx: NctxComposition,
            reps: int,
        ) -> ScreeningResult:
            call_count[0] += 1
            return ScreeningResult(
                candidate=candidate, success=True, disposition="survived",
                observation=ScreeningObservation(
                    candidate=candidate,
                    pp_samples=(42.0, 41.0, 40.0),
                    tg_samples=(43.0, 42.0, 41.0),
                ),
            )

        request = self._make_request(tmp_path, "_no_prior")
        app.screen_candidates(
            request, SCREENING_PROFILE,
            evaluator=counting_evaluator,  # type: ignore[arg-type]
        )
        expected_calls = 8
        assert call_count[0] == expected_calls, (
            f"Expected {expected_calls} calls (baseline + grid), "
            f"got {call_count[0]}"
        )


# ── Cycle 20: full shard identity (Gap 5) ───────────────────────


class TestFullShardIdentity:
    def test_fingerprint_stores_full_shard_data(self) -> None:
        shard = ShardFingerprint(
            path=Path("/tmp/model-shard-00001.gguf"),
            size_bytes=4_000_000_000,
            modified_time=1234567890.0,
            digest_sha256="abc123",
        )
        model = ModelFingerprint(
            path=Path("/tmp/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(shard,),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        spec = _make_spec(model)
        components, _ = _compute_full_fingerprint_components(spec, model)
        mm = components.get("main_model", {})
        shards_data = mm.get("shards", [])
        assert len(shards_data) == 1
        s = shards_data[0]
        assert s["path"] == "/tmp/model-shard-00001.gguf"
        assert s["size_bytes"] == 4_000_000_000
        assert s["modified_time"] == 1234567890.0
        assert s["digest_sha256"] == "abc123"

    def test_full_shard_identity_mismatch_detected(self) -> None:
        shard_a = ShardFingerprint(
            path=Path("/tmp/model-shard-00001.gguf"),
            size_bytes=4_000_000_000,
            modified_time=1234567890.0,
            digest_sha256="abc123",
        )
        shard_b = ShardFingerprint(
            path=Path("/tmp/model-shard-00001.gguf"),
            size_bytes=4_000_000_000,
            modified_time=9999999999.0,  # different modified_time
            digest_sha256="abc123",
        )
        model_a = ModelFingerprint(
            path=Path("/tmp/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(shard_a,),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        model_b = ModelFingerprint(
            path=Path("/tmp/model.gguf"), architecture="llama",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(shard_b,),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        spec_a = _make_spec(model_a)
        spec_b = _make_spec(model_b)
        components_a, _ = _compute_full_fingerprint_components(spec_a, model_a)
        components_b, _ = _compute_full_fingerprint_components(spec_b, model_b)
        reason = _compare_fingerprint_components(components_a, components_b)
        assert reason is not None
        assert "model" in reason or "shard" in reason


# ── Cycle 21: parent_session_id lineage (Fix 1) ─────────────────


class TestParentSessionIdLineage:
    def test_measurement_completed_has_parent_session_id(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal.json"
        result = ScreeningResult(
            candidate=_SAMPLE_CANDIDATE, success=True, disposition="survived",
        )
        write_measurement_completed(jpath, result, "session-x", parent_session_id="parent-1")
        events = read_journal_events(jpath)
        assert events[0]["parent_session_id"] == "parent-1"

    def test_search_decision_has_parent_session_id(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal.json"
        write_search_decision(
            jpath, candidate={"threads": 4},
            disposition="pruned", reason="test",
            session_id="session-x", parent_session_id="parent-1",
        )
        events = read_journal_events(jpath)
        assert events[0]["parent_session_id"] == "parent-1"

    def test_screen_candidates_plumbs_both_event_types(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal.json"
        binary = tmp_path / "llama-bench"
        binary.write_bytes(b"#!/bin/sh\necho ok\n")
        binary.chmod(0o755)
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF")
        call_count: list[int] = [0]

        def evaluator(
            candidate: CandidateConfig,
            n_ctx: NctxComposition,
            reps: int,
        ) -> ScreeningResult:
            call_count[0] += 1
            return ScreeningResult(
                candidate=candidate, success=True, disposition="survived",
                observation=ScreeningObservation(
                    candidate=candidate,
                    pp_samples=(45.0, 44.0, 43.0),
                    tg_samples=(46.0, 45.0, 44.0),
                ),
            )

        app = Application(FakeRunner())
        req: Any = type("Req", (), {
            "binary_path": binary,
            "model_path": model_file,
            "journal_path": jpath,
            "n_ctx": NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            "candidate": _SAMPLE_CANDIDATE,
            "binary_source": "explicit",
        })()

        app.screen_candidates(
            req, SCREENING_PROFILE,
            evaluator=evaluator,  # type: ignore[arg-type]
            session_id="session-p",
            parent_session_id="parent-p",
        )

        events = read_journal_events(jpath)
        lineage_events = [
            e for e in events
            if e.get("event_type") in ("measurement_completed", "search_decision")
        ]
        assert len(lineage_events) >= 2
        for ev in lineage_events:
            assert ev.get("parent_session_id") == "parent-p", (
                f"Event {ev['event_type']} missing parent_session_id"
            )

    def test_parent_session_id_defaults_none(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal.json"
        result = ScreeningResult(
            candidate=_SAMPLE_CANDIDATE, success=True, disposition="survived",
        )
        write_measurement_completed(jpath, result, "session-x")
        events = read_journal_events(jpath)
        assert events[0].get("parent_session_id") is None


# ── Cycle 22: early return search decision journal (Fix 2) ──────


class TestEarlyReturnSearchDecision:
    def _make_req(self, tmp_path: Path, model_path: Path) -> Any:
        return type("Req", (), {
            "binary_path": tmp_path / "llama-bench",
            "model_path": model_path,
            "journal_path": tmp_path / "journal.json",
            "n_ctx": NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            "candidate": _SAMPLE_CANDIDATE,
            "binary_source": "explicit",
        })()

    def test_toolchain_failure_journals_search_decision(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal.json"
        binary = tmp_path / "llama-bench"
        binary.write_bytes(b"#!/bin/sh\necho ok\n")
        binary.chmod(0o755)
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF")

        runner = FakeRunner()
        runner.enqueue(
            (str(binary), "--version"),
            ProcessResult(returncode=1, stdout="", stderr="not found"),
        )

        app = Application(runner)
        app.screen_candidates(
            self._make_req(tmp_path, model_file),
            SCREENING_PROFILE,
            session_id="session-tc",
        )

        events = read_journal_events(jpath)
        decisions = [e for e in events if e.get("event_type") == "search_decision"]
        assert len(decisions) == 1
        assert decisions[0]["session_id"] == "session-tc"
        assert decisions[0]["disposition"] == "unsupported"

    def test_model_failure_journals_search_decision(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal.json"
        binary = tmp_path / "llama-bench"
        binary.write_bytes(b"#!/bin/sh\necho ok\n")
        binary.chmod(0o755)
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"NOT A GGUF FILE")

        runner = FakeRunner()
        _enqueue_toolchain_std(runner, binary)

        app = Application(runner)
        app.screen_candidates(
            self._make_req(tmp_path, model_file),
            SCREENING_PROFILE,
            session_id="session-mf",
        )

        events = read_journal_events(jpath)
        decisions = [e for e in events if e.get("event_type") == "search_decision"]
        assert len(decisions) == 1
        assert decisions[0]["session_id"] == "session-mf"
        assert decisions[0]["disposition"] == "unsupported"

    def test_session_model_mismatch_journals_search_decision(self, tmp_path: Path) -> None:
        from tests.conftest import _build_minimal_gguf

        jpath = tmp_path / "journal.json"
        binary = tmp_path / "llama-bench"
        binary.write_bytes(b"#!/bin/sh\necho ok\n")
        binary.chmod(0o755)
        model_file = tmp_path / "model.gguf"
        _build_minimal_gguf(model_file, {"general.architecture": (8, "llama")})

        runner = FakeRunner()
        _enqueue_toolchain_std(runner, binary)

        different_model = ModelFingerprint(
            path=model_file, architecture="deepseek",
            parameter_count=7_000_000_000, context_length=4096,
            quantization="Q4_K_M", shards=(),
            symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        spec = SessionSpecification(
            topology=_SAMPLE_TOPOLOGY, cpu_only=True,
            selected_device_ids=(),
            n_ctx=_SAMPLE_NCTX,
            budget=ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1),
            advisory=_make_advisory(),
            model=different_model, acknowledged=True, fingerprint="x",
        )

        app = Application(runner)
        app.screen_candidates(
            self._make_req(tmp_path, model_file),
            SCREENING_PROFILE,
            session_spec=spec,
            session_id="session-sm",
        )

        events = read_journal_events(jpath)
        decisions = [e for e in events if e.get("event_type") == "search_decision"]
        assert len(decisions) == 1
        assert decisions[0]["session_id"] == "session-sm"
        assert decisions[0]["disposition"] == "unsupported"

    def test_no_session_id_no_journal(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal.json"
        binary = tmp_path / "llama-bench"
        binary.write_bytes(b"#!/bin/sh\necho ok\n")
        binary.chmod(0o755)
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"NOT A GGUF FILE")

        runner = FakeRunner()
        runner.enqueue(
            (str(binary), "--version"),
            ProcessResult(returncode=1, stdout="", stderr="not found"),
        )

        app = Application(runner)
        app.screen_candidates(
            self._make_req(tmp_path, model_file),
            SCREENING_PROFILE,
        )

        events = read_journal_events(jpath)
        assert len(events) == 0


# ── Cycle 23: fault injection on write_measurement_completed (Fix 3) ─


class TestMeasurementCompletedFault:
    def test_oserror_propagates_as_journal_error(self, tmp_path: Path, monkeypatch: Any) -> None:
        jpath = tmp_path / "journal.json"
        binary = tmp_path / "llama-bench"
        binary.write_bytes(b"#!/bin/sh\necho ok\n")
        binary.chmod(0o755)
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"GGUF")

        write_observation(jpath, _make_observation(path=tmp_path / "model.gguf"))
        prior_content = jpath.read_text()
        call_count: list[int] = [0]

        def evaluator(
            candidate: CandidateConfig,
            n_ctx: NctxComposition,
            reps: int,
        ) -> ScreeningResult:
            call_count[0] += 1
            return ScreeningResult(
                candidate=candidate, success=True, disposition="survived",
                observation=ScreeningObservation(
                    candidate=candidate,
                    pp_samples=(45.0, 44.0, 43.0),
                    tg_samples=(46.0, 45.0, 44.0),
                ),
            )

        import llama_parameter_scout.application as app_mod

        def failing_wmc(
            path: Path, result: ScreeningResult,
            session_id: str, parent_session_id: str | None = None,
        ) -> None:
            raise OSError("injected write failure")

        monkeypatch.setattr(app_mod, "write_measurement_completed", failing_wmc)

        app = Application(FakeRunner())
        req: Any = type("Req", (), {
            "binary_path": binary,
            "model_path": model_file,
            "journal_path": jpath,
            "n_ctx": NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
            "candidate": _SAMPLE_CANDIDATE,
            "binary_source": "explicit",
        })()

        with pytest.raises(JournalError, match="Failed to write measurement_completed"):
            app.screen_candidates(
                req, SCREENING_PROFILE,
                evaluator=evaluator,  # type: ignore[arg-type]
                session_id="session-f",
            )

        assert jpath.read_text() == prior_content
