from __future__ import annotations

import json
from pathlib import Path

import pytest

from llama_parameter_scout.models import (
    CandidateConfig,
    FailedValidation,
    LaunchProvenance,
    MemoryEvidence,
    ModelFingerprint,
    NctxComposition,
    NoWinnerReport,
    ProvisionalLeader,
    RankedCandidate,
    ResourceBudget,
    ScreeningObservation,
    ScreeningReport,
    ScreeningResult,
    SessionSpecification,
    SystemTopology,
    ToolchainFingerprint,
    ValidatedCandidate,
    ValidatedRecommendation,
)
from llama_parameter_scout.reporter import (
    REPORT_SCHEMA_VERSION,
    build_llama_cli_command,
    build_llama_server_command,
    build_preset_ini_section,
    build_reproduction_argv,
    join_argv,
    quote_for_shell,
    redact_all_device_ids,
    redact_all_paths,
    redact_device_id,
    redact_path,
    render_export_json,
    render_export_markdown,
)
from llama_parameter_scout.search import QUICK_PROFILE

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_topology() -> SystemTopology:
    return SystemTopology(
        cpu_count=8,
        system_memory_bytes=32_000_000_000,
        accelerators=(),
        memory_unified=True,
    )


@pytest.fixture
def sample_nctx() -> NctxComposition:
    return NctxComposition(total=2048, prompt=512, generation=1024, depth=512)


@pytest.fixture
def sample_candidate() -> CandidateConfig:
    return CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=0, flash_attn=False,
        cache_type_k="f16", cache_type_v="f16",
    )


@pytest.fixture
def sample_candidate_fa() -> CandidateConfig:
    return CandidateConfig(
        threads=8, batch_size=1024, ubatch_size=256,
        n_gpu_layers=0, flash_attn=True,
        cache_type_k="f16", cache_type_v="q8_0",
    )


@pytest.fixture
def sample_model() -> ModelFingerprint:
    return ModelFingerprint(
        path=Path("/home/user/models/model.gguf"),
        architecture="llama",
        parameter_count=7_000_000_000,
        context_length=4096,
        quantization="Q4_K_M",
        shards=(),
        symlink_warning=False,
        network_mount_detected=False,
        diagnostics=(),
    )


@pytest.fixture
def sample_toolchain() -> ToolchainFingerprint:
    return ToolchainFingerprint(
        binary_path=Path("/usr/local/bin/llama-bench"),
        version="b1234",
        supported_flags=("-m", "-p", "-n", "-d", "-t", "-b", "-ub",
                         "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r",
                         "--split-mode", "--main-gpu", "--n-cpu-moe"),
        supported_output_formats=("json",),
        device_capabilities=("cpu",),
        supported_cache_types=("f16", "f32", "q8_0"),
    )


@pytest.fixture
def sample_budget() -> ResourceBudget:
    return ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)


@pytest.fixture
def sample_session(
    sample_topology: SystemTopology, sample_nctx: NctxComposition,
    sample_budget: ResourceBudget, sample_model: ModelFingerprint,
) -> SessionSpecification:
    from llama_parameter_scout.models import ContextAdvisory
    advisory = ContextAdvisory(
        likely_fit_lower=1000, likely_fit_upper=2000, uncertain_maximum=3000,
        model_weight_bytes=4_000_000_000, kv_cache_bytes_estimate=500_000_000,
        training_context_status="known", training_context_tokens=4096,
        note="Test advisory",
    )
    return SessionSpecification(
        topology=sample_topology, cpu_only=True, selected_device_ids=(),
        n_ctx=sample_nctx, budget=sample_budget, advisory=advisory,
        model=sample_model, acknowledged=True, fingerprint="abc123",
    )


@pytest.fixture
def sample_provenance() -> LaunchProvenance:
    return LaunchProvenance(
        cwd="/home/user/projects/scout",
        shell_dialect="posix",
        executable_path="/home/user/.local/bin/llama-bench",
        argv=("llama-bench", "-m", "/home/user/models/model.gguf"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
    )


@pytest.fixture
def sample_screening_report(
    sample_candidate: CandidateConfig,
    sample_candidate_fa: CandidateConfig,
) -> ScreeningReport:
    baseline = ScreeningResult(
        candidate=sample_candidate, success=True, disposition="survived",
        observation=ScreeningObservation(
            candidate=sample_candidate,
            pp_samples=(45.0, 44.5, 45.5),
            tg_samples=(46.0, 45.5, 46.5),
        ),
    )
    survivor_result = ScreeningResult(
        candidate=sample_candidate, success=True, disposition="survived",
        observation=ScreeningObservation(
            candidate=sample_candidate,
            pp_samples=(45.0, 44.5, 45.5),
            tg_samples=(46.0, 45.5, 46.5),
        ),
    )
    survivor_result2 = ScreeningResult(
        candidate=sample_candidate_fa, success=True, disposition="survived",
        observation=ScreeningObservation(
            candidate=sample_candidate_fa,
            pp_samples=(50.0, 49.5, 50.5),
            tg_samples=(51.0, 50.5, 51.5),
        ),
    )
    ranked1 = RankedCandidate(
        candidate=sample_candidate, pp_score=1.0, tg_score=1.0, balanced=1.0,
        pp_median=45.0, tg_median=46.0, pp_relative_dispersion=0.01,
        tg_relative_dispersion=0.01, total_repetitions=3,
    )
    ranked2 = RankedCandidate(
        candidate=sample_candidate_fa, pp_score=1.1, tg_score=1.1, balanced=1.1,
        pp_median=50.0, tg_median=51.0, pp_relative_dispersion=0.02,
        tg_relative_dispersion=0.02, total_repetitions=3,
    )
    leader = ProvisionalLeader(
        candidate=sample_candidate_fa, objective="balanced",
        pp_score=1.1, tg_score=1.1, balanced=1.1,
        pp_median=50.0, tg_median=51.0,
    )
    return ScreeningReport(
        profile=QUICK_PROFILE,
        baseline_disposition=baseline,
        grid_dispositions=(survivor_result, survivor_result2),
        survivors=(ranked1, ranked2),
        leader=leader,
        attempts_used=3,
        objective="balanced",
        disposition_summary="Baseline: survived; Grid: 2 total, 2 survived",
    )


# ── Fixtures: additional ─────────────────────────────────────────


@pytest.fixture
def sample_provenance_with_home_env(
    sample_provenance: LaunchProvenance,
) -> LaunchProvenance:
    home = str(Path.home())
    return LaunchProvenance(
        cwd=sample_provenance.cwd,
        shell_dialect=sample_provenance.shell_dialect,
        executable_path=sample_provenance.executable_path,
        argv=sample_provenance.argv,
        environment={
            "MODEL_PATH": f"{home}/models/model.gguf",
            "DEVICE_ID": "550e8400-e29b-41d4-a716-446655440000",
        },
    )


@pytest.fixture
def sample_screening_report_with_notes(
    sample_candidate: CandidateConfig,
    sample_candidate_fa: CandidateConfig,
) -> ScreeningReport:
    home = str(Path.home())
    baseline = ScreeningResult(
        candidate=sample_candidate, success=True, disposition="survived",
        observation=ScreeningObservation(
            candidate=sample_candidate,
            pp_samples=(45.0, 44.5, 45.5),
            tg_samples=(46.0, 45.5, 46.5),
        ),
    )
    ranked1 = RankedCandidate(
        candidate=sample_candidate, pp_score=1.0, tg_score=1.0, balanced=1.0,
        pp_median=45.0, tg_median=46.0, pp_relative_dispersion=0.01,
        tg_relative_dispersion=0.01, total_repetitions=3,
    )
    ranked2 = RankedCandidate(
        candidate=sample_candidate_fa, pp_score=1.1, tg_score=1.1, balanced=1.1,
        pp_median=50.0, tg_median=51.0, pp_relative_dispersion=0.02,
        tg_relative_dispersion=0.02, total_repetitions=3,
    )
    leader = ProvisionalLeader(
        candidate=sample_candidate_fa, objective="balanced",
        pp_score=1.1, tg_score=1.1, balanced=1.1,
        pp_median=50.0, tg_median=51.0,
    )
    return ScreeningReport(
        profile=QUICK_PROFILE,
        baseline_disposition=baseline,
        grid_dispositions=(),
        survivors=(ranked1, ranked2),
        leader=leader,
        attempts_used=1,
        objective="balanced",
        disposition_summary="Baseline: survived; Grid: 1 total, 2 survived",
        notes=(
            f"Model path: {home}/models/model.gguf",
            "Device: 550e8400-e29b-41d4-a716-446655440000",
        ),
    )


@pytest.fixture
def sample_moe_candidate() -> CandidateConfig:
    return CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512,
        n_gpu_layers=0, flash_attn=False,
        cache_type_k="f16", cache_type_v="f16",
        n_cpu_moe=1,
    )


# ── Cycle 1: JSON schema contract ────────────────────────────────


class TestJsonSchemaContract:
    def test_schema_version_present(self, sample_session: SessionSpecification,
                                     sample_toolchain: ToolchainFingerprint,
                                     sample_model: ModelFingerprint) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
        )
        data = json.loads(raw)
        assert data["schema"] == REPORT_SCHEMA_VERSION

    def test_artifact_type_marked_private(self, sample_session: SessionSpecification,
                                           sample_toolchain: ToolchainFingerprint,
                                           sample_model: ModelFingerprint) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
        )
        data = json.loads(raw)
        assert data["artifact_type"] == "private_reproducibility_artifact"

    def test_contains_required_top_keys(self, sample_session: SessionSpecification,
                                         sample_toolchain: ToolchainFingerprint,
                                         sample_model: ModelFingerprint) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
        )
        data = json.loads(raw)
        for key in ("schema", "exported_at", "artifact_type", "result",
                     "provenance", "session", "toolchain", "model",
                     "hashes_include"):
            assert key in data, f"Missing top-level key: {key}"

    def test_screening_section_present_when_provided(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        data = json.loads(raw)
        assert "screening" in data
        assert data["screening"]["objective"] == "balanced"
        assert len(data["screening"]["survivors"]) == 2

    def test_validation_section_present_when_provided(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        vc = ValidatedCandidate(
            candidate=sample_screening_report.survivors[0].candidate,
            pp_score=0.95, tg_score=0.98, balanced=0.965,
            pp_median=42.0, tg_median=44.0,
            pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
            memory_evidence=MemoryEvidence(ram_bytes=8_000_000_000, quality="complete"),
        )
        validation = ValidatedRecommendation(
            recommendation=vc, alternatives=(),
            failures=(), attempts_used=1, budget=10,
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
            validation_result=validation,
        )
        data = json.loads(raw)
        assert data["validation"] is not None
        assert data["validation"]["type"] == "validated_recommendation"

    def test_no_winner_section(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        nw = NoWinnerReport(
            provisional_leader=sample_screening_report.leader,
            failures=(
                FailedValidation(
                    candidate=sample_screening_report.survivors[0].candidate,
                    disposition="oom", reason="OOM during full-context run",
                ),
            ),
            attempts_used=1, budget=10,
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
            validation_result=nw,
        )
        data = json.loads(raw)
        assert data["validation"] is not None
        assert data["validation"]["type"] == "no_winner"
        assert data["result"] == "no_winner"

    def test_provenance_structured_fields(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_provenance: LaunchProvenance,
    ) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            provenance=sample_provenance,
        )
        data = json.loads(raw)
        p = data["provenance"]
        assert p is not None
        assert p["cwd"] == sample_provenance.cwd
        assert p["shell_dialect"] == "posix"
        assert p["executable_path"] == sample_provenance.executable_path
        assert p["argv"] == list(sample_provenance.argv)
        assert p["environment"] == dict(sample_provenance.environment)

    def test_hashes_optional(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
            include_hashes=True,
        )
        data = json.loads(raw)
        assert data["hashes_include"] is True
        assert "hashes" in data
        assert "caveat" in data
        assert "reproducibility" in data["caveat"].lower()

    def test_hashes_false_by_default(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
        )
        data = json.loads(raw)
        assert data["hashes_include"] is False
        assert "hashes" not in data

    def test_reproduction_argv_for_survivors(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        data = json.loads(raw)
        assert "reproduction_argv" in data
        assert isinstance(data["reproduction_argv"], list)
        assert len(data["reproduction_argv"]) == 2
        entry = data["reproduction_argv"][0]
        assert "candidate" in entry
        assert "argv" in entry
        assert isinstance(entry["argv"], list)
        assert "reproduction_commands" in data
        cmds = data["reproduction_commands"]
        assert "llama_cli" in cmds
        assert "llama_server" in cmds
        assert "preset_ini" in cmds

    def test_benchmark_only_flags_noted(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        data = json.loads(raw)
        assert "benchmark_only_flags_note" in data
        assert "benchmark-only" in data["benchmark_only_flags_note"]
        assert "1:1" in data["benchmark_only_flags_note"]


# ── Cycle 2: Redaction ───────────────────────────────────────────


class TestRedaction:
    def test_redact_home_path(self) -> None:
        home = str(Path.home())
        result = redact_path(f"{home}/.llama/models/model.gguf")
        assert result.startswith("[HOME]")
        assert "/.llama/models/model.gguf" in result

    def test_redact_non_home_path_unchanged(self) -> None:
        result = redact_path("/usr/local/bin/llama-bench")
        assert result == "/usr/local/bin/llama-bench"

    def test_redact_all_paths_recursive(self) -> None:
        home = str(Path.home())
        obj = {
            "path": f"{home}/model.gguf",
            "nested": {"inner": f"{home}/cache"},
            "list": [f"{home}/shard.gguf"],
        }
        redacted = redact_all_paths(obj)
        assert isinstance(redacted, dict)
        assert redacted["path"].startswith("[HOME]")
        assert redacted["nested"]["inner"].startswith("[HOME]")
        assert redacted["list"][0].startswith("[HOME]")

    def test_redact_device_id_uuid(self) -> None:
        result = redact_device_id("device-550e8400-e29b-41d4-a716-446655440000-xyz")
        assert "[DEVICE-ID]" in result
        assert "550e8400" not in result

    def test_redact_device_id_non_uuid_unchanged(self) -> None:
        result = redact_device_id("device-12345-abc")
        assert result == "device-12345-abc"

    def test_redact_all_device_ids_recursive(self) -> None:
        obj = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "nested": {"uid": "123e4567-e89b-12d3-a456-426614174000"},
            "list": ["plain", "550e8400-e29b-41d4-a716-446655440000"],
        }
        redacted = redact_all_device_ids(obj)
        assert isinstance(redacted, dict)
        assert redacted["id"] == "[DEVICE-ID]"
        assert redacted["nested"]["uid"] == "[DEVICE-ID]"
        assert redacted["list"][0] == "plain"
        assert redacted["list"][1] == "[DEVICE-ID]"

    def test_markdown_report_redacted(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        home = str(Path.home())
        model_with_home = ModelFingerprint(
            path=Path(home) / "models/model.gguf",
            architecture="llama",
            parameter_count=7_000_000_000,
            context_length=4096,
            quantization="Q4_K_M",
            shards=(),
            symlink_warning=False,
            network_mount_detected=False,
            diagnostics=(),
        )
        session_with_home = SessionSpecification(
            topology=sample_session.topology, cpu_only=sample_session.cpu_only,
            selected_device_ids=sample_session.selected_device_ids,
            n_ctx=sample_session.n_ctx, budget=sample_session.budget,
            advisory=sample_session.advisory,
            model=model_with_home, acknowledged=sample_session.acknowledged,
            fingerprint=sample_session.fingerprint,
        )
        raw = render_export_markdown(
            session=session_with_home, toolchain=sample_toolchain, model=model_with_home,
        )
        assert "[HOME]" in raw
        assert home not in raw or "[HOME]" in raw


# ── Cycle 3: Shell quoting ───────────────────────────────────────


class TestShellQuoting:
    def test_posix_quoting(self) -> None:
        result = quote_for_shell("simple", "posix")
        assert result == "simple" or result == "'simple'"

    def test_posix_quoting_with_spaces(self) -> None:
        result = quote_for_shell("/path/with spaces/file.gguf", "posix")
        assert "'" in result

    def test_powershell_quoting(self) -> None:
        result = quote_for_shell("/path/with spaces/file.gguf", "powershell")
        assert result == "'/path/with spaces/file.gguf'"

    def test_powershell_quoting_with_single_quote(self) -> None:
        result = quote_for_shell("it's a test", "powershell")
        assert result == "'it''s a test'"

    def test_cmd_quoting(self) -> None:
        result = quote_for_shell("/path/with spaces/file.gguf", "cmd")
        assert result == '"/path/with spaces/file.gguf"'

    def test_cmd_quoting_with_double_quote(self) -> None:
        result = quote_for_shell('path"with"quotes', "cmd")
        assert result == '"path\\"with\\"quotes"'

    def test_unknown_dialect_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            quote_for_shell("test", "unknown")

    def test_join_argv_posix(self) -> None:
        argv = ["llama-bench", "-m", "/path/with spaces/model.gguf"]
        result = join_argv(argv, "posix")
        assert "llama-bench" in result
        assert "'/path/with spaces/model.gguf'" in result

    def test_join_argv_powershell(self) -> None:
        argv = ["llama-bench", "-m", "/path/with spaces/model.gguf"]
        result = join_argv(argv, "powershell")
        assert "'/path/with spaces/model.gguf'" in result

    def test_join_argv_cmd(self) -> None:
        argv = ["llama-bench", "-m", "/path/with spaces/model.gguf"]
        result = join_argv(argv, "cmd")
        assert '"/path/with spaces/model.gguf"' in result

    def test_markdown_uses_posix_quoting(self) -> None:
        argv = build_reproduction_argv(
            CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                            n_gpu_layers=0, flash_attn=False,
                            cache_type_k="f16", cache_type_v="f16"),
            NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
        )
        cmd = join_argv(argv, "posix")
        assert "llama-bench" in cmd


# ── Cycle 4: Reproduction argv ───────────────────────────────────


class TestReproductionArgv:
    def test_basic_candidate(self, sample_candidate: CandidateConfig,
                              sample_nctx: NctxComposition) -> None:
        argv = build_reproduction_argv(sample_candidate, sample_nctx)
        assert argv[0] == "llama-bench"
        assert "-m" in argv
        assert "-p" in argv
        assert str(sample_nctx.prompt) in argv
        assert "-fa" in argv

    def test_flash_attn_on(self, sample_candidate_fa: CandidateConfig,
                            sample_nctx: NctxComposition) -> None:
        argv = build_reproduction_argv(sample_candidate_fa, sample_nctx)
        fa_idx = argv.index("-fa")
        assert argv[fa_idx + 1] == "on"

    def test_quantized_cache_types(self, sample_candidate_fa: CandidateConfig,
                                    sample_nctx: NctxComposition) -> None:
        argv = build_reproduction_argv(sample_candidate_fa, sample_nctx)
        assert "-ctk" in argv
        assert "-ctv" in argv

    def test_cpu_only_default(self, sample_candidate: CandidateConfig,
                               sample_nctx: NctxComposition) -> None:
        argv = build_reproduction_argv(sample_candidate, sample_nctx)
        assert "--split-mode" not in argv

    def test_device_selected(self, sample_candidate: CandidateConfig,
                              sample_nctx: NctxComposition) -> None:
        argv = build_reproduction_argv(
            sample_candidate, sample_nctx,
            cpu_only=False, selected_device_ids=(0,),
        )
        assert "--split-mode" in argv
        assert "--main-gpu" in argv


# ── Cycle 5: llama-cli / llama-server commands ───────────────────


class TestProductionCommands:
    def test_llama_cli_basic(self, sample_candidate: CandidateConfig,
                              sample_nctx: NctxComposition) -> None:
        cmd = build_llama_cli_command(sample_candidate, sample_nctx, "/path/model.gguf")
        assert cmd[0] == "llama-cli"
        assert "-m" in cmd
        assert "/path/model.gguf" in cmd
        assert "-c" in cmd
        assert str(sample_nctx.total) in cmd

    def test_llama_cli_flash_attn(self, sample_candidate_fa: CandidateConfig,
                                   sample_nctx: NctxComposition) -> None:
        cmd = build_llama_cli_command(sample_candidate_fa, sample_nctx, "/path/model.gguf")
        assert "--flash-attn" in cmd

    def test_llama_server_basic(self, sample_candidate: CandidateConfig,
                                 sample_nctx: NctxComposition) -> None:
        cmd = build_llama_server_command(sample_candidate, sample_nctx, "/path/model.gguf")
        assert cmd[0] == "llama-server"
        assert "-m" in cmd
        assert "-c" in cmd

    def test_llama_cli_no_flash_attn(self, sample_candidate: CandidateConfig,
                                      sample_nctx: NctxComposition) -> None:
        cmd = build_llama_cli_command(sample_candidate, sample_nctx, "/path/model.gguf")
        assert "--no-flash-attn" in cmd

    def test_llama_server_cache_types(self, sample_candidate_fa: CandidateConfig,
                                       sample_nctx: NctxComposition) -> None:
        cmd = build_llama_server_command(sample_candidate_fa, sample_nctx, "/path/model.gguf")
        assert "-ctv" in cmd
        assert "-ctk" not in cmd  # cache_type_k=f16 is the default, omitted


# ── Cycle 6: preset.ini ──────────────────────────────────────────


class TestPresetIni:
    def test_preset_ini_section_has_required_keys(
        self, sample_candidate: CandidateConfig, sample_nctx: NctxComposition,
    ) -> None:
        ini = build_preset_ini_section(sample_candidate, sample_nctx)
        assert ini.startswith("[llama]")
        assert "n_ctx" in ini
        assert "n_batch" in ini
        assert "n_ubatch" in ini
        assert "n_threads" in ini
        assert "n_gpu_layers" in ini
        assert "flash_attn" in ini
        assert "cache_type_k" in ini
        assert "cache_type_v" in ini

    def test_preset_ini_values(self, sample_candidate: CandidateConfig,
                                sample_nctx: NctxComposition) -> None:
        ini = build_preset_ini_section(sample_candidate, sample_nctx)
        assert f"n_ctx = {sample_nctx.total}" in ini
        assert f"n_batch = {sample_candidate.batch_size}" in ini
        assert "flash_attn = false" in ini

    def test_preset_ini_flash_attn_true(self, sample_candidate_fa: CandidateConfig,
                                         sample_nctx: NctxComposition) -> None:
        ini = build_preset_ini_section(sample_candidate_fa, sample_nctx)
        assert "flash_attn = true" in ini

    def test_preset_ini_quantized_cache(self, sample_candidate_fa: CandidateConfig,
                                         sample_nctx: NctxComposition) -> None:
        ini = build_preset_ini_section(sample_candidate_fa, sample_nctx)
        assert "cache_type_k = f16" in ini
        assert "cache_type_v = q8_0" in ini

    def test_preset_ini_moe(self, sample_nctx: NctxComposition) -> None:
        c = CandidateConfig(
            threads=4, batch_size=512, ubatch_size=512,
            n_gpu_layers=0, flash_attn=False,
            cache_type_k="f16", cache_type_v="f16",
            n_cpu_moe=1,
        )
        ini = build_preset_ini_section(c, sample_nctx)
        assert "n_cpu_moe = 1" in ini

    def test_preset_ini_valid_format(self, sample_candidate: CandidateConfig,
                                      sample_nctx: NctxComposition) -> None:
        ini = build_preset_ini_section(sample_candidate, sample_nctx)
        lines = ini.strip().split("\n")
        assert lines[0] == "[llama]"
        for line in lines[1:]:
            if line.strip():
                assert "=" in line, f"Invalid ini line: {line}"

    def test_preset_ini_device_selected(self, sample_candidate: CandidateConfig,
                                         sample_nctx: NctxComposition) -> None:
        ini = build_preset_ini_section(
            sample_candidate, sample_nctx,
            cpu_only=False, selected_device_ids=(0,),
        )
        assert "main_gpu = 0" in ini

    def test_preset_ini_present_in_json(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        data = json.loads(raw)
        cmds = data["reproduction_commands"]
        assert cmds["preset_ini"].startswith("[llama]")


# ── Cycle 7: Markdown report ─────────────────────────────────────


class TestMarkdownReport:
    def test_markdown_has_required_sections(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        assert "# llama-parameter-scout Report" in md
        assert "## Session" in md
        assert "## Model" in md
        assert "## Toolchain" in md
        assert "## Screening Results" in md
        assert "## Reproduction Commands" in md

    def test_markdown_share_safe_label(self, sample_session: SessionSpecification,
                                        sample_toolchain: ToolchainFingerprint,
                                        sample_model: ModelFingerprint) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
        )
        assert "Share-safe report" in md

    def test_markdown_private_json_label(self, sample_session: SessionSpecification,
                                          sample_toolchain: ToolchainFingerprint,
                                          sample_model: ModelFingerprint) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
        )
        data = json.loads(raw)
        assert "Private reproducibility artifact" in data["artifact_label"]

    def test_markdown_no_winner(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        nw = NoWinnerReport(
            provisional_leader=sample_screening_report.leader,
            failures=(
                FailedValidation(
                    candidate=sample_screening_report.survivors[0].candidate,
                    disposition="oom", reason="OOM during full-context validation",
                ),
            ),
            attempts_used=1, budget=10,
        )
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
            validation_result=nw,
        )
        assert "No winner" in md
        assert "OOM during full-context validation" in md

    def test_markdown_incomplete_result(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
        )
        assert "incomplete" in md
        assert "no_winner" not in md

    def test_markdown_candidate_reproduction(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        assert "llama-bench" in md
        assert "llama-cli" in md
        assert "llama-server" in md
        assert "Preset.ini" in md

    def test_markdown_provenance_section(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_provenance: LaunchProvenance,
    ) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            provenance=sample_provenance,
        )
        assert "Launch Provenance" in md
        assert sample_provenance.shell_dialect in md


# ── Cycle 8: Validate JSON values are structured (not only strings) ──


class TestStructuredValues:
    def test_candidate_config_is_object(self, sample_session: SessionSpecification,
                                         sample_toolchain: ToolchainFingerprint,
                                         sample_model: ModelFingerprint,
                                         sample_screening_report: ScreeningReport) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        data = json.loads(raw)
        survivor = data["screening"]["survivors"][0]
        assert isinstance(survivor["candidate"]["threads"], int)
        assert isinstance(survivor["candidate"]["batch_size"], int)
        assert isinstance(survivor["candidate"]["flash_attn"], bool)

    def test_samples_are_float_lists(self, sample_session: SessionSpecification,
                                      sample_toolchain: ToolchainFingerprint,
                                      sample_model: ModelFingerprint,
                                      sample_screening_report: ScreeningReport) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        data = json.loads(raw)
        obs = data["screening"]["baseline"]["observation"]
        for s in obs["pp_samples"]:
            assert isinstance(s, (int, float))
        for s in obs["tg_samples"]:
            assert isinstance(s, (int, float))

    def test_reproduction_argv_is_structured_list(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        data = json.loads(raw)
        ra = data["reproduction_argv"]
        assert isinstance(ra, list)
        for entry in ra:
            assert "candidate" in entry
            assert "argv" in entry
            assert isinstance(entry["argv"], list)
            for item in entry["argv"]:
                assert isinstance(item, str)

    def test_environment_is_object(self, sample_session: SessionSpecification,
                                    sample_toolchain: ToolchainFingerprint,
                                    sample_model: ModelFingerprint,
                                    sample_provenance: LaunchProvenance) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            provenance=sample_provenance,
        )
        data = json.loads(raw)
        env = data["provenance"]["environment"]
        assert isinstance(env, dict)
        for k, v in env.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ── Cycle 9: Focused verification (reviewer findings) ────────────


class TestH1EnvironmentRedaction:
    def test_environment_value_path_redacted_in_markdown(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_provenance_with_home_env: LaunchProvenance,
    ) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            provenance=sample_provenance_with_home_env,
        )
        assert "[HOME]" in md
        home = str(Path.home())
        # The raw home path should NOT appear after the `=` sign in env vars
        assert f"={home}" not in md

    def test_environment_value_device_id_redacted_in_markdown(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_provenance_with_home_env: LaunchProvenance,
    ) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            provenance=sample_provenance_with_home_env,
        )
        assert "[DEVICE-ID]" in md
        assert "550e8400-e29b-41d4-a716-446655440000" not in md


class TestH2NoteRedaction:
    def test_note_path_redacted_in_markdown(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report_with_notes: ScreeningReport,
    ) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report_with_notes,
        )
        assert "[HOME]" in md
        home = str(Path.home())
        assert f"Model path: {home}" not in md

    def test_note_device_id_redacted_in_markdown(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report_with_notes: ScreeningReport,
    ) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report_with_notes,
        )
        assert "[DEVICE-ID]" in md
        assert "550e8400-e29b-41d4-a716-446655440000" not in md


class TestH3LeaderPresetIni:
    def test_preset_ini_comes_from_leader_not_first_survivor(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        # leader is sample_candidate_fa (flash_attn=True, cache_type_v=q8_0)
        assert "flash_attn = true" in md
        assert "cache_type_v = q8_0" in md
        # first survivor is sample_candidate (flash_attn=False, cache_type_v=f16)
        has_false = "flash_attn = false" in md
        if has_false:
            assert md.index("flash_attn = true") < md.index("flash_attn = false")


class TestH4MoeArgv:
    def test_reproduction_argv_emits_n_cpu_moe_when_set(
        self, sample_nctx: NctxComposition,
        sample_moe_candidate: CandidateConfig,
    ) -> None:
        argv = build_reproduction_argv(sample_moe_candidate, sample_nctx)
        assert "--n-cpu-moe" in argv
        idx = argv.index("--n-cpu-moe")
        assert argv[idx + 1] == "1"

    def test_reproduction_argv_no_n_cpu_moe_when_none(
        self, sample_nctx: NctxComposition,
        sample_candidate: CandidateConfig,
    ) -> None:
        argv = build_reproduction_argv(sample_candidate, sample_nctx)
        assert "--n-cpu-moe" not in argv


class TestM1DistinctCandidates:
    def test_reproduction_argv_has_distinct_entries_for_cache(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        c1 = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                             n_gpu_layers=0, flash_attn=False,
                             cache_type_k="f16", cache_type_v="f16")
        c2 = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                             n_gpu_layers=0, flash_attn=False,
                             cache_type_k="q8_0", cache_type_v="q8_0")
        r1 = RankedCandidate(candidate=c1, pp_score=1.0, tg_score=1.0, balanced=1.0,
                             pp_median=45.0, tg_median=46.0,
                             pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
                             total_repetitions=3)
        r2 = RankedCandidate(candidate=c2, pp_score=0.95, tg_score=0.95, balanced=0.95,
                             pp_median=43.0, tg_median=44.0,
                             pp_relative_dispersion=0.02, tg_relative_dispersion=0.02,
                             total_repetitions=3)
        obs1 = ScreeningObservation(candidate=c1, pp_samples=(45.0, 44.5, 45.5),
                                    tg_samples=(46.0, 45.5, 46.5))
        obs2 = ScreeningObservation(candidate=c2, pp_samples=(43.0, 42.5, 43.5),
                                    tg_samples=(44.0, 43.5, 44.5))
        baseline = ScreeningResult(candidate=c1, success=True, disposition="survived",
                                   observation=obs1)
        grid = ScreeningResult(candidate=c2, success=True, disposition="survived",
                               observation=obs2)
        report = ScreeningReport(
            profile=QUICK_PROFILE,
            baseline_disposition=baseline,
            grid_dispositions=(grid,),
            survivors=(r1, r2),
            leader=ProvisionalLeader(candidate=c1, objective="balanced",
                                     pp_score=1.0, tg_score=1.0, balanced=1.0,
                                     pp_median=45.0, tg_median=46.0),
            attempts_used=2, objective="balanced",
            disposition_summary="2 survivors",
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=report,
        )
        data = json.loads(raw)
        ra = data["reproduction_argv"]
        assert len(ra) == 2, "Two distinct candidates must have two entries"
        argv0 = ra[0]["argv"]
        argv1 = ra[1]["argv"]
        assert argv0 != argv1, "Distinct cache types must produce different argv"

    def test_reproduction_argv_has_distinct_entries_for_moe(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        c1 = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                             n_gpu_layers=0, flash_attn=False,
                             cache_type_k="f16", cache_type_v="f16")
        c2 = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                             n_gpu_layers=0, flash_attn=False,
                             cache_type_k="f16", cache_type_v="f16",
                             n_cpu_moe=2)
        r1 = RankedCandidate(candidate=c1, pp_score=1.0, tg_score=1.0, balanced=1.0,
                             pp_median=45.0, tg_median=46.0,
                             pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
                             total_repetitions=3)
        r2 = RankedCandidate(candidate=c2, pp_score=1.05, tg_score=1.05, balanced=1.05,
                             pp_median=47.0, tg_median=48.0,
                             pp_relative_dispersion=0.02, tg_relative_dispersion=0.02,
                             total_repetitions=3)
        obs1 = ScreeningObservation(candidate=c1, pp_samples=(45.0, 44.5, 45.5),
                                    tg_samples=(46.0, 45.5, 46.5))
        obs2 = ScreeningObservation(candidate=c2, pp_samples=(47.0, 46.5, 47.5),
                                    tg_samples=(48.0, 47.5, 48.5))
        baseline = ScreeningResult(candidate=c1, success=True, disposition="survived",
                                   observation=obs1)
        grid = ScreeningResult(candidate=c2, success=True, disposition="survived",
                               observation=obs2)
        report = ScreeningReport(
            profile=QUICK_PROFILE,
            baseline_disposition=baseline,
            grid_dispositions=(grid,),
            survivors=(r1, r2),
            leader=ProvisionalLeader(candidate=c1, objective="balanced",
                                     pp_score=1.0, tg_score=1.0, balanced=1.0,
                                     pp_median=45.0, tg_median=46.0),
            attempts_used=2, objective="balanced",
            disposition_summary="2 survivors",
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=report,
        )
        data = json.loads(raw)
        ra = data["reproduction_argv"]
        assert len(ra) == 2, "Two distinct (MoE vs non-MoE) candidates must have two entries"
        argv0 = ra[0]["argv"]
        argv1 = ra[1]["argv"]
        assert argv0 != argv1, "MoE difference must produce different argv"

    def test_provisional_leader_label_in_json(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
        sample_screening_report: ScreeningReport,
    ) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=sample_screening_report,
        )
        data = json.loads(raw)
        leader = data["screening"]["leader"]
        assert leader is not None
        assert "label" in leader
        assert "Screening evidence only" in leader["label"]


class TestM1ShellDialectValidation:
    def test_valid_shell_dialects(self) -> None:
        for d in ("posix", "powershell", "cmd"):
            lp = LaunchProvenance(
                cwd="/tmp", shell_dialect=d,
                executable_path="/bin/sh", argv=("/bin/sh",),
                environment={},
            )
            assert lp.shell_dialect == d

    def test_invalid_shell_dialect_raises(self) -> None:
        with pytest.raises(ValueError, match="shell_dialect"):
            LaunchProvenance(
                cwd="/tmp", shell_dialect="invalid",  # type: ignore[arg-type]
                executable_path="/bin/sh", argv=("/bin/sh",),
                environment={},
            )


# ── Regression: preset.ini from validation without screening ─────


class TestRegressionPresetFromValidation:
    def test_preset_ini_renders_from_validation_without_screening(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        vc = ValidatedCandidate(
            candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                                      n_gpu_layers=0, flash_attn=True,
                                      cache_type_k="f16", cache_type_v="q8_0"),
            pp_score=1.0, tg_score=1.0, balanced=1.0,
            pp_median=45.0, tg_median=46.0,
            pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
        )
        validation = ValidatedRecommendation(
            recommendation=vc, alternatives=(), failures=(),
            attempts_used=1, budget=10,
        )
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            validation_result=validation,
        )
        assert "Preset.ini" in md
        assert "flash_attn = true" in md
        assert "cache_type_v = q8_0" in md

    def test_json_preset_from_validation_without_screening(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        vc = ValidatedCandidate(
            candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                                      n_gpu_layers=0, flash_attn=True,
                                      cache_type_k="f16", cache_type_v="f16"),
            pp_score=1.0, tg_score=1.0, balanced=1.0,
            pp_median=45.0, tg_median=46.0,
            pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
        )
        validation = ValidatedRecommendation(
            recommendation=vc, alternatives=(), failures=(),
            attempts_used=1, budget=10,
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            validation_result=validation,
        )
        data = json.loads(raw)
        cmds = data["reproduction_commands"]
        assert "preset_ini" in cmds
        assert cmds["preset_ini"].startswith("[llama]")

    def test_preset_ini_no_winner_no_screening_omitted(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        nw = NoWinnerReport(
            provisional_leader=None, failures=(),
            attempts_used=0, budget=10,
        )
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            validation_result=nw,
        )
        assert "Preset.ini" not in md


# ── Regression: hash key naming and determinism ──────────────────


class TestRegressionHashKeys:
    def test_hash_keys_renamed(self, sample_session: SessionSpecification,
                                sample_toolchain: ToolchainFingerprint,
                                sample_model: ModelFingerprint) -> None:
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            include_hashes=True,
        )
        data = json.loads(raw)
        hashes = data["hashes"]
        assert "session_reference" in hashes
        assert "session_fingerprint" not in hashes
        assert "export_digest" in hashes

    def test_export_digest_deterministic(self, sample_session: SessionSpecification,
                                          sample_toolchain: ToolchainFingerprint,
                                          sample_model: ModelFingerprint) -> None:
        raw1 = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            include_hashes=True,
        )
        raw2 = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            include_hashes=True,
        )
        data1 = json.loads(raw1)
        data2 = json.loads(raw2)
        # export_digest must be identical across calls (no timestamp in input)
        assert data1["hashes"]["export_digest"] == data2["hashes"]["export_digest"]
        assert data1["hashes"]["session_reference"] == data2["hashes"]["session_reference"]


# ── Regression: Markdown table escaping ──────────────────────────


class TestRegressionTableEscaping:
    def test_table_cell_pipe_escaped(self) -> None:
        from llama_parameter_scout.reporter import _markdown_escape
        result = _markdown_escape("a|b|c")
        assert "\\|" in result

    def test_table_cell_backslash_escaped(self) -> None:
        from llama_parameter_scout.reporter import _markdown_escape
        result = _markdown_escape("path\\to\\file")
        # Each original backslash becomes two (doubled)
        assert result == r"path\\to\\file"

    def test_survivor_table_does_not_break_with_pipes(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        c = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                            n_gpu_layers=0, flash_attn=False,
                            cache_type_k="f16", cache_type_v="f16")
        ranked = RankedCandidate(candidate=c, pp_score=1.0, tg_score=1.0, balanced=1.0,
                                 pp_median=45.0, tg_median=46.0,
                                 pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
                                 total_repetitions=3)
        baseline = ScreeningResult(candidate=c, success=True, disposition="survived")
        report = ScreeningReport(
            profile=QUICK_PROFILE,
            baseline_disposition=baseline,
            grid_dispositions=(),
            survivors=(ranked,),
            leader=ProvisionalLeader(candidate=c, objective="balanced",
                                     pp_score=1.0, tg_score=1.0, balanced=1.0,
                                     pp_median=45.0, tg_median=46.0),
            attempts_used=1, objective="balanced",
            disposition_summary="Baseline: survived",
        )
        md = render_export_markdown(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=report,
        )
        for line in md.splitlines():
            if line.startswith("|"):
                pipe_count = line.count("|")
                assert pipe_count == 12, f"Unexpected pipe count {pipe_count} in: {line}"


# ── Focused public-behavior: reproduction argv correctness ───────


class TestReproductionArgvExact:
    """Public-behavior tests for exact reproduction argv logic."""

    def test_non_survivor_observed_in_grid_appears(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        c1 = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                             n_gpu_layers=0, flash_attn=False,
                             cache_type_k="f16", cache_type_v="f16")
        c2 = CandidateConfig(threads=8, batch_size=1024, ubatch_size=256,
                             n_gpu_layers=0, flash_attn=True,
                             cache_type_k="f16", cache_type_v="f16")
        obs1 = ScreeningObservation(candidate=c1, pp_samples=(45.0, 44.5, 45.5),
                                    tg_samples=(46.0, 45.5, 46.5))
        obs2 = ScreeningObservation(candidate=c2, pp_samples=(50.0, 49.5, 50.5),
                                    tg_samples=(51.0, 50.5, 51.5))
        baseline = ScreeningResult(candidate=c1, success=True, disposition="survived",
                                   observation=obs1)
        # c2 was observed but dominance_pruned (not a survivor)
        grid = ScreeningResult(candidate=c2, success=True, disposition="dominance_pruned",
                               observation=obs2)
        ranked1 = RankedCandidate(candidate=c1, pp_score=1.0, tg_score=1.0, balanced=1.0,
                                  pp_median=45.0, tg_median=46.0,
                                  pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
                                  total_repetitions=3)
        report = ScreeningReport(
            profile=QUICK_PROFILE,
            baseline_disposition=baseline,
            grid_dispositions=(grid,),
            survivors=(ranked1,),
            leader=ProvisionalLeader(candidate=c1, objective="balanced",
                                     pp_score=1.0, tg_score=1.0, balanced=1.0,
                                     pp_median=45.0, tg_median=46.0),
            attempts_used=2, objective="balanced",
            disposition_summary="1 survivor, 1 dominance_pruned",
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=report,
        )
        data = json.loads(raw)
        ra = data["reproduction_argv"]
        assert len(ra) == 2, (
            f"Expected 2 (survivor c1 + pruned-but-observed c2), got {len(ra)}"
        )
        cands = {tuple(sorted(e["candidate"].items())) for e in ra}
        c1_t = tuple(sorted({"threads": 4, "batch_size": 512, "ubatch_size": 512,
                             "n_gpu_layers": 0, "flash_attn": False,
                             "cache_type_k": "f16", "cache_type_v": "f16",
                             "n_cpu_moe": None}.items()))
        c2_t = tuple(sorted({"threads": 8, "batch_size": 1024, "ubatch_size": 256,
                             "n_gpu_layers": 0, "flash_attn": True,
                             "cache_type_k": "f16", "cache_type_v": "f16",
                             "n_cpu_moe": None}.items()))
        assert c1_t in cands, "Survivor c1 must appear"
        assert c2_t in cands, "Non-survivor observed c2 must appear"

    def test_deduplicate_keeps_highest_repetition_count(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        c = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                            n_gpu_layers=0, flash_attn=False,
                            cache_type_k="f16", cache_type_v="f16")
        # Baseline: 3 samples
        obs_base = ScreeningObservation(candidate=c, pp_samples=(45.0, 44.5, 45.5),
                                        tg_samples=(46.0, 45.5, 46.5))
        # Grid: 4 samples (higher)
        obs_grid = ScreeningObservation(candidate=c, pp_samples=(46.0, 45.0, 46.5, 45.5),
                                        tg_samples=(47.0, 46.0, 47.5, 46.5))
        # Adaptive: 2 samples (lower)
        obs_adapt = ScreeningObservation(candidate=c, pp_samples=(44.0, 45.0),
                                         tg_samples=(45.0, 46.0))
        baseline = ScreeningResult(candidate=c, success=True, disposition="survived",
                                   observation=obs_base)
        grid = ScreeningResult(candidate=c, success=True, disposition="survived",
                               observation=obs_grid)
        adapt = ScreeningResult(candidate=c, success=True, disposition="survived",
                                observation=obs_adapt)
        ranked = RankedCandidate(candidate=c, pp_score=1.0, tg_score=1.0, balanced=1.0,
                                 pp_median=45.0, tg_median=46.0,
                                 pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
                                 total_repetitions=4)
        report = ScreeningReport(
            profile=QUICK_PROFILE,
            baseline_disposition=baseline,
            grid_dispositions=(grid,),
            adaptive_dispositions=(adapt,),
            survivors=(ranked,),
            leader=ProvisionalLeader(candidate=c, objective="balanced",
                                     pp_score=1.0, tg_score=1.0, balanced=1.0,
                                     pp_median=45.0, tg_median=46.0),
            attempts_used=3, objective="balanced",
            disposition_summary="3 dispositions, 1 survivor",
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=report,
        )
        data = json.loads(raw)
        ra = data["reproduction_argv"]
        assert len(ra) == 1, f"Expected 1 deduplicated entry, got {len(ra)}"
        assert ra[0]["repetition_count"] == 4, (
            f"Expected highest repetition_count 4 (from RankedCandidate), "
            f"got {ra[0]['repetition_count']}"
        )
        assert "-r" in ra[0]["argv"] and "4" in ra[0]["argv"], (
            "-r flag must match repetition_count 4"
        )

    def test_validation_only_omits_r_and_provides_caveat(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        vc = ValidatedCandidate(
            candidate=CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                                      n_gpu_layers=0, flash_attn=False,
                                      cache_type_k="f16", cache_type_v="f16"),
            pp_score=1.0, tg_score=1.0, balanced=1.0,
            pp_median=45.0, tg_median=46.0,
            pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
        )
        validation = ValidatedRecommendation(
            recommendation=vc, alternatives=(), failures=(),
            attempts_used=1, budget=10,
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            validation_result=validation,
        )
        data = json.loads(raw)
        ra = data["reproduction_argv"]
        assert len(ra) == 1, f"Expected 1 validation-only entry, got {len(ra)}"
        assert ra[0]["repetition_count"] is None, (
            f"Expected null repetition_count, got {ra[0]['repetition_count']}"
        )
        assert "-r" not in ra[0]["argv"], (
            "-r must be omitted when repetition_count is unknown"
        )
        assert "reproduction_notes" in data, (
            "Expected reproduction_notes caveat when unknown repetition counts exist"
        )

    def test_every_entry_has_repetition_count(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        c1 = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                             n_gpu_layers=0, flash_attn=False,
                             cache_type_k="f16", cache_type_v="f16")
        c2 = CandidateConfig(threads=8, batch_size=1024, ubatch_size=256,
                             n_gpu_layers=0, flash_attn=True,
                             cache_type_k="f16", cache_type_v="q8_0")
        obs1 = ScreeningObservation(candidate=c1, pp_samples=(45.0, 44.5, 45.5),
                                    tg_samples=(46.0, 45.5, 46.5))
        obs2 = ScreeningObservation(candidate=c2, pp_samples=(50.0, 49.5, 50.5),
                                    tg_samples=(51.0, 50.5, 51.5))
        baseline = ScreeningResult(candidate=c1, success=True, disposition="survived",
                                   observation=obs1)
        grid = ScreeningResult(candidate=c2, success=True, disposition="survived",
                               observation=obs2)
        r1 = RankedCandidate(candidate=c1, pp_score=1.0, tg_score=1.0, balanced=1.0,
                             pp_median=45.0, tg_median=46.0,
                             pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
                             total_repetitions=3)
        r2 = RankedCandidate(candidate=c2, pp_score=1.1, tg_score=1.1, balanced=1.1,
                             pp_median=50.0, tg_median=51.0,
                             pp_relative_dispersion=0.02, tg_relative_dispersion=0.02,
                             total_repetitions=3)
        report = ScreeningReport(
            profile=QUICK_PROFILE, baseline_disposition=baseline,
            grid_dispositions=(grid,), survivors=(r1, r2),
            leader=ProvisionalLeader(candidate=c2, objective="balanced",
                                     pp_score=1.1, tg_score=1.1, balanced=1.1,
                                     pp_median=50.0, tg_median=51.0),
            attempts_used=2, objective="balanced",
            disposition_summary="2 survivors",
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=report,
        )
        data = json.loads(raw)
        for entry in data["reproduction_argv"]:
            assert "repetition_count" in entry, (
                f"Missing repetition_count in entry: {entry['candidate']}"
            )

    def test_markdown_redacts_home_paths(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        home = str(Path.home())
        model_home = ModelFingerprint(
            path=Path(home) / "models/model.gguf",
            architecture="llama", parameter_count=7_000_000_000,
            context_length=4096, quantization="Q4_K_M",
            shards=(), symlink_warning=False, network_mount_detected=False,
            diagnostics=(),
        )
        tc_home = ToolchainFingerprint(
            binary_path=Path(home) / "bin/llama-bench",
            version="b1234",
            supported_flags=("-m", "-p", "-n", "-fa", "-o", "-r"),
            supported_output_formats=("json",),
            device_capabilities=("cpu",),
        )
        session_home = SessionSpecification(
            topology=sample_session.topology, cpu_only=sample_session.cpu_only,
            selected_device_ids=sample_session.selected_device_ids,
            n_ctx=sample_session.n_ctx, budget=sample_session.budget,
            advisory=sample_session.advisory,
            model=model_home, acknowledged=sample_session.acknowledged,
            fingerprint=sample_session.fingerprint,
        )
        c = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                            n_gpu_layers=0, flash_attn=False,
                            cache_type_k="f16", cache_type_v="f16")
        obs = ScreeningObservation(candidate=c, pp_samples=(45.0, 44.5, 45.5),
                                   tg_samples=(46.0, 45.5, 46.5))
        baseline = ScreeningResult(candidate=c, success=True, disposition="survived",
                                   observation=obs)
        ranked = RankedCandidate(candidate=c, pp_score=1.0, tg_score=1.0, balanced=1.0,
                                 pp_median=45.0, tg_median=46.0,
                                 pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
                                 total_repetitions=3)
        report = ScreeningReport(
            profile=QUICK_PROFILE, baseline_disposition=baseline,
            grid_dispositions=(), survivors=(ranked,),
            leader=ProvisionalLeader(candidate=c, objective="balanced",
                                     pp_score=1.0, tg_score=1.0, balanced=1.0,
                                     pp_median=45.0, tg_median=46.0),
            attempts_used=1, objective="balanced",
            disposition_summary="1 survivor",
        )
        md = render_export_markdown(
            session=session_home, toolchain=tc_home, model=model_home,
            screening_report=report,
        )
        assert "[HOME]" in md, "Markdown must contain [HOME]-redacted paths"
        assert home not in md.replace("[HOME]", ""), (
            "Raw home paths must not leak in Markdown"
        )

    def test_unsupported_pruned_excluded(
        self, sample_session: SessionSpecification,
        sample_toolchain: ToolchainFingerprint,
        sample_model: ModelFingerprint,
    ) -> None:
        c1 = CandidateConfig(threads=4, batch_size=512, ubatch_size=512,
                             n_gpu_layers=0, flash_attn=False,
                             cache_type_k="f16", cache_type_v="f16")
        c2 = CandidateConfig(threads=8, batch_size=1024, ubatch_size=256,
                             n_gpu_layers=0, flash_attn=True,
                             cache_type_k="f16", cache_type_v="f16")
        c3 = CandidateConfig(threads=2, batch_size=256, ubatch_size=128,
                             n_gpu_layers=0, flash_attn=False,
                             cache_type_k="f16", cache_type_v="f16")
        obs1 = ScreeningObservation(candidate=c1, pp_samples=(45.0, 44.5, 45.5),
                                    tg_samples=(46.0, 45.5, 46.5))
        # c1 is a survivor (has observation)
        baseline = ScreeningResult(candidate=c1, success=True, disposition="survived",
                                   observation=obs1)
        # c2 was unsupported with no observation — must be excluded
        grid_unsupported = ScreeningResult(
            candidate=c2, success=False, disposition="unsupported",
            reason="Unsupported flash_attn", observation=None,
        )
        # c3 was pruned with no observation — must be excluded
        grid_pruned = ScreeningResult(
            candidate=c3, success=False, disposition="pruned",
            reason="Pruned by budget", observation=None,
        )
        ranked1 = RankedCandidate(candidate=c1, pp_score=1.0, tg_score=1.0, balanced=1.0,
                                  pp_median=45.0, tg_median=46.0,
                                  pp_relative_dispersion=0.01, tg_relative_dispersion=0.01,
                                  total_repetitions=3)
        report = ScreeningReport(
            profile=QUICK_PROFILE, baseline_disposition=baseline,
            grid_dispositions=(grid_unsupported, grid_pruned),
            survivors=(ranked1,),
            leader=ProvisionalLeader(candidate=c1, objective="balanced",
                                     pp_score=1.0, tg_score=1.0, balanced=1.0,
                                     pp_median=45.0, tg_median=46.0),
            attempts_used=1, objective="balanced",
            disposition_summary="1 survivor, 1 unsupported, 1 pruned",
        )
        raw = render_export_json(
            session=sample_session, toolchain=sample_toolchain, model=sample_model,
            screening_report=report,
        )
        data = json.loads(raw)
        ra = data["reproduction_argv"]
        assert len(ra) == 1, (
            f"Expected only 1 entry (survivor c1), unsupported+pruned excluded. "
            f"Got {len(ra)}: {[e['candidate'] for e in ra]}"
        )
        assert ra[0]["candidate"]["threads"] == 4, "Only c1 (threads=4) should appear"
