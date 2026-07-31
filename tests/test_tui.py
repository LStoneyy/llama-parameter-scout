from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from textual.widgets import DirectoryTree, Input, Label, RichLog, Select, TabbedContent

from llama_parameter_scout.models import (
    Accelerator,
    CandidateConfig,
    ContextAdvisory,
    ExecutionPolicy,
    FailedValidation,
    ModelFingerprint,
    NctxComposition,
    NoWinnerReport,
    ProgressEvent,
    RankedCandidate,
    ResourceBudget,
    ScreeningObservation,
    ScreeningReport,
    ScreeningResult,
    SessionResumeResult,
    SessionSpecification,
    SingleRunRequest,
    SystemTopology,
    ToolchainFingerprint,
    ValidatedCandidate,
    ValidatedRecommendation,
)
from llama_parameter_scout.runner import FakeRunner
from llama_parameter_scout.search import QUICK_PROFILE
from llama_parameter_scout.tui import (
    ErrorScreen,
    GGUFDirectoryTree,
    ModelFileBrowser,
    PreparedSession,
    ProgressScreen,
    ResultsScreen,
    ReviewScreen,
    RunComplete,
    RunError,
    RunProgress,
    ScoutTuiApp,
    WizardScreen,
    _FileBrowser,
)


def _text(widget: Any) -> str:
    if isinstance(widget, RichLog):
        if widget.lines:
            return "\n".join(strip.text for strip in widget.lines)
        return ""
    if isinstance(widget, Label):
        return getattr(widget, "text", str(widget.render()))
    if hasattr(widget, "renderable"):
        return str(widget.renderable)
    if hasattr(widget, "render"):
        return str(widget.render())
    return str(widget)


async def _wait_for_richlog(
    pilot: Any, screen: Any, selector: str, expected: str, *, attempts: int = 20,
) -> str:
    """Poll ``selector``'s rendered text until it contains ``expected``.

    Textual's ``RichLog`` defers rendering until the widget receives a
    non-zero-width ``Resize`` event. A widget inside an inactive
    ``TabbedContent`` pane only receives that ``Resize`` once its pane is
    activated and laid out, and a single ``await pilot.pause()`` after
    switching ``tabs.active`` is not guaranteed to span the whole
    display-toggle -> layout -> Resize -> deferred-flush chain. Poll for a
    bounded number of iterations instead of asserting after one pause, so the
    test stays robust without relying on unconditional sleeps.
    """
    text = _text(screen.query_one(selector))
    for _ in range(attempts):
        if expected in text:
            return text
        await pilot.pause()
        text = _text(screen.query_one(selector))
    raise AssertionError(
        f"Expected {expected!r} not found in {selector!r} after {attempts} "
        f"pilot.pause() iterations; last rendered text was {text!r}"
    )


_SAMPLE_TOPO = SystemTopology(
    cpu_count=8, system_memory_bytes=17179869184,
    accelerators=(Accelerator(kind="gpu", backend="metal", device_id=0,
                              description="Apple M1", memory_bytes=17179869184),),
    memory_unified=True,
)

_NO_DEVICE_TOPO = SystemTopology(
    cpu_count=8, system_memory_bytes=17179869184,
    accelerators=(), memory_unified=True,
)


def _fake_detect_with_devices() -> SystemTopology:
    return _SAMPLE_TOPO


def _fake_detect_no_devices() -> SystemTopology:
    return _NO_DEVICE_TOPO


def _make_model() -> ModelFingerprint:
    return ModelFingerprint(
        path=Path("/tmp/model.gguf"), architecture="llama",
        parameter_count=7_000_000_000, context_length=4096,
        quantization="Q4_K_M", shards=(), symlink_warning=False,
        network_mount_detected=False, diagnostics=(),
    )


def _make_tc() -> ToolchainFingerprint:
    return ToolchainFingerprint(
        binary_path=Path("/tmp/llama-bench"), version="test",
        supported_flags=("-m", "-p", "-n", "-d", "-t", "-b", "-ub",
                         "-ngl", "-fa", "-ctk", "-ctv", "-o", "-r",
                         "--split-mode", "--main-gpu"),
        supported_output_formats=("json",),
        device_capabilities=("cpu", "metal"),
    )


def _make_spec() -> SessionSpecification:
    return SessionSpecification(
        topology=_SAMPLE_TOPO, cpu_only=True, selected_device_ids=(),
        n_ctx=NctxComposition(total=2048, prompt=512, generation=1024, depth=512),
        budget=ResourceBudget(ram_bytes=17179869184, vram_bytes=None, headroom_ratio=0.1),
        advisory=ContextAdvisory(
            likely_fit_lower=1000, likely_fit_upper=10000,
            uncertain_maximum=12000, model_weight_bytes=4000000000,
            kv_cache_bytes_estimate=5000000,
            note="Test advisory note",
        ),
        model=_make_model(), acknowledged=True,
        fingerprint="test-fp-1234-full-fingerprint",
    )


def _make_base() -> CandidateConfig:
    return CandidateConfig(
        threads=4, batch_size=512, ubatch_size=512, n_gpu_layers=0,
        flash_attn=False, cache_type_k="f16", cache_type_v="f16",
    )


def _make_vr() -> ValidatedRecommendation:
    return ValidatedRecommendation(
        recommendation=ValidatedCandidate(
            candidate=_make_base(),
            pp_score=1.0, tg_score=1.0, balanced=1.0,
            pp_median=45.0, tg_median=45.0,
            pp_relative_dispersion=0.05, tg_relative_dispersion=0.05,
            label="Full-context validation - verified at requested context",
        ),
        alternatives=(), failures=(), attempts_used=1, budget=10,
    )


def _make_prepared(app_instance: Any) -> PreparedSession:
    spec = _make_spec()
    return PreparedSession(
        app=app_instance, spec=spec, toolchain=_make_tc(),
        model=_make_model(),
        request=SingleRunRequest(
            binary_path=Path("/tmp/llama-bench"),
            model_path=Path("/tmp/model.gguf"),
            journal_path=Path("/tmp/j.json"),
            n_ctx=spec.n_ctx, candidate=_make_base(),
        ),
        profile=QUICK_PROFILE, runner=FakeRunner(),
        base_candidate=_make_base(), n_ctx=spec.n_ctx,
    )


class FakeApp:
    def __init__(self) -> None:
        self.calls: dict[str, list[Any]] = {
            "setup_session": [],
            "start_or_resume_session": [],
            "screen_candidates": [],
            "validate_candidates": [],
            "collect_mtp_profile": [],
            "export_report_markdown": [],
        }
        self.screening_began: threading.Event | None = None
        self.screening_policy: ExecutionPolicy | None = None

    def setup_session(self, **kw: Any) -> SessionSpecification:
        self.calls["setup_session"].append(kw)
        return _make_spec()

    def start_or_resume_session(self, **kw: Any) -> SessionResumeResult:
        self.calls["start_or_resume_session"].append(kw)
        return SessionResumeResult(action="new", session_id="fake-sid")

    def screen_candidates(
        self, request: Any, profile: Any,
        *, policy: ExecutionPolicy | None = None,
        on_progress: Any = None, **kw: Any,
    ) -> ScreeningReport:
        self.calls["screen_candidates"].append((request, profile, kw))
        if policy is not None:
            self.screening_policy = policy
        if self.screening_began is not None:
            self.screening_began.set()
            if policy is not None and policy.cancellation is not None:
                policy.cancellation.wait(timeout=10)
        if on_progress:
            on_progress(ProgressEvent(stage="screening", message="fake eval"))
        return ScreeningReport(
            profile=profile,
            baseline_disposition=ScreeningResult(
                candidate=_make_base(), success=True, disposition="survived",
            ),
            grid_dispositions=(), survivors=(), leader=None,
            attempts_used=1, objective="balanced", disposition_summary="Fake",
        )

    def validate_candidates(
        self, request: Any, screening_report: Any,
        *, on_progress: Any = None, **kw: Any,
    ) -> ValidatedRecommendation | NoWinnerReport:
        self.calls["validate_candidates"].append((request, screening_report, kw))
        if on_progress:
            on_progress(ProgressEvent(stage="validation", message="fake validate"))
        return _make_vr()

    def collect_mtp_profile(self, **kw: Any) -> None:
        self.calls["collect_mtp_profile"].append(kw)
        return None

    def export_report_markdown(self, **kw: Any) -> str:
        self.calls["export_report_markdown"].append(kw)
        return "# Validated Report\n\n## Results\n\nn_ctx = 2048"


def _tui_kw() -> dict[str, Any]:
    return {
        "inspect_toolchain_fn": lambda r, b: _make_tc(),
        "inspect_model_fn": lambda p: _make_model(),
    }


def _setup_wizard(app: ScoutTuiApp, tmp_path: Path) -> None:
    ws = app.screen
    assert isinstance(ws, WizardScreen)
    bench = tmp_path / "llama-bench"
    bench.write_text("#!/bin/sh\necho ok")
    bench.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_text("GGUF")
    ws.query_one("#llama_bench_path", Input).value = str(bench)
    ws.query_one("#model_path", Input).value = str(model)
    ws.query_one("#journal_path", Input).value = str(tmp_path / "j.json")


# ── Wizard screen tests ─────────────────────────────────────────


async def test_wizard_title() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert "Wizard" in _text(app.screen.query_one("#wizard-title"))


async def test_wizard_mtp_fields_hidden_by_default() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.screen.query_one("#draft_model_row").display
        assert not app.screen.query_one("#server_path").display


async def test_wizard_mtp_toggle_shows_fields() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#mtp_enabled")
        await pilot.pause()
        assert app.screen.query_one("#draft_model_row").display
        assert app.screen.query_one("#server_path").display


async def test_wizard_validation_empty_paths() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws.action_next_screen()
        await pilot.pause()
        text = _text(ws.query_one("#wizard_errors"))
        assert "llama-bench path is required" in text
        assert "Model path is required" in text
        assert "Journal path is required" not in text


async def test_wizard_default_journal_path(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        expected = str(tmp_path / "journal.json")
        assert ws.query_one("#journal_path", Input).value == expected
        assert app.wizard_state.journal_path == expected


async def test_wizard_validation_cleared_journal_shows_required() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws.query_one("#journal_path", Input).value = ""
        ws.action_next_screen()
        await pilot.pause()
        text = _text(ws.query_one("#wizard_errors"))
        assert "Journal path is required" in text


async def test_wizard_validation_bad_nctx() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws.query_one("#n_ctx_total", Input).value = "500"
        ws.query_one("#n_ctx_prompt", Input).value = "600"
        ws.query_one("#n_ctx_generation", Input).value = "10"
        ws.query_one("#n_ctx_depth", Input).value = "10"
        ws.action_next_screen()
        await pilot.pause()
        assert "do not sum to total" in _text(ws.query_one("#wizard_errors"))


async def test_wizard_validation_mtp_requires_paths() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#mtp_enabled")
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws.action_next_screen()
        await pilot.pause()
        text = _text(ws.query_one("#wizard_errors"))
        assert "Draft model path is required" in text
        assert "llama-server path is required" in text


async def test_wizard_quit_button_exits(monkeypatch: Any) -> None:
    app = ScoutTuiApp()
    async with app.run_test(size=(80, 80)) as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        exit_calls: list[Any] = []
        monkeypatch.setattr(app, "exit", lambda *a, **kw: exit_calls.append((a, kw)))
        await pilot.click("#quit_btn")
        await pilot.pause()
        assert len(exit_calls) == 1


# ── Device detection tests ──────────────────────────────────────


async def test_wizard_device_detection_shows_accelerators(tmp_path: Path) -> None:
    app = ScoutTuiApp(detect_topology_fn=_fake_detect_with_devices)
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(50):
            await pilot.pause()
            status = _text(app.screen.query_one("#device_status"))
            if "accelerator" in status.lower() or "detected" in status.lower():
                break
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        status = _text(app.screen.query_one("#device_status"))
        assert "accelerator" in status or "detected" in status
        ws.query_one("#device", Select).value = "gpu:0"
        ws.query_one("#llama_bench_path", Input).value = "/tmp/llama-bench"
        ws.query_one("#model_path", Input).value = "/tmp/model.gguf"
        ws.query_one("#journal_path", Input).value = "/tmp/j.json"
        ws._gather_and_validate()
        assert not app.wizard_state.cpu_only
        assert app.wizard_state.device_id == "0"


async def test_wizard_device_detection_no_devices() -> None:
    app = ScoutTuiApp(detect_topology_fn=_fake_detect_no_devices)
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(50):
            await pilot.pause()
            status = _text(app.screen.query_one("#device_status"))
            if "accelerator" in status.lower() or "No" in status:
                break
        status = _text(app.screen.query_one("#device_status"))
        assert "No accelerators" in status
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws.query_one("#device", Select).value = "cpu_only"
        ws.query_one("#llama_bench_path", Input).value = "/tmp/llama-bench"
        ws.query_one("#model_path", Input).value = "/tmp/model.gguf"
        ws.query_one("#journal_path", Input).value = "/tmp/j.json"
        ws._gather_and_validate()
        assert app.wizard_state.cpu_only
        assert app.wizard_state.device_id == ""


async def test_wizard_device_detection_failure() -> None:
    def _fail() -> SystemTopology:
        raise RuntimeError("Probing failed")

    app = ScoutTuiApp(detect_topology_fn=_fail)
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(50):
            await pilot.pause()
            status = _text(app.screen.query_one("#device_status"))
            if "Detection" in status or "Detecting" not in status:
                break
        status = _text(app.screen.query_one("#device_status"))
        assert "Detection" in status


# ── Model file browser tests ────────────────────────────────────


async def test_model_file_browser_root_home() -> None:
    browser = ModelFileBrowser()
    assert browser._root == Path.home()


async def test_model_file_browser_custom_root(tmp_path: Path) -> None:
    browser = ModelFileBrowser(root=tmp_path)
    assert browser._root == tmp_path


async def test_model_file_browser_filters_non_gguf(tmp_path: Path) -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        browser = ModelFileBrowser(root=tmp_path)
        app.push_screen(browser)
        await pilot.pause()

        txt_file = tmp_path / "data.txt"
        txt_file.write_text("")
        gguf_file = tmp_path / "model.gguf"
        gguf_file.write_text("")

        tree = browser.query_one(GGUFDirectoryTree)
        filtered = list(tree.filter_paths([txt_file, gguf_file, tmp_path]))
        assert txt_file not in filtered
        assert gguf_file in filtered
        assert tmp_path in filtered


async def test_model_file_browser_select_gguf_callback(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_text("")
    selected: list[Path] = []
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        browser = ModelFileBrowser(root=tmp_path)
        app.push_screen(browser, lambda p: selected.append(p) if p else None)
        await pilot.pause()
        browser._on_file_selected(
            type("_E", (), {"path": gguf})()
        )
        await pilot.pause()
        assert len(selected) == 1
        assert selected[0] == gguf


async def test_model_file_browser_select_non_gguf_noop(tmp_path: Path) -> None:
    txt = tmp_path / "data.txt"
    txt.write_text("")
    selected: list[Path] = []
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        browser = ModelFileBrowser(root=tmp_path)
        app.push_screen(browser, lambda p: selected.append(p) if p else None)
        await pilot.pause()
        browser._on_file_selected(
            type("_E", (), {"path": txt})()
        )
        await pilot.pause()
        assert len(selected) == 0


async def test_wizard_browse_model_populates_input(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_text("")
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws._open_model_browser()
        await pilot.pause()
        assert isinstance(app.screen, ModelFileBrowser)
        app.screen._on_file_selected(type("_E", (), {"path": gguf})())
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert ws.query_one("#model_path", Input).value == str(gguf)


async def test_browser_escape_cancel_returns_without_selection(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_text("")
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        initial = ws.query_one("#model_path", Input).value
        ws._open_model_browser()
        await pilot.pause()
        assert isinstance(app.screen, ModelFileBrowser)
        app.screen.action_cancel()
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert ws.query_one("#model_path", Input).value == initial


async def test_browser_cancel_button_returns_without_selection(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_text("")
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        initial = ws.query_one("#model_path", Input).value
        ws._open_model_browser()
        await pilot.pause()
        assert isinstance(app.screen, ModelFileBrowser)
        await pilot.click("#cancel_btn")
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert ws.query_one("#model_path", Input).value == initial


# ── llama-bench and journal browse tests ──────────────────────────


async def test_wizard_browse_bench_populates_input(tmp_path: Path) -> None:
    bench = tmp_path / "my-llama-bench"
    bench.write_text("")
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws._open_bench_browser()
        await pilot.pause()
        assert isinstance(app.screen, _FileBrowser)
        app.screen._on_file_selected(type("_E", (), {"path": bench})())
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert ws.query_one("#llama_bench_path", Input).value == str(bench)


async def test_wizard_browse_journal_populates_input(tmp_path: Path) -> None:
    jfile = tmp_path / "journal.json"
    jfile.write_text("")
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws._open_journal_browser()
        await pilot.pause()
        assert isinstance(app.screen, _FileBrowser)
        app.screen._on_file_selected(type("_E", (), {"path": jfile})())
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert ws.query_one("#journal_path", Input).value == str(jfile)


async def test_wizard_browse_journal_select_non_json_noop(tmp_path: Path) -> None:
    txt = tmp_path / "data.txt"
    txt.write_text("")
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        initial = ws.query_one("#journal_path", Input).value
        ws._open_journal_browser()
        await pilot.pause()
        assert isinstance(app.screen, _FileBrowser)
        app.screen._on_file_selected(type("_E", (), {"path": txt})())
        await pilot.pause()
        # Browser stays open (no dismiss) — input unchanged
        assert not isinstance(app.screen, WizardScreen)
        assert isinstance(app.screen, _FileBrowser)
        app.screen.action_cancel()
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert ws.query_one("#journal_path", Input).value == initial


async def test_wizard_mmproj_input_present() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        mmproj_input = ws.query_one("#mmproj_path", Input)
        assert mmproj_input is not None
        assert mmproj_input.placeholder == "/path/to/mmproj.gguf"


async def test_wizard_mmproj_browse_populates_input(tmp_path: Path) -> None:
    gguf = tmp_path / "mmproj.gguf"
    gguf.write_text("")
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws._open_mmproj_browser()
        await pilot.pause()
        assert isinstance(app.screen, ModelFileBrowser)
        app.screen._on_file_selected(type("_E", (), {"path": gguf})())
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert ws.query_one("#mmproj_path", Input).value == str(gguf)


async def test_wizard_mmproj_empty_is_valid(tmp_path: Path) -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        bench = tmp_path / "llama-bench"
        bench.write_text("#!/bin/sh\necho ok")
        bench.chmod(0o755)
        model = tmp_path / "model.gguf"
        model.write_text("GGUF")
        ws.query_one("#llama_bench_path", Input).value = str(bench)
        ws.query_one("#model_path", Input).value = str(model)
        ws.query_one("#journal_path", Input).value = str(tmp_path / "j.json")
        ws.query_one("#mmproj_path", Input).value = ""
        errs = ws._gather_and_validate()
        assert not any("MMProj" in e for e in errs)


async def test_wizard_mmproj_nonexistent_reports_error(tmp_path: Path) -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        bench = tmp_path / "llama-bench"
        bench.write_text("#!/bin/sh\necho ok")
        bench.chmod(0o755)
        model = tmp_path / "model.gguf"
        model.write_text("GGUF")
        ws.query_one("#llama_bench_path", Input).value = str(bench)
        ws.query_one("#model_path", Input).value = str(model)
        ws.query_one("#journal_path", Input).value = str(tmp_path / "j.json")
        nonexistent = str(tmp_path / "nonexistent.gguf")
        ws.query_one("#mmproj_path", Input).value = nonexistent
        errs = ws._gather_and_validate()
        assert any("MMProj not found" in e for e in errs)


async def test_wizard_mmproj_browse_filters_non_gguf(tmp_path: Path) -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws._open_mmproj_browser()
        await pilot.pause()
        assert isinstance(app.screen, ModelFileBrowser)
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("")
        gguf_file = tmp_path / "mmproj.gguf"
        gguf_file.write_text("")
        tree = app.screen.query_one(GGUFDirectoryTree)
        filtered = list(tree.filter_paths([txt_file, gguf_file, tmp_path]))
        assert txt_file not in filtered
        assert gguf_file in filtered


async def test_wizard_browse_journal_filters_non_json(tmp_path: Path) -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws._open_journal_browser()
        await pilot.pause()
        assert isinstance(app.screen, _FileBrowser)
        tree = app.screen.query_one("#file_tree", DirectoryTree)
        txt = tmp_path / "data.txt"
        txt.write_text("")
        json_lower = tmp_path / "data.json"
        json_lower.write_text("")
        json_upper = tmp_path / "data.JSON"
        json_upper.write_text("")
        filtered = list(tree.filter_paths([txt, json_lower, json_upper, tmp_path]))
        assert txt not in filtered
        assert json_lower in filtered
        assert json_upper in filtered
        assert tmp_path in filtered


async def test_wizard_browse_journal_cancel_leaves_input_unchanged(tmp_path: Path) -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        initial = ws.query_one("#journal_path", Input).value
        ws._open_journal_browser()
        await pilot.pause()
        assert isinstance(app.screen, _FileBrowser)
        app.screen.action_cancel()
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert ws.query_one("#journal_path", Input).value == initial


async def test_device_select_stays_cpu_only_after_no_device_detection() -> None:
    app = ScoutTuiApp(detect_topology_fn=_fake_detect_no_devices)
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(50):
            await pilot.pause()
            status = _text(app.screen.query_one("#device_status"))
            if "No accelerators" in status or "Detecting" not in status:
                break
        select = app.screen.query_one("#device", Select)
        assert select.value == "cpu_only"


async def test_device_select_stays_cpu_only_after_detection_with_devices() -> None:
    app = ScoutTuiApp(detect_topology_fn=_fake_detect_with_devices)
    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(50):
            await pilot.pause()
            status = _text(app.screen.query_one("#device_status"))
            if "accelerator" in status.lower() or "detected" in status.lower():
                break
        select = app.screen.query_one("#device", Select)
        assert select.value == "cpu_only"


# ── Context preset tests ────────────────────────────────────────


async def test_wizard_context_preset_updates_values() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws.query_one("#context_preset", Select).value = "4096"
        await pilot.pause()
        assert ws.query_one("#n_ctx_total", Input).value == "4096"
        assert ws.query_one("#n_ctx_prompt", Input).value == "1024"
        assert ws.query_one("#n_ctx_generation", Input).value == "2048"
        assert ws.query_one("#n_ctx_depth", Input).value == "1024"


async def test_wizard_context_preset_hides_custom_fields() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        assert not ws.query_one("#lbl_n_ctx_total").display
        ws.query_one("#context_preset", Select).value = "custom"
        await pilot.pause()
        assert ws.query_one("#lbl_n_ctx_total").display
        assert ws.query_one("#n_ctx_total", Input).display
        ws.query_one("#context_preset", Select).value = "8192"
        await pilot.pause()
        assert not ws.query_one("#lbl_n_ctx_total").display


async def test_wizard_context_preset_bad_sum_still_validated() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws.query_one("#context_preset", Select).value = "custom"
        await pilot.pause()
        ws.query_one("#n_ctx_total", Input).value = "500"
        ws.query_one("#n_ctx_prompt", Input).value = "600"
        ws.query_one("#n_ctx_generation", Input).value = "10"
        ws.query_one("#n_ctx_depth", Input).value = "10"
        ws.action_next_screen()
        await pilot.pause()
        assert "do not sum to total" in _text(ws.query_one("#wizard_errors"))


async def test_wizard_context_preset_restores_from_state() -> None:
    app = ScoutTuiApp()
    state = app.wizard_state
    state.n_ctx_total = "8192"
    state.n_ctx_prompt = "2048"
    state.n_ctx_generation = "4096"
    state.n_ctx_depth = "2048"
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        assert ws.query_one("#context_preset", Select).value == "8192"
        assert not ws.query_one("#lbl_n_ctx_total").display


async def test_wizard_context_preset_restores_custom() -> None:
    app = ScoutTuiApp()
    state = app.wizard_state
    state.n_ctx_total = "3000"
    state.n_ctx_prompt = "1000"
    state.n_ctx_generation = "1500"
    state.n_ctx_depth = "500"
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        assert ws.query_one("#context_preset", Select).value == "custom"
        assert ws.query_one("#lbl_n_ctx_total").display


# ── KV Cache type tests ──────────────────────────────────────────


async def test_wizard_kv_cache_defaults() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        assert ws.query_one("#cache_type_k", Select).value == "f16"
        assert ws.query_one("#cache_type_v", Select).value == "f16"
        assert app.wizard_state.cache_type_k == "f16"
        assert app.wizard_state.cache_type_v == "f16"


async def test_wizard_kv_cache_selection_reaches_setup_session(tmp_path: Path) -> None:
    fa = FakeApp()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    async with app.run_test() as pilot:
        await pilot.pause()
        _setup_wizard(app, tmp_path)
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        ws.query_one("#cache_type_k", Select).value = "q4_0"
        ws.query_one("#cache_type_v", Select).value = "q8_0"
        await pilot.pause()
        ws._gather_and_validate()
        assert app.wizard_state.cache_type_k == "q4_0"
        assert app.wizard_state.cache_type_v == "q8_0"
        await pilot.press("f10")
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        for _ in range(100):
            if not app.screen.query_one("#start_btn").disabled:
                break
            await pilot.pause()
        call = fa.calls["setup_session"][-1]
        candidate = call["candidate"]
        assert candidate.cache_type_k == "q4_0"
        assert candidate.cache_type_v == "q8_0"
        content = _text(app.screen.query_one("#review-content"))
        assert "KV cache: K=q4_0, V=q8_0" in content


# ── Review screen ───────────────────────────────────────────────


async def test_review_shows_full_spec(tmp_path: Path) -> None:
    fa = FakeApp()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    async with app.run_test() as pilot:
        await pilot.pause()
        _setup_wizard(app, tmp_path)
        await pilot.press("f10")
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        for _ in range(100):
            if not app.screen.query_one("#start_btn").disabled:
                break
            await pilot.pause()
        content = _text(app.screen.query_one("#review-content"))
        assert "llama-bench" in content
        assert "test-fp-1234-full-fingerprint" in content
        assert "Test advisory note" in content


async def test_review_shows_mmproj_when_set(tmp_path: Path) -> None:
    fa = FakeApp()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    async with app.run_test() as pilot:
        await pilot.pause()
        _setup_wizard(app, tmp_path)
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        mmproj_file = tmp_path / "mmproj.gguf"
        mmproj_file.write_text("GGUF")
        ws.query_one("#mmproj_path", Input).value = str(mmproj_file)
        await pilot.press("f10")
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        for _ in range(100):
            if not app.screen.query_one("#start_btn").disabled:
                break
            await pilot.pause()
        content = _text(app.screen.query_one("#review-content"))
        assert "MMProj" in content
        assert str(mmproj_file) in content


async def test_review_mmproj_not_shown_when_empty(tmp_path: Path) -> None:
    fa = FakeApp()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    async with app.run_test() as pilot:
        await pilot.pause()
        _setup_wizard(app, tmp_path)
        await pilot.press("f10")
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        for _ in range(100):
            if not app.screen.query_one("#start_btn").disabled:
                break
            await pilot.pause()
        content = _text(app.screen.query_one("#review-content"))
        assert "MMProj" not in content


async def test_review_never_calls_start_or_resume(tmp_path: Path) -> None:
    fa = FakeApp()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    async with app.run_test() as pilot:
        await pilot.pause()
        _setup_wizard(app, tmp_path)
        ws2 = app.screen
        assert isinstance(ws2, WizardScreen)
        ws2.action_next_screen()
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        for _ in range(100):
            if not app.screen.query_one("#start_btn").disabled:
                break
            await pilot.pause()
        assert len(fa.calls.get("start_or_resume_session", [])) == 0
        assert len(fa.calls.get("setup_session", [])) > 0


async def test_review_quit_button_exits(tmp_path: Path, monkeypatch: Any) -> None:
    fa = FakeApp()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        _setup_wizard(app, tmp_path)
        await pilot.press("f10")
        await pilot.pause()
        rs = app.screen
        assert isinstance(rs, ReviewScreen)
        exit_calls: list[Any] = []
        monkeypatch.setattr(app, "exit", lambda *a, **kw: exit_calls.append((a, kw)))
        await pilot.click("#quit_btn")
        await pilot.pause()
        assert len(exit_calls) == 1


# ── End-to-end flow via Pilot ────────────────────────────────────


async def test_end_to_end_full_flow(tmp_path: Path) -> None:
    fa = FakeApp()
    app = ScoutTuiApp(
        app_factory=lambda r: fa, runner_factory=lambda: FakeRunner(),
        detect_topology_fn=_fake_detect_no_devices,
        **_tui_kw(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        _setup_wizard(app, tmp_path)
        # F10 from Wizard -> Review
        await pilot.press("f10")
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        # Wait for review preparation
        for _ in range(100):
            if not app.screen.query_one("#start_btn").disabled:
                break
            await pilot.pause()
        assert len(fa.calls.get("setup_session", [])) == 1
        assert len(fa.calls.get("start_or_resume_session", [])) == 0
        # Check review content
        content = _text(app.screen.query_one("#review-content"))
        assert "test-fp-1234-full-fingerprint" in content
        # F10 from Review -> Progress (starts worker)
        await pilot.press("f10")
        await pilot.pause()
        # Wait for natural worker completion -> Results
        for _ in range(200):
            if isinstance(app.screen, ResultsScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, ResultsScreen)
        # Assert all seam calls made exactly once
        assert len(fa.calls.get("setup_session", [])) == 1
        assert len(fa.calls.get("start_or_resume_session", [])) == 1
        assert len(fa.calls.get("screen_candidates", [])) == 1
        assert len(fa.calls.get("validate_candidates", [])) == 1
        assert len(fa.calls.get("export_report_markdown", [])) == 1
        # Assert content visible in results
        md_text = _text(app.screen.query_one("#md-content"))
        assert "Validated" in md_text
        # Assert export was called
        assert len(fa.calls["export_report_markdown"]) == 1


async def test_end_to_end_with_mtp(tmp_path: Path) -> None:
    fa = FakeApp()
    app = ScoutTuiApp(
        app_factory=lambda r: fa, runner_factory=lambda: FakeRunner(),
        detect_topology_fn=_fake_detect_no_devices,
        **_tui_kw(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        _setup_wizard(app, tmp_path)
        draft = tmp_path / "draft.gguf"
        draft.write_text("GGUF")
        srv = tmp_path / "llama-server"
        srv.write_text("#!/bin/sh\necho ok")
        srv.chmod(0o755)
        ws.query_one("#draft_model_path", Input).value = str(draft)
        ws.query_one("#server_path", Input).value = str(srv)
        await pilot.click("#mtp_enabled")
        await pilot.pause()
        await pilot.press("f10")
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        for _ in range(100):
            if not app.screen.query_one("#start_btn").disabled:
                break
            await pilot.pause()
        await pilot.press("f10")
        await pilot.pause()
        for _ in range(200):
            if isinstance(app.screen, ResultsScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, ResultsScreen)
        assert len(fa.calls.get("setup_session", [])) == 1
        assert len(fa.calls.get("start_or_resume_session", [])) == 1
        assert len(fa.calls.get("screen_candidates", [])) == 1
        assert len(fa.calls.get("validate_candidates", [])) == 1
        assert len(fa.calls.get("collect_mtp_profile", [])) == 1
        assert len(fa.calls.get("export_report_markdown", [])) == 1


async def test_cancellation_during_screening(tmp_path: Path) -> None:
    fa = FakeApp()
    fa.screening_began = threading.Event()
    app = ScoutTuiApp(
        app_factory=lambda r: fa, runner_factory=lambda: FakeRunner(),
        detect_topology_fn=_fake_detect_no_devices,
        **_tui_kw(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        _setup_wizard(app, tmp_path)
        await pilot.press("f10")
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        for _ in range(100):
            if not app.screen.query_one("#start_btn").disabled:
                break
            await pilot.pause()
        await pilot.press("f10")
        await pilot.pause()
        # Wait for fake screening to begin (screen_candidates called)
        fa.screening_began.wait(timeout=10)
        # Verify policy cancellation event captured
        assert fa.screening_policy is not None
        assert fa.screening_policy.cancellation is not None
        orig_event = fa.screening_policy.cancellation
        assert not orig_event.is_set()
        # Press Esc to cancel
        await pilot.press("escape")
        # Wait for cancellation to be processed and Results to appear
        await pilot.pause()
        orig_event.wait(timeout=5)
        assert orig_event.is_set()
        for _ in range(100):
            if isinstance(app.screen, ResultsScreen):
                break
            await pilot.pause()
        assert isinstance(app.screen, ResultsScreen)
        md_text = _text(app.screen.query_one("#md-content"))
        assert "Cancelled" in md_text


async def test_progress_quit_button_exits(monkeypatch: Any) -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        monkeypatch.setattr(ProgressScreen, "run_benchmark", lambda self: None)
        ps = ProgressScreen()
        app.push_screen(ps)
        await pilot.pause()
        assert not ps._cancellation_event.is_set()
        exit_calls: list[Any] = []
        monkeypatch.setattr(app, "exit", lambda *a, **kw: exit_calls.append((a, kw)))
        await pilot.click("#quit_btn")
        await pilot.pause()
        assert ps._cancellation_event.is_set()
        assert len(exit_calls) == 1

async def test_progress_ctrlq_triggers_quit(monkeypatch: Any) -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        monkeypatch.setattr(ProgressScreen, "run_benchmark", lambda self: None)
        ps = ProgressScreen()
        app.push_screen(ps)
        await pilot.pause()
        assert not ps._cancellation_event.is_set()
        exit_calls: list[Any] = []
        monkeypatch.setattr(app, "exit", lambda *a, **kw: exit_calls.append((a, kw)))
        await pilot.press("ctrl+q")
        await pilot.pause()
        assert ps._cancellation_event.is_set()
        assert len(exit_calls) == 1


# ── Progress screen (unit) ──────────────────────────────────────


class _ManualProgressScreen(ProgressScreen):
    """ProgressScreen whose on_mount does not start the benchmark worker.

    The real screen starts ``run_benchmark`` from ``on_mount``; with
    ``app.prepared`` left unset that worker posts a spurious ``RunError``.
    Tests that drive ``RunComplete``/``RunError`` messages directly must
    mount this stub so no background worker races the manually posted
    message (which was the source of a Windows CI flake).
    """

    def on_mount(self) -> None:
        pass


async def test_progress_displays_events() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        ps = _ManualProgressScreen()
        app.push_screen(ps)
        await pilot.pause()
        ps.post_message(RunProgress(ProgressEvent(
            stage="screening", message="Evaluating...",
        )))
        await pilot.pause()
        assert "screening" in _text(ps.query_one("#progress-log"))


async def test_progress_displays_error() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        ps = _ManualProgressScreen()
        app.push_screen(ps)
        await pilot.pause()
        ps.post_message(RunError("Something broke"))
        await pilot.pause()
        assert "ERROR" in _text(ps.query_one("#progress-log"))


async def test_progress_complete_goes_to_results() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        ps = _ManualProgressScreen()
        app.push_screen(ps)
        await pilot.pause()
        ps.post_message(RunComplete(
            session_spec=None, screening_report=None,
            validation_result=None, mtp_profile=None,
            toolchain=None, model=None, session_result=None, request=None,
        ))
        await pilot.pause()
        assert isinstance(app.screen, ResultsScreen)
        tabs = app.screen.query_one(TabbedContent)
        assert tabs.active == "tab-markdown"
        md_text = await _wait_for_richlog(pilot, app.screen, "#md-content", "# Results")
        assert "encountered an error" in md_text


async def test_progress_cancelled_shows_results() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        ps = _ManualProgressScreen()
        app.push_screen(ps)
        await pilot.pause()
        ps.post_message(RunComplete(
            session_spec=None, screening_report=None,
            validation_result=None, mtp_profile=None,
            toolchain=None, model=None, session_result=None, request=None,
            cancelled=True,
        ))
        await pilot.pause()
        assert isinstance(app.screen, ResultsScreen)
        tabs = app.screen.query_one(TabbedContent)
        assert tabs.active == "tab-markdown"
        md_text = await _wait_for_richlog(
            pilot, app.screen, "#md-content", "# Benchmark Cancelled"
        )
        assert "cancelled before completion" in md_text


async def test_cancellation_sets_event() -> None:
    ps = ProgressScreen()
    assert not ps._cancellation_event.is_set()
    ps.action_cancel_run()
    assert ps._cancellation_event.is_set()


# ── Results screen ──────────────────────────────────────────────


async def test_results_cancelled_placeholder() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        app.run_result = {"cancelled": True}
        app.push_screen(ResultsScreen())
        await pilot.pause()
        assert "Cancelled" in _text(app.screen.query_one("#md-content"))


async def test_results_no_results() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        app.push_screen(ResultsScreen())
        await pilot.pause()
        assert "No results" in _text(app.screen.query_one("#md-content"))


async def test_results_clipboard_fallback(monkeypatch: Any) -> None:
    monkeypatch.setattr("llama_parameter_scout.tui._copy_to_clipboard", lambda _: False)
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        app.run_result = {"cancelled": True}
        app.push_screen(ResultsScreen())
        await pilot.pause()
        assert isinstance(app.screen, ResultsScreen)
        app.screen.action_copy_markdown()
        await pilot.pause()
        assert "Clipboard unavailable" in _text(app.screen.query_one("#clipboard-msg"))


async def test_results_save_all(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("llama_parameter_scout.tui._copy_to_clipboard", lambda _: False)
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        app.run_result = {"cancelled": True}
        app.push_screen(ResultsScreen())
        await pilot.pause()
        cwd = Path.cwd()
        try:
            import os
            os.chdir(str(tmp_path))
            assert isinstance(app.screen, ResultsScreen)
            app.screen.action_save_all()
            await pilot.pause()
            assert (tmp_path / "scout-report.md").exists()
        finally:
            os.chdir(str(cwd))


async def test_results_validated_shows_report() -> None:
    fa = FakeApp()
    spec = _make_spec()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    app.prepared = _make_prepared(fa)
    app.run_result = {
        "session_spec": spec, "screening_report": None,
        "validation_result": _make_vr(), "mtp_profile": None,
        "toolchain": _make_tc(), "model": _make_model(),
        "session_result": None, "request": None, "cancelled": False,
    }
    async with app.run_test() as pilot:
        app.push_screen(ResultsScreen())
        await pilot.pause()
        md = _text(app.screen.query_one("#md-content"))
        assert "Validated" in md or "Report" in md


async def test_results_preset_export_plain_text(tmp_path: Path, monkeypatch: Any) -> None:
    fa = FakeApp()
    spec = _make_spec()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    app.prepared = _make_prepared(fa)
    app.run_result = {
        "session_spec": spec, "screening_report": None,
        "validation_result": _make_vr(), "mtp_profile": None,
        "toolchain": _make_tc(), "model": _make_model(),
        "session_result": None, "request": None, "cancelled": False,
    }

    def _fake_copy(t: str) -> bool:
        clipboard_text.append(t)
        return True

    clipboard_text: list[str] = []
    monkeypatch.setattr(
        "llama_parameter_scout.tui._copy_to_clipboard",
        _fake_copy,
    )

    async with app.run_test() as pilot:
        app.push_screen(ResultsScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ResultsScreen)
        tabs = screen.query_one(TabbedContent)
        tabs.active = "tab-preset"
        visible = await _wait_for_richlog(pilot, screen, "#preset-content", "ctx-size =")
        assert visible.startswith("[model]")

        screen.action_copy_preset()
        await pilot.pause()
        assert len(clipboard_text) == 1
        copied = clipboard_text[0]
        assert copied.startswith("[model]")
        assert "Strip(" not in copied, (
            f"BUG: action_copy_preset exported Strip( repr.\n{copied}"
        )

        cwd = Path.cwd()
        try:
            import os
            os.chdir(str(tmp_path))
            screen.action_save_all()
            await pilot.pause()
            saved = (tmp_path / "preset.ini").read_text()
            assert saved.startswith("[model]")
            assert "Strip(" not in saved, (
                f"BUG: action_save_all wrote Strip( repr.\n{saved}"
            )
        finally:
            os.chdir(str(cwd))


async def test_results_preset_includes_mmproj_when_set(tmp_path: Path) -> None:
    fa = FakeApp()
    spec = _make_spec()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    mmproj_file = tmp_path / "mmproj.gguf"
    mmproj_file.write_text("GGUF")
    prepared = _make_prepared(fa)
    prepared.mmproj_path = str(mmproj_file)
    app.prepared = prepared
    app.run_result = {
        "session_spec": spec, "screening_report": None,
        "validation_result": _make_vr(), "mtp_profile": None,
        "toolchain": _make_tc(), "model": _make_model(),
        "session_result": None, "request": None, "cancelled": False,
    }
    async with app.run_test() as pilot:
        app.push_screen(ResultsScreen())
        await pilot.pause()
        tabs = app.screen.query_one(TabbedContent)
        tabs.active = "tab-preset"
        visible = await _wait_for_richlog(pilot, app.screen, "#preset-content", "mmproj =")
        assert str(mmproj_file) in visible
        assert "mmproj =" in visible


async def test_results_preset_no_mmproj_when_empty(tmp_path: Path) -> None:
    fa = FakeApp()
    spec = _make_spec()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    app.prepared = _make_prepared(fa)
    app.run_result = {
        "session_spec": spec, "screening_report": None,
        "validation_result": _make_vr(), "mtp_profile": None,
        "toolchain": _make_tc(), "model": _make_model(),
        "session_result": None, "request": None, "cancelled": False,
    }
    async with app.run_test() as pilot:
        app.push_screen(ResultsScreen())
        await pilot.pause()
        tabs = app.screen.query_one(TabbedContent)
        tabs.active = "tab-preset"
        visible = await _wait_for_richlog(pilot, app.screen, "#preset-content", "ctx-size =")
        assert "mmproj =" not in visible


async def test_results_shell_includes_mmproj_in_commands(tmp_path: Path) -> None:
    fa = FakeApp()
    spec = _make_spec()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    mmproj_file = tmp_path / "mmproj.gguf"
    mmproj_file.write_text("GGUF")
    prepared = _make_prepared(fa)
    prepared.mmproj_path = str(mmproj_file)
    app.prepared = prepared
    app.run_result = {
        "session_spec": spec, "screening_report": None,
        "validation_result": _make_vr(), "mtp_profile": None,
        "toolchain": _make_tc(), "model": _make_model(),
        "session_result": None, "request": None, "cancelled": False,
    }
    async with app.run_test() as pilot:
        app.push_screen(ResultsScreen())
        await pilot.pause()
        tabs = app.screen.query_one(TabbedContent)
        tabs.active = "tab-shell"
        visible = await _wait_for_richlog(pilot, app.screen, "#shell-content", "--mmproj")
        assert str(mmproj_file) in visible
        assert "--mmproj" in visible


async def test_results_nowinner_shows_no_winner() -> None:
    class NoWinnerFakeApp(FakeApp):
        def export_report_markdown(self, **kw: Any) -> str:
            self.calls["export_report_markdown"].append(kw)
            return "# No validated recommendation\n\nAll candidates failed validation."

    fa = NoWinnerFakeApp()
    spec = _make_spec()
    nw = NoWinnerReport(
        provisional_leader=None, failures=(
            FailedValidation(candidate=_make_base(), disposition="oom",
                             reason="Out of memory"),
        ),
        attempts_used=3, budget=10,
    )
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    app.prepared = _make_prepared(fa)
    app.run_result = {
        "session_spec": spec, "validation_result": nw,
        "screening_report": None, "mtp_profile": None,
        "toolchain": _make_tc(), "model": _make_model(),
        "session_result": None, "request": None, "cancelled": False,
    }
    async with app.run_test() as pilot:
        app.push_screen(ResultsScreen())
        await pilot.pause()
        assert len(fa.calls["export_report_markdown"]) == 1
        assert "No validated recommendation" in _text(app.screen.query_one("#md-content"))
        tabs = app.screen.query_one(TabbedContent)
        tabs.active = "tab-preset"
        await _wait_for_richlog(
            pilot, app.screen, "#preset-content", "No validated recommendation",
        )
        tabs.active = "tab-shell"
        await _wait_for_richlog(
            pilot, app.screen, "#shell-content", "No validated recommendation",
        )


# ── Narrow viewport ─────────────────────────────────────────────


async def test_narrow_viewport_wizard() -> None:
    app = ScoutTuiApp()
    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)
        assert "Wizard" in _text(app.screen.query_one("#wizard-title"))


async def test_narrow_viewport_results() -> None:
    app = ScoutTuiApp()
    async with app.run_test(size=(40, 20)) as pilot:
        app.run_result = {"cancelled": True}
        app.push_screen(ResultsScreen())
        await pilot.pause()
        assert "Cancelled" in _text(app.screen.query_one("#md-content"))


# ── Error screen ────────────────────────────────────────────────


async def test_error_screen_back_button() -> None:
    app = ScoutTuiApp()
    async with app.run_test() as pilot:
        app.push_screen(ErrorScreen("Test error"))
        await pilot.pause()
        assert isinstance(app.screen, ErrorScreen)
        app.screen.action_back()
        await pilot.pause()
        assert isinstance(app.screen, WizardScreen)


# ── Keyboard navigation ─────────────────────────────────────────


async def test_keyboard_f10_advances(tmp_path: Path) -> None:
    fa = FakeApp()
    app = ScoutTuiApp(
        app_factory=lambda r: fa,
        detect_topology_fn=_fake_detect_no_devices,
        **_tui_kw(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        ws = app.screen
        assert isinstance(ws, WizardScreen)
        _setup_wizard(app, tmp_path)
        ws.focus()
        await pilot.press("f10")
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)


# ── Application validation progress (unit) ──────────────────────


async def test_validate_candidates_emits_progress(tmp_path: Path) -> None:
    """Application.validate_candidates with injected evaluator emits before/after events."""
    from llama_parameter_scout.application import Application

    events: list[ProgressEvent] = []

    def capture(event: ProgressEvent) -> None:
        events.append(event)

    def fake_eval(candidate: CandidateConfig, n_ctx: NctxComposition) -> ScreeningResult:
        return ScreeningResult(
            candidate=candidate, success=True, disposition="survived",
            observation=ScreeningObservation(
                candidate=candidate,
                pp_samples=(45.0, 46.0, 44.0),
                tg_samples=(42.0, 43.0, 41.0),
                observed_prompt=n_ctx.prompt,
                observed_generation=n_ctx.generation,
                observed_depth=n_ctx.depth,
                observed_total=n_ctx.total,
            ),
        )

    base = _make_base()
    report = ScreeningReport(
        profile=QUICK_PROFILE,
        baseline_disposition=ScreeningResult(
            candidate=base, success=True, disposition="survived",
        ),
        grid_dispositions=(),
        survivors=(RankedCandidate(
            candidate=base, pp_score=1.0, tg_score=1.0, balanced=1.0,
            pp_median=45.0, tg_median=45.0,
            pp_relative_dispersion=0.05, tg_relative_dispersion=0.05,
        ),),
        leader=None,
        attempts_used=0,
        objective="balanced",
    )

    from llama_parameter_scout.runner import FakeRunner, ProcessResult

    bp = tmp_path / "llama-bench"
    bp.touch()
    bp.chmod(0o755)
    mp = tmp_path / "model.gguf"
    mp.touch()
    journal_path = tmp_path / "j.json"

    runner = FakeRunner()
    runner.enqueue(
        (str(bp), "--version"),
        ProcessResult(returncode=0, stdout=(
            "version: test\nbackends: cpu\n"
            "supported_flags: -m,-p,-n,-d,-t,-b,-ub,-ngl,-fa,-ctk,-ctv,-o,-r\n"
            "supported_output_formats: json\n"
        ), stderr=""),
    )
    runner.enqueue(
        (str(bp), "--help"),
        ProcessResult(returncode=0, stdout="usage: llama-bench\n", stderr=""),
    )
    app = Application(runner=runner)

    request = SingleRunRequest(
        binary_path=bp, model_path=mp,
        journal_path=journal_path,
        n_ctx=NctxComposition(total=64, prompt=16, generation=32, depth=16),
        candidate=base,
    )

    result = app.validate_candidates(
        request, report,
        validation_evaluator=fake_eval,
        on_progress=capture,
    )
    assert len(events) >= 2
    assert any(e.stage == "validation" and "Validating" in e.message for e in events)
    assert any(e.stage == "validation" and "Validated" in e.message for e in events)
    assert isinstance(result, ValidatedRecommendation)


async def test_results_passes_mmproj_to_export_report_markdown() -> None:
    fa = FakeApp()
    spec = _make_spec()
    app = ScoutTuiApp(app_factory=lambda r: fa, **_tui_kw())
    mmproj_file = "/opt/mmproj/mmproj.gguf"
    prepared = _make_prepared(fa)
    prepared.mmproj_path = mmproj_file
    app.prepared = prepared
    app.run_result = {
        "session_spec": spec, "screening_report": None,
        "validation_result": _make_vr(), "mtp_profile": None,
        "toolchain": _make_tc(), "model": _make_model(),
        "session_result": None, "request": None, "cancelled": False,
    }
    async with app.run_test() as pilot:
        app.push_screen(ResultsScreen())
        await pilot.pause()
        assert len(fa.calls["export_report_markdown"]) == 1
        kwargs = fa.calls["export_report_markdown"][0]
        assert kwargs.get("mmproj_path") == mmproj_file
