from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, cast, runtime_checkable

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import get_current_worker

from .application import Application
from .models import (
    Accelerator,
    CandidateConfig,
    ExecutionPolicy,
    ModelFingerprint,
    MtpProfileReport,
    MtpSearchProfile,
    NctxComposition,
    NoWinnerReport,
    ProgressEvent,
    ScreeningReport,
    SearchProfile,
    SessionResumeResult,
    SessionSpecification,
    SingleRunRequest,
    ToolchainFingerprint,
    ValidatedRecommendation,
)
from .runner import ProcessRunner, ProductionRunner
from .search import get_profile
from .topology import DetectTopologyFn, detect_topology

# ── Clipboard helpers ────────────────────────────────────────────


def _copy_to_clipboard(text: str) -> bool:
    try:
        if sys.platform == "darwin":
            proc = subprocess.run(["pbcopy"], input=text, text=True, timeout=5)
            return proc.returncode == 0
        elif sys.platform == "linux":
            try:
                proc = subprocess.run(
                    ["wl-copy"], input=text, text=True, timeout=5
                )
                return proc.returncode == 0
            except FileNotFoundError:
                try:
                    proc = subprocess.run(
                        ["xclip", "-selection", "clipboard"], input=text, text=True, timeout=5
                    )
                    return proc.returncode == 0
                except FileNotFoundError:
                    proc = subprocess.run(["xsel", "-b"], input=text, text=True, timeout=5)
                    return proc.returncode == 0
        elif sys.platform == "win32":
            proc = subprocess.run(["clip"], input=text, text=True, timeout=5)
            return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


# ── Wizard state ─────────────────────────────────────────────────


@dataclass
class WizardState:
    llama_bench_path: str = ""
    model_path: str = ""
    mmproj_path: str = ""
    journal_path: str = field(default_factory=lambda: str(Path.cwd() / "journal.json"))
    draft_model_path: str = ""
    server_path: str = ""
    cpu_only: bool = True
    device_id: str = ""
    n_ctx_total: str = "2048"
    n_ctx_prompt: str = "512"
    n_ctx_generation: str = "1024"
    n_ctx_depth: str = "512"
    acknowledged: bool = False
    profile_label: str = "standard"
    objective: str = "balanced"
    budget_ram: str = ""
    budget_vram: str = ""
    mtp_enabled: bool = False
    headroom_ratio: str = "0.1"
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"


# ── Application protocol ─────────────────────────────────────────


@runtime_checkable
class TuiApplication(Protocol):
    def setup_session(
        self,
        model: ModelFingerprint,
        n_ctx: NctxComposition,
        candidate: CandidateConfig,
        cpu_only: bool,
        device_ids: tuple[int, ...] = (),
        budget_overrides: tuple[int | None, int | None] | None = None,
        headroom_ratio: float = 0.1,
        acknowledged: bool = False,
    ) -> SessionSpecification: ...

    def start_or_resume_session(
        self,
        journal_path: Path,
        spec: SessionSpecification,
        model: ModelFingerprint,
        candidate: CandidateConfig | None = None,
        profile: SearchProfile | None = None,
        objective: str = "balanced",
        toolchain: ToolchainFingerprint | None = None,
    ) -> SessionResumeResult: ...

    def screen_candidates(
        self,
        request: SingleRunRequest,
        profile: SearchProfile,
        *,
        evaluator: Any | None = None,
        session_spec: SessionSpecification | None = None,
        policy: ExecutionPolicy | None = None,
        objective: str = "balanced",
        session_id: str | None = None,
        parent_session_id: str | None = None,
        completed_search_results: tuple[Any, ...] = (),
        on_progress: Callable[[ProgressEvent], Any] | None = None,
    ) -> ScreeningReport: ...

    def validate_candidates(
        self,
        request: SingleRunRequest,
        screening_report: ScreeningReport,
        *,
        budget: int = 10,
        objective: str = "balanced",
        validation_evaluator: Any | None = None,
        session_spec: SessionSpecification | None = None,
        policy: ExecutionPolicy | None = None,
        on_progress: Callable[[ProgressEvent], Any] | None = None,
    ) -> ValidatedRecommendation | NoWinnerReport: ...

    def collect_mtp_profile(
        self,
        *,
        validation_result: ValidatedRecommendation,
        draft_model_path: Path,
        server_path: Path,
        session_spec: SessionSpecification,
        profile: MtpSearchProfile | None = None,
        server_lifecycle: Any | None = None,
        main_fingerprint: ModelFingerprint | None = None,
        draft_fingerprint: ModelFingerprint | None = None,
        profile_evaluator: Any | None = None,
        _warmup_fn: Any | None = None,
        _measure_fn: Any | None = None,
        on_progress: Callable[[ProgressEvent], Any] | None = None,
    ) -> MtpProfileReport: ...

    def export_report_markdown(
        self,
        *,
        session: SessionSpecification,
        toolchain: ToolchainFingerprint,
        model: ModelFingerprint,
        screening_report: ScreeningReport | None = None,
        validation_result: ValidatedRecommendation | NoWinnerReport | None = None,
        mtp_profile: MtpProfileReport | None = None,
        provenance: Any | None = None,
        include_hashes: bool = False,
        mmproj_path: str | None = None,
    ) -> str: ...


# ── Prepared session (stored on App for reuse) ───────────────────


@dataclass
class PreparedSession:
    app: TuiApplication
    spec: SessionSpecification
    toolchain: ToolchainFingerprint
    model: ModelFingerprint
    request: SingleRunRequest
    profile: SearchProfile
    runner: ProcessRunner
    base_candidate: CandidateConfig
    n_ctx: NctxComposition
    mmproj_path: str | None = None


# ── Messages ─────────────────────────────────────────────────────


class RunProgress(Message):
    def __init__(self, event: ProgressEvent) -> None:
        super().__init__()
        self.event = event


class RunComplete(Message):
    def __init__(
        self,
        session_spec: SessionSpecification | None,
        screening_report: ScreeningReport | None,
        validation_result: ValidatedRecommendation | NoWinnerReport | None,
        mtp_profile: Any,
        toolchain: ToolchainFingerprint | None,
        model: ModelFingerprint | None,
        session_result: SessionResumeResult | None,
        request: SingleRunRequest | None,
        cancelled: bool = False,
    ) -> None:
        super().__init__()
        self.session_spec = session_spec
        self.screening_report = screening_report
        self.validation_result = validation_result
        self.mtp_profile = mtp_profile
        self.toolchain = toolchain
        self.model = model
        self.session_result = session_result
        self.request = request
        self.cancelled = cancelled


class RunError(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


# ── App ──────────────────────────────────────────────────────────


class ScoutTuiApp(App[None]):
    TITLE = "llama-parameter-scout"
    SUB_TITLE = "Benchmark Session Wizard"
    CSS = """
    Screen {
        background: $surface;
    }
    .section-title {
        text-style: bold;
        padding: 1 0 0 0;
        color: $accent;
    }
    .button-row {
        align: center middle;
        padding: 1;
        height: auto;
    }
    #wizard-title, #review-title, #progress-title, #prepare-label {
        text-style: bold;
        padding: 1;
        text-align: center;
    }
    Input, Select { margin: 0 2; }
    Label { margin: 0 2; padding: 0; }
    Checkbox { margin: 0 2; }
    RichLog { border: solid $primary; margin: 1 2; }
    #clipboard-msg { text-align: center; padding: 0 2; }
    #error-screen { align: center middle; }
    .browse-row { height: auto; margin: 0 2; }
    .browse-row > Input { width: 1fr; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
    ]

    wizard_state: WizardState
    prepared: PreparedSession | None
    run_result: dict[str, Any] | None

    def __init__(
        self,
        app_factory: Callable[..., Any] | None = None,
        runner_factory: Callable[[], ProcessRunner] | None = None,
        inspect_toolchain_fn: Callable[..., ToolchainFingerprint] | None = None,
        inspect_model_fn: Callable[..., ModelFingerprint] | None = None,
        detect_topology_fn: DetectTopologyFn | None = None,
    ) -> None:
        super().__init__()
        self.wizard_state = WizardState()
        self.prepared = None
        self.run_result = None
        self._app_factory = app_factory or (lambda r: Application(runner=r))
        self._runner_factory = runner_factory or (lambda: ProductionRunner())
        self._inspect_toolchain_fn = inspect_toolchain_fn
        self._inspect_model_fn = inspect_model_fn
        self._detect_topology_fn = detect_topology_fn or detect_topology

    def on_mount(self) -> None:
        self.push_screen(WizardScreen())


# ── Error screen ─────────────────────────────────────────────────


class ErrorScreen(Screen[ScoutTuiApp]):
    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__()
        self._msg = message
        self._detail = detail

    def compose(self) -> ComposeResult:
        with Vertical(id="error-screen"):
            yield Label("-- Error --", id="error-title")
            yield Label(self._msg, id="error-msg")
            if self._detail:
                yield Label(self._detail, id="error-detail")
            yield Horizontal(
                Button("Back", id="back_btn"),
                Button("Quit", id="quit_btn"),
                classes="button-row",
            )

    @on(Button.Pressed, "#back_btn")
    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#quit_btn")
    def action_quit(self) -> None:
        self.app.exit()


class _FilteredDirectoryTree(DirectoryTree):
    """DirectoryTree with a configurable file filter."""

    def __init__(
        self, path: str, *, filter_fn: Callable[[Path], bool] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(path, **kwargs)
        self._filter_fn = filter_fn

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        if self._filter_fn is None:
            yield from paths
            return
        for p in paths:
            if p.is_dir() or self._filter_fn(p):
                yield p


class GGUFDirectoryTree(_FilteredDirectoryTree):
    """DirectoryTree that only shows .gguf files and directories."""

    def __init__(self, path: str, **kwargs: Any) -> None:
        super().__init__(path, filter_fn=lambda p: p.suffix.lower() == ".gguf", **kwargs)


class _FileBrowser(ModalScreen[Path | None]):
    """Generic modal file browser with configurable title, tree, and filter."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        title: str,
        *,
        root: Path | None = None,
        tree_cls: type[DirectoryTree] = DirectoryTree,
        is_valid: Callable[[Path], bool] | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._root = root or Path.home()
        self._tree_cls = tree_cls
        self._is_valid = is_valid

    def compose(self) -> ComposeResult:
        yield Label(self._title)
        yield self._tree_cls(str(self._root), id="file_tree")
        yield Horizontal(
            Button("Cancel", id="cancel_btn", variant="default"),
            classes="button-row",
        )

    @on(Button.Pressed, "#cancel_btn")
    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(DirectoryTree.FileSelected)
    def _on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = Path(event.path)
        if path.is_file() and (self._is_valid is None or self._is_valid(path)):
            self.dismiss(path)


class ModelFileBrowser(_FileBrowser):
    """Modal browser for .gguf model files."""

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(
            "Select a .gguf model file:",
            root=root,
            tree_cls=GGUFDirectoryTree,
            is_valid=lambda p: p.suffix.lower() == ".gguf",
        )


class _BenchFileBrowser(_FileBrowser):
    """Browser for selecting any regular file (e.g. llama-bench binary)."""

    def __init__(self) -> None:
        super().__init__("Select llama-bench binary:")


class _JournalDirectoryTree(_FilteredDirectoryTree):
    """DirectoryTree that only shows .json files."""

    def __init__(self, path: str, **kwargs: Any) -> None:
        super().__init__(path, filter_fn=lambda p: p.suffix.lower() == ".json", **kwargs)


class _JournalFileBrowser(_FileBrowser):
    """Browser for selecting .json journal files."""

    def __init__(self) -> None:
        super().__init__(
            "Select a .json journal file:",
            tree_cls=_JournalDirectoryTree,
            is_valid=lambda p: p.suffix.lower() == ".json",
        )


class DeviceDetected(Message):
    def __init__(self, accelerators: tuple[Accelerator, ...]) -> None:
        super().__init__()
        self.accelerators = accelerators


class DeviceDetectionFailed(Message):
    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason


class PrepareDone(Message):
    def __init__(self, prepared: PreparedSession) -> None:
        super().__init__()
        self.prepared = prepared


class PrepareFailed(Message):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


# ── Wizard screen ────────────────────────────────────────────────


class WizardScreen(Screen[ScoutTuiApp]):
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("f10", "next_screen", "Next"),
    ]

    @property
    def _tui(self) -> ScoutTuiApp:
        return cast(ScoutTuiApp, self.app)

    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            yield Label("-- llama-parameter-scout Wizard --", id="wizard-title")
            yield Label("Binary & Model", classes="section-title")
            yield Label("llama-bench path")
            yield Horizontal(
                Input(placeholder="/path/to/llama-bench", id="llama_bench_path"),
                Button("Browse", id="browse_bench_btn"),
                classes="browse-row",
            )
            yield Label("Model path")
            yield Horizontal(
                Input(placeholder="/path/to/model.gguf", id="model_path"),
                Button("Browse", id="browse_model_btn"),
                classes="browse-row",
            )
            yield Label("MMProj path (optional)")
            yield Horizontal(
                Input(placeholder="/path/to/mmproj.gguf", id="mmproj_path"),
                Button("Browse", id="browse_mmproj_btn"),
                classes="browse-row",
            )
            yield Label("Journal path")
            yield Horizontal(
                Input(placeholder="/path/to/journal.json", id="journal_path"),
                Button("Browse", id="browse_journal_btn"),
                classes="browse-row",
            )

            yield Label("MTP (Multi-Token Prediction)", classes="section-title")
            yield Checkbox("Enable MTP", id="mtp_enabled")
            yield Label("Draft model path", id="lbl_draft_model_path")
            yield Horizontal(
                Input(placeholder="/path/to/draft.gguf", id="draft_model_path"),
                Button("Browse", id="browse_draft_btn"),
                classes="browse-row",
                id="draft_model_row",
            )
            yield Label("llama-server path", id="lbl_server_path")
            yield Input(placeholder="/path/to/llama-server", id="server_path")

            yield Label("Device", classes="section-title")
            yield Label("Detecting...", id="device_status")
            yield Select(
                [("CPU-only", "cpu_only")],
                prompt="Device",
                value="cpu_only",
                allow_blank=False,
                id="device",
            )

            yield Label("Context", classes="section-title")
            yield Select(
                [(str(n), str(n)) for n in (2048, 4096, 8192, 16384, 32768)]
                + [("custom", "custom")],
                prompt="Context preset",
                value="2048",
                allow_blank=False,
                id="context_preset",
            )
            yield Label("n_ctx total", id="lbl_n_ctx_total")
            yield Input(value="2048", id="n_ctx_total")
            yield Label("n_ctx prompt", id="lbl_n_ctx_prompt")
            yield Input(value="512", id="n_ctx_prompt")
            yield Label("n_ctx generation", id="lbl_n_ctx_generation")
            yield Input(value="1024", id="n_ctx_generation")
            yield Label("n_ctx depth", id="lbl_n_ctx_depth")
            yield Input(value="512", id="n_ctx_depth")
            yield Checkbox("Acknowledge context risk", id="acknowledged")

            yield Label("KV Cache", classes="section-title")
            yield Label("Key cache type")
            yield Select(
                [("f16 (higher precision)", "f16"),
                 ("q8_0 (8-bit)", "q8_0"),
                 ("q4_0 (4-bit, smallest)", "q4_0")],
                prompt="Key cache type",
                value="f16",
                allow_blank=False,
                id="cache_type_k",
            )
            yield Label("Value cache type")
            yield Select(
                [("f16 (higher precision)", "f16"),
                 ("q8_0 (8-bit)", "q8_0"),
                 ("q4_0 (4-bit, smallest)", "q4_0")],
                prompt="Value cache type",
                value="f16",
                allow_blank=False,
                id="cache_type_v",
            )

            yield Label("Search Profile", classes="section-title")
            yield Select(
                [(p, p) for p in ("quick", "standard", "thorough")],
                prompt="Profile",
                value="standard",
                id="profile",
            )
            yield Select(
                [(o, o) for o in ("balanced", "pp", "tg", "memory")],
                prompt="Objective",
                value="balanced",
                id="objective",
            )

            yield Label("Budget (optional)", classes="section-title")
            yield Label("RAM override (bytes)")
            yield Input(placeholder="e.g. 17179869184", id="budget_ram")
            yield Label("VRAM override (bytes)")
            yield Input(placeholder="e.g. 8589934592", id="budget_vram")
            yield Label("Headroom ratio")
            yield Input(value="0.1", id="headroom_ratio")

            yield Label("", id="wizard_errors")
            yield Horizontal(
                Button("Review & Start", id="next_btn", variant="primary"),
                Button("Quit (Ctrl+Q)", id="quit_btn"),
                classes="button-row",
            )

    def on_mount(self) -> None:
        self._populate_from_state()
        self._update_mtp_visibility()
        self._detect_devices()

    def _populate_from_state(self) -> None:
        state = self._tui.wizard_state
        self.query_one("#llama_bench_path", Input).value = state.llama_bench_path
        self.query_one("#model_path", Input).value = state.model_path
        self.query_one("#mmproj_path", Input).value = state.mmproj_path
        self.query_one("#journal_path", Input).value = state.journal_path
        self.query_one("#draft_model_path", Input).value = state.draft_model_path
        self.query_one("#server_path", Input).value = state.server_path
        self.query_one("#n_ctx_total", Input).value = state.n_ctx_total
        self.query_one("#n_ctx_prompt", Input).value = state.n_ctx_prompt
        self.query_one("#n_ctx_generation", Input).value = state.n_ctx_generation
        self.query_one("#n_ctx_depth", Input).value = state.n_ctx_depth
        self.query_one("#acknowledged", Checkbox).value = state.acknowledged
        self.query_one("#budget_ram", Input).value = state.budget_ram
        self.query_one("#budget_vram", Input).value = state.budget_vram
        self.query_one("#headroom_ratio", Input).value = state.headroom_ratio
        self.query_one("#mtp_enabled", Checkbox).value = state.mtp_enabled
        self.query_one("#cache_type_k", Select).value = state.cache_type_k
        self.query_one("#cache_type_v", Select).value = state.cache_type_v
        self._restore_context_preset()

    def _restore_context_preset(self) -> None:
        state = self._tui.wizard_state
        presets = [2048, 4096, 8192, 16384, 32768]
        try:
            total = int(state.n_ctx_total)
            prompt = int(state.n_ctx_prompt)
            gen = int(state.n_ctx_generation)
            depth = int(state.n_ctx_depth)
        except ValueError:
            self.query_one("#context_preset", Select).value = "custom"
            self._show_context_custom_fields()
            return
        for p in presets:
            expected_prompt = p // 4
            expected_gen = p // 2
            expected_depth = p - expected_prompt - expected_gen
            if (total == p and prompt == expected_prompt
                    and gen == expected_gen and depth == expected_depth):
                self.query_one("#context_preset", Select).value = str(p)
                self._hide_context_custom_fields()
                return
        self.query_one("#context_preset", Select).value = "custom"
        self._show_context_custom_fields()

    def _update_mtp_visibility(self) -> None:
        mtp_on = self.query_one("#mtp_enabled", Checkbox).value
        self.query_one("#lbl_draft_model_path", Label).display = mtp_on
        self.query_one("#draft_model_row", Horizontal).display = mtp_on
        self.query_one("#lbl_server_path", Label).display = mtp_on
        self.query_one("#server_path", Input).display = mtp_on

    @on(Checkbox.Changed, "#mtp_enabled")
    def _on_mtp_toggle(self) -> None:
        self._update_mtp_visibility()

    @on(Button.Pressed, "#browse_model_btn")
    def _open_model_browser(self) -> None:
        def on_select(path: Path | None) -> None:
            if path is not None:
                self.query_one("#model_path", Input).value = str(path)
        self.app.push_screen(ModelFileBrowser(), on_select)

    @on(Button.Pressed, "#browse_draft_btn")
    def _open_draft_browser(self) -> None:
        def on_select(path: Path | None) -> None:
            if path is not None:
                self.query_one("#draft_model_path", Input).value = str(path)
        self.app.push_screen(ModelFileBrowser(), on_select)

    @on(Button.Pressed, "#browse_bench_btn")
    def _open_bench_browser(self) -> None:
        def on_select(path: Path | None) -> None:
            if path is not None:
                self.query_one("#llama_bench_path", Input).value = str(path)
        self.app.push_screen(_BenchFileBrowser(), on_select)

    @on(Button.Pressed, "#browse_mmproj_btn")
    def _open_mmproj_browser(self) -> None:
        def on_select(path: Path | None) -> None:
            if path is not None:
                self.query_one("#mmproj_path", Input).value = str(path)
        self.app.push_screen(ModelFileBrowser(), on_select)

    @on(Button.Pressed, "#browse_journal_btn")
    def _open_journal_browser(self) -> None:
        def on_select(path: Path | None) -> None:
            if path is not None:
                self.query_one("#journal_path", Input).value = str(path)
        self.app.push_screen(_JournalFileBrowser(), on_select)

    @work(thread=True)
    def _detect_devices(self) -> None:
        tui = self._tui
        worker = get_current_worker()
        try:
            topology = tui._detect_topology_fn()
            if worker.is_cancelled:
                return
            self.post_message(DeviceDetected(topology.accelerators))
        except Exception as e:
            if worker.is_cancelled:
                return
            self.post_message(DeviceDetectionFailed(str(e)))

    @on(DeviceDetected)
    def _on_devices_detected(self, message: DeviceDetected) -> None:
        accels = message.accelerators
        select = self.query_one("#device", Select)
        options: list[tuple[str, str]] = [("CPU-only", "cpu_only")]
        for a in accels:
            label = f"{a.description} (id={a.device_id}, {a.backend})"
            options.append((label, f"gpu:{a.device_id}"))
        select.set_options(options)
        state = self._tui.wizard_state
        restored = False
        if not state.cpu_only and state.device_id:
            prev_val = f"gpu:{state.device_id}"
            if any(v == prev_val for _, v in options):
                select.value = prev_val
                restored = True
        if not restored:
            select.value = "cpu_only"
        if not accels:
            self.query_one("#device_status", Label).update(
                "No accelerators detected"
            )
        else:
            n = len(accels)
            label = f"{n} accelerator(s) detected"
            self.query_one("#device_status", Label).update(label)

    @on(DeviceDetectionFailed)
    def _on_devices_failed(self, message: DeviceDetectionFailed) -> None:
        self.query_one("#device_status", Label).update(
            f"Detection: {message.reason}"
        )

    @on(Select.Changed, "#context_preset")
    def _on_context_preset_changed(self, event: Select.Changed) -> None:
        value = event.value
        if not isinstance(value, str):
            return
        if value == "custom":
            self._show_context_custom_fields()
        else:
            try:
                total = int(value)
            except (ValueError, TypeError):
                return
            prompt = total // 4
            gen = total // 2
            depth = total - prompt - gen
            self.query_one("#n_ctx_total", Input).value = str(total)
            self.query_one("#n_ctx_prompt", Input).value = str(prompt)
            self.query_one("#n_ctx_generation", Input).value = str(gen)
            self.query_one("#n_ctx_depth", Input).value = str(depth)
            self._hide_context_custom_fields()

    def _show_context_custom_fields(self) -> None:
        for lbl_id in (
            "lbl_n_ctx_total", "lbl_n_ctx_prompt",
            "lbl_n_ctx_generation", "lbl_n_ctx_depth",
        ):
            self.query_one(f"#{lbl_id}", Label).display = True
        for inp_id in (
            "n_ctx_total", "n_ctx_prompt",
            "n_ctx_generation", "n_ctx_depth",
        ):
            self.query_one(f"#{inp_id}", Input).display = True

    def _hide_context_custom_fields(self) -> None:
        for lbl_id in (
            "lbl_n_ctx_total", "lbl_n_ctx_prompt",
            "lbl_n_ctx_generation", "lbl_n_ctx_depth",
        ):
            self.query_one(f"#{lbl_id}", Label).display = False
        for inp_id in (
            "n_ctx_total", "n_ctx_prompt",
            "n_ctx_generation", "n_ctx_depth",
        ):
            self.query_one(f"#{inp_id}", Input).display = False

    def action_next_screen(self) -> None:
        self._do_next()

    @on(Button.Pressed, "#next_btn")
    def _on_next(self) -> None:
        self._do_next()

    @on(Button.Pressed, "#quit_btn")
    def action_quit(self) -> None:
        self.app.exit()

    def _do_next(self) -> None:
        errors = self._gather_and_validate()
        if errors:
            self.query_one("#wizard_errors", Label).update(
                "\n".join(f"[red]ERROR: {e}[/]" for e in errors)
            )
            return
        self.query_one("#wizard_errors", Label).update("")
        self.app.push_screen(ReviewScreen())

    def _gather_and_validate(self) -> list[str]:
        errs: list[str] = []
        state = self._tui.wizard_state
        state.llama_bench_path = self.query_one("#llama_bench_path", Input).value.strip()
        state.model_path = self.query_one("#model_path", Input).value.strip()
        state.mmproj_path = self.query_one("#mmproj_path", Input).value.strip()
        state.journal_path = self.query_one("#journal_path", Input).value.strip()
        state.draft_model_path = self.query_one("#draft_model_path", Input).value.strip()
        state.server_path = self.query_one("#server_path", Input).value.strip()
        device_val = self.query_one("#device", Select).value
        if isinstance(device_val, str) and device_val.startswith("gpu:"):
            state.cpu_only = False
            state.device_id = device_val.removeprefix("gpu:")
        else:
            state.cpu_only = True
            state.device_id = ""
        state.n_ctx_total = self.query_one("#n_ctx_total", Input).value.strip()
        state.n_ctx_prompt = self.query_one("#n_ctx_prompt", Input).value.strip()
        state.n_ctx_generation = self.query_one("#n_ctx_generation", Input).value.strip()
        state.n_ctx_depth = self.query_one("#n_ctx_depth", Input).value.strip()
        state.acknowledged = self.query_one("#acknowledged", Checkbox).value
        state.budget_ram = self.query_one("#budget_ram", Input).value.strip()
        state.budget_vram = self.query_one("#budget_vram", Input).value.strip()
        state.headroom_ratio = self.query_one("#headroom_ratio", Input).value.strip()
        state.mtp_enabled = self.query_one("#mtp_enabled", Checkbox).value
        state.cache_type_k = str(self.query_one("#cache_type_k", Select).value or "f16")
        state.cache_type_v = str(self.query_one("#cache_type_v", Select).value or "f16")

        profile_widget = self.query_one("#profile", Select)
        if profile_widget.value is not None:
            state.profile_label = str(profile_widget.value)
        obj_widget = self.query_one("#objective", Select)
        if obj_widget.value is not None:
            state.objective = str(obj_widget.value)

        if not state.llama_bench_path:
            errs.append("llama-bench path is required")
        elif not Path(state.llama_bench_path).exists():
            errs.append(f"llama-bench not found: {state.llama_bench_path}")

        if not state.model_path:
            errs.append("Model path is required")
        elif not Path(state.model_path).exists():
            errs.append(f"Model not found: {state.model_path}")

        if state.mmproj_path and not Path(state.mmproj_path).exists():
            errs.append(f"MMProj not found: {state.mmproj_path}")

        if not state.journal_path:
            errs.append("Journal path is required")

        if state.mtp_enabled:
            if not state.draft_model_path:
                errs.append("Draft model path is required when MTP is enabled")
            elif not Path(state.draft_model_path).exists():
                errs.append(f"Draft model not found: {state.draft_model_path}")
            if not state.server_path:
                errs.append("llama-server path is required when MTP is enabled")
            elif not Path(state.server_path).exists():
                errs.append(f"llama-server not found: {state.server_path}")

        if not state.cpu_only and not state.device_id:
            errs.append("Device ID is required when not CPU-only")
        elif not state.cpu_only and state.device_id:
            try:
                int(state.device_id)
            except ValueError:
                errs.append("Device ID must be an integer")

        try:
            total = int(state.n_ctx_total)
            prompt = int(state.n_ctx_prompt)
            gen = int(state.n_ctx_generation)
            depth = int(state.n_ctx_depth)
            if total <= 0:
                errs.append("n_ctx total must be > 0")
            if prompt < 0 or gen < 0 or depth < 0:
                errs.append("Context components must be >= 0")
            if prompt + gen + depth != total:
                errs.append(
                    f"n_ctx components ({prompt}+{gen}+{depth}) "
                    f"do not sum to total ({total})"
                )
        except ValueError:
            errs.append("Invalid n_ctx values (must be integers)")

        if state.budget_ram:
            try:
                int(state.budget_ram)
            except ValueError:
                errs.append("RAM override must be an integer")

        if state.budget_vram:
            try:
                int(state.budget_vram)
            except ValueError:
                errs.append("VRAM override must be an integer")

        try:
            hr = float(state.headroom_ratio)
            if not 0.0 <= hr < 1.0:
                errs.append("Headroom ratio must be in [0.0, 1.0)")
        except ValueError:
            errs.append("Headroom ratio must be a number")

        return errs


# ── Review screen ────────────────────────────────────────────────


class ReviewScreen(Screen[ScoutTuiApp]):
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("f10", "start_run", "Start"),
    ]

    @property
    def _tui(self) -> ScoutTuiApp:
        return cast(ScoutTuiApp, self.app)

    def compose(self) -> ComposeResult:
        with ScrollableContainer():
            yield Label("-- Preparing session... --", id="prepare-label")
            yield Static(id="review-content")
            yield Label("", id="review-errors")
            yield Horizontal(
                Button("Back", id="back_btn"),
                Button("Start Benchmark", id="start_btn", variant="primary", disabled=True),
                Button("Quit (Ctrl+Q)", id="quit_btn"),
                classes="button-row",
            )

    def on_mount(self) -> None:
        self._prepare_session()

    @work(thread=True)
    def _prepare_session(self) -> None:
        state = self._tui.wizard_state
        tui = self._tui
        worker = get_current_worker()

        try:
            runner = tui._runner_factory()
            app = tui._app_factory(runner)

            n_ctx = NctxComposition(
                total=int(state.n_ctx_total),
                prompt=int(state.n_ctx_prompt),
                generation=int(state.n_ctx_generation),
                depth=int(state.n_ctx_depth),
            )

            budget_overrides: tuple[int | None, int | None] | None = None
            ram_override = int(state.budget_ram) if state.budget_ram else None
            vram_override = int(state.budget_vram) if state.budget_vram else None
            if ram_override is not None or vram_override is not None:
                budget_overrides = (ram_override, vram_override)

            base_candidate = CandidateConfig(
                threads=4,
                batch_size=512,
                ubatch_size=512,
                n_gpu_layers=0,
                flash_attn=False,
                cache_type_k=state.cache_type_k,
                cache_type_v=state.cache_type_v,
            )

            profile = get_profile(state.profile_label)

            binary_path = Path(state.llama_bench_path)
            model_path = Path(state.model_path)

            if tui._inspect_toolchain_fn is not None:
                toolchain = tui._inspect_toolchain_fn(runner, binary_path)
            else:
                from .toolchain import inspect_toolchain as _it
                toolchain = _it(runner, binary_path)

            if tui._inspect_model_fn is not None:
                model = tui._inspect_model_fn(model_path)
            else:
                from .model_inspector import inspect_model as _im
                model = _im(model_path)

            device_ids: tuple[int, ...] = ()
            if not state.cpu_only and state.device_id:
                device_ids = (int(state.device_id),)

            spec = app.setup_session(
                model=model,
                n_ctx=n_ctx,
                candidate=base_candidate,
                cpu_only=state.cpu_only,
                device_ids=device_ids,
                budget_overrides=budget_overrides,
                headroom_ratio=float(state.headroom_ratio),
                acknowledged=state.acknowledged,
            )

            # Build request WITHOUT calling start_or_resume_session
            journal_path = Path(state.journal_path)
            request = SingleRunRequest(
                binary_path=binary_path,
                model_path=model_path,
                journal_path=journal_path,
                n_ctx=n_ctx,
                candidate=base_candidate,
                binary_source="explicit",
            )

            mmproj = state.mmproj_path if state.mmproj_path else None

            prepared = PreparedSession(
                app=app,
                spec=spec,
                toolchain=toolchain,
                model=model,
                request=request,
                profile=profile,
                runner=runner,
                base_candidate=base_candidate,
                n_ctx=n_ctx,
                mmproj_path=mmproj,
            )

            if worker.is_cancelled:
                return
            self.post_message(PrepareDone(prepared))

        except Exception as e:
            if worker.is_cancelled:
                return
            self.post_message(PrepareFailed(str(e)))

    @on(PrepareDone)
    def _on_prepare_done(self, message: PrepareDone) -> None:
        self._tui.prepared = message.prepared
        self.query_one("#prepare-label", Label).update(
            "-- Review Session Specification --"
        )
        self._render_review(message.prepared)
        self.query_one("#start_btn", Button).disabled = False

    @on(PrepareFailed)
    def _on_prepare_failed(self, message: PrepareFailed) -> None:
        self.app.push_screen(ErrorScreen("Preparation failed", message.message))

    def _render_review(self, prepared: PreparedSession) -> None:
        spec = prepared.spec
        state = self._tui.wizard_state
        lines: list[str] = []
        lines.append(f"llama-bench: {prepared.toolchain.binary_path}")
        lines.append(f"Model: {spec.model.path}")
        lines.append(f"Model name: {spec.model.path.name} ({spec.model.architecture})")
        if state.mmproj_path:
            lines.append(f"MMProj: {state.mmproj_path}")
        lines.append(f"Journal: {state.journal_path}")
        if spec.cpu_only:
            lines.append("Device: CPU-only")
        elif spec.selected_device_ids:
            device_id = spec.selected_device_ids[0]
            for a in spec.topology.accelerators:
                if a.device_id == device_id:
                    lines.append(f"Device: {a.description} (id={device_id}, {a.backend})")
                    break
            else:
                lines.append(f"Device: id={device_id}")
        lines.append(
            f"Context: total={spec.n_ctx.total}, prompt={spec.n_ctx.prompt}, "
            f"gen={spec.n_ctx.generation}, depth={spec.n_ctx.depth}"
        )
        lines.append(f"Acknowledged: {spec.acknowledged}")
        budget = spec.budget
        lines.append(f"RAM budget: {budget.ram_bytes} bytes")
        if budget.vram_bytes is not None:
            lines.append(f"VRAM budget: {budget.vram_bytes} bytes")
        else:
            lines.append("VRAM: unified with RAM")
        lines.append(f"Headroom: {budget.headroom_ratio * 100:.0f}%")
        advisory = spec.advisory
        status = advisory.training_context_status
        status_line = f"Training context: {status}"
        if advisory.training_context_tokens is not None:
            status_line += f" ({advisory.training_context_tokens})"
        lines.append(status_line)
        lines.append(f"Likely-fit range: {advisory.likely_fit_lower} - {advisory.likely_fit_upper}")
        lines.append(f"Uncertain maximum: {advisory.uncertain_maximum}")
        lines.append(f"Fingerprint: {spec.fingerprint}")
        lines.append(f"Advisory note: {advisory.note}")
        lines.append(f"Profile: {state.profile_label}")
        lines.append(f"Objective: {state.objective}")
        lines.append(f"MTP enabled: {state.mtp_enabled}")
        if state.mtp_enabled:
            lines.append(f"Draft model: {state.draft_model_path}")
            lines.append(f"Server: {state.server_path}")
        lines.append(f"KV cache: K={state.cache_type_k}, V={state.cache_type_v}")
        self.query_one("#review-content", Static).update("\n".join(lines))

    @on(Button.Pressed, "#start_btn")
    def action_start_run(self) -> None:
        if self._tui.prepared is not None:
            self.app.push_screen(ProgressScreen())

    @on(Button.Pressed, "#back_btn")
    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#quit_btn")
    def action_quit(self) -> None:
        self.app.exit()


# ── Progress screen ──────────────────────────────────────────────


class ProgressScreen(Screen[ScoutTuiApp]):
    BINDINGS = [
        Binding("escape", "cancel_run", "Cancel"),
        Binding("ctrl+c", "cancel_run", "Cancel"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._cancellation_event = threading.Event()

    @property
    def _tui(self) -> ScoutTuiApp:
        return cast(ScoutTuiApp, self.app)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("-- Benchmark in Progress --", id="progress-title")
            yield RichLog(id="progress-log", highlight=True, markup=True)
            yield Label("", id="progress-status")
            yield Horizontal(
                Button("Cancel (Esc)", id="cancel_btn", variant="error"),
                Button("Quit (Ctrl+Q)", id="quit_btn"),
                classes="button-row",
            )

    def on_mount(self) -> None:
        self.run_benchmark()

    def action_cancel_run(self) -> None:
        self._cancellation_event.set()
        self.post_message(RunProgress(ProgressEvent(
            stage="cancellation", message="Cancelling benchmark...",
        )))

    def action_quit(self) -> None:
        self._cancellation_event.set()
        self.app.exit()

    @on(Button.Pressed, "#quit_btn")
    def _on_quit_pressed(self) -> None:
        self.action_quit()

    @work(thread=True)
    def run_benchmark(self) -> None:
        self._run_benchmark_impl(self._cancellation_event)

    def _run_benchmark_impl(self, cancellation_event: threading.Event) -> None:
        tui = self._tui
        prepared = tui.prepared
        if prepared is None:
            self.post_message(RunError("No prepared session"))
            return

        app = prepared.app
        spec = prepared.spec
        toolchain = prepared.toolchain
        model = prepared.model
        request = prepared.request
        profile = prepared.profile
        state = tui.wizard_state

        def on_progress(event: ProgressEvent) -> None:
            self.post_message(RunProgress(event))

        try:
            self.post_message(RunProgress(ProgressEvent(
                stage="setup", message="Starting or resuming session..."
            )))

            session_result = app.start_or_resume_session(
                journal_path=request.journal_path,
                spec=spec,
                model=model,
                candidate=request.candidate,
                profile=profile,
                objective=state.objective,
                toolchain=toolchain,
            )

            self.post_message(RunProgress(ProgressEvent(
                stage="setup",
                message=f"Session: {session_result.action} (id={session_result.session_id})"
            )))

            if cancellation_event.is_set():
                self.post_message(RunComplete(
                    session_spec=spec, screening_report=None,
                    validation_result=None, mtp_profile=None,
                    toolchain=toolchain, model=model,
                    session_result=session_result, request=request,
                    cancelled=True,
                ))
                return

            policy = ExecutionPolicy(
                cancellation=cancellation_event,
            )
            reusable_results = session_result.completed_search_results

            self.post_message(RunProgress(ProgressEvent(
                stage="screening", message="Starting candidate screening..."
            )))

            if cancellation_event.is_set():
                self.post_message(RunComplete(
                    session_spec=spec, screening_report=None,
                    validation_result=None, mtp_profile=None,
                    toolchain=toolchain, model=model,
                    session_result=session_result, request=request,
                    cancelled=True,
                ))
                return

            screening_report = app.screen_candidates(
                request,
                profile=profile,
                session_spec=spec,
                policy=policy,
                objective=state.objective,
                session_id=session_result.session_id,
                completed_search_results=reusable_results,
                on_progress=on_progress,
            )

            if cancellation_event.is_set():
                self.post_message(RunComplete(
                    session_spec=spec, screening_report=screening_report,
                    validation_result=None, mtp_profile=None,
                    toolchain=toolchain, model=model,
                    session_result=session_result, request=request,
                    cancelled=True,
                ))
                return

            self.post_message(RunProgress(ProgressEvent(
                stage="validation", message="Starting candidate validation..."
            )))

            validation_result = app.validate_candidates(
                request,
                screening_report,
                objective=state.objective,
                session_spec=spec,
                policy=policy,
                on_progress=on_progress,
            )

            if cancellation_event.is_set():
                self.post_message(RunComplete(
                    session_spec=spec, screening_report=screening_report,
                    validation_result=validation_result, mtp_profile=None,
                    toolchain=toolchain, model=model,
                    session_result=session_result, request=request,
                    cancelled=True,
                ))
                return

            mtp_profile = None
            if state.mtp_enabled and isinstance(validation_result, ValidatedRecommendation):
                self.post_message(RunProgress(ProgressEvent(
                    stage="mtp_profile", message="Collecting MTP profile..."
                )))
                mtp_profile = app.collect_mtp_profile(
                    validation_result=validation_result,
                    draft_model_path=Path(state.draft_model_path),
                    server_path=Path(state.server_path),
                    session_spec=spec,
                    profile=MtpSearchProfile(),
                    on_progress=on_progress,
                )

            self.post_message(RunComplete(
                session_spec=spec,
                screening_report=screening_report,
                validation_result=validation_result,
                mtp_profile=mtp_profile,
                toolchain=toolchain,
                model=model,
                session_result=session_result,
                request=request,
                cancelled=False,
            ))

        except Exception as e:
            self.post_message(RunError(str(e)))

    def on_run_progress(self, message: RunProgress) -> None:
        event = message.event
        log = self.query_one("#progress-log", RichLog)
        prefix = {
            "setup": "[bold blue]",
            "screening": "[bold cyan]",
            "validation": "[bold green]",
            "mtp_profile": "[bold magenta]",
            "cancellation": "[bold yellow]",
        }.get(event.stage, "[bold]")
        icon = ""
        if event.is_error:
            icon = "[red]!![/] "
        elif event.is_warning:
            icon = "[yellow]!![/] "
        log.write(f"{prefix}{event.stage}[/]: {icon}{event.message}")
        self.query_one("#progress-status", Label).update(
            f"Stage: {event.stage} | {event.message}"
        )

    def on_run_error(self, message: RunError) -> None:
        self.query_one("#progress-log", RichLog).write(
            f"[red]ERROR: {message.message}[/]"
        )
        self.query_one("#progress-status", Label).update("Error during benchmark")
        self.app.push_screen(ErrorScreen("Benchmark failed", message.message))

    def on_run_complete(self, message: RunComplete) -> None:
        self._tui.run_result = {
            "session_spec": message.session_spec,
            "screening_report": message.screening_report,
            "validation_result": message.validation_result,
            "mtp_profile": message.mtp_profile,
            "toolchain": message.toolchain,
            "model": message.model,
            "session_result": message.session_result,
            "request": message.request,
            "cancelled": message.cancelled,
        }
        self.app.switch_screen(ResultsScreen())


# ── Results screen ───────────────────────────────────────────────


class ResultsScreen(Screen[ScoutTuiApp]):
    BINDINGS = [
        Binding("escape", "quit", "Quit"),
        Binding("f3", "copy_markdown", "Copy MD"),
        Binding("f4", "copy_preset", "Copy INI"),
        Binding("f5", "copy_shell", "Copy Shell"),
        Binding("f6", "save_all", "Save All"),
    ]

    @property
    def _tui(self) -> ScoutTuiApp:
        return cast(ScoutTuiApp, self.app)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Header()
            with TabbedContent(initial="tab-markdown"):
                with TabPane("Markdown Report", id="tab-markdown"):
                    yield RichLog(id="md-content", highlight=True, markup=False)
                with TabPane("Preset.ini", id="tab-preset"):
                    yield RichLog(id="preset-content", highlight=True, markup=False)
                with TabPane("Shell Commands", id="tab-shell"):
                    yield RichLog(id="shell-content", highlight=True, markup=False)
            yield Static(id="clipboard-msg")
            yield Horizontal(
                Button("Copy MD (F3)", id="copy_md_btn"),
                Button("Copy INI (F4)", id="copy_preset_btn"),
                Button("Copy Shell (F5)", id="copy_shell_btn"),
                Button("Save All (F6)", id="save_all_btn"),
                Button("Quit (Esc)", id="quit_btn"),
                classes="button-row",
            )
            yield Footer()

    def on_mount(self) -> None:
        self._render_results()

    def _render_results(self) -> None:
        result = self._tui.run_result
        if result is None:
            self.query_one("#md-content", RichLog).write("No results available.")
            return

        cancelled = result.get("cancelled", False)
        session_spec = result.get("session_spec")
        screening_report = result.get("screening_report")
        validation_result = result.get("validation_result")
        mtp_profile = result.get("mtp_profile")
        toolchain = result.get("toolchain")
        model = result.get("model")
        prepared = self._tui.prepared

        md_lines: list[str] = []
        if cancelled:
            md_lines.append("# Benchmark Cancelled")
            md_lines.append("")
            md_lines.append("The benchmark was cancelled before completion.")
            md_lines.append("")
            md_lines.append("Partial results may be available in the journal.")
        else:
            try:
                assert session_spec is not None
                assert toolchain is not None
                assert model is not None
                if prepared is not None:
                    md = prepared.app.export_report_markdown(
                        session=session_spec,
                        toolchain=toolchain,
                        model=model,
                        screening_report=screening_report,
                        validation_result=validation_result,
                        mtp_profile=mtp_profile,
                        mmproj_path=prepared.mmproj_path,
                    )
                    md_lines.append(md)
                else:
                    md_lines.append("# Results")
                    md_lines.append("No prepared Application available for export.")
            except Exception:
                md_lines.append("# Results")
                md_lines.append("(Report generation encountered an error.)")
                if validation_result is not None:
                    if isinstance(validation_result, ValidatedRecommendation):
                        vc = validation_result.recommendation
                        md_lines.append(f"Validated: {vc.label}")
                        md_lines.append(
                            f"  PP={vc.pp_score:.4f} TG={vc.tg_score:.4f}"
                        )
                    elif isinstance(validation_result, NoWinnerReport):
                        md_lines.append("No winner -- all candidates failed validation.")
                        if validation_result.provisional_leader is not None:
                            md_lines.append(
                                "  Screening leader available (not validated)."
                            )
                elif screening_report is not None:
                    md_lines.append(f"Candidates evaluated: "
                                    f"{len(screening_report.grid_dispositions)}")

        self.query_one("#md-content", RichLog).write("\n".join(md_lines))

        validated_vr: ValidatedRecommendation | None = None
        if (not cancelled
                and validation_result is not None
                and isinstance(validation_result, ValidatedRecommendation)):
            validated_vr = validation_result

        preset_lines: list[str] = []
        if validated_vr is not None and session_spec is not None:
            try:
                target = validated_vr.recommendation.candidate
                from .reporter import build_preset_ini_section
                preset_lines.append(
                    build_preset_ini_section(
                        target,
                        session_spec.n_ctx,
                        str(session_spec.model.path),
                        mmproj_path=prepared.mmproj_path if prepared else None,
                        cpu_only=session_spec.cpu_only,
                        selected_device_ids=session_spec.selected_device_ids,
                    )
                )
            except Exception:
                preset_lines.append("; Could not build preset.ini")
        elif not cancelled:
            preset_lines.append("; No validated recommendation -- preset.ini unavailable.")
        else:
            preset_lines.append("; Benchmark was cancelled -- no preset available.")
        self.query_one("#preset-content", RichLog).write("\n".join(preset_lines))

        shell_lines: list[str] = []
        if validated_vr is not None and session_spec is not None and toolchain is not None:
            try:
                target = validated_vr.recommendation.candidate
                mp = str(session_spec.model.path)
                from .reporter import (
                    build_llama_cli_command,
                    build_llama_server_command,
                    build_reproduction_argv,
                    join_argv,
                )
                mmproj = prepared.mmproj_path if prepared else None
                cli_cmd = build_llama_cli_command(
                    target, session_spec.n_ctx, mp,
                    mmproj_path=mmproj,
                    cpu_only=session_spec.cpu_only,
                    selected_device_ids=session_spec.selected_device_ids,
                )
                shell_lines.append("# llama-cli")
                shell_lines.append(join_argv(cli_cmd, "posix"))
                shell_lines.append("")
                srv_cmd = build_llama_server_command(
                    target, session_spec.n_ctx, mp,
                    mmproj_path=mmproj,
                    cpu_only=session_spec.cpu_only,
                    selected_device_ids=session_spec.selected_device_ids,
                )
                shell_lines.append("# llama-server")
                shell_lines.append(join_argv(srv_cmd, "posix"))
                shell_lines.append("")
                repro_argv = build_reproduction_argv(
                    target, session_spec.n_ctx,
                    binary_path=str(toolchain.binary_path) if toolchain else "llama-bench",
                    model_path=mp,
                    cpu_only=session_spec.cpu_only,
                    selected_device_ids=session_spec.selected_device_ids,
                )
                shell_lines.append("# llama-bench (reproduction)")
                shell_lines.append(join_argv(repro_argv, "posix"))
            except Exception:
                shell_lines.append("# Could not build shell commands")
        elif not cancelled:
            shell_lines.append(
                "# No validated recommendation -- shell commands unavailable."
            )
        else:
            shell_lines.append(
                "# Benchmark was cancelled -- no commands available."
            )
        self.query_one("#shell-content", RichLog).write("\n".join(shell_lines))

    @on(Button.Pressed, "#copy_md_btn")
    def action_copy_markdown(self) -> None:
        self._do_copy("#md-content", "markdown")

    @on(Button.Pressed, "#copy_preset_btn")
    def action_copy_preset(self) -> None:
        self._do_copy("#preset-content", "preset.ini")

    @on(Button.Pressed, "#copy_shell_btn")
    def action_copy_shell(self) -> None:
        self._do_copy("#shell-content", "shell commands")

    def _do_copy(self, widget_id: str, label: str) -> None:
        widget = self.query_one(widget_id, RichLog)
        text = "\n".join(str(line.text) for line in widget.lines)
        if _copy_to_clipboard(text):
            self.query_one("#clipboard-msg", Static).update(
                f"[green]Copied {label} to clipboard[/]"
            )
        else:
            self.query_one("#clipboard-msg", Static).update(
                f"[yellow]Clipboard unavailable. Use Save to write {label} to a file.[/]"
            )

    @on(Button.Pressed, "#save_all_btn")
    def action_save_all(self) -> None:
        result = self._tui.run_result
        if result is None:
            self._show_save_msg("[red]No results to save.[/]")
            return

        base_dir = Path.cwd()
        try:
            md_text = "\n".join(
                str(line.text) for line in self.query_one("#md-content", RichLog).lines
            )
            preset_text = "\n".join(
                str(line.text) for line in self.query_one("#preset-content", RichLog).lines
            )
            shell_text = "\n".join(
                str(line.text) for line in self.query_one("#shell-content", RichLog).lines
            )
            (base_dir / "scout-report.md").write_text(md_text)
            (base_dir / "preset.ini").write_text(preset_text)
            (base_dir / "shell-commands.txt").write_text(shell_text)
            self._show_save_msg(f"[green]Saved files in {base_dir}[/]")
        except OSError as e:
            self._show_save_msg(f"[red]Save failed: {e}[/]")

    def _show_save_msg(self, msg: str) -> None:
        self.query_one("#clipboard-msg", Static).update(msg)

    @on(Button.Pressed, "#quit_btn")
    def action_quit(self) -> None:
        self.app.exit()
