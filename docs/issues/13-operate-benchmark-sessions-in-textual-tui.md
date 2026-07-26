# 13: Operate Benchmark Sessions in the Textual TUI
**Status:** ready-for-agent
**User stories covered:** 47, 62-66
## What to build
Expose all completed Application behavior through a keyboard-driven Textual workflow. Build a wizard that guides the user through toolchain and model input, draft model input, hardware and device selection, context advisory and acknowledgement, search profile and objective selection, resource limits, and an MTP checkbox. Provide a review and confirmation screen that shows the complete Session Specification before the search begins. Add a live progress view showing search stage, candidate observations, validation status, errors, and warnings as they arrive. Support cancellation of a running session and later resume through the TUI. After completion or cancellation, display results in read-only tabs or panes for the Markdown report, preset.ini, and platform-specific shell commands, with copy and save actions. When the system clipboard is unavailable, degrade gracefully to save-to-file and export. Never duplicate core decision logic; the TUI must use the same Application seam as the CLI. Keep the CLI behavior unchanged. Tests must drive the Application seam with fake adapters, covering the wizard flow, keyboard navigation, input validation, error display, cancellation, resume detection, no-winner state, MTP mode, and narrow-terminal behavior where applicable.
## Acceptance criteria
- [ ] A keyboard-driven wizard covers: toolchain path, model path, draft model path (optional), hardware/device selection, context advisory and acknowledgement, profile, objective, resource limits, and MTP checkbox
- [ ] A review screen shows the complete Session Specification before the search starts
- [ ] A live progress view shows stage, observations, validation status, errors, and warnings
- [ ] Running sessions can be cancelled; cancelled sessions can be resumed
- [ ] Results are displayed in read-only tabs for Markdown, preset.ini, and platform shell commands
- [ ] Copy and save actions work; clipboard-unavailable environments offer save and export fallback
- [ ] The TUI uses the same Application seam as the CLI; core decision logic is not duplicated
- [ ] CLI behavior is unchanged when the TUI is not invoked
- [ ] Tests drive the Application seam with fake adapters, covering: wizard flow, keyboard navigation, input validation, error display, cancellation, resume, no-winner, MTP, and narrow-terminal behavior
- [ ] README documents the TUI workflow, key bindings, and clipboard handling
## Blocked by
- [10-export-reproducible-share-safe-artifacts.md](10-export-reproducible-share-safe-artifacts.md)
- [12-optimize-bounded-mtp-profile.md](12-optimize-bounded-mtp-profile.md)
