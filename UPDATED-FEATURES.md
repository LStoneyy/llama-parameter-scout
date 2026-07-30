# UPDATED-FEATURES — Implementation Plan

**Status:** Proposed — awaiting approval before implementation.
**Author:** planning pass (orchestrated).
**Scope:** four phases. Each phase is independently reviewable, tested, and committed.

This document is a *plan*, not a changelog. Nothing here is implemented yet. It
exists so the maintainer can approve the direction and the concrete surface
changes before any code is written.

---

## Baseline assessment

The codebase is in good health and this plan does **not** exist to fix breakage:

- `ruff check src tests` — clean.
- `mypy` (strict) — clean, 36 source files.
- `pytest` (default markers) — **845 passed**, 7 deselected.
- CI matrix — 3 OS × 3 Python versions, plus a CPU smoke job and a release gate.
- `.gitignore` correctly excludes runtime artifacts (`journal.json`, `dist/`,
  models, reports, `preset.ini`, `scout-report.md`).

The improvements below are therefore about **product completeness,
architecture, CLI ergonomics, and documentation** — not defect repair.

Source sizes at time of writing:

| File | LOC | Note |
|------|-----|------|
| `application.py` | 2791 | Monolith — orchestrator class + ~40 free functions |
| `tui.py` | 1606 | Textual UI |
| `reporter.py` | 1176 | Report/export rendering |
| `search.py` | 1169 | Staged search / screening |
| `mtp.py` | 1059 | Multi-token-prediction comparison/profile |
| `models.py` | 649 | Typed domain model |
| `cli.py` | 587 | argparse front-end (`main()` is 194 lines) |

---

## Key finding driving Phase 1

The PRD promises *"a non-interactive CLI for scripting and automation"* and the
README documents Screening, Validation, MTP, and Export at length. **However,
the CLI today exposes only three things:**

1. A single-candidate benchmark (`setup_session` → `start_or_resume_session` →
   `evaluate_candidate`).
2. `--tui`.
3. `--list-sessions`.

The flagship capabilities — the staged adaptive search (`Application.run` /
`screen_candidates` → `validate_candidates`), MTP comparison/profiling, and
report export (`export_report_json` / `export_report_markdown`) — are **only
reachable through the TUI**. For the stated audience (scripting, automation,
CI), this is the single biggest gap. Phase 1 closes it; Phase 2 gives the
grown CLI a coherent subcommand shape.

Verified: no `app.run` / `app.screen_candidates` / `app.validate_candidates` /
`app.compare_mtp` / `app.collect_mtp_profile` / `app.export_report_*` call
exists anywhere in `cli.py`.

---

## Phase 1 — Expose the core search + export in the non-interactive CLI

**Goal:** Make the staged search, its validated recommendation, and report
export usable without the TUI, with machine-readable output for CI.

**Why first:** highest product value, lowest architectural risk, and it defines
the surface Phase 2 will restructure.

### Core interfaces to wire (already implemented, signatures confirmed)

```
setup_session(model, n_ctx, candidate, cpu_only, device_ids, budget_overrides, acknowledged)
start_or_resume_session(journal_path, spec, model, candidate, profile, objective, toolchain)
screen_candidates(request, profile, *, objective, session_spec, policy, ...) -> ScreeningReport
validate_candidates(request, screening_report, *, budget, objective, session_spec, policy, ...)
    -> ValidatedRecommendation | NoWinnerReport
export_report_json(*, session, toolchain, model, screening_report, validation_result, ...) -> str
export_report_markdown(*, ...) -> str
```

The TUI's proven orchestration order (mirror it in the CLI):
`setup_session → start_or_resume_session → screen_candidates →
validate_candidates → (optional collect_mtp_profile) → export_report_*`.

### New CLI capabilities (surface names finalized in Phase 2)

- **Staged search:** run screening then validation, print the Validated
  Recommendation or an explicit no-winner state (with the Provisional
  Screening Leader when present).
  - `--profile {quick,standard,thorough}` (already parsed; wire it to the search).
  - `--objective {pp,tg,balanced,memory}` (already parsed).
  - `--validation-budget N` (maps to `validate_candidates(budget=...)`, default 10).
- **Export:** `--export-json PATH` and/or `--export-md PATH`, plus
  `--export-include-hashes` (maps to `include_hashes`). Export is available for
  both single-run and staged-search results.
- **JSON result to stdout:** `--json` prints a structured result object
  (recommendation + provenance + per-candidate summary) for CI consumption,
  suppressing human-formatted prose. Exit code reflects outcome
  (winner = 0, no-winner = documented non-zero).
- **Progress:** reuse the existing `ProgressCallback`/`on_progress` hooks to
  emit line-based progress to stderr; silence with `--quiet`.

### Explicitly out of scope for Phase 1

- MTP compare/profile in the CLI — deferred to Phase 1b (below) because it
  additionally needs a `llama-server` path and a draft model, i.e. more new
  flags and a `ServerLifecycle`. Kept separate to keep Phase 1 tight.

### Phase 1b (optional, same phase family) — MTP in the CLI

- `--draft-model PATH`, `--server-path PATH`, then wire `compare_mtp` and
  `collect_mtp_profile`. Only pursued after Phase 1 lands and if desired.

### Tests (Phase 1)

- New `tests/test_cli_search.py`: staged-search happy path with a fake runner,
  no-winner path, `--json` shape, `--export-json/--export-md` file creation,
  exit codes, and `--objective`/`--profile`/`--validation-budget` plumbing.
- Reuse existing fake-runner/evaluator fixtures from `conftest.py` and the
  `test_issue06/07/10` suites so we do not invoke real binaries.

### Deliverable / acceptance

- `llama-parameter-scout` can run a full search and emit a recommendation +
  exported report from the shell, no TUI.
- ruff + mypy clean; new tests green; existing 845 tests unaffected.

---

## Phase 2 — CLI subcommands + slimmed `main()`

**Decision (approved):** move to real **subcommands**. The tool is not yet on
PyPI, so there are no released users to break — this is the correct moment.

**Goal:** replace the single flat `argparse` namespace and the 194-line
`main()` with a subcommand dispatcher and one thin handler per command.

### Proposed command surface

```
llama-parameter-scout run       # single explicit candidate (today's default path)
llama-parameter-scout search    # staged adaptive search -> validated recommendation (Phase 1)
llama-parameter-scout export    # re-export a stored session's report as json/md
llama-parameter-scout sessions  # list journaled sessions (today's --list-sessions)
llama-parameter-scout tui       # launch the Textual UI
llama-parameter-scout mtp       # MTP compare/profile (only if Phase 1b is done)
```

- Shared flags (binary discovery, model, n_ctx composition, device, budget,
  `--yes`, `--journal-path`) live on a common parent parser and are attached to
  each subcommand that needs them — no duplication.
- `main()` becomes: parse → dispatch to `_cmd_run` / `_cmd_search` /
  `_cmd_export` / `_cmd_sessions` / `_cmd_tui`. Each handler is independently
  testable and < ~60 lines.

### Backward-compatibility shim

- Keep a thin translation so a bare invocation with the old single-run flags
  (no subcommand) still routes to `run`, printing a one-line deprecation notice
  to stderr. Removed before/at a later minor version. This protects any local
  scripts and the README examples during the transition.
- `--tui` and `--list-sessions` remain as hidden aliases mapping to the `tui`
  and `sessions` subcommands for one release.

### Docs impact

- Update every README/`docs/` invocation example to the subcommand form
  (coordinated with Phase 4).

### Tests (Phase 2)

- Extend `tests/test_cli.py`: subcommand routing, shared-parent-parser flag
  inheritance, back-compat shim (old flags → `run` + deprecation notice),
  `--help` for each subcommand exits 0.

### Deliverable / acceptance

- All modes reachable as subcommands; `main()` reduced to a dispatcher;
  old-style single-run invocation still works with a notice; tests green.

---

## Phase 3 — Modularize `application.py`

**Goal:** split the 2791-line module into cohesive units with the
`Application` class staying as a thin orchestrator. **Behavior is unchanged** —
this is a pure structural refactor, guarded by the 845-test suite.

### Constraint: preserve import paths

12 test files and both `cli.py` and `tui.py` import from
`llama_parameter_scout.application`, including **private** symbols
(`_journal_search_decisions`, `_build_workload_session_spec`,
`make_memory_pressure_fn`, `list_session_summaries`, `_dict_to_observation`,
…). The refactor must not break these imports.

**Strategy:** move implementations into new modules, then **re-export** them
from `application.py` (`from .execution import _build_argv  # noqa: F401`, etc.)
so existing import paths keep working. In the same phase, update the test and
UI imports to the new canonical modules; keep the re-exports for one release as
a compatibility layer, then remove.

### Proposed module split

| New module | Moved functions (from `application.py`) |
|------------|------------------------------------------|
| `execution.py` | `_build_argv`, `_env_for_device`, `_build_launch_env`, `_is_oom`, `_classify_launch`, `_warmup_and_classify`, `_launch_and_classify`, `_make_observation`, `_provenance_for` |
| `measurement.py` | `_parse_measurements`, `_validate_warmup_output`, `_compute_stats` |
| `fingerprint.py` | `_canonical_json`, `_compute_session_fingerprint`, `_compute_full_fingerprint_components`, `_compare_fingerprint_components`, `_capture_environment` |
| `estimation.py` | `_cache_element_size`, `_is_quantized_cache`, `_compute_model_weight_bytes`, `_estimate_advisory`, `_detect_available_memory_bytes`, `make_memory_pressure_fn`, `_check_resource_budget`, `_validate_budget`, `pre_execution_disposition` |
| `serialization.py` | `_dict_to_observation`, `_dict_to_screening_result` |
| `application.py` (kept) | `Application` class, `setup_session`/`start_or_resume_session`/`evaluate_candidate`/`screen_candidates`/`validate_candidates`/`run`/`compare_mtp`/`collect_mtp_profile`/`export_report_*`, `_validate_request`, `_screening_preflight`, `_run_screening_workload`, `_build_workload_session_spec`, `_journal_*`, `list_session_summaries` |

Exact assignment refined during implementation to respect call graphs (some
helpers may sit with their sole caller). Target: `application.py` under ~1200
lines, no module over ~600.

### Tests (Phase 3)

- No new behavior tests. Update imports; run the full suite (including
  `real_binary`/`smoke` markers locally where feasible). A green diff with
  identical test outcomes is the acceptance signal.
- Add a small `tests/test_module_boundaries.py` asserting the re-export shim
  still resolves the legacy symbols (guards the compat layer).

### Deliverable / acceptance

- Same behavior, smaller modules, no import breakage, ruff + mypy clean,
  845 tests still green.

---

## Phase 4 — Documentation split

**Goal:** turn the 1208-line / 61 KB single-file README into a lean entry
point plus focused topic pages, and align all examples with the Phase 2
subcommand CLI.

### Plan

- Keep `README.md` as: intro, install, 60-second quick start, a capability
  overview table, and links out. Target < ~250 lines.
- Move deep sections into `docs/`:
  - `docs/cli.md` — full subcommand reference.
  - `docs/search.md` — screening, validation, objectives, profiles.
  - `docs/mtp.md` — MTP comparison and profile.
  - `docs/export.md` — report/export formats, `include_hashes`.
  - `docs/journal.md` — journal schema (v5), resume/derive semantics.
  - `docs/troubleshooting.md` — GPU/backend matrix, network safety, errors.
- Cross-link from the README table of contents.
- Resolve the PyPI **TODO**: either document the real `pipx`/`uv tool` install
  once published, or clearly mark it "from source only until v0.1 is on PyPI".

### PyPI note (maintainer action, not code)

- `publish.yml` already exists. Actually publishing needs the maintainer's PyPI
  project + Trusted Publishing (OIDC) config. This plan documents the readiness
  and updates install docs; the publish action itself is a maintainer decision
  and is **not** performed as part of these phases.

### Tests (Phase 4)

- Doc-only; CI unaffected. Optional: a link-check is out of scope unless wanted.

### Deliverable / acceptance

- Lean README, topic pages under `docs/`, examples on the new CLI, no broken
  internal links.

---

## Sequencing, risk, and workflow

| Phase | Value | Risk | Depends on |
|-------|-------|------|------------|
| 1 — CLI core features | High | Low | — |
| 2 — Subcommands | Med-High | Med (CLI surface change, mitigated by shim) | 1 |
| 3 — Modularize `application.py` | High (maintainability) | Low-Med (pure refactor, well tested) | — (independent, but easier after 2) |
| 4 — Docs split | Med | Low | 2 (examples reference new CLI) |

Recommended order: **1 → 2 → 3 → 4**. Phase 3 could run in parallel with 1/2
but touches `application.py`, which Phase 1 also edits, so it is sequenced
after to avoid conflicting edits to the same file.

Per-phase workflow (orchestrated): map files → delegate implementation →
independent review → run targeted then full tests → fix findings → commit.
Each phase is its own branch/PR-sized unit and is committed only when
ruff + mypy + the full default test suite are green.

---

## Explicit non-goals

- No change to the journal schema (v5) or the core search algorithm.
- No behavior change in Phase 3 (structure only).
- No new runtime dependencies beyond the current `textual`.
- No automatic PyPI publish — readiness only.
- No detection-evasion, network, or model-download features (out of project scope).
