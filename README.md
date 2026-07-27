# llama-parameter-scout

Benchmark llama.cpp models across candidate configurations by driving `llama-bench` with explicit settings, collecting throughput samples, and journaling results.

> **Status:** issue 09 — resume or derive benchmark sessions with
> fingerprint-based exact-match reuse and CLI session listing.
> No TUI or MTP yet.

## Setup

```bash
uv sync --all-extras
```

## Single-run CLI

```bash
uv run llama-parameter-scout \
  --llama-bench /path/to/llama-bench \
  --model-path /path/to/model.gguf \
  --journal-path ./journal.json \
  --n-ctx 2048 \
  --n-prompt 512 \
  --n-generation 1024 \
  --n-depth 512 \
  --threads 4 \
  --batch-size 512 \
  --ubatch-size 512 \
  --n-gpu-layers 0 \
  --flash-attn \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --n-cpu-moe 1
```

### Binary resolution precedence

`--llama-bench` (explicit path), `--install-dir` (directory search), and `PATH` lookup are supported with the following precedence:

1. **`--llama-bench`** — used exactly as provided; no further lookup.
2. **`--install-dir`** — regular executable files in `{install_dir}` are checked if `--llama-bench` is absent. Falls through to PATH if absent or non-executable.
3. **`PATH`** — each directory in `PATH` is searched for a regular executable file.

Only regular executable files are accepted during search. On Windows both `llama-bench.exe` and `llama-bench` are tried.

### Device selection

The CLI requires an explicit device choice when accelerators are detected:

- **`--cpu-only`** — run on CPU only, even if accelerators (GPUs) are present.
- **`--device <id>`** — select a specific accelerator by device ID (0, 1, etc.).

If no accelerators are detected, CPU-only is automatic and no flag is required. If accelerators exist and no flag is given, an error is raised.

`--cpu-only` and `--device` are mutually exclusive.

### Session specification and confirmation

Before running the benchmark, the CLI builds a **Session Specification** containing the detected system topology, selected device, resource budget, and a context advisory. The full spec is printed for review:

```
Session Specification:
  CPU-only: True
  Context: 2048 tokens (prompt=512, gen=1024, depth=512)
  Resource Budget:
    RAM: 17179869184 bytes (16.0 GB)
    VRAM: unified with RAM
    Headroom: 10%
  Context Advisory:
    Likely-fit range: 10000 - 20000 tokens
    Uncertain maximum: 22000 tokens
    Note: This is an estimate ...
  Fingerprint: <sha256>
```

Use **`--yes`** to skip the confirmation prompt (for scripting). Without `--yes`, the CLI prints the spec and waits for user confirmation.

### Context warning

When the requested context exceeds the model's known training context, or the model's context length is unknown or internally conflicted, a warning is printed and explicit acknowledgement is required before proceeding.

### Budget overrides

Default budgets use the detected system memory (and zero VRAM for discrete GPUs). Override with:

- **`--budget-ram <bytes>`** — override the RAM budget for the context advisory.
- **`--budget-vram <bytes>`** — override the VRAM budget for the context advisory.

### What happens

1. **Binary resolution:** the executable is located via the precedence chain above.
2. **Toolchain fingerprint:** `llama-bench --version` is parsed for version, supported flags, device backends, and supported output formats. `--help` is verified to confirm the binary is responsive.
3. **Model inspection:** the GGUF file header is read to extract architecture, parameter count, training context, and quantization type. Split shard sets (e.g. `model-00001-of-00003.gguf`) are resolved and fingerprinted with path, size, mtime, and SHA-256 digest. Symlinks and network mounts produce warnings. Remote URLs, mmproj files, and LoRA adapters (by extension or GGUF metadata) are rejected.
4. **Session setup:** system topology is detected (CPU count, system memory, accelerator count and type, unified vs discrete memory). Device selection is validated, a resource budget is built, and a context advisory is estimated. The full specification is printed and requires confirmation (unless `--yes`).
5. **Preflight validation:** binary must exist and be executable; model must exist; n_ctx must be > 0, components nonnegative and sum exactly to total; candidate values must be in range.
6. **Warmup:** one `llama-bench` invocation (`-r 1`), output validated as parseable JSON/JSONL.
7. **Measurement:** second `llama-bench` invocation (`-r 3`), output parsed for `samples_ts` (requires at least 3 finite positive throughput samples).
8. **Statistics:** median throughput and population standard deviation (dispersion).
9. **Atomic journal (v5):** a schema-versioned JSON envelope with an ordered events list is written atomically (write-temp + fsync + `os.replace`). Each append preserves all prior committed events.
10. **Terminal output:** session specification, toolchain fingerprint and model fingerprint printed before warmup, followed by measurement results and journal path.

**Dispersion** is defined as the population standard deviation of the per-repetition throughput samples (tokens/second).

### llama-bench flags used

- `-p <n_prompt>`, `-n <n_gen>`, `-d <n_depth>`: context composition (n_ctx total is validated explicitly but not passed as a separate flag; depth is passed directly as `-d`)
- `-t <threads>`, `-b <batch>`, `-ub <ubatch>`: compute parameters
- `-ngl <layers>`: GPU offload
- `--main-gpu <id>`: selected accelerator device (when a specific device is chosen)
- `-fa on|off`: flash attention
- `-ctk <type>`, `-ctv <type>`: KV cache types
- `--n-cpu-moe <n>`: MoE expert CPU placement (MoE models only)
- `-o json`: machine-readable output
- `-r <repetitions>`: repetition count

### Supported model formats

| Format | Status |
|--------|--------|
| Single `.gguf` file | Supported |
| Split shards (`.gguf`, e.g. `model-00001-of-00003.gguf`) | Supported |
| `.mmproj` (multimodal projector) | Rejected with out-of-scope message |
| `.lora` (LoRA adapter) | Rejected with out-of-scope message |
| GGUF LoRA adapters (`general.type`, `adapter.type` metadata) | Rejected with out-of-scope message |
| Remote URLs (`http://`, `https://`, `s3://`, etc.) | Rejected with out-of-scope message |

GGUF versions 2 and 3 are supported. Missing optional metadata fields (parameter count, context length, quantization) are recorded as unknown with a diagnostic rather than causing rejection.

## Session Specification

The session specification bundles the detected system topology, device selection, context composition, resource budget, context advisory, and a deterministic fingerprint into a single immutable object.

### Resource budget defaults

| Parameter | Default |
|-----------|---------|
| RAM | Detected system memory (or 8 GB fallback) |
| VRAM (discrete) | 0 bytes (must be overridden explicitly with `--budget-vram`) |
| VRAM (unified) | Unified with RAM (no separate tracking) |
| Headroom | 10% of total budget |

Default budgets include a configurable headroom ratio to account for operating system and framework overhead. The headroom is subtracted from the total available budget before computing the context advisory.

### Context advisory

The advisory estimates a likely-fit context range based on:

- **Model weight footprint:** total size of all model shard files, with a 10% runtime overhead.
- **KV cache estimate:** computed from parameter count, cache type element size, and a heuristic bytes-per-token factor.
- **Resource budget:** total available RAM/VRAM after subtracting headroom.
- **Uncertain maximum:** the budget minus headroom, representing the theoretical upper bound.

The advisory includes a note stating clearly that it is an estimate, not a tested guarantee.

### Fingerprint

The session specification fingerprint is a SHA-256 hash of the device selection, context composition, budget, and topology parameters. It is deterministic and included in the journal for reproducibility.

The journal-level fingerprint (stored in `session_start` events) is a SHA-256 digest of an expanded component set covering toolchain, search profile, objective, environment, candidate config, and schema version for exact-match session resume.

## Search / Screening

The `search` module provides a deterministic staged candidate search that evaluates multiple candidate configurations in a single pass:

### Search Profiles

Three built-in profiles with documented finite budgets:

| Profile | `max_attempts` | `min_repetitions` | `max_repetitions_per_candidate` | `tie_bound_coefficient` | `dispersion_threshold` |
|---------|---------------|-------------------|-------------------------------|------------------------|----------------------|
| Quick   | 8            | 3                 | 5                             | 0.05                   | 0.15                 |
| Standard| 20           | 4                 | 6                             | 0.03                   | 0.12                 |
| Thorough| 40           | 5                 | 8                             | 0.02                   | 0.10                 |

All profiles use a screening context of `NctxComposition(total=512, prompt=128, generation=256, depth=128)` — a short-context evaluation before any full-context validation.

### Screening flow

1. **Baseline** — the supplied base `CandidateConfig` is evaluated first with `min_repetitions`.
2. **Expanded grid** — candidates are generated by crossing core variations (threads, batch_size, ubatch_size) with new parameter dimensions: GPU layers, Flash Attention on/off, all independently paired supported K/V cache types, and optional `n_cpu_moe` for MoE models. Core grid candidates come first, followed by new-parameter variations with base core values. The profile's `max_attempts` budget limits how many candidates are actually launched; the remainder receive `not_attempted` with a budget-exhausted reason.
3. **Capability preflight** — before any process launch, each candidate passes through capability screening:
   - **Flash Attention** requires the `-fa` flag in the toolchain's supported flags.
   - **Flash + cache compatibility**: Flash Attention is incompatible with quantized cache types (`q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`); such pairings are pruned with an explicit reason.
   - **Cache types**: both `cache_type_k` and `cache_type_v` must appear in the toolchain's `supported_cache_types`. Unsupported values produce an `unsupported` disposition with capability provenance.
   - **n_cpu_moe**: only emitted for MoE models when the toolchain supports `--n-cpu-moe`. Setting `n_cpu_moe` on a dense model is pruned. Values exceeding the model's expert count are pruned.
   - Device backend match, CPU-only vs GPU-layer consistency, and resource budget are checked as before.
4. **Evaluation** — each surviving candidate receives a short-context screening run. Failed candidates (OOM, crash, timeout, budget exhaustion, malformed output, unsupported, cancellation) are recorded with a disposition and reason. Duplicates of the base candidate are noted as `matches_baseline` and not re-evaluated. Every disposition has a non-empty reason.
5. **Dominance pruning** — candidates strictly dominated in both PP and TG (Pareto: >= in both and > in at least one) are removed from the ranked list and receive a `dominance_pruned` disposition.
6. **Adaptive repetitions** — if any surviving candidate has relative dispersion above the profile's `dispersion_threshold`, additional single-repetition evaluations are added until either the dispersion falls below the threshold, the candidate reaches `max_repetitions_per_candidate`, or the total `max_attempts` budget is exhausted. If the leading survivors are tied (within the objective-specific tie bound), extra evaluations are also added to separate them. All observations are pooled for aggregate scoring.
7. **Scoring** — PP and TG scores are normalized relative to baseline: `pp_score = candidate_pp / baseline_pp`. The Balanced score is the geometric mean: `sqrt(pp_score * tg_score)`.
8. **Ranking** — survivors are sorted by the selected objective (`pp`, `tg`, or `balanced`).
9. **Tie-breaking** — ties are identified using an objective-specific dynamic bound based on observed relative dispersion: PP dispersion for PP objective, TG dispersion for TG objective, average of both for Balanced. Tie order: stability (lower pooled relative dispersion), then comparable memory (lower known memory wins; unknown memory cannot win), then simpler flags (fewer threads, then smaller batch_size, then smaller ubatch_size).
10. **Provisional leader** — the top-ranked candidate is reported with the label: `⚠ Screening evidence only — not validated at requested context`.
11. **Quantized KV disclosure** — if any successful candidate uses a quantized KV cache type (`q8_0`, `q4_0`, etc.), the report includes a note stating that "quantized KV cache type quality was not measured; only throughput was assessed."
12. **Zero survivors** — if all grid candidates fail, the report has an empty shortlist, no leader, and each failure includes a reason.

### Capability detection and provenance

Capabilities are detected deterministically from the `llama-bench` binary:

- **Supported flags**: parsed from the `supported_flags:` line in `--version` output.
- **Supported cache types**: parsed from `-ctk`/`-ctv` flag help in `--help` output (e.g. `{f16,f32,q8_0,q4_0}`). If not enumerated, a conservative minimum of `("f16",)` is used.
- **Device backends**: parsed from the `backends:` line in `--version` output.

When a candidate is pruned or marked unsupported due to a capability mismatch, the `ScreeningResult.provenance` field and `reason` field document the specific capability that triggered the outcome (e.g. `flash_attn=unsupported`, `cache_k=q8_0=unsupported`).

### CandidateConfig parameter space

| Field | Type | Description |
|-------|------|-------------|
| `threads` | int | CPU thread count |
| `batch_size` | int | Batch size |
| `ubatch_size` | int | Micro-batch size |
| `n_gpu_layers` | int | GPU offload layers (0 = CPU, -1 = all) |
| `flash_attn` | bool | Flash Attention on/off |
| `cache_type_k` | str | KV cache type for K (e.g. `f16`, `f32`, `q8_0`) |
| `cache_type_v` | str | KV cache type for V (e.g. `f16`, `f32`, `q8_0`) |
| `n_cpu_moe` | int\|None | MoE experts on CPU (None for dense models) |

### CPU-only behavior

When `cpu_only=True`:
- `n_gpu_layers` is fixed to 0 for all candidates.
- `--split-mode` and `--main-gpu` are never emitted in argv.
- Only CPU-eligible capability checks apply.

### Selected-device isolation

When a GPU device is selected:
- Only that single device is used (`--split-mode none`, `--main-gpu <id>`).
- No tensor split is configured.
- Device visibility is enforced via `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` environment variables.

### MoE model behavior

- MoE is detected from GGUF metadata (e.g. `llama.expert_count` key). The exact key and value are preserved in `ModelFingerprint.is_moe` and `ModelFingerprint.moe_expert_count`.
- For MoE models, the screening grid includes candidates with `n_cpu_moe` set to `None`, `0`, and `1` (where `None` omits the flag entirely).
- Dense models never carry `n_cpu_moe`; attempting to set it produces a `pruned` disposition.
- The `--n-cpu-moe` flag is only emitted in argv when `candidate.n_cpu_moe is not None` and the toolchain supports it.

### Quantized KV quality disclosure

Any screening report that includes successful candidates using quantized KV cache types (`q8_0`, `q4_0`, etc.) carries a disclosure note:

> "Quantized KV cache type quality was not measured; only throughput was assessed."

This reflects that the screening pipeline measures throughput (tokens/second) and does not evaluate output quality differences between cache types.

### Usage via Application

```python
from llama_parameter_scout.application import Application
from llama_parameter_scout.search import STANDARD_PROFILE
from llama_parameter_scout.models import SingleRunRequest

app = Application(runner)
report = app.screen_candidates(
    request,
    profile=STANDARD_PROFILE,
)

# report.survivors — ranked list of RankedCandidate
# report.leader — ProvisionalLeader with scores and screening-evidence label
# report.baseline_disposition — baseline ScreeningResult
# report.grid_dispositions — every grid candidate's ScreeningResult (survivor or failure)
```

### Injectable evaluator

The `ScreeningEvaluator` protocol allows injecting a custom evaluator for testing:

```python
from llama_parameter_scout.search import ScreeningEvaluator, screen

class MyEvaluator:
    def __call__(self, candidate, n_ctx, repetitions) -> ScreeningResult:
        ...

report = screen(
    base_candidate=base,
    profile=QUICK_PROFILE,
    evaluator=MyEvaluator(),
    screening_n_ctx=SCREENING_N_CTX,
)
```

See `tests/test_search.py` for comprehensive deterministic fake-evaluator tests covering pruning, ranking, tie-breaking, budget enforcement, adaptive repetitions, and zero-survivor handling.

## Validation (issue 07)

The validation module pulls candidates from the ranked screening shortlist and evaluates them at the exact total n_ctx requested by the user, not at the short screening context (512).

### Validation process

1. **Accept a ScreeningReport** — the public `Application.validate_candidates()` method takes an existing report and does not repeat screening.
2. **Iterate survivors** — candidates are pulled from `ScreeningReport.survivors` in rank order.
3. **Full-context evaluation** — each candidate is evaluated at the exact requested `NctxComposition` (prompt + generation + depth = total).
4. **Context verification** — the effective observation context fields (prompt, generation, depth, total) are verified to match the requested composition exactly. A mismatch (any component differs or the sum does not equal total) is recorded as `context_mismatch` and the candidate is not validated.
5. **Bounded attempts** — at most 3 successful validations are collected. The process stops early when the shortlist is exhausted or a configurable finite attempt budget is reached.
6. **Re-ranking** — after each successful validation, all successful candidates are re-ranked by the selected objective (pp, tg, or balanced). The updated ranking is used to determine the recommendation order.
7. **Discriminated result**:
   - **`ValidatedRecommendation`** — emitted when at least one candidate completes validation. Contains the top-ranked `recommendation`, `alternatives`, and `failures`.
   - **`NoWinnerReport`** — emitted when zero candidates validate. Contains the `ProvisionalLeader` from screening and the failure reasons.

### No-winner state

When all candidates fail validation, the result is a `NoWinnerReport` that carries:
- The `ProvisionalLeader` from screening (labeled "Screening evidence only — not validated at requested context").
- Every validation attempt failure with its disposition category (OOM, timeout, crash, malformed_output, context_mismatch, validation_budget_exhausted, systemic_error) and a human-readable reason.

### Language rules

- Validated candidates carry the label: `"Full-context validation - verified at requested context"`.
- Unvalidated screening evidence (the `ProvisionalLeader`) is never described as the best at the requested context. Its label always reads: `"Screening evidence only - not validated at requested context"`.
- The two labels are mutually exclusive: a validated recommendation replaces the provisional leader; they never merge.

### Injection seam

```python
from llama_parameter_scout.search import ValidationEvaluator, validate_shortlist

class MyValidator:
    def __call__(self, candidate, n_ctx) -> ScreeningResult:
        ...

result = validate_shortlist(
    screening_report=report,
    requested_n_ctx=requested_ctx,
    evaluator=MyValidator(),
)
```

See `tests/test_issue07.py` for comprehensive validation tests covering OOM finalists, rank reversal, budget exhaustion, all-fail/no-winner, context mismatch, exactly three successes, and objective-specific ranking (pp, tg, balanced).

## Resource Budget (issue 08)

The `memory` objective selects the fastest stable candidate with complete comparable evidence that fits within the resource budget.

### Memory evidence model

```python
@dataclass(frozen=True)
class MemoryEvidence:
    ram_bytes: int | None = None        # system RAM usage
    vram_bytes: int | None = None       # device VRAM usage (None for unified)
    quality: MemoryQuality | None = None  # None = missing

MemoryQuality = Literal["complete", "partial", "estimated", "incomparable"]
```

- `complete`: measured process memory during the run.
- `partial`: only one of RAM/VRAM was measured (discrete systems).
- `estimated`: derived from heuristics, not measured.
- `incomparable`: measured but under incomparable conditions.
- `None` (missing): not measured at all (e.g. production workload cannot track process memory).

Missing values are `None`, never zero.

### Memory objective behavior

When `objective="memory"`:

1. **Complete evidence required** — only candidates with `quality="complete"` on all constrained resources are eligible. Partial/estimated/incomparable/missing evidence cannot establish under-limit status and cannot select a memory leader.
2. **Budget enforcement** — on discrete GPU systems, both `ram_bytes` and `vram_bytes` must be present and independently within their configured caps (after headroom). On unified/CPU topology, the single RAM measurement must fit the RAM cap.
3. **Speed measure** — balanced throughput (`sqrt(pp_score * tg_score)`) is used as the speed measure.
4. **Stability** — existing profile dispersion policy applies (no leader if dispersion exceeds threshold).
5. **Eligible candidates** — survivors list may retain candidates regardless of memory eligibility, but only a fully qualifying candidate can be the memory leader.

### No-winner semantics

When no validated candidate has complete in-budget stable evidence, `validate_shortlist` returns a `NoWinnerReport` with actionable failure reasons that distinguish:
- **No fit**: candidate memory exceeds budget.
- **Incomplete metrics**: memory evidence is missing or not `complete`.

Screening returns `leader=None` when no candidate qualifies.

### Pareto frontier (speed vs memory)

A speed-vs-memory Pareto frontier is computed for every screening and validation report. It includes all candidates with complete comparable metrics (not only PP/TG dominance survivors). A candidate is dominated when another has `>=` balanced speed and `<=` every constrained memory resource, with at least one strict improvement.

```python
@dataclass(frozen=True)
class ParetoEntry:
    candidate: CandidateConfig
    balanced: float
    memory: MemoryEvidence
```

The frontier is available as `ScreeningReport.pareto_frontier` and `ValidatedRecommendation.pareto_frontier`.

### Usage

```python
from llama_parameter_scout.search import screen
from llama_parameter_scout.models import ResourceBudget

budget = ResourceBudget(ram_bytes=32_000_000_000, vram_bytes=None, headroom_ratio=0.1)
report = screen(
    base_candidate=base,
    profile=STANDARD_PROFILE,
    evaluator=evaluator,
    screening_n_ctx=SCREENING_N_CTX,
    objective="memory",
    resource_budget=budget,
    memory_unified=True,
)
# report.leader — fastest stable candidate with complete in-budget evidence, or None
# report.pareto_frontier — speed-vs-memory frontier entries
```

`Application.screen_candidates()` and `Application.validate_candidates()` derive `resource_budget` and `memory_unified` from the `SessionSpecification` when provided.

### Production workload limitation

The production workload (`_run_screening_workload`) currently cannot measure process memory. Its resulting evidence is `None` (missing) — never synthesized from advisory estimates as observed complete evidence. This means **production alone cannot claim a memory winner** until an injected or real measurement source supplies complete metrics. To use the memory objective in production, inject an evaluator that attaches measured `MemoryEvidence(quality="complete")` observations.

### Tests

Run the issue 08 test suite:

```bash
uv run pytest tests/test_issue08.py -v
```

See `tests/test_issue08.py` for acceptance tests covering: fastest stable fit, no-fit, no-complete-metrics, discrete RAM/VRAM enforcement, unified enforcement, Pareto frontier (dominated exclusion, incomplete exclusion), evidence quality labels, validation no-winner semantics, and Application seam budget/topology propagation.

## Tests

```bash
# All unit/acceptance tests (excludes real-binary smoke via auto-deselect)
uv run pytest -v

# Targeted files
uv run pytest tests/test_search.py -v
uv run pytest tests/test_session.py -v
uv run pytest tests/test_topology.py -v
uv run pytest tests/test_application.py -v
uv run pytest tests/test_cli.py -v
uv run pytest tests/test_issue09.py -v

# Type check
uv run mypy src tests

# Lint
uv run ruff check src tests
```

### Real-binary smoke test

An opt-in test that invokes an actual `llama-bench` binary and a real GGUF model:

```bash
LLAMA_BENCH_PATH=/path/to/llama-bench \
MODEL_PATH=/path/to/model.gguf \
  uv run pytest tests/test_real_binary.py -v -m real_binary
```

The test is skipped unless both environment variables are set. It is excluded from the default `uv run pytest` run.

## Journal schema (v5)

Each run writes a v5 JSON envelope containing an ordered `events` collection:

```json
{
  "schema_version": 5,
  "events": [
    {
      "event_type": "session_start",
      "event_version": 1,
      "session_id": "<uuid>",
      "parent_session_id": null,
      "timestamp": "2026-07-27T12:00:00",
      "fingerprint_digest": "<sha256>",
      "fingerprint_components": {
        "schema_version": 5,
        "main_model": { "architecture": "llama", "shard_digests": [], ... },
        "draft_model": null,
        "toolchain": { "binary_path": "...", "version": "1234", ... },
        "topology": { "cpu_count": 8, "accelerators": [], ... },
        "device_selection": { "cpu_only": true, "selected_device_ids": [] },
        "context": { "total": 2048, "prompt": 512, "generation": 1024, "depth": 512 },
        "profile": null,
        "objective": "balanced",
        "environment": {},
        "budget": { "ram_bytes": 17179869184, "vram_bytes": null, "headroom_ratio": 0.1 },
        "candidate": { "threads": 4, ... }
      },
      "session_spec": { ... }
    },
    {
      "event_type": "observation_completed",
      "event_version": 1,
      "session_id": "<uuid>",
      "parent_session_id": null,
      "timestamp": "2026-07-27T12:05:00",
      "observation": {
        "disposition": "completed",
        "candidate": { "threads": 4, ... },
        "samples_tps": [45.5, 44.8, 45.3],
        "toolchain": { ... },
        "model": { ... }
      }
    }
  ]
}
```

`shards` contains one fingerprint entry per shard file path (with `size_bytes`, `modified_time`, `digest_sha256`) when the model is a split shard set; an empty array for a single file.

Event types: `session_start`, `observation_completed`, `outcome_recorded`, `search_decision`.

### Session resume / derive

When a journal already exists, the application inspects the full fingerprint components to decide:

- **Exact match:** fingerprint digest and all component families match → the session is **resumed**. No warmup or measurement is performed; the prior completed observation is reconstructed from the journal. CLI prints `Resumed session <id> — exact fingerprint match.`
- **Any mismatch:** a new session ID is created, referencing the parent session. Each differing family (model, toolchain, topology, context, budget, etc.) produces a human-readable reason. CLI prints `Derived session <id> from <parent> — Mismatch in: <family>.`
- **New journal:** a new session is started. CLI prints `New session <id>.`
- **Legacy v4 journal:** classified as incompatible/historical; a new session is derived from it. The v4 content is preserved and never overwritten.

### Fingerprint families

The `fingerprint_components` dict covers all relevant dimensions for session identity:

| Family | Contents |
|--------|----------|
| `schema_version` | Journal schema version |
| `main_model` | Model path, architecture, parameter count, context length, quantization, shard digests |
| `draft_model` | Reserved for future MTP; always `null` |
| `toolchain` | Binary path, version, supported flags, device capabilities, cache types |
| `topology` | CPU count, system memory, accelerators (kind/backend/device_id/description/memory) |
| `device_selection` | `cpu_only`, `selected_device_ids` |
| `context` | `total`, `prompt`, `generation`, `depth` of the n_ctx composition |
| `profile` | Search profile label and parameters (present during screening) |
| `objective` | Search objective (balanced, pp, tg, memory) |
| `environment` | Relevant environment variables (e.g. `CUDA_VISIBLE_DEVICES`) |
| `budget` | `ram_bytes`, `vram_bytes`, `headroom_ratio` |
| `candidate` | Single-candidate `CandidateConfig` (present in explicit single-run mode) |

### List sessions

```bash
uv run llama-parameter-scout --list-sessions --journal-path ./journal.json
```
```
Known sessions:
  <session_id>  [current]
  v4-legacy  [historical]  Legacy v4 journal format — incompatible schema
```

`--list-sessions` does not require any benchmark arguments.

## Error handling

| Condition | Error |
|-----------|-------|
| Binary not found via any resolution method | `ToolchainError` (preflight, no process launch) |
| Binary missing / not executable | `BinaryNotFoundError` (preflight, no process launch) |
| Model missing | `ModelNotFoundError` (preflight, no process launch) |
| Model path is a remote URL, mmproj, LoRA, or GGUF LoRA adapter | `ModelRejectedError` (preflight, no process launch) |
| Missing, incomplete, or inconsistent shard set | `ModelRejectedError` (preflight, no process launch) |
| Invalid GGUF file, bad magic, unsupported version, unknown type, or truncation | `ModelRejectedError` (preflight, no process launch) |
| Invalid n_ctx composition | `InvalidNctxError` (preflight, no process launch) |
| Candidate value out of range | `CandidateError` (preflight, no process launch) |
| llama-bench `--version` or `--help` fails | `ToolchainError` (preflight, no process launch) |
| Missing required `backends:` or `supported_output_formats:` in version output | `ToolchainError` |
| llama-bench exits nonzero | `SubprocessError` including captured stderr |
| Unparseable / malformed output | `MalformedOutputError` including raw output |
| GPU offload requested in CPU-only mode | `InconsistentNglError` (preflight, no process launch) |
| Multiple accelerators detected but no device selected | `NoAcceleratorSelectedError` (preflight, no process launch) |
| Device ID not found among detected accelerators | `UnknownDeviceError` (preflight, no process launch) |

Errors are printed to stderr; exit code is 1; no traceback.

## Candidate lifecycle

Every candidate configuration receives exactly one **disposition**. Pre-execution dispositions are assigned without launching a child process. Execution outcomes apply only when a process is launched.

### Pre-execution dispositions

| Disposition | Meaning | Next action |
|-------------|---------|-------------|
| `unsupported` | Binary not found, model not found, invalid configuration, unsupported device/backend | Check binary, model path, and configuration validity |
| `pruned` | Candidate parameter inconsistent with session (e.g. n_gpu_layers > 0 in cpu-only mode) | Adjust candidate parameter bounds |
| `not_attempted` | Candidate skipped by session logic | Ensure session is set up |
| `budget_exhausted` | Resource budget insufficient based on model footprint + KV cache estimate vs RAM/VRAM after headroom | Increase budget or reduce workload |

### Execution outcomes (process launched)

| Disposition | Meaning | Next action |
|-------------|---------|-------------|
| `completed` | Process exited with status 0; output parsed successfully | (none) |
| `oom` | OOM evidence: known exit codes (-9, 137) or stderr OOM keywords; also triggered by repeated memory-pressure backoff | Reduce context/model size or increase memory |
| `timeout` | Process exceeded `--candidate-timeout` and was terminated | Increase timeout or reduce workload |
| `crash` | Process exited nonzero without OOM evidence | Check binary compatibility, model integrity |
| `malformed_output` | Process exited 0 but output is not parseable as JSON/JSONL | Verify llama-bench version compatibility |
| `cancellation` | Process terminated by cancellation event | The run was cancelled by user or scheduler |
| `systemic_error` | Process adapter raised an exception during launch or monitoring | Investigate system/driver/hardware state |

### Resource budget gate

Before launching the candidate process, a conservative pre-execution check compares
the estimated resource need (model weight + KV cache estimate) against the available
budget (RAM minus headroom, plus discrete VRAM when applicable). If the budget is
insufficient, the candidate receives `budget_exhausted` with `process_launched=False`
and no child process is started.

For unified memory systems, the check is: model weight + KV estimate ≤ available RAM.
For discrete GPU systems, model weight is placed in VRAM first, KV cache fills remaining
VRAM, and any excess spills to RAM.

### Memory pressure detection

A stdlib cross-platform best-effort memory pressure detector monitors available system
memory against the session budget (after headroom). On Linux it reads `/proc/meminfo`,
on macOS it parses `vm_stat`. If detection is unavailable it returns false (no pressure).

Repeated pressure during a candidate run leads to bounded backoff (`--candidate-timeout`
respects backoff intervals) then `oom` termination. The detector is wired into the CLI
default evaluation policy when a session spec exists, and is preserved as an injectable
`memory_pressure_fn` on `ExecutionPolicy`.

### Execution policy

The `--candidate-timeout` CLI option enforces a per-process wall-clock timeout:

```
--candidate-timeout 120.0
```

Default: no timeout. Memory pressure backoff is built into the policy when a session
specification is available.

### Process containment

The tool launches native child processes via `Popen` with process-group management.
Process trees are terminated on timeout, cancellation, memory pressure, and caller
interruption. Cross-platform strategy:

- **POSIX:** `os.killpg` with `SIGTERM`; 3-second grace period; `SIGKILL` fallback.
- **Windows:** `taskkill /F /T` directly for the entire process tree (no SIGTERM).

This containment is best-effort. The tool never promises absolute host isolation
— child processes may still trigger host OOM or driver reset.

### Journal schema (v5 events)

Every journal event includes:

- `"event_type"` — one of `session_start`, `observation_completed`, `outcome_recorded`, `search_decision`
- `"event_version"` — version of the event schema
- `"session_id"` — session identifier
- `"parent_session_id"` — parent session (for derived sessions)
- `"timestamp"` — UTC ISO-8601 timestamp

`observation_completed` events contain measurement data (`samples_tps`,
`median_tps`, `dispersion_tps`), toolchain fingerprint, and model fingerprint.
`search_decision` events contain candidate, disposition, and reason for every
screening decision made during candidate search.

### Repeated / systemic stop policy

The `StopTracker` class tracks consecutive systemic failures (crash, OOM,
systemic_error). After `max_stops` consecutive failures it signals stop. A
successful completion resets the counter.

```python
from llama_parameter_scout.models import StopTracker
tracker = StopTracker(max_stops=3)
```

## CLI output

Every candidate evaluation prints the disposition, diagnostic, and next action:

```
Disposition: completed

Warmup complete.
Measurements: 3 repetitions
Median throughput: 45.30 t/s
Dispersion (population stddev): 0.29 t/s
Journal: /path/to/journal.json
```

For failures:

```
Disposition: crash
Diagnostic: Process exited with code 1.
Segmentation fault
Next action: Check binary compatibility, model integrity, and system resources.
```

Exit code is 1 for any disposition other than `completed`.
