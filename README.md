# llama-parameter-scout

Benchmark llama.cpp models across candidate configurations by driving `llama-bench` with explicit settings, collecting throughput samples, and journaling results.

> **Status:** issue 06 - GPU layers, Flash Attention, K/V cache type pairing, and MoE expert CPU placement screening.
> No resource-budget recommendations, resume, or TUI yet.

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
9. **Atomic journal:** a versioned JSON file is written atomically (write-temp + fsync + `os.replace`).
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

## Journal schema

Each successful run writes a versioned JSON document (version 4):

```json
{
  "version": 4,
  "candidate": {
    "threads": 4,
    "batch_size": 512,
    "ubatch_size": 512,
    "n_gpu_layers": 0,
    "flash_attn": false,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "n_cpu_moe": null
  },
  "total_ctx": 2048,
  "prompt_ctx": 512,
  "generation_ctx": 1024,
  "depth_ctx": 512,
  "samples_tps": [45.5, 44.8, 45.3],
  "median_tps": 45.3,
  "dispersion_tps": 0.286,
  "n_measurements": 3,
  "binary_path": "/path/to/llama-bench",
  "binary_source": "explicit",
  "model_path": "/path/to/model.gguf",
  "toolchain": {
    "binary_path": "/path/to/llama-bench",
    "version": "1234",
    "supported_flags": ["-fa", "-ctk"],
    "supported_output_formats": ["json"],
    "device_capabilities": ["cpu", "metal"],
    "supported_cache_types": ["f16", "f32", "q8_0"]
  },
  "model": {
    "path": "/path/to/model.gguf",
    "architecture": "llama",
    "parameter_count": 7000000000,
    "context_length": 4096,
    "quantization": "Q4_K_M",
    "shards": [],
    "symlink_warning": false,
    "network_mount_detected": false,
    "diagnostics": [],
    "is_moe": false
  },
  "session_spec": {
    "cpu_only": false,
    "selected_device_ids": [],
    "n_ctx": {"total": 2048, "prompt": 512, "generation": 1024, "depth": 512},
    "fingerprint": "abc123...",
    "topology": {
      "cpu_count": 8,
      "system_memory_bytes": 17179869184,
      "accelerators": [],
      "memory_unified": false
    },
    "budget": {
      "ram_bytes": 17179869184,
      "vram_bytes": null,
      "headroom_ratio": 0.1
    },
    "advisory": {
      "likely_fit_lower": 10000,
      "likely_fit_upper": 20000,
      "uncertain_maximum": 22000,
      "model_weight_bytes": 4500000000,
      "kv_cache_bytes_estimate": 500000000,
      "note": "This is an estimate..."
    }
  }
}
```

`shards` contains one fingerprint entry per shard file path (with `size_bytes`, `modified_time`, `digest_sha256`) when the model is a split shard set; an empty array for a single file.

`session_spec` is present when the run used session setup (default). It contains the full session specification, including the system topology, resource budget, context advisory, and fingerprint for provenance.

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

### Journal schema (version 4)

Every journal entry includes:

- `"version": 4`
- `"disposition"` — the candidate disposition string
- `"process_launched"` — whether a child process was started
- `"diagnostic"` — human-readable diagnostic details
- `"safe_next_action"` — suggested next step

Successful (`completed`) outcomes also include measurement data (`samples_tps`,
`median_tps`, `dispersion_tps`), toolchain fingerprint, and model fingerprint.

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
