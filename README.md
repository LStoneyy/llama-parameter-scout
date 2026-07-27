# llama-parameter-scout

Benchmark llama.cpp models across candidate configurations by driving `llama-bench` with explicit settings, collecting throughput samples, and journaling results.

> **Status:** issue 03 - topology detection, device selection, session specification, context advisory.
> No adaptive search, full report rendering, TUI, or timeout/resource containment yet.

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
  --cache-type-v f16
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

## Tests

```bash
# All unit/acceptance tests (excludes real-binary smoke via auto-deselect)
uv run pytest -v

# Targeted files
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

Each successful run writes a versioned JSON document (version 3):

```json
{
  "version": 3,
  "candidate": {
    "threads": 4,
    "batch_size": 512,
    "ubatch_size": 512,
    "n_gpu_layers": 0,
    "flash_attn": false,
    "cache_type_k": "f16",
    "cache_type_v": "f16"
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
    "device_capabilities": ["cpu", "metal"]
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
    "diagnostics": []
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
