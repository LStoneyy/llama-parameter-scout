# llama-parameter-scout

Benchmark llama.cpp models across candidate configurations by driving `llama-bench` with explicit settings, collecting throughput samples, and journaling results.

> **Status:** issue 02 - binary resolution, toolchain fingerprint, model inspection.
> No system/device detection, adaptive search, full report rendering, TUI, or timeout/resource containment yet.

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

### What happens

1. **Binary resolution:** the executable is located via the precedence chain above.
2. **Toolchain fingerprint:** `llama-bench --version` is parsed for version, supported flags, device backends, and supported output formats. `--help` is verified to confirm the binary is responsive.
3. **Model inspection:** the GGUF file header is read to extract architecture, parameter count, training context, and quantization type. Split shard sets (e.g. `model-00001-of-00003.gguf`) are resolved and fingerprinted with path, size, mtime, and SHA-256 digest. Symlinks and network mounts produce warnings. Remote URLs, mmproj files, and LoRA adapters (by extension or GGUF metadata) are rejected.
4. **Preflight validation:** binary must exist and be executable; model must exist; n_ctx must be > 0, components nonnegative and sum exactly to total; candidate values must be in range.
5. **Warmup:** one `llama-bench` invocation (`-r 1`), output validated as parseable JSON/JSONL.
6. **Measurement:** second `llama-bench` invocation (`-r 3`), output parsed for `samples_ts` (requires at least 3 finite positive throughput samples).
7. **Statistics:** median throughput and population standard deviation (dispersion).
8. **Atomic journal:** a versioned JSON file is written atomically (write-temp + fsync + `os.replace`).
9. **Terminal output:** toolchain fingerprint and model fingerprint printed before warmup, followed by measurement results and journal path.

**Dispersion** is defined as the population standard deviation of the per-repetition throughput samples (tokens/second).

### llama-bench flags used

- `-p <n_prompt>`, `-n <n_gen>`, `-d <n_depth>`: context composition (n_ctx total is validated explicitly but not passed as a separate flag; depth is passed directly as `-d`)
- `-t <threads>`, `-b <batch>`, `-ub <ubatch>`: compute parameters
- `-ngl <layers>`: GPU offload
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

## Tests

```bash
# All unit/acceptance tests (excludes real-binary smoke via auto-deselect)
uv run pytest -v

# Targeted file
uv run pytest tests/test_application.py -v

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

Each successful run writes a versioned JSON document (version 2):

```json
{
  "version": 2,
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
  }
}
```

`shards` contains one fingerprint entry per shard file path (with `size_bytes`, `modified_time`, `digest_sha256`) when the model is a split shard set; an empty array for a single file.

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

Errors are printed to stderr; exit code is 1; no traceback.
