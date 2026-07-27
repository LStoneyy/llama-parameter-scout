# llama-parameter-scout

Benchmark llama.cpp models across candidate configurations by driving `llama-bench` with explicit settings, collecting throughput samples, and journaling results.

> **Status:** issue 01 - single explicit candidate configuration.
> No PATH discovery, GGUF metadata/shards, system/device detection, adaptive search, full report rendering, TUI, or timeout/resource containment yet.

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

All flags are kebab-case. `--llama-bench`, `--model-path`, `--journal-path`, `--n-ctx`, `--n-prompt`, `--n-generation`, `--n-depth`, `--threads`, `--batch-size`, `--ubatch-size`, and `--n-gpu-layers` are required. `--flash-attn` defaults to off. `--cache-type-k` and `--cache-type-v` default to `f16`.

### What happens

1. **Preflight validation:** binary must exist and be executable; model must exist; n_ctx must be > 0, components nonnegative and sum exactly to total; candidate values must be in range.
2. **Warmup:** one `llama-bench` invocation (`-r 1`), output validated as parseable JSON/JSONL.
3. **Measurement:** second `llama-bench` invocation (`-r 3`), output parsed for `samples_ts` (requires at least 3 finite positive throughput samples).
4. **Statistics:** median throughput and population standard deviation (dispersion).
5. **Atomic journal:** a versioned JSON file is written atomically (write-temp + fsync + `os.replace`).
6. **Terminal output:** warmup confirmation, measurement count, median, dispersion, journal path.

**Dispersion** is defined as the population standard deviation of the per-repetition throughput samples (tokens/second).

### llama-bench flags used

- `-p <n_prompt>`, `-n <n_gen>`, `-d <n_depth>`: context composition (n_ctx total is validated explicitly but not passed as a separate flag; depth is passed directly as `-d`)
- `-t <threads>`, `-b <batch>`, `-ub <ubatch>`: compute parameters
- `-ngl <layers>`: GPU offload
- `-fa on|off`: flash attention
- `-ctk <type>`, `-ctv <type>`: KV cache types
- `-o json`: machine-readable output
- `-r <repetitions>`: repetition count

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

Each successful run writes a versioned JSON document:

```json
{
  "version": 1,
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
  "n_measurements": 3
}
```

## Error handling

| Condition | Error |
|-----------|-------|
| Binary missing / not executable | `BinaryNotFoundError` (preflight, no process launch) |
| Model missing | `ModelNotFoundError` (preflight, no process launch) |
| Invalid n_ctx composition | `InvalidNctxError` (preflight, no process launch) |
| Candidate value out of range | `CandidateError` (preflight, no process launch) |
| llama-bench exits nonzero | `SubprocessError` including captured stderr |
| Unparseable / malformed output | `MalformedOutputError` including raw output |

Errors are printed to stderr; exit code is 1; no traceback.
