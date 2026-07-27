from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .application import Application
from .models import (
    CandidateConfig,
    ModelFingerprint,
    NctxComposition,
    ScoutError,
    SingleRunRequest,
    ToolchainFingerprint,
)
from .runner import ProductionRunner
from .toolchain import resolve_binary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llama-parameter-scout",
        description="Benchmark a single llama.cpp candidate configuration via llama-bench.",
    )
    parser.add_argument(
        "--llama-bench",
        required=False,
        type=Path,
        default=None,
        dest="binary_path",
        help="Explicit path to llama-bench executable (overrides --install-dir and PATH)",
    )
    parser.add_argument(
        "--install-dir",
        required=False,
        type=Path,
        default=None,
        help="Directory containing llama-bench (searched if --llama-bench is not given)",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--journal-path",
        required=True,
        type=Path,
        help="Path to journal output file",
    )
    parser.add_argument(
        "--n-ctx",
        required=True,
        type=int,
        help="Total context size (n_ctx = prompt + generation + depth)",
    )
    parser.add_argument(
        "--n-prompt",
        required=True,
        type=int,
        help="Prompt token count",
    )
    parser.add_argument(
        "--n-generation",
        required=True,
        type=int,
        help="Generation token count",
    )
    parser.add_argument(
        "--n-depth",
        required=True,
        type=int,
        help="Depth token count passed to llama-bench",
    )
    parser.add_argument(
        "--threads",
        required=True,
        type=int,
        help="Number of threads",
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--ubatch-size",
        required=True,
        type=int,
        help="Micro-batch (ubatch) size",
    )
    parser.add_argument(
        "--n-gpu-layers",
        required=True,
        type=int,
        help="Number of GPU layers (-1 = all)",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Enable flash attention",
    )
    parser.add_argument(
        "--cache-type-k",
        default="f16",
        type=str,
        help="KV cache type for K (default: f16)",
    )
    parser.add_argument(
        "--cache-type-v",
        default="f16",
        type=str,
        help="KV cache type for V (default: f16)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    try:
        resolution = resolve_binary(
            llama_bench=args.binary_path,
            install_dir=args.install_dir,
        )
    except ScoutError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    request = SingleRunRequest(
        binary_path=resolution.binary_path,
        model_path=args.model_path,
        journal_path=args.journal_path,
        n_ctx=NctxComposition(
            total=args.n_ctx,
            prompt=args.n_prompt,
            generation=args.n_generation,
            depth=args.n_depth,
        ),
        candidate=CandidateConfig(
            threads=args.threads,
            batch_size=args.batch_size,
            ubatch_size=args.ubatch_size,
            n_gpu_layers=args.n_gpu_layers,
            flash_attn=args.flash_attn,
            cache_type_k=args.cache_type_k,
            cache_type_v=args.cache_type_v,
        ),
        binary_source=resolution.source,
    )

    runner = ProductionRunner()
    app = Application(runner)

    def _on_ready(tc: ToolchainFingerprint, m: ModelFingerprint) -> None:
        _print_toolchain_fingerprint(tc, request.binary_source)
        _print_model_fingerprint(m)

    try:
        observation = app.run(request, on_ready=_on_ready)
    except ScoutError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Warmup complete.\n"
        f"Measurements: {len(observation.samples_tps)} repetitions\n"
        f"Median throughput: {observation.median_tps:.2f} t/s\n"
        f"Dispersion (population stddev): {observation.dispersion_tps:.2f} t/s\n"
        f"Journal: {request.journal_path}",
    )


def _print_toolchain_fingerprint(tc: ToolchainFingerprint, source: str) -> None:
    print(f"Toolchain (from {source}):")
    print(f"  Binary: {tc.binary_path}")
    print(f"  Version: {tc.version}")
    if tc.supported_flags:
        print(f"  Flags: {', '.join(tc.supported_flags)}")
    print(f"  Backends: {', '.join(tc.device_capabilities)}")
    print(f"  Output formats: {', '.join(tc.supported_output_formats)}")


def _print_model_fingerprint(m: ModelFingerprint) -> None:
    print("Model:")
    print(f"  Path: {m.path}")
    print(f"  Architecture: {m.architecture}")
    if m.parameter_count is not None:
        print(f"  Parameters: {m.parameter_count}")
    if m.context_length is not None:
        print(f"  Context length: {m.context_length}")
    if m.quantization is not None:
        print(f"  Quantization: {m.quantization}")
    if m.shards:
        print(f"  Shards: {len(m.shards)}")
    for diag in m.diagnostics:
        print(f"  Note: {diag}")


if __name__ == "__main__":
    main()
