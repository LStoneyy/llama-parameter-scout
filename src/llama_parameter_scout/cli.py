from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .application import (
    Application,
    ScoutError,
)
from .models import CandidateConfig, NctxComposition, SingleRunRequest
from .runner import ProductionRunner


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llama-parameter-scout",
        description="Benchmark a single llama.cpp candidate configuration via llama-bench.",
    )
    parser.add_argument(
        "--llama-bench",
        required=True,
        type=Path,
        dest="binary_path",
        help="Path to llama-bench executable",
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

    request = SingleRunRequest(
        binary_path=args.binary_path,
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
    )

    runner = ProductionRunner()
    app = Application(runner)

    try:
        observation = app.run(request)
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


if __name__ == "__main__":
    main()
