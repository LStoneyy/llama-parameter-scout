from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .application import Application
from .model_inspector import inspect_model
from .models import (
    CandidateConfig,
    ContextAdvisory,
    ContextExceededWithoutAcknowledgementError,
    InconsistentNglError,
    ModelFingerprint,
    MultipleAcceleratorsSelectedError,
    NctxComposition,
    NoAcceleratorSelectedError,
    ResourceBudget,
    ScoutError,
    SessionSetupError,
    SessionSpecification,
    SingleRunRequest,
    ToolchainFingerprint,
    UnknownDeviceError,
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

    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run on CPU only, even if accelerators are detected",
    )
    device_group.add_argument(
        "--device",
        type=int,
        action="append",
        default=None,
        help="Select a specific accelerator device by ID (may be specified only once)",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge context risk and confirm non-interactively",
    )
    parser.add_argument(
        "--budget-ram",
        type=int,
        default=None,
        help="Override system RAM budget in bytes for the context advisory",
    )
    parser.add_argument(
        "--budget-vram",
        type=int,
        default=None,
        help="Override device VRAM budget in bytes for the context advisory",
    )
    return parser.parse_args(argv)


def _print_session_spec(spec: SessionSpecification) -> None:
    print("Session Specification:")
    print(f"  CPU-only: {spec.cpu_only}")
    if spec.selected_device_ids:
        device_id = spec.selected_device_ids[0]
        for a in spec.topology.accelerators:
            if a.device_id == device_id:
                print(f"  Device: {a.description} (id={a.device_id}, backend={a.backend})")
                break
    print(f"  Context: {spec.n_ctx.total} tokens "
          f"(prompt={spec.n_ctx.prompt}, gen={spec.n_ctx.generation}, depth={spec.n_ctx.depth})")
    m = spec.model
    print(f"  Model: {m.path.name} ({m.architecture}, {m.context_length or 'unknown'} ctx)")
    _print_budget(spec.budget)
    _print_advisory(spec.advisory)
    print(f"  Acknowledged: {spec.acknowledged}")
    print(f"  Fingerprint: {spec.fingerprint}")


def _print_budget(budget: ResourceBudget) -> None:
    ram_gb = budget.ram_bytes / (1024 ** 3)
    print("  Resource Budget:")
    print(f"    RAM: {budget.ram_bytes} bytes ({ram_gb:.1f} GB)")
    if budget.vram_bytes is not None:
        vram_gb = budget.vram_bytes / (1024 ** 3)
        print(f"    VRAM: {budget.vram_bytes} bytes ({vram_gb:.1f} GB)")
    else:
        print("    VRAM: unified with RAM")
    print(f"    Headroom: {budget.headroom_ratio * 100:.0f}%")


def _print_advisory(advisory: ContextAdvisory) -> None:
    print("  Context Advisory:")
    status_line = f"    Training context: {advisory.training_context_status}"
    if advisory.training_context_tokens is not None:
        status_line += f" ({advisory.training_context_tokens})"
    print(status_line)
    print(f"    Likely-fit range: {advisory.likely_fit_lower} - {advisory.likely_fit_upper} tokens")
    print(f"    Uncertain maximum: {advisory.uncertain_maximum} tokens")
    print(f"    Note: {advisory.note}")


def _get_context_warning(model: ModelFingerprint, n_ctx_total: int) -> str | None:
    conflicted = [d for d in model.diagnostics if "conflict" in d.lower()]
    if conflicted:
        return (
            f"Model context length metadata is internally conflicted: "
            f"{'; '.join(conflicted)}. "
            "The context advisory is labelled as conflicted."
        )

    ctx = model.context_length
    if ctx is not None:
        if n_ctx_total > ctx:
            return (
                f"Requested context ({n_ctx_total}) exceeds the model's known "
                f"training context ({ctx}). Performance and quality may degrade. "
                "Proceeding may produce unreliable results."
            )
        return None

    return (
        "Model training context length is unknown. "
        "The context advisory is labelled as unknown."
    )


def _prompt_acknowledgement(warning: str) -> bool:
    print(f"Warning: {warning}", file=sys.stderr)
    try:
        response = input("Acknowledge and proceed? [y/N]: ")
        return response.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _prompt_confirmation(spec: SessionSpecification) -> bool:
    _print_session_spec(spec)
    try:
        response = input("Proceed with this session specification? [y/N]: ")
        return response.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


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

    n_ctx = NctxComposition(
        total=args.n_ctx,
        prompt=args.n_prompt,
        generation=args.n_generation,
        depth=args.n_depth,
    )

    candidate = CandidateConfig(
        threads=args.threads,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=args.flash_attn,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
    )

    runner = ProductionRunner()
    app = Application(runner)

    try:
        model = inspect_model(args.model_path)
    except ScoutError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    budget_overrides = None
    if args.budget_ram is not None or args.budget_vram is not None:
        budget_overrides = (args.budget_ram, args.budget_vram)

    acknowledged = args.yes
    if not acknowledged:
        warning = _get_context_warning(model, args.n_ctx)
        if warning is not None:
            if not _prompt_acknowledgement(warning):
                print("Aborted by user.", file=sys.stderr)
                sys.exit(1)
            acknowledged = True

    device_ids: tuple[int, ...] = tuple(args.device) if args.device is not None else ()

    try:
        spec = app.setup_session(
            model=model,
            n_ctx=n_ctx,
            candidate=candidate,
            cpu_only=args.cpu_only,
            device_ids=device_ids,
            budget_overrides=budget_overrides,
            acknowledged=acknowledged,
        )
    except (
        NoAcceleratorSelectedError,
        MultipleAcceleratorsSelectedError,
        UnknownDeviceError,
        InconsistentNglError,
        ContextExceededWithoutAcknowledgementError,
        SessionSetupError,
    ) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.yes:
        if not _prompt_confirmation(spec):
            print("Aborted by user.", file=sys.stderr)
            sys.exit(1)
    else:
        _print_session_spec(spec)

    request = SingleRunRequest(
        binary_path=resolution.binary_path,
        model_path=args.model_path,
        journal_path=args.journal_path,
        n_ctx=n_ctx,
        candidate=candidate,
        binary_source=resolution.source,
    )

    def _on_ready(tc: ToolchainFingerprint, m: ModelFingerprint) -> None:
        _print_toolchain_fingerprint(tc, request.binary_source)
        _print_model_fingerprint(m)

    try:
        observation = app.run(request, on_ready=_on_ready, session_spec=spec)
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
