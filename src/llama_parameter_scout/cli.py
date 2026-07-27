from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .application import Application, list_session_summaries, make_memory_pressure_fn
from .model_inspector import inspect_model
from .models import (
    CandidateConfig,
    CandidateOutcome,
    ContextAdvisory,
    ContextExceededWithoutAcknowledgementError,
    ExecutionPolicy,
    InconsistentNglError,
    JournalError,
    ModelFingerprint,
    MultipleAcceleratorsSelectedError,
    NctxComposition,
    NoAcceleratorSelectedError,
    ResourceBudget,
    ScoutError,
    SearchProfile,
    SessionSetupError,
    SessionSpecification,
    SingleRunRequest,
    ToolchainFingerprint,
    UnknownDeviceError,
)
from .runner import ProductionRunner
from .search import get_profile
from .toolchain import inspect_toolchain, resolve_binary


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
        required=False,
        type=Path,
        default=None,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--journal-path",
        required=False,
        type=Path,
        default=None,
        help="Path to journal output file",
    )
    parser.add_argument(
        "--n-ctx",
        required=False,
        type=int,
        default=None,
        help="Total context size (n_ctx = prompt + generation + depth)",
    )
    parser.add_argument(
        "--n-prompt",
        required=False,
        type=int,
        default=None,
        help="Prompt token count",
    )
    parser.add_argument(
        "--n-generation",
        required=False,
        type=int,
        default=None,
        help="Generation token count",
    )
    parser.add_argument(
        "--n-depth",
        required=False,
        type=int,
        default=None,
        help="Depth token count passed to llama-bench",
    )
    parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=None,
        help="Number of threads",
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--ubatch-size",
        required=False,
        type=int,
        default=None,
        help="Micro-batch (ubatch) size",
    )
    parser.add_argument(
        "--n-gpu-layers",
        required=False,
        type=int,
        default=None,
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
    parser.add_argument(
        "--n-cpu-moe",
        type=int,
        default=None,
        help="Number of MoE experts to offload to CPU (MoE models only)",
    )

    device_group = parser.add_mutually_exclusive_group(required=False)
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
    parser.add_argument(
        "--candidate-timeout",
        type=float,
        default=None,
        help="Maximum seconds for each candidate process launch (default: no timeout)",
    )

    # ── TUI mode ─────────────────────────────────────────────────
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the Textual TUI wizard",
    )

    # ── Session listing mode ─────────────────────────────────────
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all known sessions in the journal and exit",
    )

    # ── Search profile and objective (for fingerprinting) ─────────
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=("quick", "standard", "thorough"),
        help="Search profile label (default: none; used for session fingerprint)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="balanced",
        choices=("balanced", "pp", "tg", "memory"),
        help="Search objective (default: balanced; used for session fingerprint)",
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


def _print_outcome(outcome: CandidateOutcome, journal_path: Path | None = None) -> None:
    print(f"\nDisposition: {outcome.disposition}")
    if outcome.diagnostic:
        print(f"Diagnostic: {outcome.diagnostic[:200]}")
    if outcome.safe_next_action:
        print(f"Next action: {outcome.safe_next_action}")
    if outcome.observation is not None:
        obs = outcome.observation
        print("\nWarmup complete.")
        print(f"Measurements: {len(obs.samples_tps)} repetitions")
        print(f"Median throughput: {obs.median_tps:.2f} t/s")
        print(f"Dispersion (population stddev): {obs.dispersion_tps:.2f} t/s")
    if journal_path is not None:
        print(f"Journal: {journal_path}")


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


def _print_list_sessions(sessions: list[dict[str, Any]]) -> None:
    if not sessions:
        print("No sessions found.")
        return
    print("Known sessions:")
    for s in sessions:
        status = s.get("status", "unknown")
        reason = s.get("reason", "")
        sid = s.get("session_id", "?")
        parent = s.get("parent_session_id")
        line = f"  {sid}  [{status}]"
        if parent and parent != "v4-legacy":
            line += f"  parent: {parent}"
        if reason:
            line += f"  {reason}"
        print(line)


def _require_benchmark_args(args: argparse.Namespace) -> None:
    required = [
        ("--model-path", args.model_path),
        ("--journal-path", args.journal_path),
        ("--n-ctx", args.n_ctx),
        ("--n-prompt", args.n_prompt),
        ("--n-generation", args.n_generation),
        ("--n-depth", args.n_depth),
        ("--threads", args.threads),
        ("--batch-size", args.batch_size),
        ("--ubatch-size", args.ubatch_size),
        ("--n-gpu-layers", args.n_gpu_layers),
    ]
    for name, val in required:
        if val is None:
            print(f"Error: {name} is required in benchmark mode", file=sys.stderr)
            sys.exit(1)
    if not args.cpu_only and args.device is None:
        print(
            "Error: either --cpu-only or --device is required in benchmark mode",
            file=sys.stderr,
        )
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # ── TUI mode ────────────────────────────────────────────────
    if args.tui:
        from .tui import ScoutTuiApp as _ScoutTuiApp
        tui_app = _ScoutTuiApp()
        tui_app.run()
        return

    # ── List-sessions mode ──────────────────────────────────────
    if args.list_sessions:
        jp = args.journal_path
        if jp is None:
            print("Error: --journal-path is required with --list-sessions", file=sys.stderr)
            sys.exit(1)
        try:
            summaries = list_session_summaries(jp)
        except JournalError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        _print_list_sessions(summaries)
        sys.exit(0)

    _require_benchmark_args(args)

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
        n_cpu_moe=args.n_cpu_moe,
    )

    runner = ProductionRunner()

    toolchain: ToolchainFingerprint | None = None
    try:
        toolchain = inspect_toolchain(runner, resolution.binary_path)
    except ScoutError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

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

    # ── Resolve profile from --profile flag ─────────────────────
    profile: SearchProfile | None = None
    if args.profile is not None:
        try:
            profile = get_profile(args.profile)
        except ValueError:
            pass

    # ── Session seam: start / resume / derive ──────────────────
    try:
        session_result = app.start_or_resume_session(
            journal_path=request.journal_path,
            spec=spec,
            model=model,
            candidate=candidate,
            profile=profile,
            objective=args.objective,
            toolchain=toolchain,
        )
    except JournalError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    obs = session_result.observation
    if session_result.action == "resumed" and obs is not None:
        print(f"Resumed session {session_result.session_id} — exact fingerprint match.")
        print("\nWarmup complete.")
        print(f"Measurements: {len(obs.samples_tps)} repetitions")
        print(f"Median throughput: {obs.median_tps:.2f} t/s")
        print(f"Dispersion (population stddev): {obs.dispersion_tps:.2f} t/s")
        print(f"Journal: {request.journal_path}")
        return

    if session_result.action == "resumed":
        print(
            f"Resumed incomplete session {session_result.session_id} "
            f"— exact fingerprint match; proceeding to measurement."
        )
    elif session_result.action == "derived":
        print(
            f"Derived session {session_result.session_id} "
            f"from {session_result.parent_session_id} — "
            f"{session_result.mismatch_reason}."
        )
    else:
        print(f"New session {session_result.session_id}.")

    memory_pressure_fn = make_memory_pressure_fn(spec) if spec is not None else None
    policy = ExecutionPolicy(
        candidate_timeout_seconds=args.candidate_timeout,
        memory_pressure_fn=memory_pressure_fn,
    )

    def _on_ready(tc: ToolchainFingerprint, m: ModelFingerprint) -> None:
        _print_toolchain_fingerprint(tc, request.binary_source)
        _print_model_fingerprint(m)

    outcome = app.evaluate_candidate(
        request,
        on_ready=_on_ready,
        session_spec=spec,
        policy=policy,
        session_id=session_result.session_id,
    )

    _print_outcome(outcome, journal_path=request.journal_path)

    if outcome.disposition != "completed":
        sys.exit(1)


def _print_toolchain_fingerprint(tc: ToolchainFingerprint, source: str) -> None:
    print(f"Toolchain (from {source}):")
    print(f"  Binary: {tc.binary_path}")
    print(f"  Version: {tc.version}")
    if tc.supported_flags:
        print(f"  Flags: {', '.join(tc.supported_flags)}")
    print(f"  Backends: {', '.join(tc.device_capabilities)}")
    print(f"  Output formats: {', '.join(tc.supported_output_formats)}")
    if tc.supported_cache_types:
        print(f"  Cache types: {', '.join(tc.supported_cache_types)}")


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
