# Product Requirements Document: llama-parameter-scout

**Version:** 0.1 (draft)
**License:** Apache-2.0
**Language:** English
**Distribution:** PyPI (pipx / uv tool) for v0.1; standalone binaries later
**Target audience:** llama.cpp power users with local binaries and local GGUF models

---

## Problem Statement

Finding optimal llama.cpp runtime settings is hardware-dependent and model-dependent. Users who already have local llama.cpp binaries and GGUF files must manually probe parameter combinations (GPU layers, threads, batch sizes, flash attention, KV cache types) for every new model or hardware change. The process is slow, repetitive, irreproducible, and grows worse for large models, Mixture-of-Experts architectures, and speculative decoding setups.

Existing search is ad hoc: no structured comparison, no ranking against a chosen objective, no explicit memory constraints, no record of what was tried or why it failed. Users lack a tool that systematically discovers safe, well-performing configurations for their exact system, model, and target context.

---

## Solution

llama-parameter-scout is a cross-platform Python 3.11+ application that takes a local llama.cpp installation (or llama-bench executable), a local GGUF text model, and a requested context size, then runs a staged adaptive search across capability-eligible parameter combinations. It benchmarks candidates under a stated objective (prompt processing speed, token generation speed, balanced performance, or speed under memory limits) and attempts requested-context validation. When a candidate completes validation, it produces a Validated Recommendation with full provenance. Otherwise, it reports an explicit no-winner state and may show a Provisional Screening Leader with the reasons validation could not be completed.

The tool provides two interfaces: a non-interactive CLI for scripting and automation, and a keyboard-driven Textual TUI for interactive sessions. Both cross a common Application interface that starts or resumes Benchmark Sessions and loads results. The core is UI-independent and typed.

Every observation is journaled atomically. Sessions can be cancelled and resumed when the full toolchain, model, hardware, device, context composition, search profile, objective, environment, resource budget, and schema fingerprint match. Failed configurations are categorised and recorded; isolated failures continue, repeated systemic errors stop.

---

## User Stories

1. As a user, I want to install llama-parameter-scout with a single pipx or uv tool command, so that I can start using it without manual dependency resolution.

2. As a user, I want the tool to discover llama.cpp installations through PATH, an install directory, or an explicit llama-bench path, so that I can use whichever build I already have.

3. As a user, I want an explicit executable path to override any discovered installation, so that I can pin a specific build for reproducibility.

4. As a user, I want the selected binary and its version to be fingerprinted and displayed before any benchmark runs, so that I can confirm the correct executable is used.

5. As a user, I want the tool to detect which flags and formats my llama-bench binary supports, so that only valid parameter combinations are generated.

6. As a user, I want the tool to reject or clearly report unsupported flags or malformed binary output, so that I am not misled by heuristics against human-readable output.

7. As a user, I want to provide a single GGUF model file or a shard set, so that the tool can resolve and fingerprint all shards before benchmarking.

8. As a user with a symlinked or network-mounted model, I want a warning about potential latency or instability, so that I can decide whether to proceed.

9. As a user, I want the tool to reject remote URLs, multimodal mmproj files, and LoRA adapters in v0.1, so that I get clear feedback about out-of-scope inputs.

10. As a user on a CPU-only system, I want capability-eligible CPU candidates and parameters to remain supported within the staged search, so that I still receive a useful recommendation.

11. As a user with multiple accelerators (GPUs), I want to explicitly select exactly one device before the search begins, so that no configuration uses an unintended device or splits across GPUs.

12. As a user, I want runs to enforce only the selected device, so that results reflect a single-device configuration.

13. As a user, I want the effective context to be the total n_ctx with prompt, generation, and depth components summing to my requested value, so that I know exactly what context size is validated.

14. As a user, I want an advisory likely-fit range and an uncertain maximum shown before I confirm context, so that I can avoid configurations that will trivially fail memory constraints.

15. As a user, I want the context estimate to draw on model metadata, training context, KV cache cost, main and draft model footprints, selected hardware, and conservative memory headroom, so that the estimate is as informed as possible without running a benchmark.

16. As a user, I want the advisory to state clearly that it is an estimate and not a tested guarantee, so that I do not mistake it for proven fit.

17. As a user requesting a context larger than the model's training context, I want an expert opt-in with a warning that no quality or RoPE guarantee applies, so that I knowingly proceed beyond the model's designed range.

18. As a user requesting a large context for a model whose training context is missing or internally conflicted, I want the context labelled unknown and the same acknowledgement required, so that no invented limit is assumed.

19. As a user, I want to choose between Quick, Standard, and Thorough search profiles, so that I can trade search breadth and confidence against time and compute budget.

20. As a user, I want to select an optimisation objective (prompt processing speed, token generation speed, balanced performance, or fastest stable configuration under memory limits), so that the recommendation matches my priority.

21. As a user, I want a staged search that establishes baselines, screens candidates at short context, prunes infeasible or dominated configurations, refines promising ones, and validates survivors at the requested context, so that I get explainable progressive results.

22. As a user, I want short-context screening results labelled as screening evidence only, so that I do not mistake them for proven behaviour at the requested context.

23. As a user, I want the tool to generate candidate configurations from a parameter space that includes GPU layers, CPU threads, batch size, micro-batch size, flash attention enabled/disabled, and all capability-eligible KV cache type pairings, so that the search covers the most impactful knobs.

24. As a user with an MoE model, I want n-cpu-moe added dynamically to the search parameters, so that MoE-specific expert placement and offloading on CPU is explored.

25. As a user, I want each candidate's KV cache type state and flash attention state tracked separately, so that recommendations distinguish these choices explicitly.

26. As a user, I want unsupported KV cache pairings recorded as unsupported or pruned rather than as failed benchmarks, so that the journal distinguishes capability gaps from runtime failures.

27. As a user, I want up to three successful requested-context validations attempted, with replacements pulled in only until three candidates validate, the capability-eligible shortlist is exhausted, or the configured finite time or attempt budget is reached, so that the recommendation reflects the strongest evidence available within bounded resources.

28. As a user, I want the best validated candidate to win, so that the recommendation is backed by measurements at my requested context.

29. As a user whose session ends with no validated candidate at the requested context, I want a clear statement that no validated recommendation exists, plus a provisional screening leader with the reasons it could not be validated, so that I have an informed starting point despite incomplete evidence.

30. As a user, I want the tool never to call an unvalidated result the best at the requested context, so that the output language is always honest about evidence limits.

31. As a user, I want every benchmark measurement to include a warmup phase and at least three completed repetitions, with median and dispersion reported, so that results are statistically minimally sound.

32. As a user, I want tie bounds derived from the observed dispersion of completed repetitions under a documented profile policy, with unstable or overlapping candidates receiving adaptive repetitions until they are separated or the finite budget ends, and with interference and thermal drift detected and warned about, so that noise does not drive the recommendation.

33. As a user evaluating prompt processing speed, token generation speed, or balanced performance, I want differences within noise reported as ties, with stability, lower comparable memory, and simpler flags as tie-breakers, so that the recommendation is robust to measurement variance.

34. As a user, I want "best" always qualified by the selected objective and validation status, so that no recommendation implies universal or quality superiority.

35. As a user with memory limits, I want the fastest stable configuration under explicit resource-specific caps, alongside a speed-or-memory Pareto frontier, so that I can decide the trade-off myself.

36. As a user on a discrete GPU system, I want system RAM and selected-device VRAM tracked against separate limits and headroom, so that each resource constraint is enforced independently.

37. As a user on a unified-memory system, I want a single shared memory budget without double-counting, so that the memory model matches the hardware.

38. As a user, I want partial, estimated, or incomparable memory metrics clearly labelled as such, so that I know they cannot prove under-limit status or select a memory winner.

39. As a user, I want the tool to apply conservative preflight checks and resource headroom before launching child processes, with timeout and backoff, so that the host is protected from trivially dangerous configurations.

40. As a user, I want the tool to stop on out-of-memory events and observable pressure, so that runaway configurations are contained, while understanding that child native processes may still trigger host OOM or driver reset.

41. As a user, I want failures categorised as unsupported, pruned, not attempted, OOM, timeout, crash, malformed output, cancellation, budget exhausted, or systemic error, so that I know exactly what happened to each candidate.

42. As a user, I want isolated failures journaled with the session continuing, so that one bad candidate does not abort the entire search.

43. As a user, I want repeated or systemic errors to stop the session with a clear explanation, so that I do not wait through fruitless retries.

44. As a user, I want the session journal to record every observation atomically, so that cancellation, terminal loss, or reboot preserves all completed work.

45. As a user resuming a session, I want the tool to verify that the main model shards, draft model shards, exact binaries, hardware and backend identity, selected device, context composition, search profile, objective, environment-relevant settings, resource budget, and result schema fingerprint all match, so that resume is safe and meaningful.

46. As a user whose system no longer matches the session fingerprint, I want the tool to create a derived new session from the nearest journal state, so that I can proceed without losing prior observation history.

47. As a user who wants to explore speculative decoding, I want an optional MTP mode with a checkbox and a local draft GGUF, so that I can opt into draft-model benchmarking.

48. As a user in MTP mode, I want the tool to require a capability-compatible exact llama-server binary and run a separate ephemeral loopback-only server benchmark, so that speculative decoding throughput can be measured without outbound network traffic.

49. As a user in MTP mode, I want a bounded main-model profile that may adjust main-model settings to leave memory and compute room for the draft, with the normal non-MTP winner always preserved for comparison, so that MTP and non-MTP outcomes remain separable.

50. As a user in MTP mode, I want MTP-on and MTP-off compared with identical main settings, the exact server binary, selected device, effective context, deterministic synthetic tokens, consistent generation and sampling settings, and the same timing window, excluding startup and model-loading, with warmup and at least three repetitions, so that the comparison is fair.

51. As a user in MTP mode, I want the MTP search to cover draft GPU layers, conditional draft n-cpu-moe, and a small n-max set, recording all draft shard metadata, memory, and fingerprints, so that the draft model's impact is fully captured.

52. As a user in MTP mode, I want the tool to validate main-draft model compatibility before benchmarking, so that incompatible pairs are rejected early.

53. As a user in MTP mode, I want the tool to prefer MTP off on a statistical tie, and to state clearly if feasibility or throughput benefit is not established, so that the recommendation does not overclaim speculative-decoding advantage.

54. As a user in MTP mode, I want the MTP recommendation explicitly labelled as limited to measured synthetic-token generation throughput, with no claim about prompt processing, output quality, representative workloads, acceptance under real prompts, or universal end-to-end speedup, so that I understand the narrow scope of the evidence.

55. As a user, I want a full local JSON output with complete provenance, so that I can audit or re-run every decision the tool made.

56. As a user, I want a share-safe Markdown report with home paths and unique device identifiers redacted by default, so that I can share results without exposing personal or system-identifying information.

57. As a user, I want optional inclusion of full model hashes in the output, so that stronger exact model identity is recorded for provenance, understanding that performance reproducibility still depends on full provenance and is never guaranteed.

58. As a user, I want the output to include exact local llama-bench reproduction argv plus compatible llama-cli and llama-server invocations (with draft and MTP flags where applicable), with the understanding that benchmark-only flags may be omitted or explained rather than claiming blind 1:1 equivalence, so that I can reproduce any benchmark manually.

59. As a user, I want a valid version-compatible llama-server preset.ini section for the recommended configuration (including draft and MTP flags where applicable), plus platform- and shell-specific llama-server invocation commands, with executable path, argv, environment, cwd, and shell dialect stored structurally rather than as a single pretence of a universal command string, so that I can deploy the recommendation on my exact platform.

60. As a user or contributor, I want the README to document uv-based setup, run, test, supported toolchain and backend matrix, constraints, result interpretation, and troubleshooting, and to be kept current as functionality changes, so that I can get started and understand the project without guessing.

61. As a CLI user, I want non-interactive flags for every setup step, so that I can script benchmark sessions in automation pipelines.

62. As a TUI user, I want a keyboard-driven wizard that guides me through setup, binary and model discovery, validation, profile and objective selection, and resource limits, so that I do not need to memorise flag names.

63. As a TUI user, I want a review and confirmation step before the search starts, so that I can verify every input before committing to a potentially long benchmark run.

64. As a TUI user, I want a real-time progress view with live result details and error details, so that I can monitor the search and act on problems early.

65. As a TUI user, I want to cancel a running session and resume it later, so that I can reclaim my machine for other work without losing progress.

66. As a TUI user, I want results displayed in read-only tabs or panes for Markdown, preset.ini, and platform-specific shell commands, with copy and save actions, and clipboard-absent environments degrading to save and export, so that I can use the results on any system.

67. As a Windows user, I want the same behaviour contract as on macOS and Linux, so that I can rely on consistent tool semantics regardless of my OS.

68. As a user, I want the tool to perform no outbound network access during runtime and to use loopback only for the optional temporary MTP server, so that my privacy and network security are preserved.

69. As a user, I want the tool to detect that no candidate could produce a validated recommendation and explain why, so that I understand that my system, model, or context request is the limiting factor rather than a tool bug.

---

## Implementation Decisions

### Deep-Module Architecture

The codebase is organised around deep modules whose interfaces hide complexity. The highest testing seam is a deep Application module that starts or resumes Benchmark Sessions and loads results. Both the CLI and the TUI cross only this seam.

Supporting deep modules:

- **Toolchain Probe**: discovers, identifies, fingerprints, and capability-detects llama.cpp binaries and llama-bench or llama-server executables.
- **Model Inspector**: reads GGUF metadata, resolves shards, fingerprints models, and extracts training context and architecture type (dense, MoE).
- **System Detector**: enumerates CPU, memory, accelerators, backends, driver versions, and unified vs. discrete memory topology.
- **Search Engine**: generates candidate parameter spaces, stages the adaptive search (baseline, screening, pruning, refinement, validation), and applies profile limits.
- **llama-bench Adapter**: translates candidates into validated llama-bench invocations, parses machine-readable output, and handles all version-dependent flag and format differences.
- **MTP Adapter**: manages the separate loopback-only llama-server benchmark lifecycle for speculative decoding, with its own input, evidence, and claim contracts distinct from the llama-bench adapter.
- **Results Engine**: ranks candidates by objective, applies tie-breaking, computes Pareto frontiers, and selects validated recommendations.
- **Session Journal**: records every observation atomically, implements fingerprint-based resume, and handles session derivation when fingerprints diverge.
- **Report Renderer**: produces local full JSON, redacted share-safe Markdown, preset.ini sections, and platform-specific shell commands from structured data.

### Seam Strategy

External process adapters (llama-bench and MTP) are real seams because both production implementations and fake adapters exist. The Application module injects these adapters, plus deterministic clocks and resource readings, for testing.

No shallow pass-through modules exist; each module hides one dimension of complexity (binary versioning, model format, hardware topology, search strategy, output format).

### Tooling, Distribution, and Documentation

- **Language and runtime**: Python 3.11+.
- **UI framework**: Textual for the TUI.
- **Core design**: Typed, UI-independent core with a non-interactive CLI that shares the same Application interface as the TUI.
- **Package management**: uv-managed virtual environment, dependencies, and lockfile; all commands run via `uv run`.
- **Distribution**: PyPI with pipx and uv tool installation; standalone artifacts planned for later versions.
- **Documentation**: README updated in every coherent phase to reflect current functionality.

### Resource Budget

Default resource budgets are backend-aware: they account for the memory topology (discrete GPU VRAM plus system RAM, or unified memory) and the capabilities of the detected binaries. Budget defaults are visible before execution, documented, configurable by the user, and included in the Session Specification and fingerprint. They must leave explicit headroom against known hardware limits. Exact calibrated default constants are established during implementation and hardware validation rather than prescribed here.

### Vocabulary

The following terms are used consistently throughout the codebase and documentation:

- **Benchmark Session**: one end-to-end invocation from input to output recommendation.
- **Session Specification**: the complete set of inputs and settings that define a session.
- **Candidate Configuration**: one parameter combination evaluated during the search.
- **Benchmark Observation**: the measurement record for one candidate, including repetitions, dispersion, memory, and failure state.
- **Screening Run**: a short-context measurement used for coarse candidate evaluation.
- **Requested-Context Validation**: measurement at the exact requested context that qualifies a candidate for recommendation.
- **Validated Recommendation**: a candidate confirmed through requested-context validation.
- **Provisional Screening Leader**: the best screening candidate when no validated recommendation exists.
- **MTP Profile**: a bounded search profile for speculative-decoding candidates.
- **Resource Budget**: the memory and runtime limits governing the search.
- **Capability-Eligible Pairing**: a parameter combination that the installed binary supports based on capability detection.
- **Session Journal**: the atomic log of all observations and decisions.
- **Toolchain Fingerprint**: a composite identity for the exact binary, version, capabilities, and platform.
- **Model Fingerprint**: a composite identity for the model's resolved shard set, metadata, stable file identity, and optional full content hash.

---

## Testing Decisions

- Good tests assert externally visible behaviour through the highest Application seam, not through internal implementation method calls.
- Fake process adapters and deterministic clocks and resource readings are injected through internal seams. The Application interface remains the primary test surface.
- Adapter-level contract fixtures are justified for external llama-bench and llama-server versioned output formats.
- Parser fixtures validate that the tool correctly interprets known-good and known-bad machine-readable output from multiple binary versions.
- Process lifecycle tests verify timeout, backoff, cancellation, process isolation, and OOM handling.
- Property-based and invariant tests cover ranking correctness, tie-breaking, memory-budget enforcement, and no-winner state generation.
- Resume fingerprint tests confirm that matching and non-matching session states produce the correct resume or derive behaviour.
- Report snapshot and schema tests confirm that JSON and Markdown output maintain their declared structure and redaction rules.
- TUI smoke tests exercise the wizard flow, progress views, cancellation, and result display without real binaries.
- The same behaviour contract applies on Windows, macOS, and Linux. CPU smoke paths release-gate every operating system. GPU and backend support is listed explicitly and validated where hardware exists, without implying that every backend is release-gated.
- OS-level CI runs tests and lint on Windows, macOS, and Linux. An opt-in real-hardware smoke suite runs on each OS when GPU hardware is available.
- Release checks run via uv and include tests, lint, typing, and packaging. Exact tool choices beyond pytest and ruff are not overprescribed until decided.
- There is no prior test suite in the empty repository. Upstream llama.cpp machine-readable output contracts and fixture-driven process test tools are the relevant prior art; this project establishes the pattern.

---

## Out of Scope

- Training, fine-tuning, or model-weight quantization (runtime quantized KV cache types remain in scope).
- Downloading models or binaries, or managing their licenses.
- Replacing llama.cpp or its tools.
- GUI (graphical interface beyond TUI and CLI).
- Multimodal models or mmproj files.
- LoRA adapters or control vectors.
- Remote model URLs or any outbound network download at runtime.
- Telemetry or usage reporting.
- Multi-GPU tensor split or RPC distribution.
- Parallel concurrent Benchmark Sessions.
- NUMA topology optimisation.
- RoPE or YaRN optimisation or quality validation beyond the training context.
- Exhaustive or Bayesian hyperparameter search.
- Privileged tuning (e.g., CPU governor, GPU clock locking).
- Universal reproducibility, performance, safety, or quality guarantees.
- Standalone installer binaries in v0.1 (planned for later).

---

## Further Notes

### Acceptance Claims: Qualifications

- **Validated Recommendation exists only after required completed measurements at the exact requested context.** Screening evidence alone does not constitute a validated recommendation. The output must always state the validation status.
- **A memory-under-limit claim requires complete comparable coverage of the constrained resource.** Partial or estimated metrics cannot prove under-limit status.
- **An MTP recommendation is valid only for paired synthetic-token generation evidence.** It may result in "benefit not established" when throughput does not improve or feasibility cannot be determined.
- **A quantized-KV recommendation always discloses that quality was not measured.** No quality-equivalence claim is made.
- **Local full JSON and redacted Markdown have different reproducibility and privacy contracts.** Full JSON preserves exact paths and device identifiers for reproducibility; the Markdown report redacts these for sharing.

### Staged Implementation Sequence

1. **Foundation**: Project structure, configuration models, typed core, process runner, cross-platform system detection, Toolchain Probe, Model Inspector foundation.
2. **One-Run Tracer Bullet**: End-to-end single-candidate benchmark through the Application seam, running a real (or fake) llama-bench, producing a journal entry and report.
3. **Model, System, and Safety**: Full Model Inspector (shard resolution, metadata, fingerprinting), System Detector (CPU, memory, GPU topology, unified vs. discrete), preflight safety checks, resource budgets.
4. **Adaptive Search and Full Validation**: Search Engine with staged screening, pruning, refinement, requested-context validation, ranking, tie-breaking, Pareto frontiers, and no-winner states.
5. **MTP**: Separate MTP Adapter, loopback server lifecycle, MTP profile, main-draft compatibility validation, synthetic-token comparison, combined reporting.
6. **TUI**: Textual application with wizard, review, progress, live result and error detail, cancellation, resume, and result tabs with copy-save.
7. **Release Hardening**: CI and OS matrix, opt-in hardware smoke, packaging for PyPI, README workflows, documentation, lint, typing, and final test suite.
