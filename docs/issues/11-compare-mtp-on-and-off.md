# 11: Compare MTP On and Off
**Status:** ready-for-agent
**User stories covered:** 47-48, 50, 52, 54, 68
## What to build
Add the narrow MTP (Multi-Token Prediction) tracer bullet. Accept a user-supplied local draft GGUF and an exact capability-compatible llama-server binary. Validate main-model and draft-model compatibility, resolve all draft shards, and produce a draft Model Fingerprint before any benchmarking. Launch an ephemeral loopback-only llama-server process with no outbound or non-loopback network traffic. Compare MTP off against MTP on using identical main-model settings, the selected device, effective context, deterministic synthetic tokens, consistent generation and sampling settings, and the same timing window. Exclude startup and model-loading time. Warm up the server and collect at least three completed measurement repetitions for each mode. Journal the paired evidence separately from main llama-bench evidence. Report measured synthetic generation throughput only, with explicit caveats that the measurement does not cover prompt processing, output quality, representative workloads, acceptance under real prompts, or universal end-to-end speedup. A fixed main Candidate Configuration is sufficient for this slice; no adaptive MTP search is required yet. Tests must cover fake server lifecycle, port-race handling, server cleanup on exit, incompatible draft model rejection, statistical tie producing no-benefit result, and loopback-only network binding.
## Acceptance criteria
- [ ] A local draft GGUF and exact llama-server path are accepted as optional inputs
- [ ] Main-model and draft-model compatibility is validated before any server process is launched
- [ ] All draft shards are resolved and fingerprinted
- [ ] An ephemeral llama-server is launched on loopback only; no outbound or non-loopback traffic occurs
- [ ] MTP off and MTP on are compared with identical main settings, device, effective context, deterministic synthetic tokens, and generation/sampling settings
- [ ] Startup and model-loading time are excluded from the measurement window
- [ ] Warmup and at least three completed repetitions are collected for each mode
- [ ] MTP evidence is journaled separately from main llama-bench evidence
- [ ] Output reports measured synthetic generation throughput only, with all quality caveats
- [ ] If benefit is not established, the output states that clearly rather than claiming advantage
- [ ] Tests cover: fake server lifecycle, port-race detection and recovery, server cleanup, incompatible draft rejection, tie/no-benefit, and loopback-only binding
- [ ] README documents MTP mode, server requirements, and evidence caveats
## Blocked by
- [10-export-reproducible-share-safe-artifacts.md](10-export-reproducible-share-safe-artifacts.md)
