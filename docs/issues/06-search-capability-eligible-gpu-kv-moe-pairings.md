# 06: Search Capability-Eligible GPU, KV, and MoE Pairings
**Status:** ready-for-agent
**User stories covered:** 10-12, 23-26
## What to build
Extend the Screening Run path to include selected-device GPU layers, Flash Attention on/off, K cache type and V cache type tracked independently, and conditional n-cpu-moe expert CPU placement for MoE models. Generate every capability-eligible K/V type pairing that the installed binary supports. Prune pairings that are unsupported by the backend or incompatible with the model or hardware, recording each pruning with an explicit reason rather than a failure. Enforce that only the selected accelerator is used and no tensor split occurs. Keep CPU-only search valid and functional. Disclose in output that quantized KV type quality is not measured. Rankings, journal entries, CLI output, and tests must show the exact pairing and its capability provenance. Tests must include fixture coverage for dense models, MoE models, CPU-only, discrete-GPU, unified-memory, asymmetric K/V type combinations, and Flash Attention compatibility.
## Acceptance criteria
- [ ] GPU layers, Flash Attention state, K cache type, and V cache type are independently tracked for every candidate
- [ ] n-cpu-moe is added as a search parameter for MoE models; it is absent for dense models
- [ ] Every capability-eligible K/V type pairing is considered; unsupported or incompatible pairings are pruned with an explicit capability reason
- [ ] Only the selected accelerator is used; tensor split is never configured
- [ ] CPU-only search remains valid and produces a ranked result without accelerator-specific parameters
- [ ] Quantized KV output includes a disclosure that quality was not measured
- [ ] Rankings and journal entries show the exact pairing state and capability provenance for every candidate
- [ ] Tests cover dense models, MoE models, CPU-only, discrete-GPU, unified-memory, asymmetric K/V, and Flash Attention compatibility
- [ ] README documents the parameter space, capability detection, and the quantized-KV quality disclosure
## Blocked by
- [05-screen-and-rank-core-candidates.md](05-screen-and-rank-core-candidates.md)
