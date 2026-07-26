# 08: Recommend Within Resource Budgets
**Status:** ready-for-agent
**User stories covered:** 35-40
## What to build
Add the memory-efficiency objective to the screening and validation flow, and enforce explicit resource-specific limits throughout. When a memory-limiting objective is selected, prefer the fastest stable validated Candidate Configuration that fits under the caps. Produce a speed-versus-memory Pareto frontier across all candidates with complete comparable metrics. On discrete-GPU systems require separate complete system RAM and selected-device VRAM evidence for any under-limit or memory-winner claim; on unified-memory systems use the single shared measure. Label partial, estimated, or incomparable memory metrics clearly; never treat missing metrics as zero. Never claim under-limit status or select a memory winner without complete comparable constrained-resource coverage. Apply conservative preflight checks and resource headroom before candidate execution. Stop or back off on observed memory pressure or OOM events; do not claim absolute safety. Tests must cover the memory-efficiency objective, no-fit outcome, no-complete-metrics outcome, discrete-GPU budget enforcement, unified-memory budget enforcement, and Pareto frontier output.
## Acceptance criteria
- [ ] A memory-limiting objective selects the fastest stable validated candidate that fits under explicit resource budgets
- [ ] A speed-versus-memory Pareto frontier is shown for all candidates with complete comparable metrics
- [ ] Discrete-GPU systems track complete system RAM and selected-device VRAM as separate constraints
- [ ] Unified-memory systems use a single shared budget
- [ ] Partial, estimated, or incomparable memory metrics are labelled as such; missing metrics are never treated as zero
- [ ] Under-limit status or memory-winner selection requires complete comparable constrained-resource coverage
- [ ] Conservative preflight checks with headroom run before every candidate execution
- [ ] Observed memory pressure or OOM stops or backs off the candidate
- [ ] Tests cover: memory-efficiency objective, no-fit, no-complete-metrics, discrete-GPU enforcement, unified-memory enforcement, and Pareto frontier
- [ ] README documents the memory-efficiency objective, budget enforcement rules, and metric labelling
## Blocked by
- [07-validate-finalists-at-requested-context.md](07-validate-finalists-at-requested-context.md)
