# 12: Optimize a Bounded MTP Profile
**Status:** ready-for-agent
**User stories covered:** 49-54, 58-59
## What to build
Extend the fixed MTP comparison into a bounded MTP Profile search. Preserve the normal non-MTP winner from the main search so it remains available for separate comparison. Allow a separate main-model configuration that reserves combined main-model and draft-model memory when MTP is on. Search draft GPU layers, conditional draft n-cpu-moe for MoE draft models, and a small capability-eligible n-max (maximum number of draft tokens per speculative iteration) set, all under finite profile limits. Every MTP-on result must be paired with an MTP-off measurement using identical main settings. Prefer MTP off on statistical ties. Produce a separate MTP recommendation or state that benefit was not established. Never merge synthetic MTP evidence into the main llama-bench ranking or recommendation. Extend the full JSON and Markdown report outputs with draft-model flags and combined provenance. Extend the preset.ini output with draft and MTP flags. Tests must cover memory-driven main-model adjustment for MTP headroom, no-fit for MTP candidates, rank shifts, partial metrics, statistical tie producing off preference, and successful MTP benefit.
## Acceptance criteria
- [ ] The normal non-MTP winner is preserved and does not change when MTP profiling is enabled
- [ ] A separate main-model configuration that reserves combined main+draft memory is searched when MTP is on
- [ ] Draft GPU layers, conditional draft n-cpu-moe, and a small n-max set are searched under finite profile limits
- [ ] Every MTP-on result is paired with MTP-off on identical main settings
- [ ] MTP off is preferred on statistical ties
- [ ] Separate MTP recommendation or benefit-not-established state is produced; MTP evidence is never merged into main ranking
- [ ] JSON and Markdown reports include draft-model flags, MTP flags, and combined provenance
- [ ] preset.ini output includes draft and MTP flags
- [ ] Tests cover: memory-driven main adjustment, no-fit, rank shifts, partial metrics, tie (off preferred), and successful benefit
- [ ] README documents the MTP profile, memory reservation, and the evidence separation policy
## Blocked by
- [08-recommend-within-resource-budgets.md](08-recommend-within-resource-budgets.md)
- [11-compare-mtp-on-and-off.md](11-compare-mtp-on-and-off.md)
