# 05: Screen and Rank Core Candidates
**Status:** ready-for-agent
**User stories covered:** 19-22, 31-34
## What to build
Execute an explainable staged set of candidate evaluations rather than a single fixed run. Offer three finite search profiles: Quick, Standard, and Thorough. Before screening, establish a baseline measurement and compute performance bounds. Generate capability-eligible core candidates varying threads, batch size, and micro-batch size at currently supported fixed offload, attention, and cache settings. Run short-context Screening Runs on each candidate. Prune infeasible or dominated configurations. Adapt repetition counts dynamically: increase repetitions when observed dispersion is high, stop when candidates separate or the finite budget is exhausted. Calculate prompt processing speed (PP), token generation speed (TG), and a normalized geometric-mean Balanced score for each candidate. Rank by the selected objective. Derive tie bounds from observed dispersion under a documented profile policy. Break ties using stability (lower dispersion), then comparable memory, then simpler flags. Output a Provisional Screening Leader explicitly labelled as not validated at requested context, plus a full disposition list showing every candidate and its screening outcome. If no candidate survives the Screening Runs, stop cleanly: report an empty shortlist, produce no Provisional Screening Leader, and provide an actionable disposition summary explaining why every candidate was eliminated. Include deterministic fake search tests that verify correct pruning, ranking, tie-breaking, and profile budget enforcement.
## Acceptance criteria
- [ ] Quick, Standard, and Thorough profiles are available; each enforces a documented finite budget
- [ ] A baseline measurement is taken before the staged search begins
- [ ] Core candidates are generated from threads, batch size, and micro-batch size at fixed offload/attention/cache settings
- [ ] Each candidate runs a short-context Screening Run
- [ ] Pruned or infeasible candidates are recorded with a reason and not ranked
- [ ] Repetition count adapts to dispersion within the profile budget
- [ ] PP, TG, and normalized geometric-mean Balanced scores are calculated for every surviving candidate
- [ ] Tie bounds are derived from observed dispersion under the documented profile policy
- [ ] Ties are broken by stability, then comparable memory, then simpler flags
- [ ] The Provisional Screening Leader is labelled as screening evidence only, not as validated at requested context
- [ ] The full disposition list shows every candidate with its screening outcome and reason
- [ ] When no candidate survives Screening Runs, the tool reports an empty shortlist with no Provisional Screening Leader and an actionable disposition summary
- [ ] Deterministic fake search tests verify pruning, ranking, tie-breaking, budget enforcement, and zero-survivor handling
- [ ] README documents the search profiles, objective selection, ranking method, and tie-breaking policy
## Blocked by
- [04-contain-and-classify-candidate-execution.md](04-contain-and-classify-candidate-execution.md)
