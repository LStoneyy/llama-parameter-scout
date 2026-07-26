# 07: Validate Finalists at Requested Context
**Status:** ready-for-agent
**User stories covered:** 13, 27-30, 69
## What to build
Pull candidates from the ranked screening shortlist and run Requested-Context Validation at the exact total n_ctx requested by the user. Attempt up to three successful validations, or stop when the entire capability-eligible shortlist is exhausted, or when the configured finite time or attempt budget is reached. Verify that the prompt, generation, and depth components sum to exactly the requested total n_ctx; record the effective values. Re-rank only those candidates that successfully complete validation; candidates that fail are recorded with their outcome category. When a top-ranked finalist fails validation, pull the next candidate from the shortlist as a bounded replacement. If rank changes occur after a replacement, the updated ranking is used for further replacements. Emit a Validated Recommendation only if at least one candidate completes validation; otherwise emit an explicit no-winner result showing the Provisional Screening Leader with the reasons that validation could not be completed. Never describe unvalidated evidence as best at the requested context. Tests must cover OOM failure of a top finalist, rank reversal after replacement, budget exhaustion, all candidates failing, exact context mismatch detection, and the case of exactly three successful validations.
## Acceptance criteria
- [ ] Candidates are pulled from the ranked screening shortlist for Requested-Context Validation
- [ ] At most three successful validations are attempted; the process stops early when the shortlist is exhausted or the budget is reached
- [ ] The effective prompt+generation+depth n_ctx is verified to sum to the exact requested total; mismatch is recorded and the candidate is not considered validated
- [ ] Only successfully validated candidates appear in the re-ranked final list
- [ ] Failed finalists are replaced by the next candidate from the shortlist
- [ ] Rank changes after a replacement trigger re-ranking for subsequent replacements
- [ ] A Validated Recommendation is emitted only when at least one candidate completes validation
- [ ] A no-winner result shows the Provisional Screening Leader and explains why validation could not be completed
- [ ] Unvalidated evidence is never described as best at the requested context
- [ ] Tests cover: OOM finalist, rank reversal, budget exhaustion, all fail, exact context mismatch, and three successful validations
- [ ] README documents the validation process, the no-winner state, and the language rules around unvalidated evidence
## Blocked by
- [06-search-capability-eligible-gpu-kv-moe-pairings.md](06-search-capability-eligible-gpu-kv-moe-pairings.md)
