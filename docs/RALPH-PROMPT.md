Inspect the latest commits, ./PRD.md, and ./issues/ to determine which issues
are already implemented. Implement exactly the next unfinished issue -- exactly
one issue per invocation.

At the very start of this session, use the `todowrite` tool to create a concise
todo list covering selection, implementation, review, verification, and commit.
Keep the list up to date as you work. Mark at most one entry as `in_progress`
at any time. Do not try to produce JSON output artificially.

Where practical, use the skill tool to load and follow the `tdd` skill at seams
already established or agreed in the spec, issue, architecture, or tests.

During implementation, regularly run typechecking and relevant individual test
files.

After implementation, delegate a review to `reviewer-cloud`. Address relevant
findings and rerun affected targeted checks as needed. Once review findings are
handled, run a final typechecking pass and the full test suite as the last
verification step before committing. If either final check fails, fix the cause
and rerun the failed check until it passes.

Commit to the current branch. Push only after committing the final remaining
issue.

Loop completion protocol:
- If all issues were already implemented when this invocation started, do not
  modify the repository and print a final line containing exactly:
  RALPH_COMPLETE
- If you just implemented, committed, and successfully pushed the final
  remaining issue, print a final line containing exactly: RALPH_COMPLETE
- If any required step fails (review, fix, verification, commit, or the final
  push), do not print RALPH_COMPLETE.
- Otherwise, do not print RALPH_COMPLETE.
