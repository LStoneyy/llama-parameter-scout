# 04: Contain and Classify Candidate Execution
**Status:** ready-for-agent
**User stories covered:** 39-43
## What to build
Harden single Candidate Configuration execution end to end. Enforce the Session Specification: the selected device, environment variables, and resource boundaries. Use process-tree timeout and cancellation so that runaway child processes do not block the tool indefinitely. Monitor observable memory pressure and apply bounded backoff when pressure is detected. Classify every Candidate Configuration into exactly one disposition. Pre-execution/search dispositions (unsupported, pruned, not attempted, budget exhausted) are assigned without launching a child process. Execution/session outcomes (OOM, timeout, crash, malformed output, cancellation, systemic error) apply only when a process launches. Process containment (timeout, cancellation, memory-pressure backoff) applies only after process launch. Journal the status and diagnostic details atomically. Continue the session after isolated failures but stop on repeated or systemic errors with a clear explanation. Never promise absolute host isolation; the tool contains native child processes that may trigger host OOM or driver reset. Terminal output and the report must expose the outcome category and a safe next action for each candidate. Tests must simulate every failure category and verify correct process cleanup on Windows, macOS, and Linux process semantics.
## Acceptance criteria
- [ ] Every candidate execution enforces the device, environment, and resource limits from the Session Specification
- [ ] Process-tree timeout and cancellation work: a candidate exceeding its time budget is killed and classified as timeout
- [ ] Observable memory pressure triggers bounded backoff; repeated pressure events are classified and stop the candidate
- [ ] Every Candidate Configuration receives exactly one disposition from the approved set
- [ ] Isolated failures are journaled and the session continues to the next candidate
- [ ] Repeated systemic errors stop the session with a clear explanation
- [ ] The terminal output shows the outcome category and a safe suggested next action for every candidate
- [ ] Pre-execution dispositions (unsupported, pruned, not attempted, budget exhausted) are assigned without launching a child process
- [ ] Execution/session outcomes (OOM, timeout, crash, malformed output, cancellation, systemic error) apply only when a process launches
- [ ] Process containment and monitoring apply only after process launch
- [ ] Tests simulate every disposition category via a fake process adapter, including both pre-execution and execution categories
- [ ] Process cleanup tests verify that child process trees are terminated after timeout, cancellation, and crash on Windows, macOS, and Linux
- [ ] README documents the candidate lifecycle, outcome categories, and process-safety caveats
## Blocked by
- [03-build-single-device-session-specification.md](03-build-single-device-session-specification.md)
