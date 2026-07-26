# 09: Resume or Derive Benchmark Sessions
**Status:** ready-for-agent
**User stories covered:** 44-46
## What to build
Persist every completed Benchmark Observation and search decision atomically so that a Ctrl-C, terminal loss, or reboot preserves all completed work. When the user starts a session with a journal file present, verify that the full fingerprint set matches: main-model shards (and draft shards where applicable), exact binary paths and capabilities, hardware identity, backend and driver, selected device, context composition, search profile, objective, environment-relevant settings, Resource Budget, and result schema version. On a full fingerprint match, resume the session from the last journal entry without re-running completed observations. On any mismatch, preserve the prior journal as history and create a derived Benchmark Session with a new identity. Prior measurements are excluded from the derived session's evidence collection; they remain as historical reference only. The derived session collects fresh evidence. Non-evidentiary user inputs may be reused only after explicit user review. The CLI must list all prior sessions and show the resumable or derived status and the reason for each. Tests must cover interrupted journal writes, cancellation during write, matching resume, every mismatch family, and schema-migration safety with a versioned schema that does not silently reuse incompatible data. Keep this slice compatible with later search and MTP observations by using versioned event types rather than special-case handling.
## Acceptance criteria
- [ ] Every completed Benchmark Observation and search decision is persisted atomically to the journal
- [ ] Starting a session with an existing journal verifies the full fingerprint set before resuming
- [ ] On a full fingerprint match, the session resumes from the last journal entry without re-running completed observations
- [ ] On any fingerprint mismatch, the prior journal is preserved and a derived session is created with a new identity; prior observations are excluded from derived-session evidence and remain as historical reference only
- [ ] The CLI lists all prior sessions with resumable or derived status and the reason
- [ ] Tests cover: interrupted journal write, cancellation during write, matching resume, every fingerprint mismatch family, derivation with fresh evidence collection, and schema migration versioning
- [ ] Journal events are versioned to allow forward-compatible schema evolution
- [ ] README documents session persistence, resume, and derivation semantics
## Blocked by
- [05-screen-and-rank-core-candidates.md](05-screen-and-rank-core-candidates.md)
