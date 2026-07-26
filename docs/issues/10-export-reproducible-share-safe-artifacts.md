# 10: Export Reproducible and Share-Safe Artifacts
**Status:** ready-for-agent
**User stories covered:** 55-60
## What to build
From a completed Benchmark Session or a no-winner state, export two artifact types. The first is a versioned full local JSON file with complete provenance: every input, fingerprint, observation, decision, and result. The second is a share-safe Markdown report that redacts home directory paths and unique device identifiers. Store the executable path, argv, environment variables, working directory, and shell dialect as structured fields rather than a single opaque command string. Render an exact llama-bench reproduction argv for every benchmarked candidate. Render compatible llama-cli and llama-server invocations, and a valid capability- and version-compatible llama-server preset.ini section. Omit or explain benchmark-only flags rather than claiming a 1:1 mapping to production usage. Support optional inclusion of full content hashes for stronger provenance, noting that performance reproducibility still requires full provenance and is never guaranteed. Clearly distinguish the full JSON as a private reproducibility artifact from the redacted Markdown as a share-safe report. Handle provisional screening and no-winner output states honestly, reflecting the evidence without overclaiming. Update the README to document uv workflows, the supported toolchain and backend matrix, constraints, result interpretation, and troubleshooting. Include schema snapshot tests, redaction tests, shell-quoting tests across all OS dialects, and preset contract tests.
## Acceptance criteria
- [ ] Full local JSON output contains versioned schema, every input, fingerprint, observation, decision, and result
- [ ] Share-safe Markdown report redacts home directory paths and unique device identifiers
- [ ] Executable path, argv, environment, cwd, and shell dialect are stored as structured fields
- [ ] Exact llama-bench reproduction argv is rendered for every benchmarked candidate
- [ ] Compatible llama-cli and llama-server invocations are rendered
- [ ] A valid llama-server preset.ini section is rendered for the recommended configuration
- [ ] Benchmark-only flags are omitted or explained; no claim of 1:1 mapping to production is made
- [ ] Optional full content hashes are supported with a caveat about performance reproducibility
- [ ] Full JSON is labelled as private reproducibility artifact; Markdown is labelled as share-safe report
- [ ] Provisional screening and no-winner states are handled honestly in both artifact types
- [ ] Tests cover: schema snapshot, redaction rules, shell quoting on all OS dialects, and preset contract
- [ ] README documents uv workflows, support matrix, constraints, result interpretation, and troubleshooting
## Blocked by
- [07-validate-finalists-at-requested-context.md](07-validate-finalists-at-requested-context.md)
- [08-recommend-within-resource-budgets.md](08-recommend-within-resource-budgets.md)
- [09-resume-or-derive-benchmark-sessions.md](09-resume-or-derive-benchmark-sessions.md)
