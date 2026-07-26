# 02: Validate a Local Toolchain and Model
**Status:** ready-for-agent
**User stories covered:** 2-9
## What to build
Turn raw paths into a trustworthy ready/not-ready result before any benchmark execution. Resolve the llama-bench executable by precedence: explicit path overrides an install directory which overrides PATH discovery. Inspect the exact binary to obtain its version, help text, supported output formats, device capabilities, and produce a Toolchain Fingerprint. Display the fingerprint and persist it in the journal for provenance. Never scrape or heuristically parse human-readable output. Inspect a local single GGUF model file or a shard set: resolve every shard in the set, produce a Model Fingerprint with metadata (architecture, parameter count where available, training context where available, quantization type where available), warn for symlinked or network-mounted inputs, reject missing or inconsistent shard sets, reject remote URLs, mmproj files, LoRA adapters, and any format that is not a supported single or split-shard GGUF. Missing optional metadata fields are recorded as unknown with a clear diagnostic; they do not cause automatic rejection. The exact selected binary path must remain visible throughout the session. Integrate through the CLI session setup flow, persist fingerprints in the journal, and include tests with versioned binary and model fixtures.
## Acceptance criteria
- [ ] Explicit `--llama-bench` path is used as-is; absent that, `--install-dir` is searched; absent that, PATH is searched
- [ ] A Toolchain Fingerprint is displayed on the terminal and persisted in the journal, containing binary path, version, supported flags, and device capabilities
- [ ] The tool rejects a llama-bench binary that fails version/help inspection
- [ ] Human-readable output parsing is not used for any capability or result detection
- [ ] A single GGUF file is fingerprinted with model metadata where available; missing optional fields are recorded as unknown with a diagnostic; split shards are resolved and each shard is fingerprinted
- [ ] A warning is emitted for symlinked or network-mounted model paths
- [ ] Remote URLs, mmproj files, and LoRA adapters are rejected with a clear out-of-scope message
- [ ] Missing shards or inconsistent shard sets are rejected with a diagnostic message
- [ ] The selected binary path is visible in the terminal output and the Session Specification
- [ ] Tests use versioned binary output fixtures and model metadata fixtures to validate every detection and rejection path
- [ ] A model with unknown parameter count, training context, or quantization type is accepted with a diagnostic rather than rejected
- [ ] README documents the resolution precedence and supported model formats
## Blocked by
- [01-run-one-explicit-candidate-configuration.md](01-run-one-explicit-candidate-configuration.md)
