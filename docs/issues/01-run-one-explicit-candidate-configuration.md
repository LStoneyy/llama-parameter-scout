# 01: Run One Explicit Candidate Configuration
**Status:** ready-for-agent
**User stories covered:** 3, 13, 31, 44
## What to build
Accept an exact llama-bench executable path, one local single-file GGUF, an explicit requested total n_ctx composition, and one explicit Candidate Configuration from a non-interactive CLI. Execute the candidate through llama-bench using machine-readable output, perform a warmup phase and at least three completed measurements, produce a single Benchmark Observation, atomically journal it, and return a clear terminal result. Establish the deep Application seam and the production/fake process adapter seam only as needed for this end-to-end behavior. Include uv project setup, uv-managed run and test workflows, and an initial README with the workflow shown. Do not claim PATH discovery, shard support, adaptive search, complete reports, or TUI. Fail cleanly with clear error messages on missing inputs, nonzero exit from the child binary, or malformed machine-readable output. The acceptance criteria must include an end-to-end test that drives the Application seam with a fake process adapter and an opt-in real-binary smoke path.
## Acceptance criteria
- [ ] A noninteractive CLI invocation accepting an exact binary path, model path, requested n_ctx composition, and explicit Candidate Configuration completes and reports a single Benchmark Observation on the terminal
- [ ] The output shows warmup phase indication, at least three completed measurements, median, and dispersion
- [ ] The Benchmark Observation is atomically written to the Session Journal
- [ ] Missing executable path produces a clear error before any process launch
- [ ] Missing model path produces a clear error before any process launch
- [ ] Invalid n_ctx composition produces a clear error before any process launch
- [ ] Nonzero exit from llama-bench produces a clean error with the captured stderr
- [ ] Malformed machine-readable output produces a clean error with the captured raw output
- [ ] An end-to-end test injects a fake process adapter and verifies the full Application seam produces a correctly structured Benchmark Observation
- [ ] An opt-in real-binary smoke path exists and is documented in the README
- [ ] README documents the uv project setup, the single-run workflow, and how to run tests
## Blocked by
None - can start immediately
