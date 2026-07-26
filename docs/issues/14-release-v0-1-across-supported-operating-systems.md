# 14: Release v0.1 Across Supported Operating Systems
**Status:** ready-for-agent
**User stories covered:** 1, 60, 67-68
## What to build
Make the approved end-to-end behavior installable and supportable as v0.1. Prepare the package for PyPI distribution with pipx and uv tool installation. Commit a uv lockfile so that all dependency versions are pinned. Provide `uv run` developer commands for running, testing, linting, type-checking, and packaging. Set up CI on Windows, macOS, and Linux that runs tests, lint, and type checks, and includes a CPU-only smoke path as a release gate. Produce an explicit GPU and backend support matrix that is validated where hardware exists in CI and listed honestly otherwise. Include an opt-in real-hardware smoke suite. Verify that no outbound network access occurs at runtime (outside the optional loopback MTP server). Update the README with complete installation instructions for pipx and uv tool, run, test, troubleshooting, and result interpretation. Verify that the installed CLI and TUI function correctly from a packaged artifact using representative fake-adapter flows and full-contract smoke paths. Standalone binaries remain out of scope for this release.
## Acceptance criteria
- [ ] The package is publishable to PyPI and installable via pipx and uv tool
- [ ] A committed uv lockfile pins all dependency versions
- [ ] `uv run` commands exist for running, testing, linting, type-checking, and packaging
- [ ] CI runs on Windows, macOS, and Linux: tests, lint, type checks, and a CPU-only smoke path
- [ ] The CPU-only smoke path is a release gate
- [ ] A GPU and backend support matrix is documented; entries are validated where CI hardware exists
- [ ] An opt-in real-hardware smoke suite exists
- [ ] No outbound network access occurs at runtime (loopback MTP server is the only network binding)
- [ ] README includes installation, run, test, troubleshooting, and result interpretation sections
- [ ] The installed CLI and TUI are verified from a packaged artifact with representative flows
- [ ] Tests cover: package import, installed entry point, and representative fake-adapter flows against the installed artifact
- [ ] README documents the outbound-network-safety property
## Blocked by
- [13-operate-benchmark-sessions-in-textual-tui.md](13-operate-benchmark-sessions-in-textual-tui.md)
