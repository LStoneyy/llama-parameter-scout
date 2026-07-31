# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately through GitHub's
"Report a vulnerability" feature on the repository Security tab. Do not open
a public issue for an unpatched vulnerability.

Include reproduction steps, affected versions, and the expected impact when
possible. You should receive an initial response within seven days.

## Supported Versions

Security fixes are provided for the latest released version.

## Trust Boundaries

llama-parameter-scout does not bundle `llama-bench` or `llama-server`.
Resolution differs by binary:

- **llama-bench** is resolved via `--llama-bench` (explicit path), then
  `--install-dir` (directory search), then `PATH` lookup.
- **llama-server** is supplied as an explicit path in the TUI; it is not
  resolved through `--install-dir` or `PATH`.

Operating-system helpers (`pbcopy`, `wl-copy`, `xclip`, `xsel`, `clip`,
`nvidia-smi`, `rocm-smi`) are invoked by bare name and resolved through
`PATH` at runtime with no explicit-path override.

When running in an environment with an untrusted `PATH`, provide an explicit
path to `--llama-bench` and to `llama-server` in the TUI, and ensure `PATH`
is trusted for the OS helpers listed above.
