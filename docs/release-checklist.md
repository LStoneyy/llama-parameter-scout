# Release Checklist

The items below describe GitHub repository settings and PyPI trusted-publisher
configuration required for a release. These settings must be configured by a
repository administrator and require **separate explicit approval** before they
are changed. This document does not change any settings.

## Administrative settings

- [ ] Create or verify the GitHub `pypi` environment.
- [ ] Require at least one environment reviewer for the `pypi` environment.
- [ ] Restrict deployment branches to `main` on the `pypi` environment.
- [ ] Restrict the PyPI trusted publisher to this repository,
      `.github/workflows/publish.yml`, and the `pypi` environment.
- [ ] Protect the `main` branch with pull request reviews and required CI checks
      (quality and smoke jobs).
- [ ] Enable private vulnerability reporting on the repository.
- [ ] Enable secret scanning and push protection where available.

## Publishing

Once the administrative settings above are in place, publishing is triggered
automatically by either:

- Creating a GitHub release in the web UI, or
- Manually dispatching the `Publish to PyPI` workflow from the `main` branch.

The workflow verifies that the selected commit is reachable from `origin/main`
before building and publishing.

## Verification

After a published release, confirm:

- The package is installable via `pipx install llama-parameter-scout` and
  `uv tool install llama-parameter-scout`.
- `llama-parameter-scout --help` works on each supported OS.
