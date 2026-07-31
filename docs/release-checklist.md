# Release Checklist

The items below describe the steps to publish a new release.
These steps are divided into **administrative settings** (one-time setup) and
**per-release steps** (repeat for every release).

---

## Administrative settings (one-time)

These must be configured by a repository administrator and require
**separate explicit approval** before they are changed. This document does not
change any settings.

- [ ] Create or verify the GitHub `pypi` environment.
- [ ] Require at least one environment reviewer for the `pypi` environment.
- [ ] Set deployment branches to **release tags matching `v*`** on the `pypi`
      environment (release tags, not just `main`).
- [ ] Configure PyPI **trusted publishing** for this project:
      - **Owner / Repository:** `LStoneyy/llama-parameter-scout`
      - **Workflow:** `.github/workflows/publish.yml`
      - **Environment:** `pypi`
      - **Package name:** `llama-parameter-scout`
      - Enable "PEP 740 attestations" (enabled by default via the PyPA action).
- [ ] Protect the `main` branch with pull request reviews and required CI
      checks (quality and smoke jobs).
- [ ] Enable private vulnerability reporting on the repository.
- [ ] Enable secret scanning and push protection where available.

---

## Per-release steps

Follow this order for every release:

1. **Configure Trusted Publisher** (if not already done)
   - Open **Publishing** in the PyPI account sidebar and add a pending publisher.
   - Add a trusted publisher with the details above.

2. **Bump version and update CHANGELOG**
   - Update `version` in `pyproject.toml` to the new version (e.g. `0.2.0`).
   - Add an entry to `CHANGELOG.md` under the new version header.

3. **Merge to `main`**
   - Create a PR with the version bump and CHANGELOG, get reviews, merge.

4. **Create and push an annotated/signed tag**
   ```bash
   git switch main
   git pull --ff-only origin main
   git tag -s v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

5. **Publish on GitHub**
   - Go to GitHub -> Releases -> "Draft a new release"
   - Select the tag created above, set release title, add release notes
   - Click **Publish release**
   - The `publish.yml` workflow triggers automatically on release publish.

6. **Verify after publish**
   - Confirm the package is installable:
     ```bash
     pipx install llama-parameter-scout
     uv tool install llama-parameter-scout
     ```
   - Run `llama-parameter-scout --help` on each supported OS.
   - Check the published package hash and attestation on PyPI:
     - Open the package page on PyPI and inspect the per-file attestation
       link on each wheel/sdist (click the "Verified attestation" badge).
     - Alternatively, verify locally with the `pypi-attestations` tool:
       ```bash
       WHEEL_URL="$(python -c 'import json, urllib.request; data=json.load(urllib.request.urlopen("https://pypi.org/pypi/llama-parameter-scout/0.1.0/json")); print(next(item["url"] for item in data["urls"] if item["packagetype"] == "bdist_wheel"))')"
       uvx --from pypi-attestations pypi-attestations verify pypi \
         --repository https://github.com/LStoneyy/llama-parameter-scout "$WHEEL_URL"
       ```

---

## Verification

After a published release, confirm:

- The package is installable via `pipx install llama-parameter-scout` and
  `uv tool install llama-parameter-scout`.
- `llama-parameter-scout --help` works on each supported OS.
