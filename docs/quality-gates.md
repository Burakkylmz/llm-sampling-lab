# Quality Gates and Release Readiness

The repository uses four GitHub Actions workflows. A green local test run does
not replace GitHub CI; CI must pass on the actual proposed pull-request commit.

## 1. Python Quality

`.github/workflows/python-quality.yml` runs on pull requests, pushes to
`main`, and manual dispatch.

- Tests the **locked** `uv.lock` environment on Linux and Windows (Python 3.12).
- Runs the full pytest suite, syntax compilation, and read-only release preflight.
- Runs a **scoped** Ruff lint check and strict mypy analysis for the new release
  validator. The existing project source has **not** been declared fully lint-
  clean or mypy-clean. Extending the static-analysis scope is future work.

Local equivalent:

```powershell
uv sync --locked --group dev --python 3.12
uv run --no-sync pytest -q
uv run --no-sync python -m compileall -q src scripts streamlit_app.py
uv run --no-sync python scripts/release_check.py
uvx ruff@0.17.5 check --select E4,E7,E9,F scripts/release_check.py tests/test_release_check.py
uvx mypy@2.4.1 --strict scripts/release_check.py
```

## 2. Docker Smoke

`.github/workflows/docker-smoke.yml` builds the UI image, starts the
container using its automatically allocated loopback host port, calls
`/_stcore/health`, verifies `streamlit_app.py` is available in the running
container, and checks the packaged CLI's `--help` output.

The workflow does **not** make live Groq/OpenAI calls or require Ollama models.
The CI container is cleaned up even after failure. On Windows, the local helper
`scripts/start-ui.ps1` performs a comparable Streamlit HTTP-readiness check.

```powershell
docker compose config --quiet
.\scripts\start-ui.ps1
docker compose ps
```

## 3. Dependency Review

`.github/workflows/dependency-review.yml` checks **new dependencies introduced
in a pull request** for high-severity or critical known vulnerabilities.
GitHub Dependency Graph/Dependency Review must be available and enabled.
This is not a substitute for periodic full-environment vulnerability scanning,
secret scanning, pinning workflow actions to immutable SHAs, or code review.

Workflow permissions are scoped to read-only repository contents. Pull request
workflows do not need provider API secrets.

## 4. Manual Release Candidate

`.github/workflows/release-candidate.yml` can be launched manually once a
real Git tag exists. It checks out that tag, verifies that the tag equals the
version in `pyproject.toml` (for example `v0.1.0`), runs pytest, builds the
source and wheel distributions using `uv build`, validates their package
metadata using Twine, and uploads short-lived workflow artifacts.

This workflow **does not publish to PyPI or create a GitHub Release**.

Before tagging, decide on the new semantic version, update `pyproject.toml`,
refresh the dependency lock if required, merge passing CI changes, and create
an immutable annotated tag on the approved commit. Avoid rewriting published
release tags.

## Recommended branch protection

After the new workflows have run successfully on the upstream repository,
enable branch protection/rulesets for `main`: require pull requests, enforce
successful Python and Docker checks, and review changes before merging. Exact
required-check names should come from the *actual* successful CI run. Do not
select checks that do not exist.

## Known limitations

- CI is new and must be validated on GitHub-hosted runners.
- Ruff and mypy checks are deliberately scoped; the whole codebase is not yet
  statically verified.
- Docker smoke verifies the UI startup and CLI wiring, not model quality.
- Existing reported performance differences may be affected by provider load,
  hardware contention, sample size, and incomplete runtime metadata.
