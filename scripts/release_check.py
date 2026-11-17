"""Read-only project and release-tag validation; uses Python's standard library.

Run locally with:
    uv run python scripts/release_check.py
    uv run python scripts/release_check.py --tag v0.1.0
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

REQUIRED_FILES = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "Dockerfile",
    "docker-compose.yaml",
    "streamlit_app.py",
    ".streamlit/config.toml",
)


def validate_release(root: Path, tag: str | None = None) -> list[str]:
    """Return validation errors rather than modifying the repository."""

    errors: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (root / relative_path).is_file():
            errors.append(f"Missing release file: {relative_path}")

    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return errors

    try:
        config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"Invalid pyproject.toml: {type(exc).__name__}")
        return errors

    project = config.get("project")
    if not isinstance(project, dict):
        errors.append("Missing [project] metadata.")
        return errors

    if project.get("name") != "llm-sampling-lab":
        errors.append("Unexpected package name.")

    version = project.get("version")
    if not isinstance(version, str) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+", version
    ):
        errors.append("Version must follow MAJOR.MINOR.PATCH.")
    elif tag is not None and tag != f"v{version}":
        errors.append(
            f"Release tag {tag!r} must match pyproject version v{version}."
        )

    if project.get("requires-python") != ">=3.12":
        errors.append("Project must retain the supported Python >=3.12 baseline.")

    for filename in (".gitignore", ".dockerignore"):
        file_path = root / filename
        if not file_path.is_file():
            errors.append(f"Missing ignore file: {filename}")
            continue
        entries = {
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if ".env" not in entries:
            errors.append(f"{filename} must exclude local .env secrets.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate LLM Sampling Lab release prerequisites."
    )
    parser.add_argument("--tag", help="Optional Git tag, e.g. v0.1.0")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    errors = validate_release(root, tag=args.tag)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Release preflight: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
