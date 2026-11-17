"""Tests for version and asset preflight without external services."""

from pathlib import Path

from scripts.release_check import REQUIRED_FILES, validate_release


def make_project(root: Path, version: str = "0.1.0") -> None:
    for filename in REQUIRED_FILES:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "llm-sampling-lab"\n'
        f'version = "{version}"\n'
        'requires-python = ">=3.12"\n',
        encoding="utf-8",
    )
    for filename in (".gitignore", ".dockerignore"):
        (root / filename).write_text(".env\n", encoding="utf-8")


def test_release_preflight_matches_tag(tmp_path: Path) -> None:
    make_project(tmp_path)
    assert validate_release(tmp_path) == []
    assert validate_release(tmp_path, "v0.1.0") == []


def test_release_preflight_rejects_wrong_tag(tmp_path: Path) -> None:
    make_project(tmp_path)
    errors = validate_release(tmp_path, "v0.2.0")
    assert len(errors) == 1
    assert "must match" in errors[0]


def test_release_preflight_rejects_invalid_version(tmp_path: Path) -> None:
    make_project(tmp_path, version="not-a-release")
    assert any(
        "MAJOR.MINOR.PATCH" in error
        for error in validate_release(tmp_path)
    )


def test_release_preflight_detects_missing_entrypoint(tmp_path: Path) -> None:
    make_project(tmp_path)
    (tmp_path / "streamlit_app.py").unlink()
    assert "Missing release file: streamlit_app.py" in validate_release(
        tmp_path
    )


def test_release_preflight_requires_secret_ignores(tmp_path: Path) -> None:
    make_project(tmp_path)
    (tmp_path / ".dockerignore").write_text("results/\n", encoding="utf-8")
    assert any(
        ".dockerignore" in error and ".env" in error
        for error in validate_release(tmp_path)
    )


def test_release_preflight_catches_invalid_pyproject(tmp_path: Path) -> None:
    make_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project\n", encoding="utf-8"
    )
    assert any(
        "Invalid pyproject.toml" in error
        for error in validate_release(tmp_path)
    )
