"""Baked-in check for lesson 001 — project setup.

Run via `uv run pytest ../aml-tutor/tests/001_test_project_setup.py` with
cwd=aml-triage, so every path below is relative to the student's project root
and `import aml_triage` resolves to their package.
"""

from pathlib import Path


def project_root() -> Path:
    return Path.cwd()


def test_pyproject_and_lock_exist():
    root = project_root()
    assert (root / "pyproject.toml").is_file(), "pyproject.toml is missing — run `uv init` first."
    assert (root / "uv.lock").is_file(), (
        "uv.lock is missing — run `uv add pandas xgboost scikit-learn` and "
        "`uv add --dev pytest` (or `uv sync`) first."
    )


def test_package_layout_exists():
    root = project_root()
    assert (root / "src" / "aml_triage" / "__init__.py").is_file(), (
        "src/aml_triage/__init__.py is missing."
    )
    assert (root / "tests").is_dir(), "tests/ directory is missing."


def test_pyproject_declares_required_dependencies():
    pyproject = (project_root() / "pyproject.toml").read_text()
    for dependency in ("pandas", "xgboost", "scikit-learn", "pytest"):
        assert dependency in pyproject, (
            f"pyproject.toml does not declare a dependency on {dependency}."
        )


def test_package_imports():
    import aml_triage  # noqa: F401
