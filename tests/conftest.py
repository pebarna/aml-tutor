"""Shared pytest fixtures for aml-tutor's baked-in lesson tests."""
import re
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_path():
    return FIXTURES_DIR / "paysim_fixture.csv"


@pytest.fixture(scope="session")
def split_step():
    text = (FIXTURES_DIR / "PROVENANCE.md").read_text()
    match = re.search(r"Split-boundary step \(lesson 003\): (\d+)", text)
    if not match:
        raise RuntimeError("Could not find the split-boundary step in PROVENANCE.md")
    return int(match.group(1))
