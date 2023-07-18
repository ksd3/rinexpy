"""Shared pytest fixtures for the rinexpy test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to the bundled test corpus (copied from georinex)."""
    return DATA


def fixture(name: str) -> Path:
    """Return the path to a single named fixture file."""
    p = DATA / name
    if not p.exists():
        pytest.skip(f"missing fixture {name!r}")
    return p
