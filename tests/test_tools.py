"""Tests for validate_file, concat_files, diff_datasets."""

from __future__ import annotations

import pytest

import rinexpy as rp
from rinexpy.tools import concat_files, diff_datasets, validate_file

from .conftest import fixture


def test_validate_obs():
    rep = validate_file(fixture("demo.10o"))
    assert rep["ok"] is True
    assert rep["n_epochs"] == 2
    assert rep["info"]["rinextype"] == "obs"


def test_validate_missing_file():
    rep = validate_file("/no/such/file.10o")
    assert rep["ok"] is False
    assert any("failed" in w for w in rep["warnings"])


def test_validate_reports_interval():
    rep = validate_file(fixture("demo.10o"))
    assert rep["interval_seconds"] == 30.0


def test_diff_equal():
    a = rp.load(fixture("demo.10o"))
    b = rp.load(fixture("demo.10o"))
    out = diff_datasets(a, b)
    assert out["equal"] is True
    assert out["differences"] == []


def test_diff_finds_mismatch():
    a = rp.load(fixture("demo.10o"))
    b = a.copy(deep=True)
    # Re-assign the whole C1 array to make sure the mutation sticks.
    new_vals = b["C1"].values.copy()
    # Pick the first non-NaN cell to bump.
    import numpy as np

    nonnan = np.argwhere(np.isfinite(new_vals))
    assert nonnan.size, "fixture has no finite C1 cells"
    i, j = nonnan[0]
    new_vals[i, j] += 100.0
    b = b.assign({"C1": (b["C1"].dims, new_vals)})
    out = diff_datasets(a, b)
    assert out["equal"] is False
    assert any(d["var"] == "C1" for d in out["differences"])


def test_concat_two_copies(tmp_path):
    """Concatenating a file with itself dedupes back to the original."""
    a = rp.load(fixture("demo.10o"))
    combined = concat_files([fixture("demo.10o"), fixture("demo.10o")])
    assert combined.time.size == a.time.size  # dedup


def test_concat_no_files_raises():
    with pytest.raises(ValueError):
        concat_files([])
