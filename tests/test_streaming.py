"""Tests for the per-epoch streaming iterator."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from pytest import approx

import rinexpy as rp
from rinexpy.streaming import iter_obs3_epochs

from .conftest import fixture


def test_iter_yields_one_per_epoch():
    """Number of yields equals number of epochs in the file."""
    fn = fixture("obs3.01gage.10o")
    epochs = list(iter_obs3_epochs(fn))
    bulk = rp.load(fn)
    assert len(epochs) == bulk.time.size


def test_iter_yields_correct_times():
    """Yielded datetime matches the bulk-loaded time coord."""
    fn = fixture("obs3.01gage.10o")
    yielded_times = [t for t, _ in iter_obs3_epochs(fn)]
    bulk = rp.load(fn)
    bulk_times = rp.to_datetime(bulk.time).tolist()
    if not isinstance(bulk_times, list):
        bulk_times = [bulk_times]
    assert yielded_times == bulk_times


def test_iter_per_epoch_values_match_bulk():
    """One SV's per-epoch value (via streaming) matches the bulk dataset."""
    fn = fixture("obs3.01gage.10o")
    bulk = rp.load(fn)
    bulk_g07 = bulk.C1C.sel(sv="G07").values

    streamed: list[float] = []
    for _, ds in iter_obs3_epochs(fn):
        if "G07" in ds.sv.values:
            # ds has a length-1 time axis; squeeze to scalar.
            streamed.append(float(ds.C1C.sel(sv="G07").item()))
        else:
            streamed.append(float("nan"))

    np.testing.assert_allclose(streamed, bulk_g07, equal_nan=True)


def test_iter_use_filter_drops_other_systems():
    """``use={'G'}`` only yields GPS SVs."""
    fn = fixture("obs3.01gage.10o")
    for _, ds in iter_obs3_epochs(fn, use={"G"}):
        for sv in ds.sv.values:
            assert sv.startswith("G")


def test_iter_tlim_skips_before_and_breaks_after():
    """``tlim`` cuts off before-start and after-stop epochs."""
    fn = fixture("obs3.01gage.10o")
    bulk = rp.load(fn)
    times = rp.to_datetime(bulk.time)
    if not isinstance(times, list):
        times = [times] if isinstance(times, datetime) else times.tolist()
    if len(times) < 2:
        pytest.skip("fixture has too few epochs to slice")
    second = times[1]
    out = list(iter_obs3_epochs(fn, tlim=(second, second)))
    assert len(out) == 1
    assert out[0][0] == second


def test_iter_zip_input():
    """Streaming through a zip-compressed input produces non-empty SV sets."""
    fn = fixture("ABMF00GLP_R_20181330000_01D_30S_MO.zip")
    epochs = list(iter_obs3_epochs(fn))
    assert len(epochs) >= 1
    assert all(ds.sv.size > 0 for _, ds in epochs)


def test_iter_value_matches_bulk_g07():
    """Cross-check one expected G07 value via streaming."""
    fn = fixture("obs3.01gage.10o")
    streamed_g07_first: float | None = None
    for _, ds in iter_obs3_epochs(fn):
        if "G07" in ds.sv.values:
            streamed_g07_first = float(ds.C1C.sel(sv="G07").item())
            break
    assert streamed_g07_first == approx(22227666.76)
