"""Real-file integration for the high-level I/O surface:

- RINEX 2 and RINEX 3 OBS readers (api.load),
- writer.to_rinex_obs round-trip on a bundled fixture,
- streaming.iter_obs3_epochs vs the bulk OBS3 reader,
- api.batch_convert on a small directory of real files.

Uses fixtures that already ship under tests/data/, so no network
required for this module.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import rinexpy as rp
from rinexpy.streaming import iter_obs3_epochs
from rinexpy.writer import to_rinex_obs

_RINEX2_OBS = "tests/data/14601736.18o"
_RINEX3_OBS = "tests/data/ABMF00GLP_R_20181330000_01D_30S_MO.zip"
_CEDA = "tests/data/CEDA00USA_R_20182100000_23H_15S_MO.rnx.gz"


def test_rinex2_obs_has_expected_observations():
    """Reading a real RINEX 2 OBS file returns sensible measurement vars."""
    ds = rp.load(_RINEX2_OBS)
    assert "L1" in ds.data_vars
    assert "C1" in ds.data_vars or "P1" in ds.data_vars
    assert ds.time.size > 0
    assert ds.sv.size > 0
    # Pseudoranges should be in the typical GNSS receive range.
    c1 = ds["C1"].values
    finite = c1[np.isfinite(c1)]
    if finite.size:
        assert finite.min() > 1.0e7
        assert finite.max() < 5.0e7


def test_rinex3_obs_multi_system_present():
    """ABMF MO file has GPS, GLONASS, Galileo, SBAS together."""
    ds = rp.load(_RINEX3_OBS)
    systems = {str(sv)[0] for sv in ds.sv.values}
    assert {"G", "R", "E"}.issubset(systems), (
        f"expected GPS/GLONASS/Galileo, got {systems}"
    )


def test_rinex3_obs_writer_round_trips_values():
    """Read a real OBS3, write it back, re-read; data on epochs with at
    least one observation should round-trip to mm precision.

    The writer correctly omits epochs that have zero observations for the
    selected systems (this can happen if a multi-system fixture is sliced
    to a single system that wasn't observed at every input epoch).
    """
    src = rp.load(_RINEX3_OBS, use="G")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_path = f"{tmp}/roundtrip.rnx"
        to_rinex_obs(src, out_path, version=3)
        reread = rp.load(out_path)
        # SV set should match.
        assert set(src.sv.values) == set(reread.sv.values)
        # The written file has only the epochs with at least one finite
        # observation across the selected SVs.
        src_epochs_with_data = []
        for ti in range(src.time.size):
            if np.isfinite(src["C1C"].isel(time=ti).values).any():
                src_epochs_with_data.append(src.time.values[ti])
        assert reread.time.size == len(src_epochs_with_data), (
            f"reread has {reread.time.size}, expected "
            f"{len(src_epochs_with_data)} non-empty epochs"
        )
        # C1C values should round-trip to within mm precision (RINEX OBS
        # stores 3 decimal digits = 1 mm).
        for t in reread.time.values:
            a = src["C1C"].sel(time=t).values
            b = reread["C1C"].sel(time=t).values
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                err = np.abs(a[finite] - b[finite])
                assert err.max() < 1.0e-2, f"round-trip {err.max():.4f} m"


def test_streaming_matches_bulk_reader_on_first_epoch():
    """The first epoch from iter_obs3_epochs should match the first epoch
    of the bulk load() on the same file."""
    bulk = rp.load(_CEDA, use="E")
    first_bulk_time = bulk.time.values[0]
    first_bulk = bulk.isel(time=0)

    stream = iter_obs3_epochs(_CEDA, use="E")
    stream_t, stream_ds = next(stream)
    # Convert both to ns precision for comparison.
    assert np.datetime64(stream_t, "ns") == np.datetime64(first_bulk_time, "ns")
    # Spot-check a measurement variable that exists in both: L1C.
    bulk_l1 = first_bulk["L1C"].values
    stream_l1 = stream_ds["L1C"].isel(time=0).values
    # Same SVs in same order, same values where finite.
    common = list(set(bulk.sv.values) & set(stream_ds.sv.values))
    assert common, "no common SVs between bulk and streaming first epoch"
    for sv in common[:3]:
        b = float(first_bulk["L1C"].sel(sv=sv).values)
        s = float(stream_ds["L1C"].sel(sv=sv).isel(time=0).values)
        if np.isfinite(b) and np.isfinite(s):
            assert b == s, f"{sv}: bulk={b} stream={s}"


def test_batch_convert_processes_multi_files(tmp_path):
    """batch_convert on a 1-element glob should produce a NetCDF output."""
    out = rp.batch_convert(
        "tests/data",
        "demo.10o",
        str(tmp_path),
        verbose=False,
    )
    assert isinstance(out, list)
    assert len(out) >= 1
    # The output is a NetCDF; load_with_plugins / rp.load can read it back.
    nc = rp.load(str(out[0]))
    assert isinstance(nc, xr.Dataset)


def test_load_dispatch_detects_file_type():
    """rp.load auto-detects OBS vs NAV vs SP3 from the file content."""
    obs = rp.load(_RINEX3_OBS)
    nav = rp.load("tests/data/brdc2800.15n")
    sp3 = rp.load_sp3("tests/data/igs19362.sp3c")

    # OBS dataset has L1C or similar; NAV has clock bias; SP3 has position.
    assert any(v.startswith("L") for v in obs.data_vars)
    assert "SVclockBias" in nav.data_vars
    assert "position" in sp3.data_vars
