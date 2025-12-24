"""Parity tests for the in-tree CRINEX 3 decoder vs the upstream
``hatanaka`` Python package reference.

Skipped when either rinexpy_native (for the C++ kernels) or hatanaka
(as the ground-truth reference) is unavailable.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

pytest.importorskip("rinexpy_native")
hatanaka = pytest.importorskip("hatanaka")

from rinexpy.crinex import crx2rnx  # noqa: E402

from .conftest import fixture  # noqa: E402


def test_intree_decoder_matches_hatanaka_on_short_crinex():
    """The 17-minute P43300USA capture (356 KB decoded) round-trips
    byte-for-byte against the hatanaka reference."""
    p = fixture("P43300USA_R_20190012056_17M_15S_MO.crx")
    raw = p.read_text()
    mine = crx2rnx(raw)
    ref = hatanaka.crx2rnx(raw)
    assert mine == ref, (
        f"len mismatch mine={len(mine)} ref={len(ref)}"
        if len(mine) != len(ref)
        else "byte content differs at first diff position"
    )


def test_intree_decoder_matches_hatanaka_on_full_day_gz():
    """The full-day CEBR00ESP capture (~18 MB decoded multi-GNSS) is
    bit-exact against hatanaka end-to-end."""
    p = fixture("CEBR00ESP_R_20182000000_01D_30S_MO.crx.gz")
    with gzip.open(p, "rt") as f:
        raw = f.read()
    mine = crx2rnx(raw)
    ref = hatanaka.crx2rnx(raw)
    assert mine == ref


def test_intree_decoder_handles_rp_load_pipeline():
    """End-to-end: rinexpy.load on a .crx file uses the in-tree
    decoder via _io and produces a valid xarray.Dataset."""
    import rinexpy as rp

    ds = rp.load(fixture("P43300USA_R_20190012056_17M_15S_MO.crx"))
    assert "time" in ds.dims
    assert "sv" in ds.dims
    assert ds.sizes["time"] > 0
    assert ds.sizes["sv"] > 0


def test_intree_decoder_rejects_non_crinex():
    """A plain RINEX 3 file (no CRINEX header) raises ValueError."""
    p = fixture("demo_nav3.17n")
    with pytest.raises(ValueError, match="not a CRINEX"):
        crx2rnx(p.read_text())


def test_intree_decoder_handles_gzip_via_load():
    """rp.load through the gzipped CRINEX file end-to-end."""
    import rinexpy as rp
    ds = rp.load(fixture("CEBR00ESP_R_20182000000_01D_30S_MO.crx.gz"))
    assert ds.sizes["time"] > 0 and ds.sizes["sv"] > 0


@pytest.mark.parametrize("rnx_name", [
    "14601736.18o",
    "default_time_system2.10o",
    "minimal2.10o",
    "badtime.10o",
])
def test_intree_decoder_round_trips_rinex2_via_hatanaka(rnx_name):
    """Take a RINEX 2 OBS fixture, compress it with hatanaka.rnx2crx,
    then decompress it through BOTH the in-tree decoder and
    hatanaka.crx2rnx. The two outputs must be byte-for-byte equal.

    Exercises the CRINEX 1 path end-to-end: pair-level LLI/SSI
    TextDiff, the leading-`&` reinit marker, multi-line per-SV
    output, single global obs list, and the missing-obs-clears-slot
    convention.
    """
    rnx_path = fixture(rnx_name)
    rnx = rnx_path.read_text()
    crx = hatanaka.rnx2crx(rnx)
    mine = crx2rnx(crx)
    ref = hatanaka.crx2rnx(crx)
    assert mine == ref, (
        f"length mine={len(mine)} ref={len(ref)}"
        if len(mine) != len(ref)
        else "byte content differs"
    )


def test_intree_decoder_handles_event_flag_epochs():
    """Real CRINEX 1 files in the wild interleave normal obs epochs
    with event-flag epochs (kinematic-start markers, occupation
    annotations, etc.). The decoder must pass those event-text lines
    through unchanged."""
    rnx_path = fixture("14601736.18o")
    crx = hatanaka.rnx2crx(rnx_path.read_text())
    mine = crx2rnx(crx)
    # Look for the comment text that lives inside an event-flag
    # block in this fixture.
    assert "*** Start of Kinematic Data ***" in mine
    assert "*** Start of Occupation ***" in mine
