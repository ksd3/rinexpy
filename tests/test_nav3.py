"""Tests for the RINEX-3 NAV reader."""

from __future__ import annotations

from datetime import datetime

import pytest
import xarray as xr
from pytest import approx

import rinexpy as rp

from .conftest import fixture


def test_nav3_header():
    hdr = rp.rinexheader(fixture("demo_nav3.17n"))
    assert hdr["IONOSPHERIC CORR"]["GPSA"] == approx(
        [1.1176e-08, -1.4901e-08, -5.9605e-08, 1.1921e-07]
    )
    assert hdr["TIME SYSTEM CORR"]["GPUT"] == approx(
        [-3.7252902985e-09, -1.065814104e-14, 61440, 1976]
    )


def test_nav3_time():
    times = rp.gettime(fixture("VILL00ESP_R_20181700000_01D_MN.rnx.gz"))
    assert times[0] == datetime(2018, 4, 24, 8)
    assert times[-1] == datetime(2018, 6, 20, 22)


def test_nav3_tlim_past_eof():
    nav = rp.load(
        fixture("CEDA00USA_R_20182100000_01D_MN.rnx.gz"),
        tlim=("2018-07-29T23", "2018-07-29T23:30"),
    )
    times = rp.to_datetime(nav.time)
    assert times == datetime(2018, 7, 29, 23)


def test_nav3_spare_filled():
    nav = rp.load(fixture("spare_filled_nav3.rnx"))
    g01 = nav.sel(sv="G01").dropna(dim="time", how="all")
    assert g01.time.size == 2
    assert (g01["FitIntvl"] == approx([4.0, 4.0])).all()


def test_nav3_mixed_systems():
    nav = rp.load(
        fixture("ELKO00USA_R_20182100000_01D_MN.rnx.gz"),
        tlim=(datetime(2018, 7, 28, 21), datetime(2018, 7, 28, 23)),
    )
    e04 = nav.sel(sv="E04").dropna(dim="time", how="all")
    e04_dup = nav.sel(sv="E04_1").dropna(dim="time", how="all")
    assert e04["TransTime"].values.tolist() != e04_dup["TransTime"].values.tolist()
    assert isinstance(nav, xr.Dataset)
    assert set(nav.svtype) == {"C", "E", "G", "R"}
    times = rp.to_datetime(nav.time)
    assert times.size == 15


def test_nav3_mixed_full():
    nav = rp.load(fixture("ELKO00USA_R_20182100000_01D_MN.rnx.gz"))
    expected_subset = {"C06", "E04", "G01", "R01"}
    assert expected_subset.issubset(set(nav.sv.values))


def test_nav3_ionospheric_corr_gps():
    nav = rp.load(fixture("demo_nav3.17n"))
    assert nav.attrs["ionospheric_corr_GPS"] == approx(
        [
            1.1176e-08,
            -1.4901e-08,
            -5.9605e-08,
            1.1921e-07,
            9.8304e04,
            -1.1469e05,
            -1.9661e05,
            7.2090e05,
        ]
    )


def test_nav3_ionospheric_corr_gal():
    nav = rp.load(fixture("galileo3.15n"))
    assert nav.attrs["ionospheric_corr_GAL"] == approx([0.1248e03, 0.5039, 0.2377e-01])


def test_nav3_missing_fields_zero():
    """Empty cells within a present record should be 0, not NaN.

    rinexpy diverges intentionally on truncated records: trailing fields
    that were never emitted by the receiver default to 0 for required
    fields and stay NaN for the purely-decorative ``spare*`` slots, so
    this assertion only counts non-spare columns.
    """
    nav = rp.load(fixture("BRDC00IGS_R_20201360000_01D_MN.rnx"), use="E")
    df = nav.to_dataframe()
    non_spare = [c for c in df.columns if not c.startswith("spare")]
    assert df[non_spare].isna().sum().sum() == 0


def test_nav3_missing_trailing_field():
    nav = rp.load(fixture("BRDM00DLR_R_20130010000_01D_MN.rnx"), use="J")
    assert nav.to_dataframe()["FitIntvl"].to_list() == [0.0, 0.0]


@pytest.mark.parametrize(
    "rfn, ncfn",
    [
        ("demo_nav3.17n", "demo_nav3.17n.nc"),
        ("demo_nav3.10n", "demo_nav3.10n.nc"),
        ("qzss_nav3.14n", "qzss_nav3.14n.nc"),
    ],
)
def test_nav3_vs_reference_nc(rfn, ncfn):
    """Compare to upstream-georinex-emitted reference NetCDF.

    The reference files were emitted by an older georinex that left fields
    not present in a record as NaN; rinexpy fills required fields with 0
    per the RINEX 3.04 spec. We therefore skip ``spare*`` and ``FitIntvl``
    in the per-variable check.
    """
    pytest.importorskip("netCDF4")
    truth = rp.load(fixture(ncfn))
    nav = rp.load(fixture(rfn))
    skip = {"FitIntvl"}
    for v in nav.data_vars:
        if v.startswith("spare") or v in skip:
            continue
        assert truth[v].equals(nav[v]), f"mismatch on {v}"
