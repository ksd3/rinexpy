"""Cross-check rinexpy against an installed georinex on the shared corpus.

These tests are skipped if ``georinex`` is not importable. Where the two
implementations are known to diverge (specifically, NAV3 spare-field and
trailing-FitIntvl handling), the diverging columns are excluded from the
per-variable comparison and the rationale is documented in
``SCRATCHPAD.md`` / ``docs/OPTIMIZATIONS.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import rinexpy as rp

from .conftest import fixture

georinex = pytest.importorskip("georinex")

pytestmark = pytest.mark.parity


def _equal_within_tol(a, b, *, tol: float = 1e-9) -> bool:
    """Compare two xarray.DataArray values, NaN-safe and float-tolerant."""
    av = np.asarray(a.values)
    bv = np.asarray(b.values)
    if av.shape != bv.shape:
        return False
    if av.dtype.kind in "Mm" or bv.dtype.kind in "Mm":
        return bool((av == bv).all())
    if av.dtype.kind in "OUS" or bv.dtype.kind in "OUS":
        return bool((av == bv).all())
    return bool(np.allclose(av, bv, atol=tol, rtol=tol, equal_nan=True))


@pytest.mark.parametrize(
    "fname",
    ["demo.10o", "minimal2.10o", "ab430140.18o.zip", "rinex2onesat.10o"],
)
def test_obs2_parity(fname):
    fn = fixture(fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        truth = georinex.load(fn)
        mine = rp.load(fn)
    for v in truth.data_vars:
        assert v in mine, f"{fname}: {v} missing"
        assert _equal_within_tol(truth[v], mine[v]), f"{fname}: {v} differs"


@pytest.mark.parametrize(
    "fname",
    ["obs3.01gage.10o", "ABMF00GLP_R_20181330000_01D_30S_MO.zip"],
)
def test_obs3_parity(fname):
    fn = fixture(fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        truth = georinex.load(fn)
        mine = rp.load(fn)
    for v in truth.data_vars:
        assert v in mine, f"{fname}: {v} missing"
        assert _equal_within_tol(truth[v], mine[v]), f"{fname}: {v} differs"


@pytest.mark.parametrize(
    "fname",
    ["demo.10n", "ab422100.18n", "ceda2100.18e", "brdc2800.15n"],
)
def test_nav2_parity(fname):
    fn = fixture(fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        truth = georinex.load(fn)
        mine = rp.load(fn)
    for v in truth.data_vars:
        assert v in mine, f"{fname}: {v} missing"
        assert _equal_within_tol(truth[v], mine[v]), f"{fname}: {v} differs"


@pytest.mark.parametrize(
    "fname",
    ["igs19362.sp3c", "minimal.sp3d"],
)
def test_sp3_parity(fname):
    """Position/clock parity vs georinex.

    Velocity and dclock are excluded: georinex leaves those buffers
    uninitialised (``np.empty``, no fill) on the first epoch when no V
    record is present, so the 'value' is whatever junk happens to be on
    the heap. rinexpy fills with NaN, the only sane interpretation when
    V records are absent.

    ``example1.sp3a`` is intentionally not in this parametrize set: it
    has more SV header entries than actual P records, so georinex's
    position array trails junk memory for the unseen SVs in the same
    way. rinexpy emits NaN for those slots.
    """
    fn = fixture(fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        truth = georinex.load(fn)
        mine = rp.load(fn)
    for v in truth.data_vars:
        if v in {"velocity", "dclock"}:
            continue
        assert v in mine, f"{fname}: {v} missing"
        assert _equal_within_tol(truth[v], mine[v]), f"{fname}: {v} differs"


# NOTE: example1.sp3a is intentionally NOT in the parity suite: georinex's
# np.empty position buffer means the comparison is nondeterministic between
# runs (the values in unwritten cells depend on whatever else allocated heap
# memory before the read). rinexpy's behaviour for that file is verified by
# tests/test_sp3.py::test_sp3a, which only asserts intrinsic invariants.


def test_gettime_parity():
    fn = fixture("ab422100.18n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        truth = georinex.gettime(fn)
        mine = rp.gettime(fn)
    np.testing.assert_array_equal(np.asarray(truth), np.asarray(mine))
