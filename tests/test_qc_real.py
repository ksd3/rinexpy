"""Integration tests for the QC stack against the bundled RINEX fixtures.

These run the actual readers on shipped files (CEDA Galileo OBS3,
brdc2800.15n GPS NAV) and feed the results to detect_slips, hatch_filter,
multipath_rms, and tgd_from_nav. The asserts are loose sanity checks
that catch unit errors and end-to-end wiring regressions, not bit-level
parity.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

import rinexpy as rp
from rinexpy.positioning import apply_tgd_correction, tgd_from_nav
from rinexpy.qc import (
    detect_slips_geometry_free,
    detect_slips_mw,
    hatch_filter,
    mp1,
    multipath_rms,
)

# Galileo E1 / E5a (CEDA fixture is Galileo-only, so we override the
# default GPS L1/L2 frequencies in detect_slips_mw).
_C = 299_792_458.0
_F_E1 = 1575.42e6
_F_E5A = 1176.45e6
_LAMBDA_E1 = _C / _F_E1
_LAMBDA_E5A = _C / _F_E5A

CEDA = "tests/data/CEDA00USA_R_20182100000_23H_15S_MO.rnx.gz"
NAV15N = "tests/data/brdc2800.15n"


@pytest.fixture(scope="module")
def ceda_obs():
    return rp.load(CEDA)


@pytest.fixture(scope="module")
def brdc_nav():
    return rp.load(NAV15N)


def _longest_contiguous_arc(obs, *vars_, min_len: int = 30):
    """Find the SV + slice with the longest run of all ``vars_`` simultaneously finite.

    Returns (sv_label, start, end) such that obs[v].sel(sv=sv).values[start:end]
    has no NaN gaps in any of ``vars_``. Slip detection on a clean arc is the
    only fair sanity check; NaN gaps look like slips to the first-difference
    detectors (correctly).
    """
    best = None
    best_len = 0
    for sv in obs.sv.values:
        masks = [np.isfinite(obs[v].sel(sv=sv).values) for v in vars_]
        ok = np.all(masks, axis=0)
        if not ok.any():
            continue
        runs = np.diff(np.concatenate([[False], ok, [False]]).astype(int))
        starts = np.flatnonzero(runs == 1)
        ends = np.flatnonzero(runs == -1)
        if len(ends) == 0:
            continue
        lens = ends - starts
        if lens.max() > best_len:
            best_len = int(lens.max())
            i = int(lens.argmax())
            best = (str(sv), int(starts[i]), int(ends[i]))
    if best is None or best_len < min_len:
        pytest.skip(
            f"No SV has a contiguous arc >= {min_len} epochs across {vars_}"
        )
    return best


def test_real_galileo_gf_runs_and_returns_sane_shape(ceda_obs):
    """GF detector runs on real Galileo data and returns a per-epoch bool mask.

    Sets a generous 5 m threshold to absorb the ionospheric variation that's
    real on a 15-s arc; the goal here is end-to-end wiring, not detection
    quality. With the threshold this loose, slips should be a clean minority.
    """
    sv, s, e = _longest_contiguous_arc(ceda_obs, "L1C", "L5Q")
    phi1 = ceda_obs["L1C"].sel(sv=sv).values[s:e]
    phi2 = ceda_obs["L5Q"].sel(sv=sv).values[s:e]
    slips = detect_slips_geometry_free(
        phi1, phi2, lambda1=_LAMBDA_E1, lambda2=_LAMBDA_E5A, threshold_m=5.0
    )
    assert slips.shape == phi1.shape
    assert slips.dtype == bool
    # With a 5 m threshold and a real Galileo arc, slips are rare.
    assert slips.sum() < 5


def test_real_galileo_mw_runs_and_returns_sane_shape(ceda_obs):
    """MW detector runs end-to-end on real Galileo dual-freq data.

    Raw MW on a single-frequency code receiver has a noise floor of several
    cycles RMS in first-difference (the (f1-f2)/(f1+f2) weight on the code
    term is ~0.15 for E1/E5a, with multi-meter code noise underneath). With
    a generous threshold, the detector is quiet enough to be meaningful but
    we don't claim a specific count on this particular fixture.
    """
    sv, s, e = _longest_contiguous_arc(ceda_obs, "L1C", "L5Q", "C1C", "C5Q")
    phi1 = ceda_obs["L1C"].sel(sv=sv).values[s:e]
    phi2 = ceda_obs["L5Q"].sel(sv=sv).values[s:e]
    p1 = ceda_obs["C1C"].sel(sv=sv).values[s:e]
    p2 = ceda_obs["C5Q"].sel(sv=sv).values[s:e]
    slips = detect_slips_mw(
        phi1, phi2, p1, p2, f1=_F_E1, f2=_F_E5A, threshold_cycles=20.0
    )
    assert slips.shape == phi1.shape
    assert slips.dtype == bool
    # At a 20-cycle threshold the detector should be silent on a clean arc.
    assert slips.sum() < 5


def test_real_hatch_filter_smooths_real_pr(ceda_obs):
    """Hatch-filtering on a contiguous arc reduces epoch-to-epoch jitter."""
    sv, s, e = _longest_contiguous_arc(ceda_obs, "L1C", "C1C")
    phi_m = ceda_obs["L1C"].sel(sv=sv).values[s:e] * _LAMBDA_E1
    pr = ceda_obs["C1C"].sel(sv=sv).values[s:e]
    smoothed = hatch_filter(pr, phi_m, window=20)
    # On real data the carrier-vs-code drift is dominated by ionosphere over
    # the arc; we just check that filter output is in-band and doesn't NaN.
    assert np.all(np.isfinite(smoothed))
    # Bias between smoothed and raw stays bounded over the short arc.
    assert np.max(np.abs(smoothed - pr)) < 50.0


def test_real_multipath_first_diff_in_noise_band(ceda_obs):
    """MP1 first-difference RMS on a real arc sits in the typical Galileo
    single-frequency code-noise band (~0.5 - 5 m), independent of the slow
    multipath drift across the arc that the raw RMS would pick up."""
    sv, s, e = _longest_contiguous_arc(ceda_obs, "L1C", "L5Q", "C1C")
    phi1_m = ceda_obs["L1C"].sel(sv=sv).values[s:e] * _LAMBDA_E1
    phi2_m = ceda_obs["L5Q"].sel(sv=sv).values[s:e] * _LAMBDA_E5A
    p1 = ceda_obs["C1C"].sel(sv=sv).values[s:e]
    series = mp1(p1, phi1_m, phi2_m, f1=_F_E1, f2=_F_E5A)
    # The raw multipath_rms picks up the slow drift, which is dominated by
    # the multipath envelope; the first-difference RMS is the per-epoch
    # noise floor.
    diff_rms = float(np.std(np.diff(series)))
    assert 0.5 < diff_rms < 10.0, f"MP1 first-diff RMS {diff_rms:.3f} m off"
    # multipath_rms itself returns a finite value.
    assert np.isfinite(multipath_rms(series))


def test_real_tgd_from_brdc(brdc_nav):
    """TGD values from a real broadcast file are in the expected ns scale and present
    for the GPS constellation."""
    epoch = datetime(2015, 10, 7, 12, 0)
    tgd = tgd_from_nav(brdc_nav, epoch)
    # Most of GPS should have a TGD at this time.
    assert len(tgd) >= 20
    # All values should be ns-scale (|TGD| < 100 ns).
    for sv, t in tgd.items():
        assert -1e-7 < t < 1e-7, f"{sv}: TGD={t:.3e} s out of band"
    # Apply it and confirm the per-SV correction is in the ~1-30 m range.
    sv_labels = list(tgd.keys())
    pr = np.full(len(sv_labels), 2.5e7)
    out = apply_tgd_correction(pr, sv_labels, tgd)
    delta = pr - out
    assert np.all(np.abs(delta) < 50.0), "TGD corrections should be tens of meters at most"
    assert np.any(np.abs(delta) > 0.1), "At least one TGD correction should be non-trivial"


def test_real_tgd_picks_latest_record(brdc_nav):
    """Asking for TGD at a time after the first record returns the latest valid one."""
    # First record at 2015-10-07T00:00:00. Query at 23:00 should pull the
    # latest entry, not the very first.
    early = tgd_from_nav(brdc_nav, datetime(2015, 10, 7, 0, 30))
    late = tgd_from_nav(brdc_nav, datetime(2015, 10, 7, 23, 0))
    # Both should be populated.
    assert early and late
    # If any SV has multiple records in the day, the values should differ
    # between the two queries.
    common = set(early) & set(late)
    differing = [sv for sv in common if early[sv] != late[sv]]
    assert differing, "expected at least one SV with a TGD update during the day"
