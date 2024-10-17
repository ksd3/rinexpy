"""Real-data integration test for static-batch carrier-phase PPP.

Loads the bundled ABMF00GLP RINEX-3 OBS and runs
``ppp_solve_static_batch`` against the IGS final SP3 + CLK + Saastamoinen
troposphere products downloaded by ``test_ppp_realdata``. Verifies the
wiring end-to-end and reports the position recovery vs the marker XYZ.

Note: ABMF only has 3 epochs in the bundled fixture, so the carrier-
phase floating ambiguities don't fully converge - the test is a
plumbing check, not an accuracy demonstration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rinexpy as rp
from rinexpy.geodesy import (
    azimuth_elevation,
    ecef_to_lla,
    saastamoinen,
)
from rinexpy.positioning import (
    apply_light_time_and_earth_rotation,
    iono_free_phase,
    iono_free_pseudorange,
    ppp_solve_static_batch,
)

_C = 299_792_458.0
_F_L1 = 1575.42e6
_F_L2 = 1227.60e6
_LAMBDA_L1 = _C / _F_L1
_LAMBDA_L2 = _C / _F_L2

_CACHE = Path("/tmp/igs_real_cache")
_SP3 = _CACHE / "igs20010.sp3"
_CLK = _CACHE / "igs20010.clk"
_ABMF = "tests/data/ABMF00GLP_R_20181330000_01D_30S_MO.zip"


def test_static_batch_runs_on_real_abmf():
    """Real-data integration: load ABMF + IGS products, build the
    iono-free code/phase observables per (epoch, SV), and let
    ppp_solve_static_batch return a finite position."""
    if not _SP3.exists() or not _CLK.exists():
        pytest.skip(
            "Need cached IGS SP3+CLK; run test_ppp_realdata first to download"
        )
    sp3 = rp.load_sp3(str(_SP3))
    clk = rp.load_clk(str(_CLK))
    obs = rp.load(_ABMF, use="G")
    hdr = rp.rinexheader(_ABMF)
    truth_rx = np.array([float(v) for v in hdr["APPROX POSITION XYZ"].split()])

    # Restrict to GPS SVs that have valid C1C/C2W/L1C/L2W at the FIRST
    # epoch (which is the densest one in ABMF). Subsequent epochs can be
    # NaN per-SV; the static-batch solver tolerates missing observations.
    gps_svs = [sv for sv in obs.sv.values if str(sv).startswith("G")]
    kept: list[str] = []
    first_t = obs.time.values[0]
    for sv in gps_svs:
        if all(
            np.isfinite(float(obs[v].sel(time=first_t, sv=sv).values))
            for v in ("C1C", "C2W", "L1C", "L2W")
        ):
            kept.append(sv)
    if len(kept) < 6:
        pytest.skip(f"Need 6+ GPS SVs at the first epoch; got {len(kept)}")

    n_epoch = obs.time.size
    n_sv = len(kept)

    # Align CLK + SP3 to the OBS epochs; the bundle is short enough that
    # SP3's 15-min grid only contains 01:30:00, so 01:30:30 and 01:31:00
    # have to be interpolated. We use apply_light_time_and_earth_rotation
    # to get the SV ECEF at signal emission for each (epoch, SV), then
    # interpolate the CLK to each receive time.
    sv_ecef = np.empty((n_epoch, n_sv, 3))
    sat_clock = np.empty((n_epoch, n_sv))
    pr_if = np.empty((n_epoch, n_sv))
    phase_if = np.empty((n_epoch, n_sv))
    tropo = np.empty((n_epoch, n_sv))

    lat, _, alt = ecef_to_lla(*truth_rx)
    for ki, t in enumerate(obs.time.values):
        epoch_ns = np.datetime64(t, "ns")
        for ji, sv in enumerate(kept):
            sv_ecef[ki, ji] = apply_light_time_and_earth_rotation(
                sp3, epoch_ns, truth_rx, sv
            )
            sat_clock[ki, ji] = float(
                rp.interpolate_clk(clk, sv, t.astype("datetime64[us]").tolist())
            )
        p1 = np.array([float(obs["C1C"].sel(time=t, sv=sv).values) for sv in kept])
        p2 = np.array([float(obs["C2W"].sel(time=t, sv=sv).values) for sv in kept])
        l1 = np.array([float(obs["L1C"].sel(time=t, sv=sv).values) for sv in kept]) * _LAMBDA_L1
        l2 = np.array([float(obs["L2W"].sel(time=t, sv=sv).values) for sv in kept]) * _LAMBDA_L2
        pr_if[ki] = iono_free_pseudorange(p1, p2)
        phase_if[ki] = iono_free_phase(l1, l2)
        _, el = azimuth_elevation(truth_rx, sv_ecef[ki])
        tropo[ki] = np.array([saastamoinen(float(e), alt) for e in el])

    sol = ppp_solve_static_batch(
        pr_if, phase_if, sv_ecef, sat_clock,
        tropo=tropo,
        initial_position=tuple(truth_rx),
        sigma_code=1.0,
        sigma_phase=0.005,
        max_iter=20,
    )

    err = np.linalg.norm(np.array(sol["position"]) - truth_rx)
    # With only 3 epochs of ABMF data spanning 60 s, the carrier-phase
    # ambiguities don't separate from the position. The recovered
    # position can be several hundred meters off; that's the expected
    # behaviour of a static-batch PPP without phase wind-up, antenna
    # PCV, or DCB corrections AND without the geometry diversity that
    # a multi-hour session would provide. The threshold below catches
    # gross wiring regressions (the synthetic-data tests in
    # test_ppp_static.py catch accuracy).
    assert err < 500.0, (
        f"static-batch PPP position: {sol['position']} vs marker "
        f"{tuple(truth_rx)}; err = {err:.3f} m"
    )
    assert sol["ambiguities_m"].shape == (n_sv,)
    assert sol["clock_per_epoch"].shape == (n_epoch,)
    assert np.all(np.isfinite(sol["ambiguities_m"]))
    # The recovered ambiguities for iono-free combinations are O(meters)
    # to O(hundreds of meters) depending on the integer Ns.
    assert np.max(np.abs(sol["ambiguities_m"])) < 1.0e6
