"""Tests for the rinexpy.ppp.ppp_solve(obs, sp3, clk) driver.

There are no public PPP-ready obs+SP3+CLK fixtures shipped in
tests/data, so these tests build matched synthetic datasets from a
known truth station and verify the driver:

- Picks an obs-code quadruple correctly.
- Drives a StaticPPPFilter that converges towards the truth position.
- Returns the documented result-dict shape.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from rinexpy.geodesy import ecef_to_lla, lla_to_ecef, saastamoinen
from rinexpy.multifreq import LAMBDA_L1, LAMBDA_L2
from rinexpy.ppp import ppp_solve


def _elev_deg(rx, sv):
    lat, lon, _ = ecef_to_lla(*rx)
    lr, gr = np.deg2rad(lat), np.deg2rad(lon)
    sl, cl = np.sin(lr), np.cos(lr)
    sg, cg = np.sin(gr), np.cos(gr)
    Rmat = np.array([
        [-sg, cg, 0.0],
        [-sl * cg, -sl * sg, cl],
        [cl * cg, cl * sg, sl],
    ])
    enu = (sv - rx) @ Rmat.T
    horiz = np.linalg.norm(enu[:, :2], axis=1)
    return np.degrees(np.arctan2(enu[:, 2], horiz))

_C = 299_792_458.0


def _synth_session(
    truth_lla=(40.0, -3.0, 100.0),
    n_epochs: int = 120,
    dt_s: float = 30.0,
    n_sv: int = 8,
    sigma_phase_m: float = 0.003,
    sigma_code_m: float = 0.3,
    seed: int = 0,
):
    """Build matched obs/SP3/CLK xarray Datasets from a known truth."""
    rng = np.random.default_rng(seed)
    truth = np.array(lla_to_ecef(*truth_lla))

    start = datetime(2024, 1, 1, 0, 0, 0)
    obs_times = np.array(
        [start + timedelta(seconds=k * dt_s) for k in range(n_epochs)],
        dtype="datetime64[ns]",
    )
    sp3_step = 900.0
    n_sp3 = int(np.ceil(n_epochs * dt_s / sp3_step)) + 4
    sp3_times = np.array(
        [start + timedelta(seconds=k * sp3_step - 2 * sp3_step) for k in range(n_sp3)],
        dtype="datetime64[ns]",
    )
    sv_labels = [f"G{i + 1:02d}" for i in range(n_sv)]

    geom_rng = np.random.default_rng(42)
    az = geom_rng.uniform(0, 2 * np.pi, size=n_sv)
    el = geom_rng.uniform(np.deg2rad(20), np.deg2rad(75), size=n_sv)
    R = 2.0e7
    topo = np.column_stack([
        R * np.cos(el) * np.sin(az),
        R * np.cos(el) * np.cos(az),
        R * np.sin(el),
    ])
    base_sv_ecef = truth + topo

    # Real GPS SVs move at ~4 km/s. Without realistic motion the filter
    # cannot disambiguate receiver position from receiver clock - the
    # two are perfectly correlated when geometry is static.
    sv_velocity = geom_rng.uniform(-3000.0, 3000.0, size=(n_sv, 3))
    sp3_pos = np.zeros((n_sp3, n_sv, 3))
    for k in range(n_sp3):
        t_s = k * sp3_step - 2 * sp3_step
        sp3_pos[k] = base_sv_ecef + sv_velocity * t_s

    sv_clocks = geom_rng.uniform(-1e-5, 1e-5, size=n_sv)
    clk_step = 30.0
    n_clk = int(np.ceil(n_epochs * dt_s / clk_step)) + 4
    clk_times = np.array(
        [start + timedelta(seconds=k * clk_step - 2 * clk_step) for k in range(n_clk)],
        dtype="datetime64[ns]",
    )
    clk_bias = np.tile(sv_clocks, (n_clk, 1))

    rx_clock = 1.234e-7
    sp3_t_s = ((sp3_times - sp3_times[0]).astype("timedelta64[s]")).astype(float)
    obs_t_s = ((obs_times - sp3_times[0]).astype("timedelta64[s]")).astype(float)

    c1 = np.full((n_epochs, n_sv), np.nan)
    c2 = np.full((n_epochs, n_sv), np.nan)
    l1 = np.full((n_epochs, n_sv), np.nan)
    l2 = np.full((n_epochs, n_sv), np.nan)
    n1_int = geom_rng.integers(-1000, 1000, size=n_sv)
    n2_int = geom_rng.integers(-1000, 1000, size=n_sv)
    truth_alt = truth_lla[2]
    for k in range(n_epochs):
        before = int(np.searchsorted(sp3_t_s, obs_t_s[k]) - 1)
        before = max(0, min(before, n_sp3 - 2))
        span = sp3_t_s[before + 1] - sp3_t_s[before]
        w = (obs_t_s[k] - sp3_t_s[before]) / span if span > 0 else 0.0
        sv_now = sp3_pos[before] * (1 - w) + sp3_pos[before + 1] * w
        rho = np.linalg.norm(sv_now - truth, axis=1)
        # Inject the same Saastamoinen tropo the solver will subtract.
        elev = _elev_deg(truth, sv_now)
        tropo = np.array([saastamoinen(float(e), truth_alt) if e > 0 else 0.0
                          for e in elev])
        pr_clean = rho + _C * (rx_clock - sv_clocks) + tropo
        c1[k] = pr_clean + rng.normal(scale=sigma_code_m, size=n_sv)
        c2[k] = pr_clean + rng.normal(scale=sigma_code_m, size=n_sv)
        l1[k] = (pr_clean / LAMBDA_L1 + n1_int) + rng.normal(
            scale=sigma_phase_m / LAMBDA_L1, size=n_sv,
        )
        l2[k] = (pr_clean / LAMBDA_L2 + n2_int) + rng.normal(
            scale=sigma_phase_m / LAMBDA_L2, size=n_sv,
        )

    obs = xr.Dataset(
        {
            "C1C": (("time", "sv"), c1),
            "C2W": (("time", "sv"), c2),
            "L1C": (("time", "sv"), l1),
            "L2W": (("time", "sv"), l2),
        },
        coords={"time": obs_times, "sv": sv_labels},
        attrs={"position": tuple(truth)},
    )
    sp3 = xr.Dataset(
        {"position": (("time", "sv", "ECEF"), sp3_pos)},
        coords={"time": sp3_times, "sv": sv_labels, "ECEF": ["x", "y", "z"]},
    )
    clk = xr.Dataset(
        {"bias": (("time", "sv"), clk_bias)},
        coords={"time": clk_times, "sv": sv_labels},
    )
    return obs, sp3, clk, truth


def test_ppp_solve_recovers_synthetic_station():
    """ROADMAP acceptance: ppp_solve(obs, sp3, clk) recovers a fixed
    station to centimetre-level on a multi-epoch synthetic session."""
    obs, sp3, clk, truth = _synth_session()
    result = ppp_solve(
        obs, sp3, clk,
        initial_position_ecef=tuple(truth + np.array([20.0, -10.0, 5.0])),
        elevation_mask_deg=5.0,
    )
    err = np.linalg.norm(np.array(result["position"]) - truth)
    assert err < 0.10, f"PPP recovered {err:.4f} m (expected < 10 cm)"
    assert result["n_epochs"] == 120
    assert result["obs_codes"] == ("C1C", "C2W", "L1C", "L2W")


def test_ppp_solve_returns_documented_shape():
    obs, sp3, clk, truth = _synth_session(n_epochs=20)
    out = ppp_solve(obs, sp3, clk, initial_position_ecef=tuple(truth))
    assert set(out) >= {
        "position", "lla", "clock_bias_s", "position_sigma_m",
        "n_epochs", "trace", "obs_codes", "filter",
    }
    assert len(out["position"]) == 3
    assert len(out["lla"]) == 3
    assert len(out["position_sigma_m"]) == 3
    assert isinstance(out["trace"], list)


def test_ppp_solve_missing_obs_codes_raises():
    obs, sp3, clk, _ = _synth_session(n_epochs=5)
    bare = obs.drop_vars(list(obs.data_vars))
    with pytest.raises(ValueError):
        ppp_solve(bare, sp3, clk)


def test_ppp_solve_picks_first_available_quadruple():
    obs, sp3, clk, truth = _synth_session(n_epochs=10)
    # Rename so C1C/L1C aren't available - the priority list should
    # fall through to the (C1W, C2W, L1W, L2W) quadruple.
    obs2 = obs.rename({"C1C": "C1W", "L1C": "L1W"})
    out = ppp_solve(obs2, sp3, clk, initial_position_ecef=tuple(truth))
    assert out["obs_codes"] == ("C1W", "C2W", "L1W", "L2W")
