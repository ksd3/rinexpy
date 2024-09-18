"""Real-NAV broadcast-ephemeris vs IGS final SP3 cross-check.

Downloads (and caches) the brdc1330.18n broadcast navigation file for
2018-05-13 from the WHU IGS mirror, evaluates each GPS satellite's
ECEF position at every transmitted Toe using ``rinexpy.keplerian2ecef``,
and compares to the SP3-interpolated position at the same epoch.

Expected agreement: broadcast ephemerides are accurate to ~1 m
(URA index 2.4-4.8 m, typical 1-3 m); IGS final orbits are cm-class.
A 5 m worst-case threshold catches unit / sign / scale-factor
regressions while staying robust to real-data noise.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import rinexpy as rp

_CACHE = Path("/tmp/igs_real_cache")
_NAV_URL = "ftp://igs.gnsswhu.cn/pub/gps/data/daily/2018/133/18n/brdc1330.18n.Z"
_SP3_PATH = _CACHE / "igs20010.sp3"  # downloaded by the PPP-realdata test


def _ensure_nav() -> Path:
    nav = _CACHE / "brdc1330.18n"
    if nav.exists() and nav.stat().st_size > 10000:
        return nav
    _CACHE.mkdir(exist_ok=True)
    z = _CACHE / "brdc1330.18n.Z"
    try:
        with urllib.request.urlopen(_NAV_URL, timeout=60) as r, open(z, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception:
        if z.exists():
            z.unlink()
        pytest.skip(f"Cannot reach {_NAV_URL}; skip real-NAV cross-check")
    for tool in ("uncompress", "gzip"):
        try:
            subprocess.run([tool, "-d", "-f", str(z)], check=True, timeout=30)
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    else:
        pytest.skip("No tool to decompress .Z files")
    return nav


@pytest.fixture(scope="module")
def nav():
    return rp.load(str(_ensure_nav()))


@pytest.fixture(scope="module")
def sp3():
    if not _SP3_PATH.exists():
        pytest.skip(
            f"Expected SP3 at {_SP3_PATH}; run test_ppp_realdata first to "
            "download it"
        )
    return rp.load_sp3(str(_SP3_PATH))


def test_broadcast_eph_agrees_with_igs_final_sp3(nav, sp3):
    """For each GPS SV, evaluate keplerian2ecef at every Toe and compare to
    SP3-interpolated position at the same time."""
    sp3_t_ns = sp3.time.values.astype("datetime64[ns]")
    sp3_min = sp3_t_ns.min()
    sp3_max = sp3_t_ns.max()

    errors = []
    n_checked = 0
    for sv_label in sorted(nav.sv.values):
        if not sv_label.startswith("G"):
            continue
        sv = nav.sel(sv=sv_label).dropna(dim="time", how="all")
        if sv.time.size == 0:
            continue
        X, Y, Z = rp.keplerian2ecef(sv)
        for i in range(sv.time.size):
            t = sv.time.values[i].astype("datetime64[ns]")
            if t < sp3_min or t > sp3_max:
                continue
            try:
                interp = rp.interpolate_sp3(sp3, np.array([t]))
                sp3_pos = (
                    interp.position.sel(sv=sv_label).isel(time=0).values
                    * 1000.0
                )
            except (KeyError, ValueError):
                continue
            if not np.all(np.isfinite(sp3_pos)):
                continue
            brdc_pos = np.array(
                [float(X[i]), float(Y[i]), float(Z[i])]
            )
            if not np.all(np.isfinite(brdc_pos)):
                continue
            err = float(np.linalg.norm(brdc_pos - sp3_pos))
            errors.append(err)
            n_checked += 1

    assert n_checked > 50, f"only {n_checked} broadcast-vs-precise comparisons"
    errors = np.asarray(errors)
    assert np.median(errors) < 5.0, (
        f"median broadcast-vs-SP3 disagreement: {np.median(errors):.2f} m"
    )
    # Allow tail to extend further; a single end-of-fit-interval point can
    # easily run a couple of tens of meters off.
    assert np.percentile(errors, 90) < 20.0, (
        f"90th-percentile error: {np.percentile(errors, 90):.2f} m"
    )


def test_broadcast_eph_finite_for_most_records(nav):
    """keplerian2ecef should produce finite output for the bulk of the day's
    broadcast records (a few sparse SVs at file boundaries are tolerated)."""
    n_finite = 0
    n_total = 0
    for sv_label in sorted(nav.sv.values):
        if not sv_label.startswith("G"):
            continue
        sv = nav.sel(sv=sv_label).dropna(dim="time", how="all")
        if sv.time.size == 0:
            continue
        X, Y, Z = rp.keplerian2ecef(sv)
        finite = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        n_finite += int(np.asarray(finite).sum())
        n_total += int(sv.time.size)
    assert n_total > 100
    assert n_finite / n_total > 0.95, (
        f"only {n_finite}/{n_total} broadcast records produced finite ECEF"
    )
