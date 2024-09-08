"""End-to-end PPP test on a real IGS day.

Downloads IGS final SP3 + CLK products for GPS week 2001 day 0
(2018-05-13) and runs ``ppp_solve_code_only`` on the bundled
ABMF00GLP RINEX-3 file. Expects the recovered position to be within
a few meters of the marker XYZ encoded in the OBS header.

Downloads are cached under ``/tmp/igs_real_cache/`` between runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
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
    iono_free_pseudorange,
    ppp_solve_code_only,
)

_C = 299_792_458.0
_F_L1 = 1575.42e6
_F_L2 = 1227.60e6

_CACHE = Path("/tmp/igs_real_cache")
# Garner / SOPAC archive serves IGS finals via anonymous FTP.
_SP3_URL = "ftp://garner.ucsd.edu/pub/products/2001/igs20010.sp3.Z"
_CLK_URL = "ftp://garner.ucsd.edu/pub/products/2001/igs20010.clk.Z"
_ABMF_FIXTURE = "tests/data/ABMF00GLP_R_20181330000_01D_30S_MO.zip"


def _download_to(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 100:
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return dest.stat().st_size > 100
    except Exception:
        if dest.exists():
            dest.unlink()
        return False


@pytest.fixture(scope="module")
def igs_products():
    """Download (or reuse cached) IGS SP3+CLK and decompress."""
    _CACHE.mkdir(exist_ok=True)
    sp3_z = _CACHE / "igs20010.sp3.Z"
    clk_z = _CACHE / "igs20010.clk.Z"
    sp3_path = _CACHE / "igs20010.sp3"
    clk_path = _CACHE / "igs20010.clk"

    if not sp3_path.exists():
        if not _download_to(_SP3_URL, sp3_z):
            pytest.skip(f"Cannot reach {_SP3_URL}; skip real-data PPP test")
        # uncompress is the system tool for .Z (LZW); fall back to ncompress.
        for tool in ("uncompress", "gzip"):
            try:
                subprocess.run(
                    [tool, "-d", "-f", str(sp3_z)], check=True, timeout=30
                )
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        else:
            pytest.skip("No tool to decompress .Z files (uncompress/gzip)")
    if not clk_path.exists():
        if not _download_to(_CLK_URL, clk_z):
            pytest.skip(f"Cannot reach {_CLK_URL}; skip real-data PPP test")
        for tool in ("uncompress", "gzip"):
            try:
                subprocess.run(
                    [tool, "-d", "-f", str(clk_z)], check=True, timeout=30
                )
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        else:
            pytest.skip("No tool to decompress .Z files")
    return sp3_path, clk_path


def test_ppp_recovers_abmf_marker_xyz_to_within_a_few_meters(igs_products):
    """Real PPP: actual ABMF observations + real IGS products."""
    sp3_path, clk_path = igs_products

    sp3 = rp.load_sp3(str(sp3_path))
    clk = rp.load_clk(str(clk_path))
    obs = rp.load(_ABMF_FIXTURE, use="G")

    # Marker position from the OBS header, converted from the
    # "APPROX POSITION XYZ" string.
    hdr = rp.rinexheader(_ABMF_FIXTURE)
    xyz_str = hdr["APPROX POSITION XYZ"]
    truth_rx = np.array([float(v) for v in xyz_str.split()])

    # Find an OBS epoch that lands on the SP3 15-minute grid AND the CLK
    # 5-minute grid, so no temporal interpolation is needed. Cast all three
    # to ns precision so datetime64's set lookup works across precisions.
    sp3_times = set(sp3.time.values.astype("datetime64[ns]").tolist())
    clk_times = set(clk.time.values.astype("datetime64[ns]").tolist())
    obs_times = obs.time.values.astype("datetime64[ns]")
    epoch = None
    for t in obs_times:
        t_int = t.tolist()
        if t_int in sp3_times and t_int in clk_times:
            epoch = np.datetime64(t)
            break
    if epoch is None:
        pytest.skip("No OBS epoch aligns with both the SP3 and CLK grids")
    assert str(epoch).startswith("2018-05-13")

    # Pull the GPS SVs that have observations at this epoch on BOTH C1C
    # and C2W (needed to form the ionosphere-free combination).
    p1_all = obs["C1C"].sel(time=epoch)
    p2_all = obs["C2W"].sel(time=epoch)
    gps_svs = [s for s in obs.sv.values if s.startswith("G")]
    kept = []
    p1_kept: list[float] = []
    p2_kept: list[float] = []
    for sv in gps_svs:
        v1 = float(p1_all.sel(sv=sv).values)
        v2 = float(p2_all.sel(sv=sv).values)
        if not (np.isfinite(v1) and np.isfinite(v2)):
            continue
        kept.append(sv)
        p1_kept.append(v1)
        p2_kept.append(v2)
    assert len(kept) >= 4, f"need >= 4 GPS SVs with C1C+C2W; got {len(kept)}"

    # Satellite positions at signal-emission time, with the light-time
    # fixed-point loop and the Earth-rotation correction during light
    # time. Both are mandatory for sub-meter PPP.
    sv_ecef = np.array(
        [
            apply_light_time_and_earth_rotation(sp3, epoch, truth_rx, sv)
            for sv in kept
        ]
    )

    # Precise satellite clocks at the receive epoch from CLK. The variable
    # name in load_clk's output is 'bias', not 'clock'.
    clk_at_epoch = clk["bias"].sel(time=epoch)
    sat_clock_s = np.array(
        [float(clk_at_epoch.sel(sv=sv).values) for sv in kept]
    )

    # Drop any SV whose precise clock isn't available at this epoch.
    keep_mask = np.isfinite(sat_clock_s) & np.all(np.isfinite(sv_ecef), axis=1)
    if keep_mask.sum() < 4:
        pytest.skip(
            f"only {keep_mask.sum()} SVs have full SP3+CLK at the first epoch"
        )
    sv_ecef = sv_ecef[keep_mask]
    sat_clock_s = sat_clock_s[keep_mask]
    p1 = np.asarray(p1_kept, float)[keep_mask]
    p2 = np.asarray(p2_kept, float)[keep_mask]

    # Per-SV slant tropospheric delay via Saastamoinen with the ICAO
    # standard atmosphere. Elevation is computed from the marker XYZ
    # since it's a fixed station; the small bias from using truth_rx
    # rather than a solved-for position is below the noise here.
    lat, _, alt = ecef_to_lla(*truth_rx)
    _, el_deg = azimuth_elevation(truth_rx, sv_ecef)
    tropo_m = np.array([saastamoinen(float(e), alt) for e in el_deg])

    pr_if = iono_free_pseudorange(p1, p2)
    sol = ppp_solve_code_only(
        pr_if,
        sv_ecef,
        sat_clock_s,
        tropospheric_delay_m=tropo_m,
        initial_guess=tuple(truth_rx),
    )

    err = np.linalg.norm(np.array(sol["position"]) - truth_rx)
    # With light-time correction, Earth-rotation correction, and
    # Saastamoinen troposphere applied, the unmodelled residuals are
    # dominated by code multipath, broadcast-vs-IGS clock differences
    # for the few-second offset between OBS time and clock time, and
    # DCBs. We expect a few meters on a clean station.
    assert err < 10.0, (
        f"PPP recovered {sol['position']} vs marker {tuple(truth_rx)}; "
        f"err = {err:.2f} m"
    )
