"""RTK baseline solution with LAMBDA integer ambiguity fixing.

Build noise-free dual-receiver synthetic data with known integer
ambiguities, run the joint baseline+ambiguity LSQ, fix the integers
via LAMBDA, and re-solve the baseline.

Run from the repo root:

    uv run python examples/07_rtk_baseline.py
"""

from __future__ import annotations

import numpy as np

import rinexpy as rp
from rinexpy.geodesy import lla_to_ecef
from rinexpy.multifreq import LAMBDA_L1
from rinexpy.rtk import rtk_fix


def main() -> None:
    rng = np.random.default_rng(2026)
    base = np.array(lla_to_ecef(40, -3, 0))
    truth_baseline = np.array([5.4, -2.1, 0.7])  # 5.4 m east, 2.1 m south, 0.7 m up
    rover = base + truth_baseline

    # 6 satellites at GPS altitude, well-spread in az AND el so the
    # geometry matrix is well-conditioned.
    sv = []
    az_el = [(10, 70), (70, 30), (130, 55), (190, 20), (250, 50), (310, 40)]
    sv_radius = 2.66e7
    lat = np.radians(40.0)
    lon = np.radians(-3.0)
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    for az, el in az_el:
        a = np.radians(az)
        elev_rad = np.radians(el)
        e = np.cos(elev_rad) * np.sin(a)
        n = np.cos(elev_rad) * np.cos(a)
        u = np.sin(elev_rad)
        x = -sl * e - sp * cl * n + cp * cl * u
        y = cl * e - sp * sl * n + cp * sl * u
        z = cp * n + sp * u
        sv.append(base + sv_radius * np.array([x, y, z]))
    sv = np.array(sv)

    # Synthetic pseudorange + phase observations.
    rho_r = np.linalg.norm(sv - rover, axis=1)
    rho_b = np.linalg.norm(sv - base, axis=1)
    true_amb = rng.integers(-200, 200, size=sv.shape[0])
    pr_r = rho_r
    pr_b = rho_b
    phase_r = rho_r / LAMBDA_L1 + true_amb
    phase_b = rho_b / LAMBDA_L1 + true_amb

    sol = rtk_fix(
        pr_r, pr_b, phase_r, phase_b, sv, tuple(base),
        wavelength=LAMBDA_L1,
        sigma_pr=1.0,
        sigma_phase=0.005,
    )
    print(f"Truth baseline:    ({truth_baseline[0]:.4f}, "
          f"{truth_baseline[1]:.4f}, {truth_baseline[2]:.4f}) m")
    print(f"Float baseline:    ({sol['float']['baseline'][0]:.4f}, "
          f"{sol['float']['baseline'][1]:.4f}, "
          f"{sol['float']['baseline'][2]:.4f}) m")
    if sol["fixed_accepted"]:
        bx, by, bz = sol["fixed"]["baseline"]
        print(f"Fixed baseline:    ({bx:.4f}, {by:.4f}, {bz:.4f}) m")
        print(f"LAMBDA ratio test: {sol['lambda']['ratio']:.2f}  -> ACCEPTED")
    else:
        print(f"LAMBDA ratio test: {sol['lambda']['ratio']:.2f}  -> rejected")

    _ = rp.__version__  # keep the rinexpy import live


if __name__ == "__main__":
    main()
