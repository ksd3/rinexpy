"""Single-point positioning (SPP) from synthetic GPS observations.

In a real workflow the satellite ECEF positions come from interpolating
SP3 (or evaluating Keplerian elements from NAV) at the signal-emission
time, and the pseudoranges come from the OBS file. Here we build a
noise-free synthetic so the example runs against a known truth.

Run from the repo root:

    uv run python examples/06_spp_positioning.py
"""

from __future__ import annotations

import numpy as np

import rinexpy as rp
from rinexpy.geodesy import lla_to_ecef

_C = 299_792_458.0


def main() -> None:
    truth_lat, truth_lon, truth_alt = 40.0, -3.0, 100.0
    truth_rx = np.array(lla_to_ecef(truth_lat, truth_lon, truth_alt))

    # Six well-spread satellites at ~26000 km from Earth's centre.
    # Vary the elevations as well as the azimuths so the geometry
    # matrix is well-conditioned.
    sv_radius = 2.66e7
    sv = []
    az_el = [(0, 70), (60, 30), (120, 50), (200, 20), (260, 60), (320, 40)]
    for az, el in az_el:
        a = np.radians(az)
        elev_rad = np.radians(el)
        e = np.cos(elev_rad) * np.sin(a)
        n = np.cos(elev_rad) * np.cos(a)
        u = np.sin(elev_rad)
        # Build ECEF unit-vector via the receiver's ECEF rotation.
        lat = np.radians(truth_lat)
        lon = np.radians(truth_lon)
        sl, cl = np.sin(lon), np.cos(lon)
        sp, cp = np.sin(lat), np.cos(lat)
        x = -sl * e - sp * cl * n + cp * cl * u
        y = cl * e - sp * sl * n + cp * sl * u
        z = cp * n + sp * u
        sv.append(truth_rx + sv_radius * np.array([x, y, z]))
    sv = np.array(sv)

    bias_s = 1e-4  # 100 microseconds
    rho = np.linalg.norm(sv - truth_rx, axis=1)
    pr = rho + _C * bias_s

    # SPP is a linearised LSQ; the (0, 0, 0) initial guess at Earth's
    # centre converges in 5-6 iterations for any receiver on the planet.
    # Bump max_iter to be safe.
    sol = rp.spp_solve(sv, pr, max_iter=20)
    print(f"Truth ECEF:    ({truth_rx[0]:.3f}, {truth_rx[1]:.3f}, {truth_rx[2]:.3f}) m")
    print(f"Solved ECEF:   ({sol['position'][0]:.3f}, "
          f"{sol['position'][1]:.3f}, {sol['position'][2]:.3f}) m")
    print(f"Truth bias:    {bias_s * 1e6:.3f} us")
    print(f"Solved bias:   {sol['clock_bias'] * 1e6:.3f} us")
    print(f"Iterations:    {sol['n_iter']}")
    lat, lon, alt = sol["lla"]
    print(f"Solved LLA:    ({lat:.6f}, {lon:.6f}, {alt:.3f} m)")


if __name__ == "__main__":
    main()
