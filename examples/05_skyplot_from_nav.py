"""Compute a satellite skyplot from a broadcast NAV file.

For each SV in the NAV dataset, evaluate Keplerian -> ECEF, project
to az/el at a hypothetical receiver position, and draw the polar
trajectory.

Run from the repo root:

    uv run python examples/05_skyplot_from_nav.py [--save sky.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import rinexpy as rp
from rinexpy.geodesy import azimuth_elevation, lla_to_ecef

MAIN_FILE = Path(__file__).resolve().parent.parent / "tests/data/brdc2800.15n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=str(MAIN_FILE))
    parser.add_argument("--lat", type=float, default=40.0)
    parser.add_argument("--lon", type=float, default=-3.0)
    parser.add_argument("--alt", type=float, default=100.0)
    parser.add_argument("--save", help="save the plot to PATH")
    ns = parser.parse_args()

    nav = rp.load(ns.file)
    rx = lla_to_ecef(ns.lat, ns.lon, ns.alt)

    sv_az_el: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sv_label in nav.sv.values:
        if sv_label[0] not in {"G", "E"}:
            continue  # skip GLONASS/SBAS (their X/Y/Z are direct, not Keplerian)
        sv = nav.sel(sv=sv_label).dropna(dim="time", how="all")
        if sv.time.size == 0:
            continue
        try:
            X, Y, Z = rp.keplerian2ecef(sv)
        except (ValueError, KeyError):
            continue
        sv_ecef = np.stack([X.values, Y.values, Z.values], axis=-1)
        az, el = azimuth_elevation(rx, sv_ecef)
        sv_az_el[sv_label] = (az, el)

    print(f"Plotted {len(sv_az_el)} satellite trajectories from "
          f"({ns.lat}, {ns.lon})")

    import matplotlib

    if ns.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rinexpy.plots import skyplot

    skyplot(sv_az_el, title=f"Sky from ({ns.lat}, {ns.lon})")
    if ns.save:
        plt.savefig(ns.save, dpi=120, bbox_inches="tight")
        print(f"Saved {ns.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
