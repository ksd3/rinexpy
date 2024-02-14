"""Load an SP3 ephemeris and interpolate satellite positions to arbitrary times.

SP3 files publish positions every 15 minutes; a typical PPP/SPP
workflow needs them at every observation epoch (often 1 Hz). The
``interpolate_sp3`` Lagrange interpolator is the IGS-recommended
default with order 10.

Run from the repo root:

    uv run python examples/04_sp3_interpolation.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

import rinexpy as rp

MAIN_FILE = Path(__file__).resolve().parent.parent / "tests/data/igs19362.sp3c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=str(MAIN_FILE))
    parser.add_argument("--sv", default="G05", help="satellite to highlight")
    ns = parser.parse_args()

    sp3 = rp.load_sp3(ns.file)
    print(f"Loaded {sp3.sv.size} SVs over {sp3.time.size} epochs "
          f"({sp3.time.values[0]} -> {sp3.time.values[-1]})")

    # Interpolate at one second past the first epoch — well inside the
    # file's time range.
    t0 = sp3.time.values[5].astype("datetime64[us]").astype(datetime)
    queries = np.array(
        [t0, t0 + np.timedelta64(60, "s"), t0 + np.timedelta64(300, "s")],
        dtype="datetime64[ns]",
    )
    interp = rp.interpolate_sp3(sp3, queries)
    sv_pos = interp.position.sel(sv=ns.sv).values  # (3 query times, 3 ECEF)
    for q, p in zip(queries, sv_pos):
        print(f"  {q}  {ns.sv} ECEF (km): "
              f"({p[0]:10.3f}, {p[1]:10.3f}, {p[2]:10.3f})")


if __name__ == "__main__":
    main()
