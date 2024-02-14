"""Load a RINEX 3 OBS file and plot the L1 carrier-phase time series.

Run from the repo root:

    uv run python examples/01_basic_load_and_plot.py [--save out.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rinexpy as rp

MAIN_FILE = Path(__file__).resolve().parent.parent / "tests/data/obs3.01gage.10o"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=str(MAIN_FILE))
    parser.add_argument("--save", help="save the plot to PATH instead of showing it")
    ns = parser.parse_args()

    obs = rp.load(ns.file)
    print(f"Loaded {obs.attrs.get('filename', ns.file)}: "
          f"{obs.time.size} epochs, {obs.sv.size} SVs, "
          f"{len(obs.data_vars)} measurement types")
    print("First 3 SVs:", obs.sv.values[:3].tolist())

    # Lazy-import plots so the bare install (no matplotlib) still parses this file.
    import matplotlib

    if ns.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rinexpy.plots import obstimeseries

    obstimeseries(obs)
    if ns.save:
        plt.savefig(ns.save, dpi=120, bbox_inches="tight")
        print(f"Saved {ns.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
