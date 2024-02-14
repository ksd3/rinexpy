"""Stream a RINEX 3 OBS file one epoch at a time without loading all of it.

For multi-GB observation files, ``rinexpy.iter_obs3_epochs`` yields
one ``(datetime, xarray.Dataset)`` per epoch. Each yielded dataset
holds the SVs present at *that* epoch only, so the memory footprint
is constant in the file size.

Run from the repo root:

    uv run python examples/03_streaming_huge_file.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rinexpy as rp

MAIN_FILE = Path(__file__).resolve().parent.parent / "tests/data/ABMF00GLP_R_20181330000_01D_30S_MO.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=str(MAIN_FILE))
    parser.add_argument("--use", help="restrict to these GNSS systems (e.g. G,E)")
    ns = parser.parse_args()

    use = set(ns.use.split(",")) if ns.use else None
    n_epochs = 0
    n_sv_total = 0
    for t, ds in rp.iter_obs3_epochs(ns.file, use=use):
        n_epochs += 1
        n_sv_total += ds.sv.size
        if n_epochs <= 3:
            print(f"epoch {n_epochs:3d}  {t}  {ds.sv.size:3d} SVs  "
                  f"vars: {list(ds.data_vars)[:4]}")
    print(f"Total: {n_epochs} epochs, average {n_sv_total / n_epochs:.1f} SVs/epoch")


if __name__ == "__main__":
    main()
