"""Convert every RINEX file in a directory to NetCDF in parallel.

Run from the repo root:

    uv run python examples/02_batch_convert.py [-j 4]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import warnings
from pathlib import Path

import rinexpy as rp

DATA_DIR = Path(__file__).resolve().parent.parent / "tests/data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("indir", nargs="?", default=str(DATA_DIR),
                        help="directory containing RINEX files")
    parser.add_argument("--glob", default="*.10o", help="filename glob")
    parser.add_argument("-j", "--workers", type=int, default=2,
                        help="number of parallel processes (0 = all CPUs)")
    parser.add_argument("--out", help="output directory (defaults to a temp dir)")
    ns = parser.parse_args()

    out_dir = Path(ns.out) if ns.out else Path(tempfile.mkdtemp(prefix="rinexpy_out_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        written = rp.batch_convert(
            ns.indir,
            ns.glob,
            out_dir,
            workers=ns.workers,
        )
    print(f"Converted {len(written)} file(s) to {out_dir}/")
    for p in written[:5]:
        print(f"  {p.name:60s}  {p.stat().st_size / 1024:6.1f} KB")

    if not ns.out:
        # Tidy up the auto-generated temp dir if the user didn't ask
        # for a specific path.
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
