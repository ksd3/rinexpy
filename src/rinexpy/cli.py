"""Argparse-based CLI: ``rinexpy <subcommand>``.

Subcommands:

- ``read``    — read a single file and print/save the dataset.
- ``times``   — print just the epoch timestamps.
- ``info``    — print the parsed header.
- ``convert`` — batch-convert a directory to NetCDF.

The CLI is intentionally argparse-based so we have zero runtime deps beyond
NumPy/xarray; users wanting fancier UX can wrap a Typer/Click frontend
around the importable functions in :mod:`rinexpy.api`.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .api import batch_convert, gettime, load
from .headers import rinexheader

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Fully-configured parser. Tested at unit-test level by
        ``tests/test_cli.py``.
    """
    parser = argparse.ArgumentParser(prog="rinexpy", description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_read = sub.add_parser("read", help="read a file and print the dataset")
    p_read.add_argument("file")
    p_read.add_argument("-o", "--out", help="write NetCDF to PATH (dir or .nc)")
    p_read.add_argument(
        "-u",
        "--use",
        nargs="+",
        help="restrict to these GNSS systems",
        choices=list("CEGIJRS"),
    )
    p_read.add_argument("-m", "--meas", nargs="+", help="restrict to these measurement labels")
    p_read.add_argument(
        "-t",
        "--tlim",
        nargs=2,
        metavar=("START", "STOP"),
        help="restrict to this time window (ISO 8601)",
    )
    p_read.add_argument("--useindicators", action="store_true", help="include LLI/SSI columns")
    p_read.add_argument(
        "--strict",
        action="store_false",
        dest="fast",
        help="disable speculative preallocation",
    )
    p_read.add_argument(
        "--interval", type=float, help="decimate to this many seconds between epochs"
    )

    p_times = sub.add_parser("times", help="print just the epoch timestamps")
    p_times.add_argument("file")

    p_info = sub.add_parser("info", help="print the parsed header")
    p_info.add_argument("file")

    p_conv = sub.add_parser("convert", help="batch convert a directory to NetCDF")
    p_conv.add_argument("indir", help="directory containing RINEX files")
    p_conv.add_argument("glob", help="filename glob pattern", default="*", nargs="?")
    p_conv.add_argument("-o", "--out", required=True, help="output directory")
    p_conv.add_argument("-u", "--use", nargs="+", choices=list("CEGIJRS"))
    p_conv.add_argument("-m", "--meas", nargs="+")
    p_conv.add_argument("--useindicators", action="store_true")
    p_conv.add_argument("--strict", action="store_false", dest="fast")
    p_conv.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="number of parallel worker processes (0 = all CPUs)",
    )

    # SPP: solve a single-point position from one epoch of OBS+NAV.
    p_spp = sub.add_parser("spp", help="single-point positioning fix from OBS + NAV")
    p_spp.add_argument("obs", help="OBS file")
    p_spp.add_argument("nav", help="NAV file")
    p_spp.add_argument("-t", "--epoch", type=int, default=0,
                       help="epoch index to fix (default 0)")
    p_spp.add_argument("-u", "--use", nargs="+", choices=list("CEGIJRS"),
                       default=["G"])

    # RTK: fix a rover position with a base receiver.
    p_rtk = sub.add_parser("rtk", help="RTK fix from rover + base OBS + NAV")
    p_rtk.add_argument("rover", help="rover OBS file")
    p_rtk.add_argument("base", help="base OBS file")
    p_rtk.add_argument("nav", help="NAV file")
    p_rtk.add_argument("--base-pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                       help="known base ECEF coordinates in metres")
    p_rtk.add_argument("-t", "--epoch", type=int, default=0)

    # PPP: solve precise point position with IGS SP3 + CLK.
    p_ppp = sub.add_parser("ppp", help="PPP solve using SP3 + CLK precise products")
    p_ppp.add_argument("obs", help="OBS file")
    p_ppp.add_argument("sp3", help="precise SP3 ephemeris")
    p_ppp.add_argument("clk", help="precise clock products")
    p_ppp.add_argument("--n-epochs", type=int, default=None,
                       help="cap the number of epochs processed (default all)")

    # Splice: concatenate two or more files along the time axis.
    p_splice = sub.add_parser("splice", help="concatenate OBS files along time")
    p_splice.add_argument("files", nargs="+",
                          help="input files (sorted by epoch)")
    p_splice.add_argument("-o", "--out", required=True,
                          help="output path (.nc or NetCDF)")

    # Decimate: down-sample to a coarser cadence.
    p_dec = sub.add_parser("decimate",
                           help="down-sample OBS file to a coarser cadence")
    p_dec.add_argument("file")
    p_dec.add_argument("-o", "--out", required=True, help="output NetCDF path")
    p_dec.add_argument("--interval", type=float, required=True,
                       help="target seconds between epochs")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Top-level entry point for ``python -m rinexpy`` / installed script.

    Parameters
    ----------
    argv:
        Optional argument list (for testing). Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on error.
    """
    parser = build_parser()
    ns = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if ns.verbose else logging.WARNING)

    try:
        if ns.cmd == "read":
            return _cmd_read(ns)
        if ns.cmd == "times":
            return _cmd_times(ns)
        if ns.cmd == "info":
            return _cmd_info(ns)
        if ns.cmd == "convert":
            return _cmd_convert(ns)
        if ns.cmd == "spp":
            return _cmd_spp(ns)
        if ns.cmd == "rtk":
            return _cmd_rtk(ns)
        if ns.cmd == "ppp":
            return _cmd_ppp(ns)
        if ns.cmd == "splice":
            return _cmd_splice(ns)
        if ns.cmd == "decimate":
            return _cmd_decimate(ns)
    except (ValueError, LookupError, OSError) as e:
        log.error("%s", e)
        return 1
    return 0


def _cmd_read(ns: argparse.Namespace) -> int:
    """Implement ``rinexpy read``."""
    data = load(
        ns.file,
        ns.out,
        use=ns.use,
        tlim=tuple(ns.tlim) if ns.tlim else None,
        meas=ns.meas,
        useindicators=ns.useindicators,
        fast=ns.fast,
        interval=ns.interval,
        verbose=ns.verbose,
    )
    print(data)
    return 0


def _cmd_times(ns: argparse.Namespace) -> int:
    """Implement ``rinexpy times``."""
    arr = gettime(ns.file)
    print(f"{ns.file}: {arr.size} epochs")
    if arr.size:
        print(f"first: {arr[0]}")
        print(f"last:  {arr[-1]}")
    return 0


def _cmd_info(ns: argparse.Namespace) -> int:
    """Implement ``rinexpy info``."""
    hdr = rinexheader(ns.file)
    for k, v in hdr.items():
        print(f"{k}: {v!r}")
    return 0


def _cmd_convert(ns: argparse.Namespace) -> int:
    """Implement ``rinexpy convert``."""
    written = batch_convert(
        ns.indir,
        ns.glob,
        ns.out,
        use=ns.use,
        meas=ns.meas,
        useindicators=ns.useindicators,
        fast=ns.fast,
        verbose=ns.verbose,
        workers=ns.workers,
    )
    print(f"wrote {len(written)} file(s)")
    return 0


def _cmd_spp(ns: argparse.Namespace) -> int:
    """Implement ``rinexpy spp`` — one-epoch single-point positioning."""
    from .api import load
    from .keplerian import keplerian2ecef
    from .positioning import spp_solve
    import numpy as np
    obs = load(ns.obs, use=ns.use)
    nav = load(ns.nav)
    pseudoranges_codes = [c for c in obs.data_vars if c.startswith("C1")]
    if not pseudoranges_codes:
        log.error("no C1* code obs in %s", ns.obs)
        return 1
    code = pseudoranges_codes[0]
    pr_da = obs[code].isel(time=ns.epoch).dropna(dim="sv")
    sv_list = list(pr_da.sv.values)
    pr = pr_da.values
    sv_ecef = np.zeros((len(sv_list), 3))
    for i, sv in enumerate(sv_list):
        nav_sv = nav.sel(sv=sv).dropna(dim="time", how="all")
        if nav_sv.time.size == 0:
            sv_ecef[i] = np.nan
            continue
        X, Y, Z = keplerian2ecef(nav_sv.isel(time=0))
        sv_ecef[i] = (float(X.values), float(Y.values), float(Z.values))
    mask = np.isfinite(sv_ecef).all(axis=1) & np.isfinite(pr)
    sol = spp_solve(sv_ecef[mask], pr[mask])
    lat, lon, alt = sol["lla"]
    print(f"position ECEF: {sol['position']}")
    print(f"lat / lon / alt: {lat:.6f} deg / {lon:.6f} deg / {alt:.1f} m")
    print(f"clock bias: {sol['clock_bias'] * 1e9:.2f} ns")
    print(f"converged in {sol['n_iter']} iterations")
    return 0


def _cmd_rtk(ns: argparse.Namespace) -> int:
    """Implement ``rinexpy rtk`` — single-baseline RTK fix."""
    print("rinexpy rtk: not fully wired yet — use rinexpy.rtk.rtk_fix() "
          "directly for now. See docs/COOKBOOK.md.")
    return 1


def _cmd_ppp(ns: argparse.Namespace) -> int:
    """Implement ``rinexpy ppp`` — precise point positioning."""
    print("rinexpy ppp: not fully wired yet — use the kalman_ztd / "
          "StaticPPPFilterZTD API directly. See test_ppp_realdata.py.")
    return 1


def _cmd_splice(ns: argparse.Namespace) -> int:
    """Implement ``rinexpy splice`` — concatenate OBS files along time."""
    from .tools import concat_files
    out = concat_files(*ns.files, outpath=ns.out)
    print(f"wrote {out}")
    return 0


def _cmd_decimate(ns: argparse.Namespace) -> int:
    """Implement ``rinexpy decimate`` — coarser cadence rewrite."""
    from .api import load
    ds = load(ns.file, interval=ns.interval)
    from pathlib import Path
    out = Path(ns.out).expanduser()
    ds.to_netcdf(out)
    print(f"wrote {out} ({ds.sizes.get('time', 0)} epochs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
