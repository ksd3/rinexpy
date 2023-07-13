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
from pathlib import Path

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
        "-u", "--use", nargs="+", help="restrict to these GNSS systems",
        choices=list("CEGIJRS"),
    )
    p_read.add_argument(
        "-m", "--meas", nargs="+", help="restrict to these measurement labels"
    )
    p_read.add_argument(
        "-t", "--tlim", nargs=2, metavar=("START", "STOP"),
        help="restrict to this time window (ISO 8601)",
    )
    p_read.add_argument(
        "--useindicators", action="store_true", help="include LLI/SSI columns"
    )
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
    )
    print(f"wrote {len(written)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
