"""Public, format-agnostic entry points: ``load``, ``rinexnav``, ``rinexobs``.

These are the functions most users will call. They auto-detect the file type
and dispatch to the right version-specific reader, mirroring
``georinex.load`` / ``georinex.rinexnav`` / ``georinex.rinexobs``.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import xarray as xr

from ._io import opener  # noqa: F401  (re-exported for advanced users)
from ._time import normalize_tlim
from ._types import FileLike, MeasSelection, SystemSelection, TimeLimit
from .headers import rinexinfo
from .nav2 import rinexnav2
from .nav3 import rinexnav3
from .netcdf import write_dataset
from .obs2 import rinexobs2
from .obs3 import rinexobs3
from .sp3 import load_sp3

log = logging.getLogger(__name__)


def load(
    rinexfn: FileLike,
    out: Path | str | None = None,
    *,
    use: SystemSelection = None,
    tlim: TimeLimit = None,
    useindicators: bool = False,
    meas: MeasSelection = None,
    verbose: bool = False,
    overwrite: bool = False,
    fast: bool = True,
    interval: float | int | timedelta | None = None,
):
    """Auto-detect the type of ``rinexfn`` and dispatch to the right reader.

    Supports RINEX 2/3 NAV and OBS, SP3-a/c/d, and pre-converted ``.nc``
    files. For ``.nc`` files containing both ``NAV`` and ``OBS`` groups, a
    ``{"nav": ..., "obs": ...}`` dict is returned.

    Parameters
    ----------
    rinexfn:
        Path or open text stream of any supported file.
    out:
        Optional output. May be a directory (the basename + ``.nc`` is used)
        or a ``.nc`` file path.
    use, tlim, useindicators, meas, verbose, fast, interval:
        Forwarded to the underlying readers; see their docstrings.
    overwrite:
        If True, allow ``out`` to be overwritten.

    Returns
    -------
    xarray.Dataset | dict
        The parsed data (or a NAV+OBS dict for ``.nc`` files containing both).

    Raises
    ------
    ValueError
        For unrecognized file types or invalid arguments.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    if isinstance(rinexfn, (str, Path)):
        rinexfn = Path(rinexfn).expanduser()

    outfn = _resolve_outfn(rinexfn, out)
    tlim = normalize_tlim(tlim)

    info = rinexinfo(rinexfn)
    rinex_type = info["rinextype"]

    if rinex_type == "nav":
        return rinexnav(rinexfn, outfn, use=use, tlim=tlim, overwrite=overwrite)
    if rinex_type == "obs":
        return rinexobs(
            rinexfn,
            outfn,
            use=use,
            tlim=tlim,
            useindicators=useindicators,
            meas=meas,
            verbose=verbose,
            overwrite=overwrite,
            fast=fast,
            interval=interval,
        )
    if rinex_type == "sp3":
        assert isinstance(rinexfn, Path)
        return load_sp3(rinexfn, outfn)

    if isinstance(rinexfn, Path) and rinexfn.suffix == ".nc":
        return _load_nc(rinexfn)

    raise ValueError(f"unknown RINEX/SP3 file: {rinexfn}")


def rinexnav(
    fn: FileLike,
    outfn: Path | None = None,
    *,
    use: SystemSelection = None,
    group: str = "NAV",
    tlim: TimeLimit = None,
    overwrite: bool = False,
) -> xr.Dataset:
    """Read any RINEX-2 or RINEX-3 NAV file (or open a ``.nc`` group).

    Parameters
    ----------
    fn:
        Path or stream of a NAV file or NetCDF file.
    outfn:
        Optional ``.nc`` path to also write to.
    use:
        Optional system-letter set passed through to the RINEX-3 reader
        (RINEX-2 NAV is single-system per file, so this is ignored there).
    group:
        HDF5 group to read from when given a ``.nc`` file. Default ``"NAV"``.
    tlim, overwrite: see :func:`load`.

    Returns
    -------
    xarray.Dataset
    """
    tlim = normalize_tlim(tlim)
    if isinstance(fn, (str, Path)):
        fn = Path(fn).expanduser()
        if fn.suffix == ".nc":
            try:
                return xr.open_dataset(fn, group=group)
            except OSError as e:
                raise LookupError(f"group {group} not found in {fn}: {e}") from e

    info = rinexinfo(fn)
    version = int(info["version"])
    if version == 2:
        nav = rinexnav2(fn, tlim=tlim)
    elif version == 3:
        nav = rinexnav3(fn, use=_normalize_use(use), tlim=tlim)
    else:
        raise LookupError(f"unsupported RINEX version: {info}")

    if outfn is not None:
        write_dataset(nav, Path(outfn), group=group, overwrite=overwrite)
    return nav


def rinexobs(
    fn: FileLike,
    outfn: Path | None = None,
    *,
    use: SystemSelection = None,
    group: str = "OBS",
    tlim: TimeLimit = None,
    useindicators: bool = False,
    meas: MeasSelection = None,
    verbose: bool = False,
    overwrite: bool = False,
    fast: bool = True,
    interval: float | int | timedelta | None = None,
) -> xr.Dataset:
    """Read any RINEX-2 or RINEX-3 OBS file (or open a ``.nc`` group).

    Parameters mirror :func:`rinexnav` plus the OBS-specific switches
    documented on :func:`rinexpy.obs2.rinexsystem2` and
    :func:`rinexpy.obs3.rinexobs3`.
    """
    tlim = normalize_tlim(tlim)
    if isinstance(fn, (str, Path)):
        fn = Path(fn).expanduser()
        if fn.suffix == ".nc":
            try:
                return xr.open_dataset(fn, group=group)
            except OSError as e:
                raise LookupError(f"group {group} not found in {fn}: {e}") from e

    info = rinexinfo(fn)
    version = int(info["version"])
    use_norm = _normalize_use(use)

    if version in {1, 2}:
        obs = rinexobs2(
            fn,
            use_norm,
            tlim=tlim,
            useindicators=useindicators,
            meas=meas,
            verbose=verbose,
            fast=fast,
            interval=interval,
        )
    elif version == 3:
        obs = rinexobs3(
            fn,
            use_norm,
            tlim=tlim,
            useindicators=useindicators,
            meas=meas,
            verbose=verbose,
            fast=fast,
            interval=interval,
        )
    else:
        raise ValueError(f"unsupported RINEX version: {info}")

    if outfn is not None:
        write_dataset(obs, Path(outfn), group=group, overwrite=overwrite)
    return obs


def gettime(fn: FileLike):
    """Extract just the timestamp axis from a RINEX file.

    Parameters
    ----------
    fn:
        Path or open stream of a RINEX OBS or NAV file.

    Returns
    -------
    numpy.ndarray
        ``datetime64[ms]`` (NAV) or ``datetime64[us]`` (OBS) sorted unique
        timestamps.
    """
    from .nav2 import navtime2
    from .nav3 import navtime3
    from .obs2 import obstime2
    from .obs3 import obstime3

    info = rinexinfo(fn)
    version = int(info["version"])
    rinex_type = info["rinextype"]
    if rinex_type == "obs":
        return obstime2(fn) if version in {1, 2} else obstime3(fn)
    if rinex_type == "nav":
        return navtime2(fn) if version in {1, 2} else navtime3(fn)
    raise ValueError(f"per-epoch times not defined for {info}")


def batch_convert(
    path: Path | str,
    glob: str,
    out: Path | str,
    *,
    use: SystemSelection = None,
    tlim: TimeLimit = None,
    useindicators: bool = False,
    meas: MeasSelection = None,
    verbose: bool = False,
    fast: bool = True,
) -> list[Path]:
    """Convert every file in ``path`` matching ``glob`` to NetCDF in ``out``.

    Errors on individual files are logged and the conversion continues with
    the next file. The list of successfully converted output paths is
    returned.

    Parameters
    ----------
    path:
        Directory to scan.
    glob:
        Filename glob pattern (e.g. ``"*o"`` or ``"*.rnx.gz"``).
    out:
        Output directory or single output file. If a directory, each input's
        basename + ``.nc`` is used.
    use, tlim, useindicators, meas, verbose, fast:
        Forwarded to :func:`load`.

    Returns
    -------
    list[Path]
        Paths that were written.
    """
    path = Path(path).expanduser()
    out_p = Path(out).expanduser()

    written: list[Path] = []
    for fn in path.glob(glob):
        if not fn.is_file():
            continue
        try:
            load(
                fn,
                out_p,
                use=use,
                tlim=tlim,
                useindicators=useindicators,
                meas=meas,
                verbose=verbose,
                fast=fast,
            )
            written.append(out_p / (fn.name + ".nc") if out_p.is_dir() else out_p)
        except (ValueError, OSError) as e:
            log.error("%s: %s", fn.name, e)
    return written


def _resolve_outfn(rinexfn: FileLike, out: Path | str | None) -> Path | None:
    """Translate the user-friendly ``out`` arg to a concrete output path.

    - ``None`` → ``None`` (no write).
    - directory → ``directory / "<input.basename>.nc"``.
    - ``*.nc`` → returned as-is.
    """
    if out is None:
        return None
    out_p = Path(out).expanduser()
    if out_p.is_dir():
        if not isinstance(rinexfn, Path):
            raise ValueError("cannot infer output filename for in-memory input")
        return out_p / (rinexfn.name + ".nc")
    if out_p.suffix == ".nc":
        return out_p
    raise ValueError(f"don't know how to handle out={out!r}")


def _load_nc(fn: Path) -> object:
    """Try to read NAV and/or OBS groups from a NetCDF file."""
    nav = obs = None
    try:
        nav = rinexnav(fn)
    except LookupError:
        pass
    try:
        obs = rinexobs(fn)
    except LookupError:
        pass
    if nav is not None and obs is not None:
        return {"nav": nav, "obs": obs}
    if nav is not None:
        return nav
    if obs is not None:
        return obs
    raise ValueError(f"no NAV or OBS data in {fn}")


def _normalize_use(use: SystemSelection) -> set[str] | None:
    """Coerce a system-selection argument into a `set[str]` or `None`."""
    if use is None:
        return None
    if isinstance(use, str):
        return {use}
    return set(use)


__all__ = [
    "batch_convert",
    "gettime",
    "load",
    "rinexnav",
    "rinexobs",
]
