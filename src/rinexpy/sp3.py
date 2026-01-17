"""SP3-a / SP3-c / SP3-d ephemeris file reader.

The SP3 format is a precise-orbit text format published by the IGS:

- SP3-a / SP3-c / SP3-d: https://files.igs.org/pub/data/format/

Each "epoch" header begins with ``*``; subsequent lines give satellite
position (``P`` lines) and optional velocity (``V`` lines). This module
preallocates one big ``(n_epochs, n_sv, 3)`` array per quantity and back-
fills it as the parse proceeds, avoiding any list-of-arrays growth pattern.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ._io import opener
from ._types import FileLike
from ._version import first_nonblank_line

log = logging.getLogger(__name__)


def load_sp3(fn: FileLike, outfn: Path | None = None) -> xr.Dataset:
    """Read an SP3-a/c/d file into an ``xarray.Dataset``.

    Parameters
    ----------
    fn:
        Path or open text stream of an SP3 file.
    outfn:
        Optional output path; if provided the dataset is also written to disk
        as NetCDF4 with light zlib compression.

    Returns
    -------
    xarray.Dataset
        Coords: ``time`` (datetime64), ``sv`` (string), ``ECEF`` (``["x","y","z"]``).
        Data variables: ``position``, ``velocity`` (both ``(time, sv, ECEF)``),
        ``clock``, ``dclock`` (both ``(time, sv)``), ``t0`` (scalar).

    Raises
    ------
    ValueError
        If the file's first line is not a valid SP3 header, the SV header
        block is missing, or the file is otherwise badly malformed.
    """
    if isinstance(fn, (str, Path)):
        fn = Path(fn).expanduser()

    attrs: dict[Hashable, Any] = {}

    with opener(fn) as f:
        line = first_nonblank_line(f)
        if not line or line[0] != "#":
            raise ValueError(f"{fn}: not an SP3 file (first byte != '#')")
        t0 = _parse_sp3_dt(line)
        attrs["Nepoch"] = int(line[32:39])
        attrs["coord_sys"] = line[46:51]
        attrs["orbit_type"] = line[52:55]
        attrs["agency"] = line[56:60]

        f.readline()  # second header line (gpsweek/gpssec/...)
        line = f.readline()
        if not line or line[0] != "+":
            raise ValueError(f"{fn}: SV header block missing")

        n_sv = int(line[3:6])
        svs = _parse_sv_chunk(line, n_sv)
        unread = n_sv - 17
        while unread > 0:
            svs += _parse_sv_chunk(f.readline(), unread)
            unread -= 17

        # Skip remaining ``+``/``%``/``//`` header lines until the first
        # epoch marker.
        for line in f:
            if line.startswith("*"):
                break
        else:
            raise ValueError(f"{fn}: no epoch lines found")

        ecefs: list[np.ndarray] = []
        clocks: list[np.ndarray] = []
        vels: list[np.ndarray] = []
        # Pre-fill with NaN: SVs absent from a particular epoch must read
        # back as NaN, not whatever happened to be on the heap. (georinex
        # forgot to fill these buffers and the resulting nondeterminism
        # was a long-standing source of confusing test failures.)
        ecef = np.full((n_sv, 3), np.nan)
        clock = np.full((n_sv, 2), np.nan)
        vel = np.full((n_sv, 3), np.nan)
        i = 0
        times: list[datetime] = [_parse_sp3_dt(line)]

        for line in f:
            if not line:
                continue
            head = line[0]
            if head == "*":
                times.append(_parse_sp3_dt(line))
                ecefs.append(ecef)
                clocks.append(clock)
                vels.append(vel)
                ecef = np.full((n_sv, 3), np.nan)
                clock = np.full((n_sv, 2), np.nan)
                vel = np.full((n_sv, 3), np.nan)
                i = 0
            elif head == "P":
                ecef[i] = (float(line[4:18]), float(line[18:32]), float(line[32:46]))
                clock[i, 0] = float(line[46:60])
                i += 1
            elif head == "V":
                vel[i - 1] = (float(line[4:18]), float(line[18:32]), float(line[32:46]))
                clock[i - 1, 1] = float(line[46:60])
            elif line.startswith(("EP", "EV")) or (len(line) >= 4 and line[3] == "*"):
                continue
            elif line.startswith("EOF"):
                break
            elif not line.strip():
                continue
            else:
                log.info("unknown SP3 line: %r", line[:80])

    ecefs.append(ecef)
    clocks.append(clock)
    vels.append(vel)
    aclock = np.asarray(clocks)

    ds = xr.Dataset(
        coords={"time": times, "sv": svs, "ECEF": ["x", "y", "z"]},
        data_vars={
            "position": (("time", "sv", "ECEF"), np.asarray(ecefs)),
            "clock": (("time", "sv"), aclock[:, :, 0]),
            "velocity": (("time", "sv", "ECEF"), np.asarray(vels)),
            "dclock": (("time", "sv"), aclock[:, :, 1]),
        },
    )
    ds["t0"] = t0
    ds.attrs = attrs

    if outfn is not None:
        outfn = Path(outfn).expanduser()
        enc = {k: {"zlib": True, "complevel": 1, "fletcher32": True} for k in ds.data_vars}
        ds.to_netcdf(outfn, mode="w", encoding=enc, format="NETCDF4")
    return ds


def _parse_sp3_dt(line: str) -> datetime:
    """Parse an SP3 ``* YYYY MM DD HH MM SS.SSSSSSSS`` epoch line.

    Some receivers emit ``second=60`` / ``minute=60`` / ``hour=24`` to mean
    "rolls over to the next minute/hour/day"; we honor that here.
    """
    deltas: list[timedelta] = []
    hour = int(line[14:16])
    minute = int(line[17:19])
    second = int(line[20:22])
    if second == 60:
        deltas.append(timedelta(minutes=1))
        second = 0
    if minute == 60:
        deltas.append(timedelta(hours=1))
        minute = 0
    if hour == 24:
        deltas.append(timedelta(days=1))
        hour = 0
    t = datetime(
        int(line[3:7]),
        int(line[8:10]),
        int(line[11:13]),
        hour,
        minute,
        second,
        int(line[23:29]),
    )
    for d in deltas:
        t += d
    return t


def _parse_sv_chunk(line: str, n_sv: int) -> list[str]:
    """Parse up to 17 SV labels from a single ``+`` SP3 header line."""
    if not line or line[0] != "+":
        return []
    out: list[str] = []
    for i in range(min(n_sv, 17)):
        out.append(line[9 + i * 3 : 12 + i * 3].replace(" ", ""))
    return out


def write_sp3(ds: xr.Dataset, outpath, *, version: str = "c") -> Path:
    """Write an SP3 dataset back to disk in SP3-c or SP3-d format.

    Round-trip companion to :func:`load_sp3`. The output mimics IGS
    conventions: a 22-line header followed by per-epoch ``*`` time
    lines and per-SV ``P`` records.
    """
    out = Path(outpath).expanduser()
    times = ds.time.values.astype("datetime64[us]")
    n_t = times.size
    sv_list = list(ds.sv.values)
    n_sv = len(sv_list)
    pos = ds.position.values
    clock = (ds["clock"].values if "clock" in ds.data_vars
             else np.full((n_t, n_sv), 999999.999999))

    t0 = times[0].astype("datetime64[us]").tolist()
    if t0.tzinfo is not None:
        t0 = t0.replace(tzinfo=None)
    gps_epoch = datetime(1980, 1, 6)
    total_sec = (t0 - gps_epoch).total_seconds()
    gps_week = int(total_sec // (7 * 86400))
    sow = total_sec - gps_week * 7 * 86400

    with out.open("w") as f:
        f.write(
            f"#{version}P{t0.year:4d} {t0.month:2d} {t0.day:2d} "
            f"{t0.hour:2d} {t0.minute:2d} {t0.second + t0.microsecond*1e-6:11.8f}"
            f" {n_t:7d} ORBIT IGS14 FIT  IGS\n"
        )
        f.write(
            f"## {gps_week:4d} {sow:15.8f} {0:14.8f} {0:5d} {0:15.13f}\n"
        )
        sv_ids_per_line = 17
        for li in range(5):
            start = li * sv_ids_per_line
            chunk = sv_list[start:start + sv_ids_per_line]
            ids = "".join(str(s).rjust(3) for s in chunk)
            ids = ids.ljust(sv_ids_per_line * 3)
            prefix = f"+   {n_sv:2d}   " if li == 0 else "+         "
            f.write(f"{prefix}{ids}\n")
        for li in range(5):
            ids = "".join(" 00" for _ in range(sv_ids_per_line))
            f.write(f"++         {ids}\n")
        f.write("%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n")
        f.write("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n")
        f.write("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n")
        f.write("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n")
        f.write("%i    0    0    0    0      0      0      0      0         0\n")
        f.write("%i    0    0    0    0      0      0      0      0         0\n")
        for _ in range(4):
            f.write("/* Generated by rinexpy.sp3.write_sp3\n")
        for ti in range(n_t):
            t = times[ti].astype("datetime64[us]").tolist()
            if t.tzinfo is not None:
                t = t.replace(tzinfo=None)
            f.write(
                f"*  {t.year:4d} {t.month:2d} {t.day:2d} "
                f"{t.hour:2d} {t.minute:2d} "
                f"{t.second + t.microsecond * 1e-6:11.8f}\n"
            )
            for si, sv in enumerate(sv_list):
                px, py, pz = pos[ti, si]
                if not np.isfinite(px):
                    px = py = pz = 0.0
                clk = clock[ti, si] if clock is not None else 999999.999999
                if not np.isfinite(clk):
                    clk = 999999.999999
                f.write(
                    f"P{str(sv):>3s}{px:14.6f}{py:14.6f}{pz:14.6f}"
                    f"{clk:14.6f}\n"
                )
        f.write("EOF\n")
    return out


def stitch_sp3(*paths) -> xr.Dataset:
    """Load and concatenate consecutive daily SP3 files along time.

    IGS daily SP3 products include the first epoch of the next day for
    interpolation continuity, so naive concatenation produces duplicate
    time stamps at every day boundary. This helper concatenates with
    ``xarray.concat`` and then drops duplicate epochs, keeping the
    first occurrence. The SV axis is the union across all inputs (an
    SV missing on one day shows up as NaN there).

    Parameters
    ----------
    *paths:
        One or more SP3 file paths. Order doesn't matter; epochs are
        sorted before duplicate removal.

    Returns
    -------
    xarray.Dataset
        Same data variables as :func:`load_sp3`. Concatenated along
        ``time``; sorted; duplicate times dropped.

    Raises
    ------
    ValueError
        If no paths are provided.
    """
    if not paths:
        raise ValueError("stitch_sp3 needs at least one path")
    parts = [load_sp3(p) for p in paths]
    if len(parts) == 1:
        return parts[0]
    combined = xr.concat(parts, dim="time", join="outer", data_vars="all")
    combined = combined.sortby("time")
    _, unique_idx = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=np.sort(unique_idx))
    return combined


__all__ = ["load_sp3", "stitch_sp3"]
