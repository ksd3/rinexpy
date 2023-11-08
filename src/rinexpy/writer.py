"""RINEX OBS / NAV writer.

Round-trips a parsed ``xarray.Dataset`` back to a RINEX 2 / RINEX 3 OBS
file. Sufficient for the read-modify-write workflow (filter, decimate,
re-emit), not a full-fidelity replacement for the source bytes — header
records that aren't in our dataset attrs (e.g. comments) are dropped.

Usage:

.. code-block:: python

    obs = rinexpy.load("input.18o", tlim=("2018-07-29", "2018-07-30"))
    rinexpy.to_rinex_obs(obs, "filtered.18o", version=2)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


def to_rinex_obs(
    obs: xr.Dataset,
    fn: Path | str,
    *,
    version: int = 3,
) -> Path:
    """Write a parsed OBS dataset back to a RINEX file.

    Parameters
    ----------
    obs:
        Dataset returned by :func:`rinexpy.rinexobs` (or filtered).
    fn:
        Output path.
    version:
        Either ``2`` (RINEX 2.11) or ``3`` (RINEX 3.04). Default 3.

    Returns
    -------
    pathlib.Path
        The (resolved) output path.

    Raises
    ------
    ValueError
        If ``version`` is not 2 or 3, or the dataset lacks ``time``/``sv``
        coords.
    """
    if version not in {2, 3}:
        raise ValueError(f"unsupported RINEX version {version}")
    if "time" not in obs.coords or "sv" not in obs.coords:
        raise ValueError("OBS dataset must have time and sv coords")

    out = Path(fn).expanduser()
    if version == 3:
        text = _format_obs3(obs)
    else:
        text = _format_obs2(obs)
    out.write_text(text)
    return out


def _format_header_field(label: str, content: str) -> str:
    """Pad ``content`` to 60 chars and append the right-justified ``label``."""
    return f"{content[:60]:<60}{label:<20}\n"


def _format_obs3(obs: xr.Dataset) -> str:
    """Write a RINEX-3 OBS file body."""
    out: list[str] = []
    # Header.
    out.append(
        _format_header_field(
            "RINEX VERSION / TYPE",
            f"{3.04:9.2f}           OBSERVATION DATA    M (MIXED)",
        )
    )
    out.append(_format_header_field("PGM / RUN BY / DATE", "rinexpy             rinexpy             "))
    if "position" in obs.attrs:
        x, y, z = obs.attrs["position"][:3]
        out.append(
            _format_header_field("APPROX POSITION XYZ", f"{x:14.4f}{y:14.4f}{z:14.4f}")
        )
    if obs.time.size:
        first = np.datetime64(obs.time.values[0], "us").astype(datetime)
        sys_name = obs.attrs.get("time_system", "GPS")
        out.append(
            _format_header_field(
                "TIME OF FIRST OBS",
                f"{first.year:6d}{first.month:6d}{first.day:6d}"
                f"{first.hour:6d}{first.minute:6d}{first.second:13.7f}"
                f"     {sys_name:<3}",
            )
        )
    # Group data variables by leading letter (assumed system-pure for now).
    sys_letters = sorted({sv[0] for sv in obs.sv.values})
    fields_by_sys: dict[str, list[str]] = {sk: [] for sk in sys_letters}
    for v in obs.data_vars:
        # Skip indicator (lli/ssi) columns - we only emit the value column.
        if v.endswith(("lli", "ssi")):
            continue
        for sk in sys_letters:
            fields_by_sys[sk].append(v)
    for sk in sys_letters:
        labels = fields_by_sys[sk]
        cont = "".join(f" {l:<3}" for l in labels)
        out.append(
            _format_header_field(
                "SYS / # / OBS TYPES",
                f"{sk}{len(labels):5d}{cont}",
            )
        )
    out.append(_format_header_field("END OF HEADER", ""))

    # Data section.
    times = obs.time.values
    svs = obs.sv.values
    fields = [v for v in obs.data_vars if not v.endswith(("lli", "ssi"))]
    arrs = {v: obs[v].values for v in fields}

    for ti, t in enumerate(times):
        dt = np.datetime64(t, "ns").astype("datetime64[us]").astype(datetime)
        # Find which SVs have any non-NaN data this epoch.
        present = [
            sv
            for sj, sv in enumerate(svs)
            if any(
                np.isfinite(arrs[v][ti, sj])
                for v in fields
                if v in fields_by_sys.get(sv[0], [])
            )
        ]
        if not present:
            continue
        secs = dt.second + dt.microsecond / 1e6
        out.append(
            f"> {dt.year:4d} {dt.month:02d} {dt.day:02d} "
            f"{dt.hour:02d} {dt.minute:02d}{secs:11.7f}  0"
            f"{len(present):3d}\n"
        )
        for sv in present:
            sk = sv[0]
            row = [f"{sv:<3}"]
            for v in fields_by_sys.get(sk, []):
                sj = int(np.where(svs == sv)[0][0])
                val = arrs[v][ti, sj]
                if np.isfinite(val):
                    row.append(f"{val:14.3f}  ")
                else:
                    row.append(" " * 16)
            out.append("".join(row) + "\n")
    return "".join(out)


def _format_obs2(obs: xr.Dataset) -> str:
    """Write a RINEX-2.11 OBS file body."""
    out: list[str] = []
    out.append(
        _format_header_field(
            "RINEX VERSION / TYPE",
            f"{2.11:9.2f}           OBSERVATION DATA    M (MIXED)",
        )
    )
    out.append(_format_header_field("PGM / RUN BY / DATE", "rinexpy             rinexpy             "))
    if "position" in obs.attrs:
        x, y, z = obs.attrs["position"][:3]
        out.append(
            _format_header_field("APPROX POSITION XYZ", f"{x:14.4f}{y:14.4f}{z:14.4f}")
        )
    if obs.time.size:
        first = np.datetime64(obs.time.values[0], "us").astype(datetime)
        sys_name = obs.attrs.get("time_system", "GPS")
        out.append(
            _format_header_field(
                "TIME OF FIRST OBS",
                f"{first.year:6d}{first.month:6d}{first.day:6d}"
                f"{first.hour:6d}{first.minute:6d}{first.second:13.7f}"
                f"     {sys_name:<3}",
            )
        )
    fields = [v for v in obs.data_vars if not v.endswith(("lli", "ssi"))]
    n_obs = len(fields)
    types_str = "".join(f"    {f:<2}" for f in fields)
    out.append(
        _format_header_field(
            "# / TYPES OF OBSERV", f"{n_obs:6d}{types_str}"
        )
    )
    out.append(_format_header_field("END OF HEADER", ""))

    times = obs.time.values
    svs = obs.sv.values
    arrs = {v: obs[v].values for v in fields}

    for ti, t in enumerate(times):
        dt = np.datetime64(t, "ns").astype("datetime64[us]").astype(datetime)
        # Year is 2-digit in RINEX 2.
        year2 = dt.year % 100
        present = [
            (sj, sv)
            for sj, sv in enumerate(svs)
            if any(np.isfinite(arrs[v][ti, sj]) for v in fields)
        ]
        if not present:
            continue
        sv_str = "".join(sv for _, sv in present[:12])
        out.append(
            f" {year2:02d} {dt.month:2d} {dt.day:2d} "
            f"{dt.hour:2d} {dt.minute:2d} {dt.second:10.7f}  0"
            f"{len(present):3d}{sv_str}\n"
        )
        for sj, _sv in present:
            row = []
            for v in fields:
                val = arrs[v][ti, sj]
                if np.isfinite(val):
                    row.append(f"{val:14.3f}  ")
                else:
                    row.append(" " * 16)
            out.append("".join(row) + "\n")
    return "".join(out)


__all__ = ["to_rinex_obs"]
