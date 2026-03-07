"""Antenna PCV calibration tool: generate ANTEX entries from residuals.

Given a calibration session's post-fit residuals tagged by satellite
elevation and azimuth, this module fits a 2-D phase-center-variation
(PCV) table on the standard ANTEX grid and writes out a valid ``.atx``
record that round-trips through :func:`rinexpy.antex.load_antex`.

The fit is a simple binning + averaging: each observation falls into
one (azimuth, zenith) cell of size ``dazi_deg`` × ``dzen_deg``; the
cell value is the mean of the residuals in that cell, in millimetres.
Empty cells fall back to the corresponding NOAZI (azimuth-averaged)
value at the same zenith. NOAZI itself is the mean residual per zenith
bin across all azimuths.

For multi-day calibration sessions, callers should pre-process
residuals to remove session-wide trends (clock, troposphere) before
passing them in - PCV is the *systematic* mis-modelling that remains
after a clean float / fixed solution. Per-day stacking and outlier
rejection are also expected to be done upstream.

The output is one ANTEX entry per call:

    {
        "type": str, "serial": str,
        "valid_from": datetime, "valid_until": datetime,
        "frequencies": {"G01": {"noazi": ndarray, "pcv": ndarray, ...}},
        "azimuth_deg": ndarray, "zenith_deg": ndarray,
        "dazi_deg": float,
    }

Pass that entry (wrapped in a list) to :func:`write_antex` to produce
an ``.atx`` file that the existing reader can consume.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def calibrate_pcv(
    residuals_m: np.ndarray,
    elevation_rad: np.ndarray,
    azimuth_rad: np.ndarray,
    *,
    antenna_type: str,
    serial: str = "",
    frequency: str = "G01",
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    dazi_deg: float = 5.0,
    dzen_deg: float = 5.0,
    zen_max_deg: float = 90.0,
) -> dict[str, Any]:
    """Fit a 2-D PCV table from observed residuals.

    Parameters
    ----------
    residuals_m:
        ``(n,)`` post-fit residuals in metres.
    elevation_rad:
        ``(n,)`` satellite elevations in radians.
    azimuth_rad:
        ``(n,)`` satellite azimuths in radians (clockwise from North,
        0..2π).
    antenna_type:
        Antenna model name (max 20 chars).
    serial:
        Antenna serial number (max 20 chars).
    frequency:
        Frequency identifier (e.g. ``"G01"`` for GPS L1).
    valid_from, valid_until:
        Validity window for the ANTEX entry. Defaults to "now" and
        "open-ended".
    dazi_deg, dzen_deg:
        Azimuth and zenith bin sizes (default 5 deg).
    zen_max_deg:
        Maximum zenith angle to fit (default 90 deg).

    Returns
    -------
    dict
        ANTEX-shaped entry ready for :func:`write_antex`.
    """
    r = np.asarray(residuals_m, dtype=float)
    elev = np.asarray(elevation_rad, dtype=float)
    azi = np.asarray(azimuth_rad, dtype=float)
    if r.shape != elev.shape or r.shape != azi.shape or r.ndim != 1:
        raise ValueError("residuals_m, elevation_rad, azimuth_rad must be matched 1-D arrays")

    zen_deg = 90.0 - np.rad2deg(elev)
    azi_deg = np.rad2deg(azi) % 360.0

    n_az = int(round(360.0 / dazi_deg)) + 1
    azimuth_axis = np.linspace(0.0, 360.0, n_az)
    n_zen = int(round(zen_max_deg / dzen_deg)) + 1
    zenith_axis = np.linspace(0.0, zen_max_deg, n_zen)

    pcv_mm = np.zeros((n_az, n_zen), dtype=float)
    azi_bin = np.clip(np.floor(azi_deg / dazi_deg).astype(int), 0, n_az - 2)
    zen_bin = np.clip(np.floor(zen_deg / dzen_deg).astype(int), 0, n_zen - 2)

    # Per-cell mean, in millimetres.
    sums = np.zeros((n_az - 1, n_zen - 1))
    counts = np.zeros((n_az - 1, n_zen - 1), dtype=int)
    for ai, zi, val in zip(azi_bin, zen_bin, r):
        if 0 <= zi < n_zen - 1:
            sums[ai, zi] += val * 1000.0
            counts[ai, zi] += 1
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    # NOAZI = azimuth-averaged PCV at each zenith.
    noazi_mm = np.zeros(n_zen)
    for zi in range(n_zen - 1):
        nz = counts[:, zi].sum()
        noazi_mm[zi] = (sums[:, zi].sum() / nz) if nz > 0 else 0.0
    # Last zenith bin (== 90) just mirrors the previous to keep the axis
    # well-defined.
    noazi_mm[-1] = noazi_mm[-2]

    # Fill the 2-D table from the bin means, with empty cells falling
    # back to NOAZI. The last azimuth row (== 360) matches row 0 for
    # the wrap-around.
    for ai in range(n_az - 1):
        for zi in range(n_zen - 1):
            pcv_mm[ai, zi] = mean[ai, zi] if counts[ai, zi] > 0 else noazi_mm[zi]
    pcv_mm[:, -1] = pcv_mm[:, -2]
    pcv_mm[-1, :] = pcv_mm[0, :]

    if valid_from is None:
        valid_from = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    return {
        "type": antenna_type[:20],
        "serial": serial[:20],
        "valid_from": valid_from,
        "valid_until": valid_until,
        "frequencies": {
            frequency: {
                "north": 0.0,
                "east": 0.0,
                "up": 0.0,
                "noazi": noazi_mm,
                "pcv": pcv_mm,
                "azimuth_deg": azimuth_axis,
                "zenith_deg": zenith_axis,
            },
        },
        "azimuth_deg": azimuth_axis,
        "zenith_deg": zenith_axis,
        "dazi_deg": dazi_deg,
    }


def _fmt_header_line(content: str, label: str) -> str:
    """Format a 60-char content field + 20-char label, no trailing
    newline. Used to assemble ANTEX records."""
    if len(content) > 60:
        content = content[:60]
    if len(label) > 20:
        label = label[:20]
    return f"{content:<60}{label:<20}\n"


def _fmt_pcv_row(prefix: str, values: np.ndarray) -> str:
    """Format one ANTEX PCV row: ``prefix(8) + values(8 each) + \n``."""
    body = "".join(f"{v:8.2f}" for v in values)
    return f"{prefix:<8}{body}\n"


def write_antex(entries: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of calibrated antenna entries to an ANTEX (.atx) file.

    The output passes through :func:`rinexpy.antex.load_antex` cleanly.
    """
    path = Path(path)
    with path.open("w") as f:
        if entries:
            dazi = entries[0].get("dazi_deg", 5.0)
        else:
            dazi = 5.0
        # Header.
        f.write(_fmt_header_line("     1.4            M", "ANTEX VERSION / SYST"))
        f.write(_fmt_header_line("A", "PCV TYPE / REFANT"))
        f.write(_fmt_header_line(f"{dazi:6.1f}", "DAZI"))
        # We use 0..90 / 5 by default; report it explicitly.
        first = entries[0] if entries else None
        if first:
            z = first["zenith_deg"]
            dzen = float(z[1] - z[0]) if z.size > 1 else 5.0
            f.write(_fmt_header_line(
                f"{z[0]:6.1f}{z[-1]:6.1f}{dzen:6.1f}",
                "ZEN1 / ZEN2 / DZEN",
            ))
        f.write(_fmt_header_line("", "END OF HEADER"))

        for entry in entries:
            f.write(_fmt_header_line("", "START OF ANTENNA"))
            f.write(_fmt_header_line(
                f"{entry.get('type', ''):<20}{entry.get('serial', ''):<20}",
                "TYPE / SERIAL NO",
            ))
            valid_from = entry.get("valid_from")
            if isinstance(valid_from, datetime):
                f.write(_fmt_header_line(
                    f"{valid_from.year:6d}{valid_from.month:6d}{valid_from.day:6d}"
                    f"{valid_from.hour:6d}{valid_from.minute:6d}"
                    f"{valid_from.second:13.7f}",
                    "VALID FROM",
                ))
            z = entry["zenith_deg"]
            dzen = float(z[1] - z[0]) if z.size > 1 else 5.0
            f.write(_fmt_header_line(
                f"{z[0]:6.1f}{z[-1]:6.1f}{dzen:6.1f}",
                "ZEN1 / ZEN2 / DZEN",
            ))
            f.write(_fmt_header_line(f"{len(entry['frequencies']):6d}", "# OF FREQUENCIES"))
            for freq_id, freq in entry["frequencies"].items():
                f.write(_fmt_header_line(f"   {freq_id:<3}", "START OF FREQUENCY"))
                f.write(_fmt_header_line(
                    f"{freq.get('north', 0.0):10.2f}{freq.get('east', 0.0):10.2f}"
                    f"{freq.get('up', 0.0):10.2f}",
                    "NORTH / EAST / UP",
                ))
                f.write(_fmt_pcv_row("NOAZI", freq["noazi"]))
                pcv = freq["pcv"]
                azi = entry["azimuth_deg"]
                for ai in range(pcv.shape[0]):
                    f.write(_fmt_pcv_row(f"{azi[ai]:8.1f}", pcv[ai]))
                f.write(_fmt_header_line(f"   {freq_id:<3}", "END OF FREQUENCY"))
            f.write(_fmt_header_line("", "END OF ANTENNA"))


__all__ = ["calibrate_pcv", "write_antex"]
