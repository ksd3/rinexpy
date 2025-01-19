"""Ocean tide loading station displacement (IERS Conventions 2010 section 7.1.2).

When the open ocean redistributes mass with the diurnal and semidiurnal
tides, the elastic loading deforms the nearby crust. At coastal stations
the resulting station displacement can reach 5 cm radial; even at inland
sites it's typically 1-2 cm. Sub-cm PPP needs the correction applied.

The standard input is a Scherneck "BLQ" file produced by the Onsala
ocean-tide loading service (`http://holt.oso.chalmers.se/loading/`) for
a given station. The file lists, for each of 11 main tidal constituents,
the (radial, west, south) amplitude in meters and Greenwich phase lag
in degrees. The displacement at any epoch is then the harmonic sum

    d_c(t) = sum_j A_cj * cos(theta_j(t) - phi_cj)

where the constituent argument ``theta_j(t)`` is built from the same
Brown / Bretagnon fundamental arguments used by the solid-earth step-2
tide code (see :mod:`rinexpy.tides`). This module ships the 11-
constituent "short list" model used by HARDISP without the admittance
spline interpolation to 71 sub-constituents — accuracy is at the few-mm
level which is adequate for static PPP at non-extreme coastal sites.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .tides import (
    _fractional_hours_ut,
    _step2_fundamental_arguments,
    _t_centuries_tt,
)

# Scherneck convention: 11 tidal constituents in the canonical order
# (M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, Ssa). Each row gives the
# coefficients of (TAU, S, H, P, ZNS, PS) for that constituent's
# Doodson argument, with TAU the lunar hour angle and (S, H, P, ZNS, PS)
# the slow Bretagnon mean elements.
_OTL_NAMES = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "Mf", "Mm", "Ssa")
_OTL_ARG_COEFS = np.array([
    (2,  0,  0,  0,  0,  0),   # M2
    (2,  2, -2,  0,  0,  0),   # S2
    (2, -1,  0,  1,  0,  0),   # N2
    (2,  2,  0,  0,  0,  0),   # K2
    (1,  0,  0,  0,  0,  0),   # K1
    (1, -2,  0,  0,  0,  0),   # O1
    (1,  0, -2,  0,  0,  0),   # P1
    (1, -3,  0,  1,  0,  0),   # Q1
    (0,  2,  0,  0,  0,  0),   # Mf
    (0,  1,  0, -1,  0,  0),   # Mm
    (0,  0,  2,  0,  0,  0),   # Ssa
], dtype=float)


def read_blq(path) -> dict[str, dict[str, Any]]:
    """Parse a Scherneck BLQ ocean-loading file.

    The format is plain text. Each station block starts with a comment
    line beginning ``$$`` followed by the station name, then 6 data lines
    each carrying 11 floats: three amplitude rows (radial, west, south,
    in meters) and three phase rows (radial, west, south, in degrees).

    Parameters
    ----------
    path:
        Path to a BLQ file.

    Returns
    -------
    dict
        Map from station name to a dict with keys
        ``amp_radial``, ``amp_west``, ``amp_south`` (numpy arrays in m)
        and ``phase_radial``, ``phase_west``, ``phase_south`` (in
        degrees). Each array has length 11 in the canonical Scherneck
        constituent order.

    Raises
    ------
    ValueError
        If the file is empty or a station block is malformed.
    """
    path = Path(path)
    stations: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    last_name: str | None = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        stripped = line.lstrip()
        if stripped.startswith("$$"):
            # Remember the most recent non-empty $$ comment as the
            # candidate station name. BLQ files put the name on the
            # last $$ line immediately before the 6 data lines.
            tail = stripped[2:].strip()
            # Station-name $$ lines have exactly one token in their tail
            # (e.g. ``$$ ALGO`` or ``$$ OSLO,``); descriptive comments
            # like ``$$ Site coordinates: ...`` have many tokens and are
            # skipped. The last single-token $$ line before the data
            # block is the station name.
            first_part = tail.split(",")[0] if tail else ""
            tokens = first_part.split()
            if len(tokens) == 1:
                last_name = tokens[0]
            i += 1
            continue
        if not stripped:
            i += 1
            continue
        # Try parsing 6 consecutive data lines of 11 floats each.
        if i + 6 > n:
            break
        try:
            data = [list(map(float, lines[i + k].split())) for k in range(6)]
        except ValueError:
            i += 1
            continue
        if any(len(row) != 11 for row in data):
            i += 1
            continue
        if last_name is None:
            i += 6
            continue
        stations[last_name] = {
            "amp_radial": np.asarray(data[0], dtype=float),
            "amp_west": np.asarray(data[1], dtype=float),
            "amp_south": np.asarray(data[2], dtype=float),
            "phase_radial": np.asarray(data[3], dtype=float),
            "phase_west": np.asarray(data[4], dtype=float),
            "phase_south": np.asarray(data[5], dtype=float),
        }
        last_name = None
        i += 6
    if not stations:
        raise ValueError(f"{path}: no station blocks found in BLQ file")
    return stations


def ocean_tide_loading_displacement(
    blq_entry: dict[str, Any],
    epoch,
) -> np.ndarray:
    """Local (east, north, up) ocean-tide-loading displacement at an epoch.

    The Scherneck convention is

        d_radial(t) = sum_j A_r,j * cos(theta_j(t) - phi_r,j)
        d_west(t)   = sum_j A_w,j * cos(theta_j(t) - phi_w,j)
        d_south(t)  = sum_j A_s,j * cos(theta_j(t) - phi_s,j)

    where ``theta_j`` is the constituent's Doodson argument in radians.
    The result is returned in ENU (east, north, up) by flipping the sign
    on the west and south components.

    Parameters
    ----------
    blq_entry:
        One station entry as returned by :func:`read_blq`.
    epoch:
        Observation epoch (``datetime`` or ``numpy.datetime64``).

    Returns
    -------
    ndarray
        ``(3,)`` ENU displacement in meters.
    """
    t = _t_centuries_tt(epoch)
    fhr = _fractional_hours_ut(epoch)
    S, H, P, ZNS, PS, TAU = _step2_fundamental_arguments(t, fhr)
    elements = np.array([TAU, S, H, P, ZNS, PS])
    theta_deg = _OTL_ARG_COEFS @ elements    # (11,)
    theta = np.radians(theta_deg)

    radial = float(np.sum(
        blq_entry["amp_radial"]
        * np.cos(theta - np.radians(blq_entry["phase_radial"]))
    ))
    west = float(np.sum(
        blq_entry["amp_west"]
        * np.cos(theta - np.radians(blq_entry["phase_west"]))
    ))
    south = float(np.sum(
        blq_entry["amp_south"]
        * np.cos(theta - np.radians(blq_entry["phase_south"]))
    ))
    return np.array([-west, -south, radial])


def ocean_tide_loading_ecef(
    blq_entry: dict[str, Any],
    station_ecef,
    epoch,
) -> np.ndarray:
    """ECEF ocean-tide-loading displacement at a station.

    Computes the ENU displacement via :func:`ocean_tide_loading_displacement`
    then rotates into ECEF using the station's geodetic latitude /
    longitude.

    Parameters
    ----------
    blq_entry:
        Station entry from :func:`read_blq`.
    station_ecef:
        ``(3,)`` station position in meters ECEF.
    epoch:
        Observation epoch.

    Returns
    -------
    ndarray
        ``(3,)`` ECEF displacement in meters.
    """
    from .geodesy import ecef_to_lla, enu_to_ecef
    station = np.asarray(station_ecef, dtype=float)
    enu = ocean_tide_loading_displacement(blq_entry, epoch)
    return enu_to_ecef(enu, station) - station


__all__ = [
    "ocean_tide_loading_displacement",
    "ocean_tide_loading_ecef",
    "read_blq",
]
