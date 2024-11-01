"""Solid-earth tide station displacement (IERS Conventions 2010, simplified).

For sub-cm Precise Point Positioning the station coordinates need to be
corrected for the tidal deformation of the Earth's crust under the
direct attraction of the sun and the moon. The dominant degree-2
contribution (IERS Conventions 2010 section 7.1.1) is:

    dr = sum over body in (sun, moon) of
         (GM_body / GM_Earth) * r^4 / R_body^3
         * { h2 * (3/2 * (R_hat . r_hat)^2 - 1/2) * r_hat
            + 3 * l2 * (R_hat . r_hat) * (R_hat - (R_hat . r_hat) * r_hat) }

with ``h2 = 0.6078`` (degree-2 Love number, radial) and
``l2 = 0.0847`` (degree-2 Shida number, horizontal). The peak vertical
displacement is ~30-50 cm at mid-latitudes; the horizontal component
is ~5 cm.

This module ships the geometry helper plus low-precision sun and moon
ECEF positions (~ 1 arcmin and ~ few hundred km accurate respectively,
which is plenty for a model whose internal precision is mm-class).
Callers that want higher accuracy should pass their own
``sun_ecef`` / ``moon_ecef`` from a JPL ephemeris or astropy.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

#: Gravitational parameters (m^3 / s^2) and Earth radius (m), WGS-84-ish.
GM_SUN = 1.32712442099e20
GM_MOON = 4.9048695e12
GM_EARTH = 3.986004418e14
R_EARTH = 6378137.0

#: Degree-2 Love + Shida numbers from IERS Conventions 2010.
H2_LOVE = 0.6078
L2_SHIDA = 0.0847


def _julian_date(epoch) -> float:
    """Julian date (UTC days from 4713 BC noon TT) of a datetime."""
    if isinstance(epoch, np.datetime64):
        epoch = epoch.astype("datetime64[us]").tolist()
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    epoch_utc = epoch.astimezone(timezone.utc)
    y = epoch_utc.year
    m = epoch_utc.month
    d = (
        epoch_utc.day
        + (epoch_utc.hour + (epoch_utc.minute + epoch_utc.second / 60.0) / 60.0) / 24.0
    )
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + B - 1524.5


def _gmst_rad(jd: float) -> float:
    """Greenwich Mean Sidereal Time (radians) at the given Julian date."""
    T = (jd - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    return (gmst_sec / 86400.0 * 2.0 * math.pi) % (2.0 * math.pi)


def _eci_to_ecef(pos_eci: np.ndarray, jd: float) -> np.ndarray:
    """Rotate a position from ECI to ECEF by GMST around z."""
    gmst = _gmst_rad(jd)
    c = math.cos(gmst)
    s = math.sin(gmst)
    return np.array(
        [
            c * pos_eci[0] + s * pos_eci[1],
            -s * pos_eci[0] + c * pos_eci[1],
            pos_eci[2],
        ]
    )


def sun_position_ecef(epoch) -> np.ndarray:
    """Approximate sun ECEF position in meters (~ 1 arcmin accurate).

    Uses the Vallado low-precision formula, accurate to within a few
    arcminutes through 2050.
    """
    jd = _julian_date(epoch)
    T = (jd - 2451545.0) / 36525.0
    lambda_M = math.radians(280.460 + 36000.771 * T)
    M = math.radians(357.5291092 + 35999.05034 * T)
    lambda_ec = lambda_M + math.radians(
        1.914666471 * math.sin(M) + 0.019994643 * math.sin(2.0 * M)
    )
    r_au = (
        1.000140612
        - 0.016708617 * math.cos(M)
        - 0.000139589 * math.cos(2.0 * M)
    )
    r_m = r_au * 1.495978707e11
    eps = math.radians(23.439291 - 0.0130042 * T)
    x = r_m * math.cos(lambda_ec)
    y = r_m * math.cos(eps) * math.sin(lambda_ec)
    z = r_m * math.sin(eps) * math.sin(lambda_ec)
    return _eci_to_ecef(np.array([x, y, z]), jd)


def moon_position_ecef(epoch) -> np.ndarray:
    """Approximate moon ECEF position in meters (~ few hundred km accurate).

    Six-term harmonic series for ecliptic longitude / latitude / parallax,
    from Vallado. Good enough for solid earth tide modelling.
    """
    jd = _julian_date(epoch)
    T = (jd - 2451545.0) / 36525.0
    lambda_ec = math.radians(
        218.32 + 481267.8813 * T
        + 6.29 * math.sin(math.radians(134.9 + 477198.85 * T))
        - 1.27 * math.sin(math.radians(259.2 - 413335.38 * T))
        + 0.66 * math.sin(math.radians(235.7 + 890534.23 * T))
        + 0.21 * math.sin(math.radians(269.9 + 954397.70 * T))
        - 0.19 * math.sin(math.radians(357.5 + 35999.05 * T))
        - 0.11 * math.sin(math.radians(186.6 + 966404.05 * T))
    )
    beta = math.radians(
        5.13 * math.sin(math.radians(93.3 + 483202.03 * T))
        + 0.28 * math.sin(math.radians(228.2 + 960400.87 * T))
        - 0.28 * math.sin(math.radians(318.3 + 6003.18 * T))
        - 0.17 * math.sin(math.radians(217.6 - 407332.20 * T))
    )
    parallax = math.radians(
        0.9508
        + 0.0518 * math.cos(math.radians(134.9 + 477198.85 * T))
        + 0.0095 * math.cos(math.radians(259.2 - 413335.38 * T))
        + 0.0078 * math.cos(math.radians(235.7 + 890534.23 * T))
        + 0.0028 * math.cos(math.radians(269.9 + 954397.70 * T))
    )
    r_m = R_EARTH / math.sin(parallax)
    eps = math.radians(23.439291 - 0.0130042 * T)
    x = r_m * math.cos(beta) * math.cos(lambda_ec)
    y = r_m * (
        math.cos(eps) * math.cos(beta) * math.sin(lambda_ec)
        - math.sin(eps) * math.sin(beta)
    )
    z = r_m * (
        math.sin(eps) * math.cos(beta) * math.sin(lambda_ec)
        + math.cos(eps) * math.sin(beta)
    )
    return _eci_to_ecef(np.array([x, y, z]), jd)


def solid_earth_tide_displacement(
    station_ecef: np.ndarray,
    *,
    sun_ecef: np.ndarray | None = None,
    moon_ecef: np.ndarray | None = None,
    epoch: datetime | np.datetime64 | None = None,
    h2: float = H2_LOVE,
    l2: float = L2_SHIDA,
) -> np.ndarray:
    """Solid-earth tide station displacement in ECEF, in meters.

    Implements IERS Conventions 2010 section 7.1.1 step-1 degree-2:
    the radial (h2 * Legendre polynomial 2) and horizontal (l2)
    contributions from direct attraction by the sun and moon. The full
    convention has frequency-dependent step-2 corrections and small
    higher-degree terms; those are sub-mm to a few mm and are not
    included here.

    Parameters
    ----------
    station_ecef:
        ``(3,)`` station position in meters ECEF.
    sun_ecef, moon_ecef:
        Optional sun and moon ECEF positions in meters. If either is
        omitted, ``epoch`` must be given and the function falls back to
        the approximate :func:`sun_position_ecef` /
        :func:`moon_position_ecef` helpers.
    epoch:
        Observation epoch. Required if ``sun_ecef`` or ``moon_ecef`` is
        not supplied.
    h2, l2:
        Love and Shida numbers. Defaults match IERS Conventions 2010.

    Returns
    -------
    ndarray
        ``(3,)`` ECEF displacement vector in meters. Add to the nominal
        station coordinates to get the tidally-displaced position;
        subtract from observed coordinates to get the tide-free
        coordinates.

    Raises
    ------
    ValueError
        If neither (``sun_ecef`` AND ``moon_ecef``) nor ``epoch`` is
        provided.
    """
    if sun_ecef is None or moon_ecef is None:
        if epoch is None:
            raise ValueError(
                "Either pass both sun_ecef and moon_ecef, or pass epoch "
                "to fall back to the approximate body-position helpers."
            )
        if sun_ecef is None:
            sun_ecef = sun_position_ecef(epoch)
        if moon_ecef is None:
            moon_ecef = moon_position_ecef(epoch)

    r = np.asarray(station_ecef, dtype=float)
    r_norm = float(np.linalg.norm(r))
    r_hat = r / r_norm

    dr = np.zeros(3)
    for body_pos, gm_body in (
        (np.asarray(sun_ecef, dtype=float), GM_SUN),
        (np.asarray(moon_ecef, dtype=float), GM_MOON),
    ):
        body_norm = float(np.linalg.norm(body_pos))
        body_hat = body_pos / body_norm
        cos_psi = float(np.dot(body_hat, r_hat))
        factor = gm_body * r_norm ** 4 / (GM_EARTH * body_norm ** 3)
        radial = h2 * (1.5 * cos_psi * cos_psi - 0.5) * r_hat
        horizontal = 3.0 * l2 * cos_psi * (body_hat - cos_psi * r_hat)
        dr = dr + factor * (radial + horizontal)
    return dr


__all__ = [
    "GM_EARTH",
    "GM_MOON",
    "GM_SUN",
    "H2_LOVE",
    "L2_SHIDA",
    "R_EARTH",
    "moon_position_ecef",
    "solid_earth_tide_displacement",
    "sun_position_ecef",
]
