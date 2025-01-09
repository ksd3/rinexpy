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


def step2_k1_displacement(
    station_ecef: np.ndarray,
    epoch: "datetime | np.datetime64",
) -> np.ndarray:
    """K1 diurnal frequency-dependent correction (IERS 2010 Table 7.3a row 1).

    This is the dominant frequency-dependent correction layered on top
    of :func:`solid_earth_tide_displacement` (which is step-1, the
    in-phase degree-2 tide). The K1 amplitude is sub-mm at mid-latitudes
    and the other 10 diurnal-band entries plus 5 long-period entries in
    the full Table 7.3a / 7.3b are all under 0.4 mm; we ship the K1
    term to expose the framework and explicitly document that the rest
    is deferred (each follows the same pattern of a Doodson argument
    multiplied by tabulated in-phase / out-of-phase amplitudes).

    Formula (IERS Conventions 2010 equation 7.12, diurnal band):

        delta_radial = dR_K1 * sin(2*phi) * sin(theta + lambda)
        delta_north  = dT_K1 * cos(2*phi) * sin(theta + lambda)
        delta_east   = dT_K1 * sin(phi)   * cos(theta + lambda)

    with ``dR_K1 = -0.253 mm``, ``dT_K1 = -0.081 mm`` from IERS Table
    7.3a, ``phi`` the geodetic latitude, ``lambda`` the east longitude,
    and ``theta`` the Doodson argument approximated as GMST (the
    full argument includes ~arcsec corrections that are well below the
    sub-mm amplitudes).

    Parameters
    ----------
    station_ecef:
        ``(3,)`` ECEF station position in meters.
    epoch:
        Observation epoch (``datetime`` or ``numpy.datetime64``).

    Returns
    -------
    ndarray
        ``(3,)`` ECEF displacement in meters.
    """
    from .geodesy import ecef_to_lla, enu_to_ecef
    station = np.asarray(station_ecef, dtype=float)
    lat_deg, lon_deg, _ = ecef_to_lla(*station)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    jd = _julian_date(epoch)
    theta = _gmst_rad(jd) + lon

    dR_K1 = -0.253e-3
    dT_K1 = -0.081e-3
    d_up = dR_K1 * math.sin(2.0 * lat) * math.sin(theta)
    d_north = dT_K1 * math.cos(2.0 * lat) * math.sin(theta)
    d_east = dT_K1 * math.sin(lat) * math.cos(theta)

    enu = np.array([d_east, d_north, d_up])
    return enu_to_ecef(enu, station) - station


def pole_tide_displacement(
    station_ecef: np.ndarray,
    eop,
    epoch,
) -> np.ndarray:
    """Solid-earth pole tide displacement (IERS 2010 section 7.1.4).

    The pole tide is the elastic crustal response to the centrifugal
    bulge that wobbles with the Earth's polar motion. The displacement
    at the station is

        delta_radial = -33 mm * sin(2*phi) * (m1*cos(lon) + m2*sin(lon))
        delta_north  = -9  mm * cos(2*phi) * (m1*cos(lon) + m2*sin(lon))
        delta_east   = +9  mm *  cos(phi)  * (m1*sin(lon) - m2*cos(lon))

    where ``m1 = x_p - x_mean`` and ``m2 = -(y_p - y_mean)`` are the
    polar-motion deviations from the long-term mean pole position (in
    arcseconds), ``x_p`` / ``y_p`` come from the EOP series, and
    ``x_mean`` / ``y_mean`` follow the IERS 2010 linear-drift mean-pole
    model. Peak amplitude ~5 mm at mid-latitudes; required for sub-cm
    PPP.

    Parameters
    ----------
    station_ecef:
        ``(3,)`` ECEF station position in meters.
    eop:
        EOP dataset from :func:`rinexpy.eop.load_eop`.
    epoch:
        Observation epoch.

    Returns
    -------
    ndarray
        ``(3,)`` ECEF displacement in meters.
    """
    from .eop import interp_eop
    from .geodesy import ecef_to_lla, enu_to_ecef
    station = np.asarray(station_ecef, dtype=float)
    lat_deg, lon_deg, _ = ecef_to_lla(*station)
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    e = interp_eop(eop, epoch)
    # Linear-drift mean-pole model (IERS 2010 section 7.1.4, table 7.7).
    # t in years from 2000.0; for epochs before 2010 the pre-2010
    # coefficients apply, after 2010 the post-2010 set.
    if isinstance(epoch, np.datetime64):
        epoch_py = epoch.astype("datetime64[us]").tolist()
    else:
        epoch_py = epoch
    t_years = (epoch_py.year + epoch_py.month / 12.0) - 2000.0
    if t_years < 10.0:
        x_mean = 0.055 + 1.677e-3 * t_years
        y_mean = 0.3205 + 3.460e-3 * t_years
    else:
        x_mean = 0.0230 + 7.6e-3 * t_years
        y_mean = 0.3543 - 0.6e-3 * t_years
    m1 = e["x"] - x_mean    # arcseconds
    m2 = -(e["y"] - y_mean)
    s2p = math.sin(2.0 * phi)
    c2p = math.cos(2.0 * phi)
    cp = math.cos(phi)
    sl = math.sin(lam)
    cl = math.cos(lam)
    d_up = -33e-3 * s2p * (m1 * cl + m2 * sl)
    d_north = -9e-3 * c2p * (m1 * cl + m2 * sl)
    d_east = 9e-3 * cp * (m1 * sl - m2 * cl)
    enu = np.array([d_east, d_north, d_up])
    return enu_to_ecef(enu, station) - station


__all__ = [
    "GM_EARTH",
    "GM_MOON",
    "GM_SUN",
    "H2_LOVE",
    "L2_SHIDA",
    "R_EARTH",
    "moon_position_ecef",
    "pole_tide_displacement",
    "solid_earth_tide_displacement",
    "step2_k1_displacement",
    "sun_position_ecef",
]
