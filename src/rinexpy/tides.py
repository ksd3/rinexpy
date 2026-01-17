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


# IERS 2010 Conventions Table 7.3a (diurnal band) coefficients, ported
# directly from the reference Fortran STEP2DIU.F (Mathews, Dehant, Gipson
# 1997 / IERS Conventions Center). Columns:
#   (n_s, n_h, n_p, n_zns, n_ps, dR_ip, dR_op, dT_ip, dT_op)
# Phase angle theta_f = TAU + n_s*S + n_h*H + n_p*P + n_zns*ZNS + n_ps*PS,
# amplitudes in millimeters.
_STEP2_DIURNAL = np.array([
    (-3, 0, 2, 0, 0,  -0.01,  0.00,  0.00,  0.00),
    (-3, 2, 0, 0, 0,  -0.01,  0.00,  0.00,  0.00),
    (-2, 0, 1,-1, 0,  -0.02,  0.00,  0.00,  0.00),
    (-2, 0, 1, 0, 0,  -0.08,  0.00, -0.01,  0.01),
    (-2, 2,-1, 0, 0,  -0.02,  0.00,  0.00,  0.00),
    (-1, 0, 0,-1, 0,  -0.10,  0.00,  0.00,  0.00),
    (-1, 0, 0, 0, 0,  -0.51,  0.00, -0.02,  0.03),
    (-1, 2, 0, 0, 0,   0.01,  0.00,  0.00,  0.00),
    ( 0,-2, 1, 0, 0,   0.01,  0.00,  0.00,  0.00),
    ( 0, 0,-1, 0, 0,   0.02,  0.00,  0.00,  0.00),
    ( 0, 0, 1, 0, 0,   0.06,  0.00,  0.00,  0.00),
    ( 0, 0, 1, 1, 0,   0.01,  0.00,  0.00,  0.00),
    ( 0, 2,-1, 0, 0,   0.01,  0.00,  0.00,  0.00),
    ( 1,-3, 0, 0, 1,  -0.06,  0.00,  0.00,  0.00),
    ( 1,-2, 0,-1, 0,   0.01,  0.00,  0.00,  0.00),
    ( 1,-2, 0, 0, 0,  -1.23, -0.07,  0.06,  0.01),
    ( 1,-1, 0, 0,-1,   0.02,  0.00,  0.00,  0.00),
    ( 1,-1, 0, 0, 1,   0.04,  0.00,  0.00,  0.00),
    ( 1, 0, 0,-1, 0,  -0.22,  0.01,  0.01,  0.00),
    ( 1, 0, 0, 0, 0,  12.00, -0.80, -0.67, -0.03),
    ( 1, 0, 0, 1, 0,   1.73, -0.12, -0.10,  0.00),
    ( 1, 0, 0, 2, 0,  -0.04,  0.00,  0.00,  0.00),
    ( 1, 1, 0, 0,-1,  -0.50, -0.01,  0.03,  0.00),
    ( 1, 1, 0, 0, 1,   0.01,  0.00,  0.00,  0.00),
    ( 0, 1, 0, 1,-1,  -0.01,  0.00,  0.00,  0.00),
    ( 1, 2,-2, 0, 0,  -0.01,  0.00,  0.00,  0.00),
    ( 1, 2, 0, 0, 0,  -0.11,  0.01,  0.01,  0.00),
    ( 2,-2, 1, 0, 0,  -0.01,  0.00,  0.00,  0.00),
    ( 2, 0,-1, 0, 0,  -0.02,  0.00,  0.00,  0.00),
    ( 3, 0, 0, 0, 0,   0.00,  0.00,  0.00,  0.00),
    ( 3, 0, 0, 1, 0,   0.00,  0.00,  0.00,  0.00),
], dtype=float)

# IERS 2010 Conventions Table 7.3b (long-period band) coefficients, ported
# from STEP2LON.F. Note the column order differs from STEP2DIU:
#   (n_s, n_h, n_p, n_zns, n_ps, dR_ip, dT_ip, dR_op, dT_op)
# Phase angle theta_f = n_s*S + n_h*H + n_p*P + n_zns*ZNS + n_ps*PS (no TAU).
_STEP2_LONG_PERIOD = np.array([
    (0, 0, 0, 1, 0,  0.47,  0.23,  0.16,  0.07),
    (0, 2, 0, 0, 0, -0.20, -0.12, -0.11, -0.05),
    (1, 0,-1, 0, 0, -0.11, -0.08, -0.09, -0.04),
    (2, 0, 0, 0, 0, -0.13, -0.11, -0.15, -0.07),
    (2, 0, 0, 1, 0, -0.05, -0.05, -0.06, -0.03),
], dtype=float)


def _t_centuries_tt(epoch) -> float:
    """TT centuries since J2000. Approximates TT - UTC = 0 (the resulting
    error in the slow fundamental arguments is ~ 1e-3 arcsec, well below
    the sub-mm amplitudes in the step-2 tables)."""
    return (_julian_date(epoch) - 2451545.0) / 36525.0


def _fractional_hours_ut(epoch) -> float:
    """UT fractional hours of the day from a datetime / numpy.datetime64."""
    if isinstance(epoch, np.datetime64):
        epoch = epoch.astype("datetime64[us]").tolist()
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    epoch = epoch.astimezone(timezone.utc)
    return epoch.hour + epoch.minute / 60.0 + (
        epoch.second + epoch.microsecond * 1e-6
    ) / 3600.0


def _step2_fundamental_arguments(t: float, fhr: float):
    """Return (S, H, P, ZNS, PS, TAU) in degrees, reduced to [0, 360).

    Implements the Brown / Bretagnon mean-element series exactly as in
    STEP2DIU.F lines 119-158: ``S`` is the moon's mean longitude,
    ``H`` the sun's mean longitude, ``P`` the longitude of the moon's
    perigee, ``ZNS`` the longitude of the moon's ascending node, ``PS``
    the longitude of the sun's perigee, and ``TAU`` the lunar hour angle.
    """
    S = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * t) * t) * t
    TAU = (
        fhr * 15.0
        + 280.4606184
        + (36000.7700536 + (0.00038793 + (-0.0000000258) * t) * t) * t
        - S
    )
    PR = (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007 * t) * t) * t) * t
    S = S + PR
    H = 280.46645 + (
        36000.7697489 + (0.00030322222 + (0.000000020 + (-0.00000000654) * t) * t) * t
    ) * t
    P = 83.35324312 + (
        4069.01363525
        + (-0.01032172222 + (-0.0000124991 + 0.00000005263 * t) * t) * t
    ) * t
    ZNS = 234.95544499 + (
        1934.13626197
        + (-0.00207561111 + (-0.00000213944 + 0.00000001650 * t) * t) * t
    ) * t
    PS = 282.93734098 + (
        1.71945766667
        + (0.00045688889 + (-0.00000001778 + (-0.00000000334) * t) * t) * t
    ) * t
    return S % 360.0, H % 360.0, P % 360.0, ZNS % 360.0, PS % 360.0, TAU % 360.0


def _step2_diurnal_core(station_ecef: np.ndarray, t: float, fhr: float) -> np.ndarray:
    """Diurnal-band step-2 displacement, low-level (T, FHR) interface.

    Direct Python port of STEP2DIU.F. Inputs ``t`` (TT centuries since
    J2000) and ``fhr`` (UT fractional hours) are decoupled from the
    epoch representation so the IERS reference test case can be hit
    exactly.
    """
    station = np.asarray(station_ecef, dtype=float)
    S, H, P, ZNS, PS, TAU = _step2_fundamental_arguments(t, fhr)
    rsta = float(np.linalg.norm(station))
    sinphi = station[2] / rsta
    cosphi = math.sqrt(station[0] ** 2 + station[1] ** 2) / rsta
    cosla = station[0] / (cosphi * rsta)
    sinla = station[1] / (cosphi * rsta)
    zla = math.atan2(station[1], station[0])

    xc = np.zeros(3)
    deg = math.pi / 180.0
    sin2pc = 2.0 * sinphi * cosphi
    csm = cosphi ** 2 - sinphi ** 2
    for row in _STEP2_DIURNAL:
        c_s, c_h, c_p, c_zns, c_ps, dRin, dRout, dTin, dTout = row
        thetaf = (TAU + c_s * S + c_h * H + c_p * P + c_zns * ZNS + c_ps * PS) * deg
        a = thetaf + zla
        sa = math.sin(a)
        ca = math.cos(a)
        dr = dRin * sin2pc * sa + dRout * sin2pc * ca
        dn = dTin * csm * sa + dTout * csm * ca
        de = dTin * sinphi * ca - dTout * sinphi * sa
        xc[0] += dr * cosla * cosphi - de * sinla - dn * sinphi * cosla
        xc[1] += dr * sinla * cosphi + de * cosla - dn * sinphi * sinla
        xc[2] += dr * sinphi + dn * cosphi
    return xc / 1000.0


def _step2_long_period_core(station_ecef: np.ndarray, t: float) -> np.ndarray:
    """Long-period-band step-2 displacement, low-level T-only interface.

    Direct Python port of STEP2LON.F. The east component is identically
    zero in the long-period band (zonal tides only deform radially and
    meridionally).
    """
    station = np.asarray(station_ecef, dtype=float)
    # TAU drops out of the long-period band; pass a dummy fhr.
    S, H, P, ZNS, PS, _ = _step2_fundamental_arguments(t, 0.0)
    rsta = float(np.linalg.norm(station))
    sinphi = station[2] / rsta
    cosphi = math.sqrt(station[0] ** 2 + station[1] ** 2) / rsta
    cosla = station[0] / (cosphi * rsta)
    sinla = station[1] / (cosphi * rsta)

    xc = np.zeros(3)
    deg = math.pi / 180.0
    p2 = (3.0 * sinphi * sinphi - 1.0) / 2.0
    sin2pc = 2.0 * sinphi * cosphi
    for row in _STEP2_LONG_PERIOD:
        c_s, c_h, c_p, c_zns, c_ps, dRin, dTin, dRout, dTout = row
        thetaf = (c_s * S + c_h * H + c_p * P + c_zns * ZNS + c_ps * PS) * deg
        ct = math.cos(thetaf)
        st = math.sin(thetaf)
        dr = dRin * p2 * ct + dRout * p2 * st
        dn = dTin * sin2pc * ct + dTout * sin2pc * st
        # east component is zero in this band
        xc[0] += dr * cosla * cosphi - dn * sinphi * cosla
        xc[1] += dr * sinla * cosphi - dn * sinphi * sinla
        xc[2] += dr * sinphi + dn * cosphi
    return xc / 1000.0


def step2_diurnal_displacement(
    station_ecef: np.ndarray,
    epoch: "datetime | np.datetime64",
) -> np.ndarray:
    """Step-2 frequency-dependent correction in the diurnal band.

    Full 31-row table from IERS Conventions 2010 (Mathews/Dehant/Gipson
    1997), ported from the reference Fortran ``STEP2DIU.F``. Peak
    contribution is the K1 row at ~12 mm, almost all of which is
    cancelled by the step-1 nominal K1 amplitude; what's left is the
    in-phase / out-of-phase anelasticity correction (sub-mm to a few
    mm depending on row, latitude and epoch).

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
    return _step2_diurnal_core(
        station_ecef, _t_centuries_tt(epoch), _fractional_hours_ut(epoch)
    )


def step2_long_period_displacement(
    station_ecef: np.ndarray,
    epoch: "datetime | np.datetime64",
) -> np.ndarray:
    """Step-2 frequency-dependent correction in the long-period band.

    Full 5-row table from IERS Conventions 2010 (Mathews/Dehant/Gipson
    1997), ported from the reference Fortran ``STEP2LON.F``. Captures
    the Mf, Mm, Ssa annual and semi-annual zonal contributions, each
    well under 1 mm amplitude.

    Parameters
    ----------
    station_ecef:
        ``(3,)`` ECEF station position in meters.
    epoch:
        Observation epoch (``datetime`` or ``numpy.datetime64``).

    Returns
    -------
    ndarray
        ``(3,)`` ECEF displacement in meters. The east component is zero
        in this band.
    """
    return _step2_long_period_core(station_ecef, _t_centuries_tt(epoch))


def step2_displacement(
    station_ecef: np.ndarray,
    epoch: "datetime | np.datetime64",
) -> np.ndarray:
    """Total step-2 displacement: diurnal + long-period bands.

    Equivalent to summing :func:`step2_diurnal_displacement` and
    :func:`step2_long_period_displacement`; the convenience wrapper
    matches the structure of the IERS reference ``DEHANTTIDEINEL.F``
    which calls STEP2DIU followed by STEP2LON and accumulates both.
    """
    return step2_diurnal_displacement(station_ecef, epoch) + step2_long_period_displacement(
        station_ecef, epoch
    )


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


def ocean_pole_tide_displacement(
    station_ecef: np.ndarray,
    eop,
    epoch,
) -> np.ndarray:
    """Ocean pole tide loading displacement (IERS 2010 section 7.1.5).

    The ocean pole tide is the ocean's response to the centrifugal
    bulge that wobbles with polar motion. Following the simplified
    Desai 2002 / IERS 2010 §7.1.5 analytical approximation:

        delta_radial =  ( K_r * (m1 * cos(lon) + m2 * sin(lon)) ) * P(lat)
        delta_north  =  ( K_n * (m1 * cos(lon) + m2 * sin(lon)) ) * Q(lat)
        delta_east   = -( K_e * (m1 * sin(lon) - m2 * cos(lon)) ) * Q(lat)

    where ``K_r = -2.10 mm``, ``K_n = -0.41 mm``, ``K_e = 0.41 mm``
    are the standard IERS 2010 amplitude coefficients (the gridded
    Desai 2002 coefficients refine these on a 1-degree grid; the
    spatial-mean values used here are accurate to ~10 % of the
    amplitude). ``P(lat) = sin(2 phi)``, ``Q(lat) = cos(phi)``.

    Peak amplitude is ~1 mm at mid-latitudes - an order of magnitude
    smaller than the solid-earth pole tide. Required only for sub-cm
    PPP / VLBI work.

    Parameters
    ----------
    station_ecef:
        ``(3,)`` ECEF station position (m).
    eop:
        EOP dataset (from :func:`rinexpy.eop.load_eop`) providing
        ``x``, ``y`` polar-motion components.
    epoch:
        Observation epoch.

    Returns
    -------
    ndarray
        ``(3,)`` ECEF displacement (m).
    """
    from .eop import interp_eop
    from .geodesy import ecef_to_lla, enu_to_ecef
    station = np.asarray(station_ecef, dtype=float)
    lat_deg, lon_deg, _ = ecef_to_lla(*station)
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    e = interp_eop(eop, epoch)
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
    m1 = e["x"] - x_mean
    m2 = -(e["y"] - y_mean)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)
    K_R = -2.10e-3   # m / arcsec
    K_N = -0.41e-3
    K_E = +0.41e-3
    longitudinal_in = m1 * cos_lam + m2 * sin_lam
    longitudinal_out = m1 * sin_lam - m2 * cos_lam
    d_up = K_R * math.sin(2.0 * phi) * longitudinal_in
    d_north = K_N * math.cos(phi) * longitudinal_in
    d_east = -K_E * math.cos(phi) * longitudinal_out
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
    "ocean_pole_tide_displacement",
    "pole_tide_displacement",
    "solid_earth_tide_displacement",
    "step2_diurnal_displacement",
    "step2_displacement",
    "step2_long_period_displacement",
    "sun_position_ecef",
]
