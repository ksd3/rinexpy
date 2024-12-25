"""Geodetic and ionospheric helpers used by the positioning code.

Three independent groups of functions:

- :func:`ecef_to_lla` / :func:`lla_to_ecef`: WGS-84 conversions.
- :func:`azimuth_elevation`: receiver-to-satellite azimuth/elevation.
- :func:`dop`: GDOP/PDOP/HDOP/VDOP/TDOP from satellite line-of-sight matrix.
- :func:`klobuchar`: GPS L1 ionospheric delay from broadcast 8-coef model.
"""

from __future__ import annotations

import numpy as np

# WGS-84 ellipsoid
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_B = _WGS84_A * (1 - _WGS84_F)
_WGS84_E2 = 1 - (_WGS84_B / _WGS84_A) ** 2

#: GPS L1 frequency in Hz.
_F1 = 1.57542e9


def ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert WGS-84 ECEF (m) to geodetic ``(lat, lon, alt)`` in degrees, m.

    Closed-form Bowring (1985). Accurate to ~mm well into orbit altitude.
    """
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    if p == 0:
        lat = np.pi / 2 if z > 0 else -np.pi / 2
        alt = abs(z) - _WGS84_B
        return np.degrees(lat), np.degrees(lon), alt
    theta = np.arctan2(z * _WGS84_A, p * _WGS84_B)
    lat = np.arctan2(
        z + _WGS84_E2 / (1 - _WGS84_E2) * _WGS84_B * np.sin(theta) ** 3,
        p - _WGS84_E2 * _WGS84_A * np.cos(theta) ** 3,
    )
    n = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def lla_to_ecef(lat_deg: float, lon_deg: float, alt: float) -> tuple[float, float, float]:
    """Convert WGS-84 ``(lat, lon, alt)`` (degrees, m) to ECEF (m)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - _WGS84_E2) + alt) * np.sin(lat)
    return float(x), float(y), float(z)


def azimuth_elevation(
    rx_ecef: tuple[float, float, float] | np.ndarray,
    sv_ecef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute azimuth and elevation (in degrees) from a receiver to one or
    more satellites.

    Parameters
    ----------
    rx_ecef:
        Receiver ECEF position ``(x, y, z)`` in meters.
    sv_ecef:
        Satellite ECEF positions, shape ``(..., 3)`` in meters.

    Returns
    -------
    az, el : ndarray
        Azimuth and elevation in degrees, with the leading shape of
        ``sv_ecef``. Azimuth is measured clockwise from north [0, 360).
        Elevation is in [-90, 90]; values < 0 are below the horizon.
    """
    rx = np.asarray(rx_ecef, dtype=float)
    sv = np.asarray(sv_ecef, dtype=float)
    los = sv - rx  # line of sight in ECEF
    lat_deg, lon_deg, _ = ecef_to_lla(*rx)
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # ENU rotation matrix.
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    east = -sl * los[..., 0] + cl * los[..., 1]
    north = -sp * cl * los[..., 0] - sp * sl * los[..., 1] + cp * los[..., 2]
    up = cp * cl * los[..., 0] + cp * sl * los[..., 1] + sp * los[..., 2]

    horiz = np.hypot(east, north)
    el = np.degrees(np.arctan2(up, horiz))
    az = np.degrees(np.arctan2(east, north)) % 360.0
    return az, el


def dop(sv_ecef: np.ndarray, rx_ecef: tuple[float, float, float]) -> dict[str, float]:
    """Compute GDOP/PDOP/HDOP/VDOP/TDOP from sat positions and receiver location.

    Parameters
    ----------
    sv_ecef:
        ``(n_sv, 3)`` ECEF satellite positions in meters.
    rx_ecef:
        Receiver ECEF position.

    Returns
    -------
    dict
        ``{"GDOP", "PDOP", "HDOP", "VDOP", "TDOP"}``. Returns NaN values
        if the geometry matrix is singular (e.g. fewer than 4 SVs).
    """
    rx = np.asarray(rx_ecef, dtype=float)
    sv = np.asarray(sv_ecef, dtype=float)
    if sv.shape[0] < 4:
        return dict.fromkeys(("GDOP", "PDOP", "HDOP", "VDOP", "TDOP"), float("nan"))

    los = rx - sv
    norms = np.linalg.norm(los, axis=1, keepdims=True)
    unit = los / norms
    # Geometry matrix in ENU; use the rotation that puts row 4 at clock.
    lat_deg, lon_deg, _ = ecef_to_lla(*rx)
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    rot = np.array(
        [
            [-sl, cl, 0.0],
            [-sp * cl, -sp * sl, cp],
            [cp * cl, cp * sl, sp],
        ]
    )
    enu = unit @ rot.T
    g = np.hstack([enu, np.ones((enu.shape[0], 1))])
    try:
        cov = np.linalg.inv(g.T @ g)
    except np.linalg.LinAlgError:
        return dict.fromkeys(("GDOP", "PDOP", "HDOP", "VDOP", "TDOP"), float("nan"))
    e2, n2, u2, t2 = float(cov[0, 0]), float(cov[1, 1]), float(cov[2, 2]), float(cov[3, 3])
    return {
        "GDOP": float(np.sqrt(e2 + n2 + u2 + t2)),
        "PDOP": float(np.sqrt(e2 + n2 + u2)),
        "HDOP": float(np.sqrt(e2 + n2)),
        "VDOP": float(np.sqrt(u2)),
        "TDOP": float(np.sqrt(t2)),
    }


def klobuchar(
    alpha: tuple[float, float, float, float],
    beta: tuple[float, float, float, float],
    rx_lla: tuple[float, float, float],
    sv_az_deg: float,
    sv_el_deg: float,
    gps_sec: float,
) -> float:
    """GPS broadcast Klobuchar ionospheric correction in meters at L1.

    Parameters
    ----------
    alpha, beta:
        4-coefficient broadcast iono parameters from the GPS NAV header
        (``ION ALPHA``, ``ION BETA``). Units per ICD-GPS-200.
    rx_lla:
        Receiver geodetic position ``(lat_deg, lon_deg, alt_m)``.
    sv_az_deg, sv_el_deg:
        Satellite azimuth and elevation (degrees) at the receiver.
    gps_sec:
        GPS seconds-of-week at observation time.

    Returns
    -------
    float
        L1 ionospheric delay in meters.
    """
    phi_u = rx_lla[0] / 180.0  # in semi-circles
    lambda_u = rx_lla[1] / 180.0
    el = sv_el_deg / 180.0
    a = sv_az_deg * np.pi / 180.0

    psi = 0.0137 / (el + 0.11) - 0.022
    phi_i = phi_u + psi * np.cos(a)
    phi_i = max(min(phi_i, 0.416), -0.416)
    lambda_i = lambda_u + psi * np.sin(a) / np.cos(phi_i * np.pi)
    phi_m = phi_i + 0.064 * np.cos((lambda_i - 1.617) * np.pi)
    t = 43200.0 * lambda_i + gps_sec
    t %= 86400
    if t < 0:
        t += 86400

    amp = sum(a_i * phi_m**i for i, a_i in enumerate(alpha))
    per = sum(b_i * phi_m**i for i, b_i in enumerate(beta))
    if amp < 0:
        amp = 0.0
    if per < 72000:
        per = 72000.0

    x = 2 * np.pi * (t - 50400) / per
    f = 1.0 + 16.0 * (0.53 - el) ** 3
    if abs(x) < 1.57:
        delay = f * (5e-9 + amp * (1 - x**2 / 2 + x**4 / 24))
    else:
        delay = f * 5e-9
    # Convert from seconds (GPS time delay) to meters via c.
    return float(delay * 299_792_458.0)


def standard_atmosphere(
    altitude_m: float,
) -> tuple[float, float, float]:
    """Return ICAO standard atmosphere (T_kelvin, P_hPa, e_water_hPa) at altitude.

    Parameters
    ----------
    altitude_m:
        Geodetic altitude in meters.

    Returns
    -------
    T, P, e:
        Temperature in Kelvin, pressure in hPa, partial water-vapour
        pressure in hPa. The water-vapour estimate is a coarse 50%-RH
        default at the surface temperature.
    """
    # Standard atmosphere lapse: T = T0 - 6.5e-3 * h, P = P0 * (T/T0)^5.26
    T0, P0 = 288.15, 1013.25
    T = T0 - 6.5e-3 * altitude_m
    P = P0 * (T / T0) ** 5.2561
    # 50% RH saturation vapour pressure (Magnus formula).
    es = 6.11 * 10 ** ((7.5 * (T - 273.15)) / (T - 35.85))
    e = 0.5 * es
    return T, P, e


def niell_mapping(
    el_deg: float,
    lat_deg: float,
    altitude_m: float,
    doy: int,
) -> tuple[float, float]:
    """Niell (1996) hydrostatic + wet mapping functions.

    Parameters
    ----------
    el_deg:
        Satellite elevation in degrees (must be > 0).
    lat_deg:
        Receiver latitude in degrees.
    altitude_m:
        Receiver altitude in meters above sea level.
    doy:
        Day-of-year (1-366), used for the seasonal interpolation of the
        hydrostatic coefficients.

    Returns
    -------
    m_dry, m_wet : float
        Dimensionless mapping factors. Multiply zenith dry/wet delays by
        these to get slant delays. Returns ``(inf, inf)`` at or below the
        horizon.

    Notes
    -----
    Implements the Niell Mapping Function (NMF) as published in
    A. Niell, "Global mapping functions for the atmosphere delay at
    radio wavelengths", JGR 101 (B2), 1996. Independent of any external
    weather grid, accurate to ~few mm at zenith and a few cm at el=5°.
    """
    if el_deg <= 0:
        return float("inf"), float("inf")

    # Latitude grid for hydrostatic coefficients (15° to 75°).
    lat_grid = np.array([15.0, 30.0, 45.0, 60.0, 75.0])
    # Average hydrostatic a, b, c (Niell Table 3).
    a_avg = np.array([1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3])
    b_avg = np.array([2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3])
    c_avg = np.array([62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3])
    # Amplitude (seasonal variation).
    a_amp = np.array([0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5])
    b_amp = np.array([0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5])
    c_amp = np.array([0.0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5])
    # Wet coefficients (Niell Table 4).
    aw = np.array([5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4])
    bw = np.array([1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3])
    cw = np.array([4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2])

    abs_lat = abs(lat_deg)
    # Cosine seasonal factor: phase 28 (Northern hemisphere); subtract
    # 0.5 yr for the Southern hemisphere.
    phase = (doy - 28) / 365.25 * 2 * np.pi
    if lat_deg < 0:
        phase += np.pi
    cos_phase = np.cos(phase)

    a = np.interp(abs_lat, lat_grid, a_avg) - cos_phase * np.interp(abs_lat, lat_grid, a_amp)
    b = np.interp(abs_lat, lat_grid, b_avg) - cos_phase * np.interp(abs_lat, lat_grid, b_amp)
    c = np.interp(abs_lat, lat_grid, c_avg) - cos_phase * np.interp(abs_lat, lat_grid, c_amp)

    el_rad = np.radians(el_deg)
    sin_e = np.sin(el_rad)
    m_dry_no_h = (1 + a / (1 + b / (1 + c))) / (sin_e + a / (sin_e + b / (sin_e + c)))

    # Height correction (Niell §4): 6378.137 km Earth radius, 6378137 m
    # height-correction coefficients.
    a_h = 2.53e-5
    b_h = 5.49e-3
    c_h = 1.14e-3
    dm = 1 / sin_e - (
        (1 + a_h / (1 + b_h / (1 + c_h))) / (sin_e + a_h / (sin_e + b_h / (sin_e + c_h)))
    )
    m_dry = m_dry_no_h + dm * (altitude_m / 1000.0)

    a_w = float(np.interp(abs_lat, lat_grid, aw))
    b_w = float(np.interp(abs_lat, lat_grid, bw))
    c_w = float(np.interp(abs_lat, lat_grid, cw))
    m_wet = (1 + a_w / (1 + b_w / (1 + c_w))) / (sin_e + a_w / (sin_e + b_w / (sin_e + c_w)))

    return float(m_dry), float(m_wet)


def vmf1(
    a_h: float,
    a_w: float,
    el_deg: float,
    lat_deg: float,
    altitude_m: float,
    doy: int,
) -> tuple[float, float]:
    """Vienna Mapping Function 1 (Boehm 2006) hydrostatic + wet mapping.

    Parameters
    ----------
    a_h, a_w:
        Hydrostatic and wet ``a`` coefficients at the receiver, typically
        from :func:`rinexpy.gpt2w` or the official VMF1 grid file.
    el_deg:
        Satellite elevation angle in degrees.
    lat_deg:
        Receiver latitude in degrees (used in the seasonal ``c_h`` term).
    altitude_m:
        Receiver altitude in meters above sea level (height correction
        for the hydrostatic mapping).
    doy:
        Day-of-year (1-366) for the seasonal term in ``c_h``.

    Returns
    -------
    m_h, m_w : float
        Dimensionless mapping factors. Multiply zenith dry/wet delays
        by these to get slant delays. Returns ``(inf, inf)`` at or
        below the horizon.

    Notes
    -----
    VMF1 is the IGS standard for cm-precision tropospheric correction.
    The ``a`` coefficients are time-varying and location-specific (use
    GPT2w empirical or VMF1 grid product); ``b``/``c`` coefficients are
    global constants except for the latitudinal+seasonal ``c_h`` term
    (Boehm et al., 2006, eq. 4-7).
    """
    if el_deg <= 0:
        return float("inf"), float("inf")
    el_rad = np.radians(el_deg)
    sin_e = np.sin(el_rad)

    b_h = 0.0029
    b_w = 0.00146
    c_w = 0.04391

    # Latitudinal + seasonal c_h (eq. 7).
    if lat_deg < 0:
        psi = np.pi
        c10_h = 0.002
        c11_h = 0.007
    else:
        psi = 0.0
        c10_h = 0.001
        c11_h = 0.005
    c0_h = 0.062
    cos_phase = np.cos((doy - 28) / 365.25 * 2 * np.pi + psi)
    c_h = c0_h + ((cos_phase + 1) * c11_h / 2 + c10_h) * (1 - np.cos(np.radians(lat_deg)))

    def _cf(a: float, b: float, c: float) -> float:
        """Marini-style continued-fraction mapping evaluator."""
        return (1 + a / (1 + b / (1 + c))) / (sin_e + a / (sin_e + b / (sin_e + c)))

    m_h = _cf(a_h, b_h, c_h)

    # Hydrostatic height correction (Niell §4 / Boehm §3).
    a_ht, b_ht, c_ht = 2.53e-5, 5.49e-3, 1.14e-3
    dm = 1 / sin_e - _cf(a_ht, b_ht, c_ht)
    m_h += dm * (altitude_m / 1000.0)

    m_w = _cf(a_w, b_w, c_w)
    return float(m_h), float(m_w)


def saastamoinen(
    el_deg: float,
    altitude_m: float,
    *,
    pressure_hpa: float | None = None,
    temperature_k: float | None = None,
    humidity_e_hpa: float | None = None,
) -> float:
    """Saastamoinen tropospheric delay model in meters.

    Parameters
    ----------
    el_deg:
        Satellite elevation angle in degrees (must be > 0).
    altitude_m:
        Receiver altitude in meters above the WGS-84 ellipsoid.
    pressure_hpa, temperature_k, humidity_e_hpa:
        Surface pressure (hPa), temperature (K), and partial water-vapour
        pressure (hPa). Any unspecified value is filled from
        :func:`standard_atmosphere` at ``altitude_m``.

    Returns
    -------
    float
        Slant tropospheric delay in meters. Returns ``inf`` for elevations
        at or below the horizon (the cot z term blows up).

    Notes
    -----
    Implements the classical Saastamoinen (1972) wet+dry slant model:

        d = (0.002277 / cos z) * [P + (1255/T + 0.05) * e - tan^2 z]

    where z = 90 - el. Accurate to ~1 cm for el >= 15 degrees; the
    elevation-mapping degenerates near the horizon (use Niell or VMF1
    for el < 5 degrees in production).
    """
    if el_deg <= 0:
        return float("inf")
    T0, P0, e0 = standard_atmosphere(altitude_m)
    P = pressure_hpa if pressure_hpa is not None else P0
    T = temperature_k if temperature_k is not None else T0
    e = humidity_e_hpa if humidity_e_hpa is not None else e0
    z = np.radians(90.0 - el_deg)
    cos_z = np.cos(z)
    if cos_z <= 0:
        return float("inf")
    tan_z2 = np.tan(z) ** 2
    return float(0.002277 / cos_z * (P + (1255.0 / T + 0.05) * e - tan_z2))


def _enu_rotation(lat_rad: float, lon_rad: float) -> np.ndarray:
    """ECEF -> ENU rotation matrix at the given geodetic lat/lon."""
    sl, cl = np.sin(lat_rad), np.cos(lat_rad)
    sg, cg = np.sin(lon_rad), np.cos(lon_rad)
    return np.array(
        [
            [-sg, cg, 0.0],
            [-sl * cg, -sl * sg, cl],
            [cl * cg, cl * sg, sl],
        ]
    )


def ecef_to_enu(
    target_ecef,
    ref_ecef,
) -> np.ndarray:
    """Convert one or more ECEF positions to a local East-North-Up frame.

    The local frame is anchored at ``ref_ecef`` (a geodetic point). The
    returned ENU vector is the displacement from ``ref_ecef`` to
    ``target_ecef`` expressed in (east, north, up) meters.

    Parameters
    ----------
    target_ecef:
        ``(3,)`` ECEF point or ``(n, 3)`` array of points in meters.
    ref_ecef:
        ``(3,)`` ECEF reference position in meters.

    Returns
    -------
    ndarray
        Same shape as ``target_ecef``: a 3-vector or ``(n, 3)`` array of
        ENU displacements in meters.
    """
    ref = np.asarray(ref_ecef, dtype=float)
    if ref.shape != (3,):
        raise ValueError(f"ref_ecef must have shape (3,), got {ref.shape}")
    tgt = np.asarray(target_ecef, dtype=float)
    lat_deg, lon_deg, _ = ecef_to_lla(ref[0], ref[1], ref[2])
    R = _enu_rotation(np.deg2rad(lat_deg), np.deg2rad(lon_deg))
    if tgt.ndim == 1:
        if tgt.shape != (3,):
            raise ValueError(f"target_ecef must have shape (3,) or (n, 3), got {tgt.shape}")
        return R @ (tgt - ref)
    if tgt.ndim == 2 and tgt.shape[1] == 3:
        return (R @ (tgt - ref).T).T
    raise ValueError(
        f"target_ecef must have shape (3,) or (n, 3), got {tgt.shape}"
    )


def enu_to_ecef(
    enu,
    ref_ecef,
) -> np.ndarray:
    """Convert a local ENU displacement back to an ECEF position.

    Inverse of :func:`ecef_to_enu`.

    Parameters
    ----------
    enu:
        ``(3,)`` or ``(n, 3)`` ENU displacement in meters.
    ref_ecef:
        ``(3,)`` ECEF reference position in meters.

    Returns
    -------
    ndarray
        ECEF point or array of points in meters.
    """
    ref = np.asarray(ref_ecef, dtype=float)
    if ref.shape != (3,):
        raise ValueError(f"ref_ecef must have shape (3,), got {ref.shape}")
    e = np.asarray(enu, dtype=float)
    lat_deg, lon_deg, _ = ecef_to_lla(ref[0], ref[1], ref[2])
    R = _enu_rotation(np.deg2rad(lat_deg), np.deg2rad(lon_deg))
    if e.ndim == 1:
        if e.shape != (3,):
            raise ValueError(f"enu must have shape (3,) or (n, 3), got {e.shape}")
        return ref + R.T @ e
    if e.ndim == 2 and e.shape[1] == 3:
        return ref + (R.T @ e.T).T
    raise ValueError(f"enu must have shape (3,) or (n, 3), got {e.shape}")


def _julian_date_utc(epoch) -> float:
    """Julian date (UTC) for a datetime / numpy.datetime64 / aware datetime."""
    from datetime import datetime, timezone
    if isinstance(epoch, np.datetime64):
        epoch = epoch.astype("datetime64[us]").tolist()
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    epoch = epoch.astimezone(timezone.utc)
    y = epoch.year
    m = epoch.month
    d = (
        epoch.day
        + (epoch.hour + (epoch.minute + epoch.second / 60.0) / 60.0) / 24.0
    )
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    import math
    return (
        math.floor(365.25 * (y + 4716))
        + math.floor(30.6001 * (m + 1))
        + d + B - 1524.5
    )


def _gmst_rad(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time (radians) at the given UT1 Julian date."""
    import math
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    return (gmst_sec / 86400.0 * 2.0 * math.pi) % (2.0 * math.pi)


def ecef_to_eci(pos_ecef, epoch, *, eop=None) -> np.ndarray:
    """Convert an ECEF position vector to ECI (TEME-style, low-precision).

    The transformation applied is

        v_ECI = R_z(-GMST(t_UT1)) @ W(t)^T @ v_ECEF

    where ``GMST(t_UT1)`` is the Greenwich Mean Sidereal Time at the
    UT1-corrected epoch and ``W(t)`` is the polar-motion rotation
    ``R_x(-y_p) R_y(-x_p)`` (only applied when ``eop`` is provided).

    Precision: sub-meter at the Earth surface without EOP, sub-cm with
    EOP. Note: this is the low-precision reduction; for sub-cm at GPS
    orbit altitudes precession and nutation also need to be applied
    (use an astronomy library for that).

    Parameters
    ----------
    pos_ecef:
        ``(3,)`` or ``(n, 3)`` ECEF position in meters.
    epoch:
        Observation epoch (``datetime`` or ``numpy.datetime64``).
    eop:
        Optional EOP dataset from :func:`rinexpy.eop.load_eop`. If
        provided, the UT1-UTC offset and polar-motion (x, y) at the
        epoch are applied; otherwise UT1 is assumed equal to UTC and
        polar motion is ignored.

    Returns
    -------
    ndarray
        ECI position, same shape as input.
    """
    import math
    pos = np.asarray(pos_ecef, dtype=float)
    jd_utc = _julian_date_utc(epoch)
    if eop is not None:
        from .eop import interp_eop
        e = interp_eop(eop, epoch)
        jd_ut1 = jd_utc + e["ut1_utc"] / 86400.0
        x_p = math.radians(e["x"] / 3600.0)
        y_p = math.radians(e["y"] / 3600.0)
    else:
        jd_ut1 = jd_utc
        x_p = 0.0
        y_p = 0.0
    gmst = _gmst_rad(jd_ut1)
    c, s = math.cos(gmst), math.sin(gmst)
    # Inverse of R_z(GMST) is R_z(-GMST) i.e. [[c, -s, 0], [s, c, 0]]
    R_eci_from_tirs = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    if x_p == 0.0 and y_p == 0.0:
        W_T = np.eye(3)
    else:
        # W^T = R_y(x_p) R_x(y_p), small rotations.
        cx, sx = math.cos(x_p), math.sin(x_p)
        cy, sy = math.cos(y_p), math.sin(y_p)
        Ry = np.array([[cx, 0.0, sx], [0.0, 1.0, 0.0], [-sx, 0.0, cx]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cy, -sy], [0.0, sy, cy]])
        W_T = Ry @ Rx
    M = R_eci_from_tirs @ W_T
    if pos.ndim == 1:
        if pos.shape != (3,):
            raise ValueError(f"pos_ecef must be (3,) or (n,3), got {pos.shape}")
        return M @ pos
    if pos.ndim == 2 and pos.shape[1] == 3:
        return (M @ pos.T).T
    raise ValueError(f"pos_ecef must be (3,) or (n,3), got {pos.shape}")


def eci_to_ecef(pos_eci, epoch, *, eop=None) -> np.ndarray:
    """Inverse of :func:`ecef_to_eci`.

    Same precision caveats apply.
    """
    import math
    pos = np.asarray(pos_eci, dtype=float)
    jd_utc = _julian_date_utc(epoch)
    if eop is not None:
        from .eop import interp_eop
        e = interp_eop(eop, epoch)
        jd_ut1 = jd_utc + e["ut1_utc"] / 86400.0
        x_p = math.radians(e["x"] / 3600.0)
        y_p = math.radians(e["y"] / 3600.0)
    else:
        jd_ut1 = jd_utc
        x_p = 0.0
        y_p = 0.0
    gmst = _gmst_rad(jd_ut1)
    c, s = math.cos(gmst), math.sin(gmst)
    R_tirs_from_eci = np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    if x_p == 0.0 and y_p == 0.0:
        W = np.eye(3)
    else:
        cx, sx = math.cos(x_p), math.sin(x_p)
        cy, sy = math.cos(y_p), math.sin(y_p)
        Ry = np.array([[cx, 0.0, -sx], [0.0, 1.0, 0.0], [sx, 0.0, cx]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cy, sy], [0.0, -sy, cy]])
        W = Rx @ Ry
    M = W @ R_tirs_from_eci
    if pos.ndim == 1:
        if pos.shape != (3,):
            raise ValueError(f"pos_eci must be (3,) or (n,3), got {pos.shape}")
        return M @ pos
    if pos.ndim == 2 and pos.shape[1] == 3:
        return (M @ pos.T).T
    raise ValueError(f"pos_eci must be (3,) or (n,3), got {pos.shape}")


def phase_wind_up_correction(
    sat_xhat,
    sat_yhat,
    rx_xhat,
    rx_yhat,
    los_rx_to_sat,
    *,
    previous_cycles: float = 0.0,
) -> float:
    """Carrier-phase wind-up correction (Wu et al. 1993), in cycles.

    Apply as ``phi_corrected_cycles = phi_observed_cycles + correction``.
    The correction tracks the rotation of the receiver antenna's
    effective dipole relative to the satellite's, projected onto the
    plane perpendicular to the line of sight. Required for cm-level
    carrier-phase PPP.

    Parameters
    ----------
    sat_xhat, sat_yhat:
        ``(3,)`` satellite body x and y axes in ECEF. Conventionally
        computed from the sun position and the satellite position; the
        body z-axis points toward Earth.
    rx_xhat, rx_yhat:
        ``(3,)`` receiver antenna x and y axes in ECEF. For a north-east-
        up mount, ``x_hat`` is east in ECEF and ``y_hat`` is north.
    los_rx_to_sat:
        ``(3,)`` line of sight from the receiver to the satellite. Need
        not be a unit vector; we normalise it internally.
    previous_cycles:
        Last correction value for this satellite, used to resolve the
        2*pi ambiguity of ``arccos`` (continuity across epochs).
        Default 0.

    Returns
    -------
    float
        Wind-up correction in cycles, tracked from ``previous_cycles``.
    """
    k = np.asarray(los_rx_to_sat, dtype=float)
    norm_k = np.linalg.norm(k)
    if norm_k == 0.0:
        return previous_cycles
    k = k / norm_k
    xs = np.asarray(sat_xhat, dtype=float)
    ys = np.asarray(sat_yhat, dtype=float)
    xr = np.asarray(rx_xhat, dtype=float)
    yr = np.asarray(rx_yhat, dtype=float)

    # Satellite effective dipole projected onto the plane perpendicular to k.
    dprime = xs - k * np.dot(k, xs) - np.cross(k, ys)
    # Receiver effective dipole projected onto the same plane.
    # Same sign as the satellite so aligned antennas (xr=xs, yr=ys) give
    # zero wind-up, which is the correct physical answer for RHCP on both ends.
    d = xr - k * np.dot(k, xr) - np.cross(k, yr)

    norm_dp = np.linalg.norm(dprime)
    norm_d = np.linalg.norm(d)
    if norm_dp == 0.0 or norm_d == 0.0:
        return previous_cycles

    cos_theta = float(np.dot(d, dprime) / (norm_d * norm_dp))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    sign_val = float(np.sign(np.dot(k, np.cross(dprime, d))))
    if sign_val == 0.0:
        sign_val = 1.0
    phi = sign_val * float(np.arccos(cos_theta)) / (2.0 * np.pi)

    # Unwrap by picking the integer offset that minimises the jump.
    n = round(previous_cycles - phi)
    return float(phi + n)


__all__ = [
    "azimuth_elevation",
    "dop",
    "ecef_to_eci",
    "ecef_to_enu",
    "ecef_to_lla",
    "eci_to_ecef",
    "enu_to_ecef",
    "klobuchar",
    "lla_to_ecef",
    "niell_mapping",
    "phase_wind_up_correction",
    "saastamoinen",
    "standard_atmosphere",
    "vmf1",
]
