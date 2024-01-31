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


__all__ = [
    "azimuth_elevation",
    "dop",
    "ecef_to_lla",
    "klobuchar",
    "lla_to_ecef",
    "niell_mapping",
    "saastamoinen",
    "standard_atmosphere",
    "vmf1",
]
