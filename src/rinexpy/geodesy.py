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


__all__ = [
    "azimuth_elevation",
    "dop",
    "ecef_to_lla",
    "klobuchar",
    "lla_to_ecef",
]
