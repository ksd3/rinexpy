"""Convert Keplerian orbital elements to ECEF coordinates.

References
----------
- ICD-GPS-200, table 20-IV
- ASCE manual chapter Ap03:
  https://ascelibrary.org/doi/pdf/10.1061/9780784411506.ap03
- Schwarz (2014), Keplerian Orbit Elements -> Cartesian State Vectors:
  https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr

# Geodetic constants (WGS-84 / GPS-ICD-200).
_GM = 3.986004418e14  # [m^3 s^-2]    Earth's gravitational parameter
_OMEGA_E = 7.2921151467e-5  # [rad s^-1]    Earth's mean angular velocity
_GPS_EPOCH = datetime(1980, 1, 6)


def keplerian2ecef(
    sv: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Convert a NAV dataset's Keplerian elements to ECEF position arrays.

    Parameters
    ----------
    sv:
        An ``xarray.Dataset`` slice for a single satellite (or stack) such as
        ``nav.sel(sv="G07")``. Must contain Keplerian variables ``sqrtA``,
        ``DeltaN``, ``Eccentricity``, ``M0``, ``omega``, ``Cuc``, ``Cus``,
        ``Cic``, ``Cis``, ``Crc``, ``Crs``, ``Io``, ``IDOT``, ``Omega0``,
        ``OmegaDot``, ``Toe``. The dataset's ``svtype`` attribute must be
        ``"G"`` (GPS) or ``"E"`` (Galileo); for GLONASS / SBAS the orbital
        state is reported directly as ECEF and is returned unchanged.

    Returns
    -------
    X, Y, Z : xarray.DataArray
        ECEF position components, in meters.

    Raises
    ------
    ValueError
        If the satellite system is unsupported (anything other than G/E or
        the directly-reported R/S).
    """
    sys_letter = sv.svtype[0]
    # GLONASS / SBAS report ECEF directly — no Keplerian conversion needed.
    if sys_letter in {"R", "S"}:
        return sv["X"], sv["Y"], sv["Z"]

    if sys_letter == "E":
        weeks = sv["GALWeek"].values - 1024
    elif sys_letter == "G":
        weeks = sv["GPSWeek"].values
    else:
        raise ValueError(f"unsupported satellite system {sys_letter!r}")

    weeks = np.atleast_1d(weeks).astype(float)
    toe = np.atleast_1d(sv["Toe"].values).astype(float)
    e = sv["Eccentricity"].values
    A = sv["sqrtA"].values ** 2

    n0 = np.sqrt(_GM / A**3)
    n = n0 + sv["DeltaN"].values

    # Vectorized t_k computation. Build a NumPy datetime64 representation of
    # each (week, toe) reference epoch, subtract from sv.time once.
    week_seconds = weeks * 7 * 86400.0
    ref_seconds = week_seconds + toe  # seconds since GPS_EPOCH
    gps_epoch_ns = np.datetime64(_GPS_EPOCH, "ns")
    ref_ns = gps_epoch_ns + (ref_seconds * 1e9).astype("timedelta64[ns]")
    tk = (sv["time"].values - ref_ns) / np.timedelta64(1, "s")
    tk = tk.astype(float)

    Mk = sv["M0"].values + n * tk
    Ek = Mk + e * np.sin(Mk)
    nu = np.arctan2(np.sqrt(1 - e**2) * np.sin(Ek), np.cos(Ek) - e)

    phi = nu + sv["omega"].values
    cos2p = np.cos(2 * phi)
    sin2p = np.sin(2 * phi)
    duk = sv["Cuc"].values * cos2p + sv["Cus"].values * sin2p
    dik = sv["Cic"].values * cos2p + sv["Cis"].values * sin2p
    drk = sv["Crc"].values * cos2p + sv["Crs"].values * sin2p

    uk = phi + duk
    ik = sv["Io"].values + sv["IDOT"].values * tk + dik
    rk = A * (1 - e * np.cos(Ek)) + drk

    Omega = sv["Omega0"].values + (sv["OmegaDot"].values - _OMEGA_E) * tk - _OMEGA_E * toe

    cos_u = np.cos(uk)
    sin_u = np.sin(uk)
    cos_O = np.cos(Omega)
    sin_O = np.sin(Omega)
    cos_i = np.cos(ik)
    sin_i = np.sin(ik)

    Xk1 = rk * cos_u
    Yk1 = rk * sin_u
    X = Xk1 * cos_O - Yk1 * sin_O * cos_i
    Y = Xk1 * sin_O + Yk1 * cos_O * cos_i
    Z = Yk1 * sin_i
    return X, Y, Z


__all__ = ["keplerian2ecef"]
