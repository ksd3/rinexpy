"""GNSS reflectometry (GNSS-R) SNR retrievals.

The dominant retrieval is *interferometric altimetry*: a GNSS antenna
mounted above a smooth reflective surface (sea, ice, soil) receives the
direct LOS signal plus a delayed copy reflected off the surface. The
interference between the two appears in the SNR observation as a slow
oscillation versus satellite elevation. Larson (2008) showed that the
frequency of that oscillation is

    f = 2 H / lambda                                              (1)

where ``H`` is the antenna-to-reflector height (m) and ``lambda`` is
the carrier wavelength. Detrending the SNR vs ``sin(elevation)`` and
finding the peak frequency of the residual gives ``H`` directly.

Functions in this module:

- :func:`detrend_snr` removes the direct-signal SNR trend (low-order
  polynomial in ``sin(elevation)``).
- :func:`snr_to_sea_height` runs the full retrieval pipeline and returns
  the reflector height.

References
----------
- Larson, K. M., E. E. Small, E. D. Gutmann, A. L. Bilich, J. J. Braun,
  and V. U. Zavorotny. (2008). *Using GPS multipath to measure soil
  moisture fluctuations: initial results.* GPS Solutions 12 (3): 173-177.
- Larson, K. M., J. Lofgren, R. Haas. (2013). *Coastal sea level
  measurements using a single geodetic GPS receiver.* Advances in
  Space Research 51 (8): 1301-1310.
"""

from __future__ import annotations

import numpy as np


def detrend_snr(
    snr_db: np.ndarray,
    elevation_rad: np.ndarray,
    *,
    order: int = 4,
) -> np.ndarray:
    """Polynomial-detrend SNR observations versus ``sin(elevation)``.

    The direct-signal SNR rises roughly with antenna gain pattern,
    which is smooth in elevation. Removing a low-order polynomial in
    ``sin(elev)`` isolates the multipath oscillation.

    Parameters
    ----------
    snr_db:
        ``(n,)`` SNR observations in dB-Hz (or linear power; the fit
        is unit-agnostic).
    elevation_rad:
        ``(n,)`` satellite elevation in radians, monotonic in time.
    order:
        Polynomial order (default 4).

    Returns
    -------
    ndarray
        Detrended SNR residual, same shape as ``snr_db``.
    """
    s = np.asarray(snr_db, dtype=float)
    x = np.sin(np.asarray(elevation_rad, dtype=float))
    coeffs = np.polyfit(x, s, order)
    trend = np.polyval(coeffs, x)
    return s - trend


def _lomb_scargle(
    x: np.ndarray, y: np.ndarray, freqs: np.ndarray
) -> np.ndarray:
    """Normalized Lomb-Scargle periodogram for unevenly-spaced samples.

    Independent re-implementation so we don't pull in scipy for this
    module. Returns the spectral power at each frequency in ``freqs``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float) - np.mean(y)
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    px = np.empty(omega.size, dtype=float)
    for i, w in enumerate(omega):
        # Time-offset tau (eq. 18, Scargle 1982).
        tan2wt = np.sum(np.sin(2 * w * x)) / np.sum(np.cos(2 * w * x))
        tau = np.arctan(tan2wt) / (2 * w) if w != 0 else 0.0
        c = np.cos(w * (x - tau))
        s = np.sin(w * (x - tau))
        cy = np.sum(y * c)
        sy = np.sum(y * s)
        cc = np.sum(c * c)
        ss = np.sum(s * s)
        px[i] = 0.5 * ((cy * cy) / cc + (sy * sy) / ss)
    return px


def snr_to_sea_height(
    snr_db: np.ndarray,
    elevation_rad: np.ndarray,
    *,
    wavelength_m: float,
    height_search_m: tuple[float, float] = (0.5, 50.0),
    n_freqs: int = 1024,
    detrend_order: int = 4,
) -> dict:
    """SNR-based interferometric altimetry retrieval (Larson 2008/2013).

    Parameters
    ----------
    snr_db:
        ``(n,)`` SNR samples (dB-Hz or linear).
    elevation_rad:
        ``(n,)`` matching satellite elevations in radians. The arc
        should span > 10 deg for a stable fit; the function does not
        enforce that.
    wavelength_m:
        Carrier wavelength (e.g. 0.1903 for GPS L1).
    height_search_m:
        ``(h_min, h_max)`` reflector-height search bounds (m). Default
        ``(0.5, 50)`` covers most ground- / sea-level geodetic
        installations.
    n_freqs:
        Number of frequencies to evaluate in the search range.
    detrend_order:
        Polynomial order forwarded to :func:`detrend_snr`.

    Returns
    -------
    dict
        ``{"height_m": float, "peak_power": float,
        "frequencies_per_sin_elev": ndarray, "power": ndarray,
        "detrended": ndarray}``.
    """
    snr = np.asarray(snr_db, dtype=float)
    elev = np.asarray(elevation_rad, dtype=float)
    if snr.shape != elev.shape or snr.ndim != 1:
        raise ValueError("snr_db and elevation_rad must be 1-D arrays of equal length")
    if snr.size < 8:
        raise ValueError("need at least 8 samples for a stable fit")

    residual = detrend_snr(snr, elev, order=detrend_order)
    x = np.sin(elev)
    # f in cycles per unit-of-sin(elev); h = f * wavelength / 2.
    h_min, h_max = height_search_m
    f_min = 2.0 * h_min / wavelength_m
    f_max = 2.0 * h_max / wavelength_m
    freqs = np.linspace(f_min, f_max, int(n_freqs))
    power = _lomb_scargle(x, residual, freqs)
    peak_idx = int(np.argmax(power))
    peak_freq = float(freqs[peak_idx])
    height_m = peak_freq * wavelength_m / 2.0
    return {
        "height_m": height_m,
        "peak_power": float(power[peak_idx]),
        "frequencies_per_sin_elev": freqs,
        "power": power,
        "detrended": residual,
    }


__all__ = ["detrend_snr", "snr_to_sea_height"]
