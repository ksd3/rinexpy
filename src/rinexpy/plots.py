"""Optional matplotlib plotting helpers.

Importing this module requires ``matplotlib``; install with the ``plot``
extra (``uv add 'rinexpy[plot]'``). For receiver-location and satellite-
ground-track plots, the geodetic conversion uses :mod:`pymap3d` (install
the ``geo`` extra), and an optional :mod:`cartopy` import gives the
familiar coastlines/borders.

The three public functions are:

- :func:`obstimeseries`: plot OBS pseudorange/carrier-phase L1/L1C vs time.
- :func:`navtimeseries`: plot satellite ground-tracks from a NAV dataset.
- :func:`receiver_locations`: scatter receiver positions on a world map.

All three are no-ops on the wrong input type so they can be wired into
a generic dispatch (``timeseries(data)``) without per-call type checks.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from .keplerian import keplerian2ecef

log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for rinexpy.plots; install with `uv add 'rinexpy[plot]'`"
    ) from e

try:
    import pymap3d as _pm
except ImportError:
    _pm = None  # type: ignore[assignment]

try:
    import cartopy
    import cartopy.feature as _cpf
except ImportError:
    cartopy = None  # type: ignore[assignment]
    _cpf = None  # type: ignore[assignment]


def timeseries(data: xr.Dataset | tuple[xr.Dataset, xr.Dataset]) -> None:
    """Dispatch ``data`` to :func:`obstimeseries` or :func:`navtimeseries`.

    Parameters
    ----------
    data:
        Either a single ``xarray.Dataset`` (dispatch on its
        ``rinextype`` attribute) or a ``(obs, nav)`` tuple, in which case
        both helpers are called.
    """
    if isinstance(data, tuple):
        obs, nav = data
        obstimeseries(obs)
        navtimeseries(nav)
        return
    if not isinstance(data, xr.Dataset):
        return
    rt = data.attrs.get("rinextype")
    if rt == "obs":
        obstimeseries(data)
    elif rt == "nav":
        navtimeseries(data)


def obstimeseries(obs: xr.Dataset) -> None:
    """Plot the L1 (or L1C) carrier-phase observable for every visible SV.

    Parameters
    ----------
    obs:
        OBS dataset returned by :func:`rinexpy.load`.

    Notes
    -----
    Silently skips if no L1/L1C variable is present (e.g. the file only
    has pseudoranges) or if all values dropna out.
    """
    if not isinstance(obs, xr.Dataset):
        return

    for label in ("L1", "L1C"):
        if label not in obs:
            continue
        dat = obs[label].dropna(how="all", dim="time")
        if dat.time.size == 0:
            continue

        ax = plt.figure().gca()
        ax.plot(dat.time.values, dat)
        ax.set_title(obs.attrs.get("filename", ""))
        ax.set_xlabel("time [UTC]")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend(dat.sv.values.astype(str), loc="best")


def navtimeseries(nav: xr.Dataset) -> None:
    """Plot satellite ground-tracks (lat/lon path) for every SV in ``nav``.

    Parameters
    ----------
    nav:
        NAV dataset returned by :func:`rinexpy.load`.

    Notes
    -----
    Requires the ``geo`` extra (``pymap3d``). For GPS / Galileo, the orbit
    is computed from Keplerian elements via :func:`rinexpy.keplerian2ecef`;
    for GLONASS / SBAS the file already contains ECEF state. Sanity-check
    warnings are logged if altitudes or latitudes fall outside the
    expected envelope for each constellation.
    """
    if not isinstance(nav, xr.Dataset):
        return
    if _pm is None:
        log.warning("pymap3d not installed; skipping nav ground-track plot")
        return

    fig = plt.figure()
    ax = _world_map_axes(fig)

    for sv_label in nav.sv.values:
        sys_letter = sv_label[0]
        sv = nav.sel(sv=sv_label)
        try:
            lat, lon = _ground_track(sv, sys_letter)
        except (KeyError, ValueError) as e:
            log.debug("skip %s: %s", sv_label, e)
            continue
        if lat is None:
            continue
        ax.plot(lon, lat, label=sv_label)

    ax.set_title(nav.attrs.get("filename", ""))


def _ground_track(sv: xr.Dataset, sys_letter: str):
    """Return (lat, lon) arrays for a single SV slice, or ``(None, None)``.

    Used by :func:`navtimeseries`. Logs a warning if the resulting
    altitudes or latitudes are outside the published envelope for the
    constellation (a useful smoke test for the source file).
    """
    assert _pm is not None
    if sys_letter in {"R", "S"}:
        x = sv["X"].dropna(dim="time", how="all")
        y = sv["Y"].dropna(dim="time", how="all")
        z = sv["Z"].dropna(dim="time", how="all")
        if x.size == 0:
            return None, None
        lat, lon, alt = _pm.ecef2geodetic(x, y, z)
        if sys_letter == "S":
            if ((alt < 35.7e6) | (alt > 35.9e6)).any():
                log.warning("unrealistic geostationary altitudes")
        else:
            if ((alt < 19.0e6) | (alt > 19.4e6)).any():
                log.warning("unrealistic GLONASS altitudes")
        return lat, lon

    if sys_letter in {"G", "E"}:
        ecef = keplerian2ecef(sv)
        lat, lon, alt = _pm.ecef2geodetic(*ecef)
        if sys_letter == "G" and ((alt < 19.4e6) | (alt > 21.0e6)).any():
            log.warning("unrealistic GPS altitudes")
        elif sys_letter == "E" and ((alt < 23e6) | (alt > 24e6)).any():
            log.warning("unrealistic Galileo altitudes")
        return lat, lon

    return None, None


def receiver_locations(locs) -> None:
    """Scatter a set of receiver locations on a world map.

    Parameters
    ----------
    locs:
        ``pandas.DataFrame`` indexed by site name with columns
        ``lat``, ``lon``, and ``interval`` (seconds between epochs).

    Notes
    -----
    Marker color and size encode the receiver's sampling interval:
    red+large for sub-5 s, orange for 5-15 s, green for 15-30 s, blue
    otherwise.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not installed; skipping receiver-locations plot")
        return
    if not isinstance(locs, pd.DataFrame):
        return

    fig = plt.figure()
    ax = _world_map_axes(fig)

    for name, loc in locs.iterrows():
        interval = loc.get("interval")
        color = _interval_color(interval)
        if interval is not None and np.isfinite(interval) and interval > 0:
            ax.scatter(loc.lon, loc.lat, s=1000.0 / interval, c=color, label=name)
        else:
            ax.scatter(loc.lon, loc.lat, c=color, label=name)


def _interval_color(interval) -> str:
    """Map a sampling interval (seconds) to a matplotlib colour code."""
    if interval is None or not np.isfinite(interval):
        return "b"
    if interval < 5:
        return "r"
    if interval < 15:
        return "orange"
    if interval < 30:
        return "g"
    return "b"


def _world_map_axes(fig):
    """Return an axes with cartopy world-map features if available, else plain.

    Used by :func:`navtimeseries` and :func:`receiver_locations` so the
    cartopy/no-cartopy fallback isn't repeated in each.
    """
    if cartopy is None:
        return fig.gca()
    ax = fig.add_subplot(projection=cartopy.crs.PlateCarree())
    ax.add_feature(_cpf.LAND)
    ax.add_feature(_cpf.OCEAN)
    ax.add_feature(_cpf.COASTLINE)
    ax.add_feature(_cpf.BORDERS, linestyle=":")
    return ax


def skyplot(
    sv_az_el: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    elevation_mask_deg: float = 5.0,
    title: str = "",
):
    """Polar skyplot of satellite trajectories above a receiver.

    Parameters
    ----------
    sv_az_el:
        Mapping from SV label (e.g. ``"G07"``) to ``(az_deg, el_deg)``
        arrays for that SV's trajectory.
    elevation_mask_deg:
        Don't plot points below this elevation (default 5 degrees).
    title:
        Figure title.

    Returns
    -------
    matplotlib.axes.Axes
        The created polar axes (zenith at center, horizon at radius=1).
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1)
    ax.set_yticks([0.0, 0.333, 0.667, 1.0])
    ax.set_yticklabels(["90", "60", "30", "0"])
    for sv, (az, el) in sv_az_el.items():
        az = np.asarray(az)
        el = np.asarray(el)
        mask = el >= elevation_mask_deg
        if not mask.any():
            continue
        theta = np.radians(az[mask])
        r = (90.0 - el[mask]) / 90.0
        ax.plot(theta, r, label=sv)
        # Mark the trajectory start with a small dot.
        ax.plot(theta[0], r[0], "o", markersize=3)
    if title:
        ax.set_title(title)
    return ax


__all__ = [
    "navtimeseries",
    "obstimeseries",
    "receiver_locations",
    "skyplot",
    "timeseries",
]
