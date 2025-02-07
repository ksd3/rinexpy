"""Fetch IGS / CDDIS daily RINEX observation files by station and date.

The IGS Continuously Operating Reference Station (CORS) network publishes
daily 30-second RINEX-3 OBS files for every tracked station. The standard
public mirrors are:

- https://cddis.nasa.gov/archive/gnss/data/daily/   (NASA / CDDIS)
- https://garner.ucsd.edu/pub/rinex/                (UCSD / SOPAC)
- https://igs.bkg.bund.de/root_ftp/IGS/obs/         (BKG)

This module assembles the standard file-name and URL conventions for
those mirrors and downloads the file with a configurable timeout, with
an on-disk cache in ``~/.cache/rinexpy/cors``. Decompression is left to
the caller (CDDIS now serves daily files as ``.crx.gz``; UNAVCO uses
``.gz``).

Public functions:

- :func:`igs_daily_url` -- build a URL for one (station, date, source).
- :func:`fetch_igs_daily` -- download to the cache, returning the path.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# Standard mirror layouts. Each entry produces a full URL from the
# (station_4char, year, day_of_year, year_2digit) inputs.
_MIRRORS: dict[str, str] = {
    "cddis": (
        "https://cddis.nasa.gov/archive/gnss/data/daily/"
        "{yyyy}/{ddd}/{yy}d/{station_lc}{ddd}0.{yy}d.gz"
    ),
    "sopac": (
        "ftp://garner.ucsd.edu/pub/rinex/{yyyy}/{ddd}/"
        "{station_lc}{ddd}0.{yy}d.Z"
    ),
    "bkg": (
        "https://igs.bkg.bund.de/root_ftp/IGS/obs/"
        "{yyyy}/{ddd}/{station_lc}{ddd}0.{yy}d.gz"
    ),
}


def igs_daily_url(
    station: str,
    when: date | datetime,
    source: Literal["cddis", "sopac", "bkg"] = "cddis",
) -> str:
    """Build the canonical daily-RINEX URL for one (station, date, mirror).

    Parameters
    ----------
    station:
        4-character IGS station code (case-insensitive; downcased
        internally to match the published file-naming convention).
    when:
        Date of the file. ``datetime`` is also accepted; only the date
        portion is used.
    source:
        Mirror to query. Default ``"cddis"``.

    Returns
    -------
    str
        Complete URL to the gzipped Hatanaka-compressed daily file
        (``*.<yy>d.gz``).

    Raises
    ------
    ValueError
        If ``station`` is not 4 characters or ``source`` is unknown.
    """
    if len(station) != 4:
        raise ValueError(f"station code must be 4 chars, got {station!r}")
    if source not in _MIRRORS:
        raise ValueError(f"unknown source {source!r}; pick from {list(_MIRRORS)}")
    if isinstance(when, datetime):
        when = when.date()
    yyyy = when.year
    yy = yyyy % 100
    ddd = when.timetuple().tm_yday
    return _MIRRORS[source].format(
        station_lc=station.lower(),
        yyyy=yyyy,
        yy=f"{yy:02d}",
        ddd=f"{ddd:03d}",
    )


def _cache_path(url: str) -> Path:
    """Translate a remote URL into a stable cache filename."""
    base = Path(
        os.environ.get(
            "RINEXPY_CACHE_DIR",
            os.path.expanduser("~/.cache/rinexpy/cors"),
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    return base / name


def fetch_igs_daily(
    station: str,
    when: date | datetime,
    source: Literal["cddis", "sopac", "bkg"] = "cddis",
    *,
    timeout: float = 60.0,
    overwrite: bool = False,
) -> Path:
    """Download an IGS daily RINEX file to the local cache.

    The cache path is ``~/.cache/rinexpy/cors/<filename>`` by default
    (override via the ``RINEXPY_CACHE_DIR`` environment variable). If
    the file already exists in the cache and ``overwrite`` is False
    (the default), no network call is made and the existing path is
    returned.

    Parameters
    ----------
    station:
        4-character IGS station code.
    when:
        Date of the file.
    source:
        Mirror to query. Default ``"cddis"``.
    timeout:
        HTTP / FTP timeout in seconds. Default 60.
    overwrite:
        If True, re-download even if the file is already cached.

    Returns
    -------
    pathlib.Path
        Local path of the downloaded file. The file is left compressed
        (typically ``.crx.gz``); pair with :func:`rinexpy.load` which
        handles ``.gz`` / ``.Z`` / Hatanaka transparently.

    Raises
    ------
    urllib.error.URLError
        On network failure (caller may want to fall back to another
        source).
    """
    url = igs_daily_url(station, when, source)
    out = _cache_path(url)
    if out.exists() and not overwrite:
        log.debug("cors: cache hit %s", out)
        return out
    log.info("cors: downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "rinexpy/0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, out.open("wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    return out


__all__ = ["fetch_igs_daily", "igs_daily_url"]
