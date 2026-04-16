"""Daily DCB / OSB SINEX product fetcher.

The IGS MGEX program publishes daily multi-GNSS observable-specific
bias (OSB) products. The official archive lives at NASA CDDIS:

    https://cddis.nasa.gov/archive/gnss/products/bias/{YYYY}/

CDDIS requires NASA Earthdata Login authentication (set up via
``~/.netrc``). For unauthenticated access the same products are
public-mirrored by the IGS BKG analysis centre:

    https://igs.bkg.bund.de/root_ftp/IGS/products/mgex/{YYYY}/{DDD}/

This module targets the BKG mirror by default and caches downloaded
files locally so subsequent calls within the cache window are free.

Typical product names (long-name format, since ~2017):

- ``CAS0MGXRAP_YYYYDDD0000_01D_01D_DCB.BSX.gz`` - CAS Rapid (~1-day
  latency).
- ``DLR0MGXFIN_YYYYDDD0000_01D_01D_DCB.BSX.gz`` - DLR Final (longer
  latency).
- ``COD0OPSFIN_YYYYDDD0000_01D_01D_DCB.BSX.gz`` - CODE Final.

Usage
-----

::

    from datetime import datetime
    from rinexpy.dcb_download import download_dcb, load_daily_dcb

    path = download_dcb(datetime(2024, 4, 15), product="CAS")
    # ...or one-shot:
    records = load_daily_dcb(datetime(2024, 4, 15))

The fetched file is cached under
``~/.cache/rinexpy/dcb/<filename>.BSX`` (the ``.gz`` is decompressed
on the way to disk) so the second call is a local read.
"""

from __future__ import annotations

import gzip
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .dcb import read_bsx

log = logging.getLogger(__name__)

#: BKG public HTTPS mirror of the IGS MGEX products tree.
BKG_MGEX_BASE = "https://igs.bkg.bund.de/root_ftp/IGS/products/mgex"

#: NASA CDDIS bias products (requires Earthdata Login + ~/.netrc).
CDDIS_BIAS_BASE = "https://cddis.nasa.gov/archive/gnss/products/bias"

#: Product file-name prefixes for the long-name MGEX layout (post 2017).
_PRODUCT_PREFIXES: dict[str, str] = {
    "CAS": "CAS0MGXRAP",
    "DLR": "DLR0MGXFIN",
    "COD": "COD0OPSFIN",
}


def _default_cache_dir() -> Path:
    """Per-user cache directory for downloaded DCB files."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "rinexpy" / "dcb"


def _build_filename(date: datetime, product: str) -> str:
    """Construct the canonical long-name MGEX filename for one
    (date, product). The on-disk cache strips the trailing ``.gz``."""
    prefix = _PRODUCT_PREFIXES.get(product.upper())
    if prefix is None:
        raise ValueError(
            f"unknown DCB product {product!r}; expected one of "
            f"{sorted(_PRODUCT_PREFIXES)}"
        )
    doy = date.timetuple().tm_yday
    return f"{prefix}_{date.year:04d}{doy:03d}0000_01D_01D_DCB.BSX.gz"


def _build_url(date: datetime, product: str, *, source: str) -> str:
    fname = _build_filename(date, product)
    doy = date.timetuple().tm_yday
    if source == "bkg":
        return f"{BKG_MGEX_BASE}/{date.year:04d}/{doy:03d}/{fname}"
    if source == "cddis":
        return f"{CDDIS_BIAS_BASE}/{date.year:04d}/{fname}"
    raise ValueError(f"unknown source {source!r}; expected 'bkg' or 'cddis'")


def download_dcb(
    date: datetime,
    *,
    product: str = "CAS",
    cache_dir: Path | None = None,
    source: str = "bkg",
    timeout: float = 60.0,
    user_agent: str = "rinexpy",
) -> Path:
    """Fetch one daily DCB SINEX file and cache it locally.

    Parameters
    ----------
    date:
        Calendar date of the DCB product (any ``datetime`` whose
        ``timetuple().tm_yday`` is the target day-of-year).
    product:
        Analysis-centre tag - ``"CAS"`` (rapid, default), ``"DLR"``
        (final), or ``"COD"`` (CODE final).
    cache_dir:
        Directory where decompressed BSX files are cached. Defaults
        to ``$XDG_CACHE_HOME/rinexpy/dcb`` (or ``~/.cache/...``).
    source:
        ``"bkg"`` (default, public HTTPS) or ``"cddis"`` (NASA, requires
        Earthdata Login + ``~/.netrc``).
    timeout:
        Network timeout in seconds. Default 60.
    user_agent:
        User-Agent header value (some IGS mirrors reject blank agents).

    Returns
    -------
    pathlib.Path
        Path to the decompressed ``.BSX`` file in the local cache.

    Raises
    ------
    ValueError
        Unknown product or source.
    URLError / HTTPError
        Network errors, including 404 (file not yet published).
    """
    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    gz_name = _build_filename(date, product)
    bsx_name = gz_name[:-3]  # strip .gz
    out_path = cache_dir / bsx_name
    if out_path.is_file():
        return out_path

    url = _build_url(date, product, source=source)
    log.info("downloading DCB product from %s", url)
    req = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = resp.read()
    except (URLError, HTTPError):
        raise

    # The remote file is gzipped; decompress into the cache so callers
    # see a plain BSX file (and read_bsx can ingest it directly).
    try:
        plain = gzip.decompress(payload)
    except OSError:
        # Already plain text (rare; some mirrors serve uncompressed).
        plain = payload
    out_path.write_bytes(plain)
    return out_path


def load_daily_dcb(
    date: datetime,
    *,
    product: str = "CAS",
    cache_dir: Path | None = None,
    source: str = "bkg",
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """One-shot: fetch the daily DCB product and parse it.

    Wraps :func:`download_dcb` + :func:`rinexpy.dcb.read_bsx`. Suitable
    for ``ppp_solve(..., dcb_records=load_daily_dcb(date))`` or
    ``spp_solve(..., dcb_records=load_daily_dcb(date), ...)``.
    """
    path = download_dcb(
        date, product=product, cache_dir=cache_dir,
        source=source, timeout=timeout,
    )
    return read_bsx(path)


__all__ = [
    "BKG_MGEX_BASE",
    "CDDIS_BIAS_BASE",
    "download_dcb",
    "load_daily_dcb",
]
