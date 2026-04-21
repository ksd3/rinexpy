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

from .dcb import read_bsx, read_code_dcb

log = logging.getLogger(__name__)

#: BKG public HTTPS mirror of the IGS MGEX products tree.
BKG_MGEX_BASE = "https://igs.bkg.bund.de/root_ftp/IGS/products/mgex"

#: NASA CDDIS bias products (requires Earthdata Login + ~/.netrc).
CDDIS_BIAS_BASE = "https://cddis.nasa.gov/archive/gnss/products/bias"

#: AIUB public HTTP mirror of the legacy CODE monthly DCB products.
AIUB_CODE_BASE = "http://ftp.aiub.unibe.ch/CODE"

#: First year for which CAS / DLR MGEX daily DCB SINEX products exist.
MGEX_FIRST_YEAR = 2017

#: Product file-name prefixes for the long-name MGEX layout (post 2017).
_PRODUCT_PREFIXES: dict[str, str] = {
    "CAS": "CAS0MGXRAP",
    "DLR": "DLR0MGXFIN",
    "COD": "COD0OPSFIN",
}

#: Legacy AIUB monthly DCB product prefixes.
_LEGACY_PREFIXES: tuple[str, ...] = ("P1P2", "P1C1", "P2C2")


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


def _legacy_filename(date: datetime, product: str) -> str:
    """Filename for an AIUB CODE monthly DCB (.Z compressed)."""
    prefix = product.upper()
    if prefix not in _LEGACY_PREFIXES:
        raise ValueError(
            f"unknown legacy CODE product {product!r}; expected one of "
            f"{list(_LEGACY_PREFIXES)}"
        )
    yy = date.year % 100
    return f"{prefix}{yy:02d}{date.month:02d}.DCB.Z"


def _legacy_url(date: datetime, product: str) -> str:
    fname = _legacy_filename(date, product)
    return f"{AIUB_CODE_BASE}/{date.year:04d}/{fname}"


def download_legacy_code_dcb(
    date: datetime,
    *,
    product: str = "P1P2",
    cache_dir: Path | None = None,
    timeout: float = 60.0,
    user_agent: str = "rinexpy",
) -> Path:
    """Fetch a monthly CODE DCB file from the AIUB public HTTP mirror.

    The legacy CODE products are LZW-compressed (.Z) and require the
    optional ``[lzw]`` extra (``uv add 'rinexpy[lzw]'``) to decompress.
    The decompressed plain-text ``.DCB`` is cached locally.

    Parameters
    ----------
    date:
        Any date in the target month (year + month are used).
    product:
        ``"P1P2"`` (default, the dominant DCB), ``"P1C1"``, or
        ``"P2C2"``.
    cache_dir, timeout, user_agent:
        Same semantics as :func:`download_dcb`.

    Returns
    -------
    pathlib.Path
        Local path to the decompressed ``.DCB`` text file.
    """
    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    z_name = _legacy_filename(date, product)
    plain_name = z_name[:-2]  # strip .Z
    out_path = cache_dir / plain_name
    if out_path.is_file():
        return out_path

    url = _legacy_url(date, product)
    log.info("downloading legacy CODE DCB from %s", url)
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        payload = resp.read()

    plain = _decompress_lzw(payload)
    out_path.write_bytes(plain)
    return out_path


def _decompress_lzw(payload: bytes) -> bytes:
    """Best-effort LZW (.Z) decompression. The ``ncompress`` library
    (soft dep under the ``[lzw]`` extra) is used when present; on
    Linux/Mac systems the ``gzip`` CLI also handles .Z and is the
    fallback when the extra is not installed."""
    # Already plain SINEX/text?
    if not payload.startswith(b"\x1f\x9d"):
        return payload
    try:
        from ncompress import decompress as _unlzw  # type: ignore
    except ImportError:
        _unlzw = None
    if _unlzw is not None:
        return bytes(_unlzw(payload))
    # Fallback: gzip CLI on most Unix systems handles .Z files.
    import subprocess
    proc = subprocess.run(
        ["gzip", "-dc"], input=payload, capture_output=True, check=False,
    )
    if proc.returncode == 0 and proc.stdout:
        return proc.stdout
    raise RuntimeError(
        "LZW decompression failed; install 'ncompress' via the [lzw] extra "
        "or ensure 'gzip' is on PATH"
    )


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


def load_monthly_code_dcb(
    date: datetime,
    *,
    product: str = "P1P2",
    cache_dir: Path | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """One-shot: fetch the monthly CODE DCB and parse it via
    :func:`rinexpy.dcb.read_code_dcb`. Output records share the same
    schema as :func:`load_daily_dcb` so callers can swap one for the
    other."""
    path = download_legacy_code_dcb(
        date, product=product, cache_dir=cache_dir, timeout=timeout,
    )
    return read_code_dcb(path, year=date.year, month=date.month)


def auto_load_dcb(
    date: datetime,
    *,
    cache_dir: Path | None = None,
    timeout: float = 60.0,
    source: str = "bkg",
) -> list[dict[str, Any]]:
    """Routes by date: pre-2017 -> AIUB monthly CODE P1-P2 file,
    2017+ -> daily MGEX CAS Rapid. Returns DCB records in the unified
    :func:`get_bias`-friendly schema regardless of source."""
    if date.year < MGEX_FIRST_YEAR:
        return load_monthly_code_dcb(
            date, product="P1P2", cache_dir=cache_dir, timeout=timeout,
        )
    return load_daily_dcb(
        date, product="CAS", cache_dir=cache_dir, timeout=timeout, source=source,
    )


__all__ = [
    "AIUB_CODE_BASE",
    "BKG_MGEX_BASE",
    "CDDIS_BIAS_BASE",
    "MGEX_FIRST_YEAR",
    "auto_load_dcb",
    "download_dcb",
    "download_legacy_code_dcb",
    "load_daily_dcb",
    "load_monthly_code_dcb",
]
