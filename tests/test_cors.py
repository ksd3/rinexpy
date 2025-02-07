"""Tests for the CORS daily-RINEX URL builder and fetcher.

URL construction is fully deterministic and tested offline. The
network-touching fetcher is skipped on connection failure -- when
internet is available the cache path is verified to exist.
"""

from __future__ import annotations

import os
import socket
from datetime import date, datetime

import pytest

from rinexpy.cors import fetch_igs_daily, igs_daily_url


def test_url_cddis_for_algo_2018_doy_133():
    url = igs_daily_url("algo", date(2018, 5, 13), "cddis")
    assert url == (
        "https://cddis.nasa.gov/archive/gnss/data/daily/"
        "2018/133/18d/algo1330.18d.gz"
    )


def test_url_sopac_for_algo_2018_doy_133():
    url = igs_daily_url("ALGO", date(2018, 5, 13), "sopac")
    # Case-insensitive station code, gets downcased.
    assert url.endswith("algo1330.18d.Z")
    assert "2018/133/" in url


def test_url_bkg_uses_https():
    url = igs_daily_url("algo", date(2024, 1, 1), "bkg")
    assert url.startswith("https://igs.bkg.bund.de/")


def test_url_accepts_datetime():
    url1 = igs_daily_url("algo", date(2020, 6, 5), "cddis")
    url2 = igs_daily_url("algo", datetime(2020, 6, 5, 12, 0, 0), "cddis")
    assert url1 == url2


def test_url_rejects_bad_station():
    with pytest.raises(ValueError, match="4 chars"):
        igs_daily_url("AL", date(2020, 1, 1), "cddis")


def test_url_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        igs_daily_url("algo", date(2020, 1, 1), "bogus")  # type: ignore[arg-type]


def _internet_ok() -> bool:
    try:
        s = socket.create_connection(("cddis.nasa.gov", 443), timeout=5)
    except (OSError, socket.timeout):
        return False
    s.close()
    return True


def test_fetch_writes_to_custom_cache(tmp_path, monkeypatch):
    """We don't necessarily have internet, so this test verifies that
    when the file already exists in the cache, no network call is
    needed and the path is returned as-is."""
    monkeypatch.setenv("RINEXPY_CACHE_DIR", str(tmp_path))
    # Pre-populate the cache.
    pretend = tmp_path / "algo1330.18d.gz"
    pretend.write_bytes(b"pretend gz")
    p = fetch_igs_daily("algo", date(2018, 5, 13), "cddis")
    assert p == pretend
    assert p.read_bytes() == b"pretend gz"


@pytest.mark.skipif(not _internet_ok(), reason="no internet")
def test_fetch_downloads_when_missing(tmp_path, monkeypatch):
    """Real download. CDDIS hosts the file for ALGO on DOY 133 of 2018.

    Skips on network failure; doesn't validate the file contents to
    avoid coupling the test to a single CDDIS naming change.
    """
    monkeypatch.setenv("RINEXPY_CACHE_DIR", str(tmp_path))
    try:
        p = fetch_igs_daily("algo", date(2018, 5, 13), "cddis", timeout=30.0)
    except Exception as e:
        pytest.skip(f"network or auth failure: {e}")
    assert p.exists()
    assert p.stat().st_size > 0
