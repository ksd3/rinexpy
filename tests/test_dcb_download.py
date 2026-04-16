"""Tests for the daily DCB downloader.

These tests must not hit the network. ``urllib.request.urlopen`` is
mocked to return a small in-memory BSX payload, and the cache is
redirected to ``tmp_path`` to avoid polluting the user's real
``~/.cache``.
"""

from __future__ import annotations

import gzip
import io
from datetime import datetime
from unittest.mock import patch

import pytest

from rinexpy.dcb_download import (
    download_dcb,
    load_daily_dcb,
    _build_filename,
    _build_url,
)


_SAMPLE_BSX = b"""\
%=BIA 1.00 IGS 2024:047:00000 IGS 2024:040:00000 2024:046:86399 R 00000003
*-------------------------------------------------------------------------------
+FILE/REFERENCE
 DESCRIPTION       TEST SINEX-BIAS FILE
-FILE/REFERENCE
*-------------------------------------------------------------------------------
+BIAS/SOLUTION
*BIAS  SVN_   PRN_ STATION__ OBS1 OBS2 BIAS_START____ BIAS_END______ UNIT __ESTIMATED_VALUE____ _STD_DEV___
 OSB   G063   G05 ----      C1W       2024:040:00000 2024:046:86400 ns                 -7.1234       0.0123
-BIAS/SOLUTION
%ENDBIA
"""


class _FakeResponse:
    """Context-manager stand-in for the object returned by urlopen."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *a) -> None:  # noqa: D401
        return None


def test_build_filename_long_name_format():
    name = _build_filename(datetime(2024, 4, 15), "CAS")
    # April 15 2024 -> DOY 106.
    assert name == "CAS0MGXRAP_20241060000_01D_01D_DCB.BSX.gz"


def test_build_filename_rejects_unknown_product():
    with pytest.raises(ValueError):
        _build_filename(datetime(2024, 1, 1), "BOGUS")


def test_build_url_targets_bkg_by_default():
    url = _build_url(datetime(2024, 4, 15), "CAS", source="bkg")
    assert url.startswith("https://igs.bkg.bund.de/")
    assert "/2024/106/" in url
    assert url.endswith("CAS0MGXRAP_20241060000_01D_01D_DCB.BSX.gz")


def test_build_url_cddis_path():
    url = _build_url(datetime(2024, 4, 15), "DLR", source="cddis")
    assert url.startswith("https://cddis.nasa.gov/")
    assert "DLR0MGXFIN_20241060000_01D_01D_DCB.BSX.gz" in url


def test_build_url_rejects_unknown_source():
    with pytest.raises(ValueError):
        _build_url(datetime(2024, 1, 1), "CAS", source="elsewhere")


def test_download_dcb_writes_decompressed_to_cache(tmp_path):
    """First call hits the (mocked) network and writes the .BSX
    to disk; the result is the cached path."""
    gz_payload = gzip.compress(_SAMPLE_BSX)
    with patch("rinexpy.dcb_download.urlopen",
               return_value=_FakeResponse(gz_payload)) as m:
        path = download_dcb(
            datetime(2024, 4, 15), product="CAS",
            cache_dir=tmp_path,
        )
    assert m.call_count == 1
    assert path.name == "CAS0MGXRAP_20241060000_01D_01D_DCB.BSX"
    assert path.read_bytes() == _SAMPLE_BSX
    assert path.parent == tmp_path


def test_download_dcb_uses_cache_on_second_call(tmp_path):
    """A cached file is reused on subsequent calls without re-fetching."""
    gz_payload = gzip.compress(_SAMPLE_BSX)
    with patch("rinexpy.dcb_download.urlopen",
               return_value=_FakeResponse(gz_payload)) as m:
        download_dcb(datetime(2024, 4, 15), cache_dir=tmp_path)
        # Second call: urlopen must NOT be hit again.
        download_dcb(datetime(2024, 4, 15), cache_dir=tmp_path)
    assert m.call_count == 1


def test_download_dcb_handles_uncompressed_payload(tmp_path):
    """Some mirrors serve the .BSX uncompressed; the downloader should
    treat the body as plain SINEX in that case."""
    with patch("rinexpy.dcb_download.urlopen",
               return_value=_FakeResponse(_SAMPLE_BSX)) as m:
        path = download_dcb(datetime(2024, 4, 15), cache_dir=tmp_path)
    assert m.call_count == 1
    assert path.read_bytes() == _SAMPLE_BSX


def test_load_daily_dcb_parses_downloaded_file(tmp_path):
    """End-to-end: fetch + parse returns a list of bias records."""
    gz_payload = gzip.compress(_SAMPLE_BSX)
    with patch("rinexpy.dcb_download.urlopen",
               return_value=_FakeResponse(gz_payload)):
        records = load_daily_dcb(
            datetime(2024, 4, 15), product="CAS", cache_dir=tmp_path,
        )
    assert len(records) == 1
    rec = records[0]
    assert rec["prn"] == "G05"
    assert rec["obs1"] == "C1W"
    assert rec["bias_type"] == "OSB"
