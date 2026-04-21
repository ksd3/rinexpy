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
    auto_load_dcb,
    download_dcb,
    download_legacy_code_dcb,
    load_daily_dcb,
    load_monthly_code_dcb,
    _build_filename,
    _build_url,
    _legacy_filename,
    _legacy_url,
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


# ---------------------------------------------------------------------------
# Legacy AIUB CODE monthly DCB autodownload (pre-2017)
# ---------------------------------------------------------------------------


_SAMPLE_CODE_TEXT = b"""\
                P1-P2 DIFFERENTIAL CODE BIASES (DCB) FOR SATELLITES AND STATIONS
                Reference: AIUB

PRN / STATION NAME        VALUE (NS)  RMS (NS)
G05                       -2.5470     0.0245
"""


def test_legacy_filename_long_name():
    name = _legacy_filename(datetime(2010, 1, 15), "P1P2")
    assert name == "P1P21001.DCB.Z"


def test_legacy_filename_rejects_unknown_product():
    with pytest.raises(ValueError):
        _legacy_filename(datetime(2010, 1, 1), "DLR")


def test_legacy_url_targets_aiub():
    url = _legacy_url(datetime(2014, 6, 1), "P1C1")
    assert url == "http://ftp.aiub.unibe.ch/CODE/2014/P1C11406.DCB.Z"


def test_download_legacy_code_dcb_plain_payload(tmp_path):
    """If the server returns a plain (uncompressed) body, the
    downloader passes it through unchanged."""
    with patch("rinexpy.dcb_download.urlopen",
               return_value=_FakeResponse(_SAMPLE_CODE_TEXT)) as m:
        path = download_legacy_code_dcb(
            datetime(2010, 1, 15), product="P1P2", cache_dir=tmp_path,
        )
    assert m.call_count == 1
    assert path.name == "P1P21001.DCB"
    assert path.read_bytes() == _SAMPLE_CODE_TEXT


def test_download_legacy_code_dcb_caches(tmp_path):
    with patch("rinexpy.dcb_download.urlopen",
               return_value=_FakeResponse(_SAMPLE_CODE_TEXT)) as m:
        download_legacy_code_dcb(datetime(2010, 1, 15), cache_dir=tmp_path)
        download_legacy_code_dcb(datetime(2010, 1, 15), cache_dir=tmp_path)
    assert m.call_count == 1


def test_load_monthly_code_dcb_parses(tmp_path):
    with patch("rinexpy.dcb_download.urlopen",
               return_value=_FakeResponse(_SAMPLE_CODE_TEXT)):
        records = load_monthly_code_dcb(
            datetime(2010, 1, 15), product="P1P2", cache_dir=tmp_path,
        )
    assert len(records) == 1
    rec = records[0]
    assert rec["prn"] == "G05"
    assert rec["obs1"] == "C1W"
    assert rec["obs2"] == "C2W"
    assert rec["bias_type"] == "DSB"


def test_auto_load_dcb_routes_pre_2017_to_aiub(tmp_path):
    """A 2010 date should route through the legacy AIUB / CODE path."""
    with patch("rinexpy.dcb_download.urlopen",
               return_value=_FakeResponse(_SAMPLE_CODE_TEXT)) as m:
        records = auto_load_dcb(datetime(2010, 6, 15), cache_dir=tmp_path)
    assert m.call_count == 1
    url = m.call_args[0][0].full_url
    assert "aiub.unibe.ch/CODE/2010/" in url
    assert records[0]["bias_type"] == "DSB"


def test_auto_load_dcb_routes_post_2017_to_bkg(tmp_path):
    """A 2024 date should route through the MGEX daily path."""
    gz_payload = gzip.compress(_SAMPLE_BSX)
    with patch("rinexpy.dcb_download.urlopen",
               return_value=_FakeResponse(gz_payload)) as m:
        records = auto_load_dcb(datetime(2024, 4, 15), cache_dir=tmp_path)
    assert m.call_count == 1
    url = m.call_args[0][0].full_url
    assert "igs.bkg.bund.de" in url
    assert records[0]["bias_type"] == "OSB"
