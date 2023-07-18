"""Tests for the transparent file opener."""

from __future__ import annotations

import io

import pytest

from rinexpy._io import opener
from rinexpy._version import first_nonblank_line, rinex_version

from .conftest import fixture


def _peek_version(stream):
    return rinex_version(first_nonblank_line(stream))[0]


def test_open_plain_obs2():
    fn = fixture("demo.10o")
    with opener(fn) as f:
        v = _peek_version(f)
    assert v == 2.11


def test_open_gz():
    fn = fixture("brdc2420.18n.gz")
    with opener(fn) as f:
        v = _peek_version(f)
    assert v == 2.11


def test_open_zip():
    fn = fixture("ab430140.18o.zip")
    with opener(fn) as f:
        v = _peek_version(f)
    assert v == 2.11


def test_open_lzw():
    pytest.importorskip("ncompress")
    fn = fixture("ac660270.18o.Z")
    with opener(fn) as f:
        v = _peek_version(f)
    assert v == 2.11


def test_open_hatanaka_gz():
    pytest.importorskip("hatanaka")
    fn = fixture("CEBR00ESP_R_20182000000_01D_30S_MO.crx.gz")
    with opener(fn) as f:
        line = first_nonblank_line(f)
    # After Hatanaka decompression the file should look like a normal RINEX 3.
    assert rinex_version(line)[0] == 3.03


def test_open_bz2():
    pytest.importorskip("hatanaka")
    fn = fixture("P43300USA_R_20190012056_17M_15S_MO.crx.bz2")
    with opener(fn) as f:
        line = first_nonblank_line(f)
    assert rinex_version(line)[0] == 3.03


def test_open_stringio_passthrough():
    text = (
        "     2.11           OBSERVATION DATA    M (MIXED)           "
        "RINEX VERSION / TYPE\n"
    )
    sio = io.StringIO(text)
    with opener(sio) as f:
        v = _peek_version(f)
    assert v == 2.11


def test_open_missing_file():
    with pytest.raises(FileNotFoundError):
        with opener("/no/such/file.10o"):
            pass
