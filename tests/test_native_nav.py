"""Parity tests for the C++ GPS LNAV + BeiDou D1/D2 subframe decoders.

Skipped when rinexpy_native isn't importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy import _native
import rinexpy.beidou as bd
import rinexpy.gps_lnav as ln

pytest.importorskip("rinexpy_native")


def _force(native_on: bool) -> None:
    if native_on:
        _native.have_decode_lnav_subframe = (
            lambda: _native._decode_lnav_subframe is not None
        )
        _native.have_decode_beidou_d1_sf1 = (
            lambda: _native._decode_beidou_d1_sf1 is not None
        )
        _native.have_decode_beidou_d2_page1 = (
            lambda: _native._decode_beidou_d2_page1 is not None
        )
    else:
        _native.have_decode_lnav_subframe = lambda: False
        _native.have_decode_beidou_d1_sf1 = lambda: False
        _native.have_decode_beidou_d2_page1 = lambda: False


def test_native_lnav_available():
    assert _native.have_decode_lnav_subframe() is True


def test_native_beidou_d1_available():
    assert _native.have_decode_beidou_d1_sf1() is True


def test_native_beidou_d2_available():
    assert _native.have_decode_beidou_d2_page1() is True


def _build_lnav(spec, sf_id, tow=0):
    prefix = [
        (ln.PREAMBLE, 8), (0, 14), (0, 2),
        (tow, 17), (0, 1), (0, 1), (sf_id, 3), (0, 2),
    ]
    return ln.encode_lnav_words(prefix + spec)


def test_lnav_sf1_matches_python():
    spec = [
        (999, 10), (1, 2), (2, 4), (0, 6), (3, 2),
        (0, 1), (0, 23), (0, 24), (0, 24),
        (0, 16), ((-7) & 0xFF, 8),
        (0xAB, 8), (100, 16),
        (5 & 0xFF, 8), ((-123) & 0xFFFF, 16),
        (0x123456 & ((1 << 22) - 1), 22), (0, 2),
    ]
    words = _build_lnav(spec, 1)

    _force(False)
    py = ln.decode_lnav_subframe1(words)
    _force(True)
    cpp = ln.decode_lnav_subframe1(words)

    for k in py:
        if isinstance(py[k], float):
            assert abs(py[k] - cpp[k]) < 1e-12 * max(1.0, abs(py[k]))
        else:
            assert py[k] == cpp[k], (k, py[k], cpp[k])


def test_lnav_sf2_matches_python():
    spec = [
        (77, 8), ((-123) & 0xFFFF, 16),
        (42 & 0xFFFF, 16), (((-100000) >> 24) & 0xFF, 8),
        ((-100000) & 0xFFFFFF, 24),
        (50 & 0xFFFF, 16), (0x10000000 >> 24, 8),
        (0x10000000 & 0xFFFFFF, 24),
        ((-50) & 0xFFFF, 16), (0x1ABCDEF0 >> 24, 8),
        (0x1ABCDEF0 & 0xFFFFFF, 24),
        (200, 16), (1, 1), (5, 5), (0, 2),
    ]
    words = _build_lnav(spec, 2)
    _force(False)
    py = ln.decode_lnav_subframe2(words)
    _force(True)
    cpp = ln.decode_lnav_subframe2(words)
    for k in py:
        if isinstance(py[k], float):
            assert abs(py[k] - cpp[k]) < 1e-12 * max(1.0, abs(py[k]))
        else:
            assert py[k] == cpp[k], (k, py[k], cpp[k])


def _spec_at(offsets):
    spec, cursor = [], 0
    for off in sorted(offsets):
        v, n = offsets[off]
        if off > cursor:
            spec.append((0, off - cursor))
        spec.append((v, n))
        cursor = off + n
    return spec


def test_beidou_d1_sf1_matches_python():
    spec = _spec_at({
        0: (bd.PREAMBLE, 11),
        23: (1, 3),
        38: (0, 1), 39: (5, 5), 44: (3, 4), 48: (1234, 13), 61: (1000, 17),
        78: (-3 & 0x3FF, 10), 88: (4 & 0x3FF, 10),
        98: (1 & 0xFF, 8), 173: (12345 & 0xFFFFFF, 24),
    })
    words = bd.encode_subframe_words(spec)
    _force(False)
    py = bd.decode_d1_subframe1(words)
    _force(True)
    cpp = bd.decode_d1_subframe1(words)
    for k in py:
        if k in ("iono_alpha", "iono_beta"):
            for i in range(4):
                assert abs(py[k][i] - cpp[k][i]) < 1e-12 * max(1.0, abs(py[k][i]))
        elif isinstance(py[k], float):
            assert abs(py[k] - cpp[k]) < 1e-12 * max(1.0, abs(py[k]))
        else:
            assert py[k] == cpp[k], (k, py[k], cpp[k])


def test_beidou_d2_page1_matches_python():
    spec = _spec_at({
        0: (bd.PREAMBLE, 11),
        23: (1, 3), 38: (1, 4), 42: (1, 1), 43: (7, 5),
        48: (2, 4), 52: (876, 13), 65: (500, 17),
        102: (0x123456, 24),
    })
    words = bd.encode_subframe_words(spec)
    _force(False)
    py = bd.decode_d2_page1(words)
    _force(True)
    cpp = bd.decode_d2_page1(words)
    for k in py:
        if isinstance(py[k], float):
            assert abs(py[k] - cpp[k]) < 1e-12 * max(1.0, abs(py[k]))
        else:
            assert py[k] == cpp[k], (k, py[k], cpp[k])


def teardown_module():
    _force(True)
