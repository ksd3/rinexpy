"""Tests for the BeiDou D1/D2 raw subframe decoder."""

from __future__ import annotations

import pytest

from rinexpy.beidou import (
    PREAMBLE,
    decode_d1_subframe1,
    decode_d2_page1,
    encode_subframe_words,
)


def _spec_at_offsets(field_offsets: dict[int, tuple[int, int]]) -> list[tuple[int, int]]:
    """Build a (value, n_bits) sequential spec list that places each field
    at the requested *data-stream* bit offset.

    Gaps between fields are filled with zeros. The decoder reads from
    the parity-stripped 224-bit data stream, so offsets here are also
    in that stream.
    """
    spec: list[tuple[int, int]] = []
    cursor = 0
    for offset in sorted(field_offsets):
        value, n = field_offsets[offset]
        if offset > cursor:
            spec.append((0, offset - cursor))
        spec.append((value, n))
        cursor = offset + n
    return spec


def test_preamble_constant():
    """ICD-BDS-OS-200 specifies the 11-bit preamble 11100010010 = 0x712."""
    assert PREAMBLE == 0x712
    assert f"{PREAMBLE:011b}" == "11100010010"


def test_d1_subframe1_round_trip():
    """Build a minimal D1 subframe 1 and verify its decoded fields."""
    spec = _spec_at_offsets({
        0: (PREAMBLE, 11),
        23: (1, 3),         # FraID = 1
        38: (0, 1),         # SatH1
        39: (5, 5),         # AODC
        44: (3, 4),         # URAI
        48: (1234, 13),     # WN
        61: (1000, 17),     # t_oc / 8 -> 8000 s
    })
    words = encode_subframe_words(spec)
    out = decode_d1_subframe1(words)
    assert out["subframe_id"] == 1
    assert out["satH1"] == 0
    assert out["AODC"] == 5
    assert out["URAI"] == 3
    assert out["week"] == 1234
    assert out["t_oc_s"] == 8000


def test_d1_subframe1_iono_coefficients():
    """The decoded iono alpha/beta arrays come back in the right slots."""
    spec = _spec_at_offsets({
        0: (PREAMBLE, 11),
        23: (1, 3),
        # Just give alpha0 a known signed value at bit 98 (8 bits, * 2**-30).
        98: (1, 8),  # +1 * 2^-30
    })
    words = encode_subframe_words(spec)
    out = decode_d1_subframe1(words)
    assert out["iono_alpha"][0] == pytest.approx(2**-30)


def test_d1_subframe1_bad_preamble_raises():
    """A non-preamble first 11 bits is rejected."""
    bad = encode_subframe_words([(0, 11)])
    with pytest.raises(ValueError, match="preamble"):
        decode_d1_subframe1(bad)


def test_d1_subframe1_wrong_id_raises():
    """An otherwise valid frame with the wrong ID is rejected."""
    bad = encode_subframe_words(
        _spec_at_offsets({0: (PREAMBLE, 11), 23: (2, 3)})
    )
    with pytest.raises(ValueError, match="subframe 1"):
        decode_d1_subframe1(bad)


def test_d2_page1_round_trip():
    """Build a minimal D2 page 1 and verify its decoded fields."""
    spec = _spec_at_offsets({
        0: (PREAMBLE, 11),
        23: (1, 3),         # FraID = 1
        38: (1, 4),         # page_num = 1
        42: (0, 1),         # SatH1
        43: (5, 5),         # AODC
        48: (3, 4),         # URAI
        52: (2345, 13),     # WN
        65: (2000, 17),     # t_oc / 8 -> 16000 s
    })
    words = encode_subframe_words(spec)
    out = decode_d2_page1(words)
    assert out["subframe_id"] == 1
    assert out["page_num"] == 1
    assert out["satH1"] == 0
    assert out["AODC"] == 5
    assert out["URAI"] == 3
    assert out["week"] == 2345
    assert out["t_oc_s"] == 16000


def test_d2_page1_wrong_page_raises():
    bad = encode_subframe_words(
        _spec_at_offsets({0: (PREAMBLE, 11), 23: (1, 3), 38: (5, 4)})
    )
    with pytest.raises(ValueError, match="page=1"):
        decode_d2_page1(bad)


def test_encode_subframe_words_returns_ten_words():
    """Helper always returns 10 30-bit words even for short inputs."""
    words = encode_subframe_words([(PREAMBLE, 11)])
    assert len(words) == 10
    assert all(0 <= w < (1 << 30) for w in words)
