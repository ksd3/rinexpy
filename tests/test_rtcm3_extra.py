"""Tests for the recently-added RTCM3 1029 and 1230 decoders."""

from __future__ import annotations

from rinexpy.rtcm3 import decode_message


def _pack(field_specs: list[tuple[int, int]], msg_id: int) -> bytes:
    """Pack ``[(value, n_bits), ...]`` into a body that starts with msg_id."""
    bits = f"{msg_id:012b}"
    for value, n in field_specs:
        if n <= 0:
            continue
        bits += f"{value & ((1 << n) - 1):0{n}b}"
    # Pad to a byte boundary.
    pad = (-len(bits)) % 8
    bits += "0" * pad
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def test_decode_1230_with_all_four_signals():
    """All four signal-mask bits set: every bias appears in the output."""
    body = _pack(
        [
            (42, 12),       # station ID
            (1, 1),         # bias indicator: aligned
            (0, 3),         # reserved
            (0b1111, 4),    # mask: every signal active
            (50, 16),       # L1 C/A bias = 50 * 0.02 = 1.0 m
            (-25, 16),      # L1 P bias = -0.5 m
            (75, 16),       # L2 C/A bias = 1.5 m
            (0, 16),        # L2 P bias = 0
        ],
        1230,
    )
    out = decode_message(1230, body)
    assert out["msg_id"] == 1230
    assert out["station_id"] == 42
    assert out["bias_indicator"] == 1
    assert out["signal_mask"] == 0b1111
    assert out["biases_m"] == {
        "L1_CA": 1.0,
        "L1_P": -0.5,
        "L2_CA": 1.5,
        "L2_P": 0.0,
    }


def test_decode_1230_partial_mask():
    """Only L1 C/A + L2 P active: just those two biases appear."""
    body = _pack(
        [
            (7, 12),
            (0, 1),
            (0, 3),
            (0b1001, 4),    # L1 C/A + L2 P
            (-100, 16),     # L1 C/A = -2.0 m
            (50, 16),       # L2 P = 1.0 m
        ],
        1230,
    )
    out = decode_message(1230, body)
    assert set(out["biases_m"]) == {"L1_CA", "L2_P"}
    assert out["biases_m"]["L1_CA"] == -2.0
    assert out["biases_m"]["L2_P"] == 1.0


def test_decode_1230_empty_mask():
    """No mask bits set: biases dict is empty."""
    body = _pack(
        [
            (99, 12),
            (1, 1),
            (0, 3),
            (0, 4),
        ],
        1230,
    )
    out = decode_message(1230, body)
    assert out["biases_m"] == {}
    assert out["signal_mask"] == 0


def test_decode_1029_ascii_text():
    """A short ASCII message round-trips through the decoder."""
    text = "hello"
    body = _pack(
        [
            (123, 12),    # station ID
            (60000, 16),  # MJD
            (12345, 17),  # SoD
            (len(text), 7),
            (len(text), 8),
            *[(ord(c), 8) for c in text],
        ],
        1029,
    )
    out = decode_message(1029, body)
    assert out["msg_id"] == 1029
    assert out["station_id"] == 123
    assert out["mjd"] == 60000
    assert out["sod_s"] == 12345
    assert out["n_chars"] == 5
    assert out["n_bytes"] == 5
    assert out["text"] == "hello"


def test_decode_1029_utf8_multibyte():
    """Multi-byte UTF-8 character ('é' = 0xC3 0xA9) decodes correctly."""
    text_bytes = "café".encode("utf-8")
    body = _pack(
        [
            (1, 12),
            (60000, 16),
            (0, 17),
            (4, 7),
            (len(text_bytes), 8),
            *[(b, 8) for b in text_bytes],
        ],
        1029,
    )
    out = decode_message(1029, body)
    assert out["text"] == "café"


def test_decode_unknown_msg_id_falls_back_to_raw():
    """An unrecognised msg_id returns the raw payload, as before."""
    body = _pack([(0, 8)], 9999)
    out = decode_message(9999, body)
    assert out["msg_id"] == 9999
    assert "payload_bytes" in out
