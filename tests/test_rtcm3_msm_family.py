"""MSM 1 / 2 / 3 / 5 / 6 decoder tests.

The bundled RTKLIB capture exercises MSM7; tests here build small
synthetic single-SV / single-signal MSM bodies for each of the
remaining MSM kinds and check the layout / scale conventions.
"""

from __future__ import annotations

import pytest

from rinexpy.rtcm3 import decode_message


def pack_bits(*fields: tuple[int, int]) -> bytes:
    s = ""
    for v, n in fields:
        if v < 0:
            v = (1 << n) + v
        s += f"{v & ((1 << n) - 1):0{n}b}"
    if len(s) % 8:
        s += "0" * (8 - len(s) % 8)
    return bytes(int(s[i:i + 8], 2) for i in range(0, len(s), 8))


def _msm_header_fields(msg_id: int, sv_idx: int, sig_idx: int) -> list[tuple[int, int]]:
    """Return the (value, n_bits) tuples for the MSM header + masks
    with a single SV at slot `sv_idx` and a single signal at slot
    `sig_idx`, both relative to 0-based mask positions. Returns
    everything up to and including the cell-mask bit."""
    sv_mask = 1 << (63 - sv_idx)
    sig_mask = 1 << (31 - sig_idx)
    return [
        (msg_id, 12),                # MSG ID
        (1, 12),                     # station_id
        (0, 30),                     # tow_ms
        (0, 1),                      # sync
        (0, 3),                      # iod
        (0, 7), (0, 2), (0, 2),      # session/clock-steering/external clock
        (0, 1), (0, 3),              # smoothing indicator + interval
        (sv_mask >> 32, 32),         # sv_mask hi
        (sv_mask & 0xFFFFFFFF, 32),  # sv_mask lo
        (sig_mask, 32),              # sig_mask
        (1, 1),                      # cell_mask: the only cell is present
    ]


def test_msm1_gps_code_only():
    """1071 (GPS MSM1): code pseudoranges, no phase / Doppler."""
    body = pack_bits(
        *_msm_header_fields(1071, sv_idx=0, sig_idx=0),
        # Sat block (18 bits): rough_int(8) + rough_mod_1ms(10)
        (75, 8),                     # rough_int_ms = 75 ms (~22500 km)
        (512, 10),                   # rough_mod_1ms = 0.5 ms
        # Cell block (26 bits): fine_pr(15s) + lock(4) + halfcyc(1) + cnr(6)
        (1000, 15), (5, 4), (0, 1), (45, 6),
    )
    out = decode_message(1071, body)
    obs = out["observations"][0]
    # Code present.
    assert obs["pseudorange_m"] == pytest.approx(
        (75 + 512 / 1024.0 + 1000 * 2 ** -24) * 299_792.458
    )
    # Phase NaN (MSM1 doesn't transmit phase).
    import math
    assert math.isnan(obs["phase_m"])
    assert math.isnan(obs["doppler_mps"])


def test_msm2_gps_phase_only():
    """1072 (GPS MSM2): phase only, no code."""
    body = pack_bits(
        *_msm_header_fields(1072, 0, 0),
        (75, 8), (512, 10),
        # Cell (33 bits): fine_phase(22s) + lock(4) + halfcyc(1) + cnr(6)
        (12345, 22), (3, 4), (1, 1), (50, 6),
    )
    out = decode_message(1072, body)
    obs = out["observations"][0]
    import math
    assert math.isnan(obs["pseudorange_m"])
    assert obs["phase_m"] == pytest.approx(
        (75 + 512 / 1024.0 + 12345 * 2 ** -29) * 299_792.458
    )


def test_msm3_gps_code_phase_no_doppler():
    """1073 (GPS MSM3): code + phase, no Doppler."""
    body = pack_bits(
        *_msm_header_fields(1073, 0, 0),
        (75, 8), (512, 10),
        # Cell: fine_pr(15s) + fine_phase(22s) + lock(4) + halfcyc(1) + cnr(6) = 48
        (1000, 15), (12345, 22), (5, 4), (0, 1), (45, 6),
    )
    out = decode_message(1073, body)
    obs = out["observations"][0]
    import math
    assert not math.isnan(obs["pseudorange_m"])
    assert not math.isnan(obs["phase_m"])
    assert math.isnan(obs["doppler_mps"])


def test_msm5_galileo_with_doppler_low_precision():
    """1095 (Galileo MSM5): MSM3 plus fine Doppler."""
    body = pack_bits(
        *_msm_header_fields(1095, 0, 0),
        # Sat block (36 bits): includes ext_info(4) and rough_doppler(14s)
        (75, 8), (3, 4), (512, 10), (-100, 14),
        # Cell block (63 bits): 15+22+4+1+6+15
        (1000, 15), (12345, 22), (5, 4), (0, 1), (45, 6), (2000, 15),
    )
    out = decode_message(1095, body)
    sat = out["satellites"][0]
    obs = out["observations"][0]
    assert sat["sv"].startswith("E")
    assert sat["rough_doppler_mps"] == -100
    assert obs["doppler_mps"] == pytest.approx(2000 * 1e-4)


def test_msm6_qzss_high_precision_no_doppler():
    """1116 (QZSS MSM6): high precision (20+24 fine, 10+10 lock+cnr), no
    Doppler."""
    body = pack_bits(
        *_msm_header_fields(1116, 0, 0),
        # Sat block (22 bits): rough_int + ext_info + rough_mod_1ms
        (75, 8), (3, 4), (512, 10),
        # Cell block (65 bits): 20+24+10+1+10
        (5000, 20), (250000, 24), (123, 10), (1, 1), (320, 10),
    )
    out = decode_message(1116, body)
    obs = out["observations"][0]
    sat = out["satellites"][0]
    assert sat["sv"].startswith("J")
    # MSM6 uses the high-precision scale factors (2^-29, 2^-31).
    assert obs["pseudorange_m"] == pytest.approx(
        (75 + 512 / 1024.0 + 5000 * 2 ** -29) * 299_792.458
    )
    # cnr is high-precision: cnr_raw / 16.
    assert obs["cnr_dbhz"] == pytest.approx(320 / 16.0)
    import math
    assert math.isnan(obs["doppler_mps"])
