"""BeiDou navigation message D1 / D2 subframe decoder.

The BeiDou broadcast nav message comes in two flavors:

- **D1** (50 bps, MEO and IGSO satellites: BDS-2 and BDS-3) — 10 words
  x 30 bits per subframe = 300 bits total. Five subframes per frame:
  1 = clock + iono, 2/3 = ephemeris, 4/5 = almanac + integrity.
- **D2** (500 bps, GEO satellites) — same 30-bit word size but with
  a paginated structure (120 pages per superframe). Page 1 carries
  the same clock parameters as D1 subframe 1.

Reference: ICD-BDS-OS-200 (released by China Satellite Navigation
Office). The bit-level field offsets here match Table 5-3 / 5-4 of
that document.

The decoder takes pre-extracted 30-bit subframe words (typically
unpacked by the receiver firmware or by RTCM3 1042 / RXM-SFRBX) and
returns the structured fields. We do **not** validate the BeiDou BCH
parity — that's the receiver's job.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _native

#: BeiDou nav-message preamble (11 bits, 0x712 = 11100010010).
PREAMBLE = 0x712

#: Speed of light scale used in the clock-bias field (s -> m).
_C = 299_792_458.0

#: Pi constant exactly as defined in the BeiDou ICD.
_PI = 3.1415926535898


def _strip_parity(words: list[int]) -> str:
    """Strip the trailing parity bits from each word, return data bitstring.

    Word 1 contributes its high 26 bits (4 trailing parity); words 2-10
    each contribute their high 22 bits (8 trailing parity). Result is a
    26 + 9*22 = 224-bit MSB-first ``str``.
    """
    out = [f"{(words[0] >> 4) & ((1 << 26) - 1):026b}"]
    for w in words[1:]:
        out.append(f"{(w >> 8) & ((1 << 22) - 1):022b}")
    return "".join(out)


def _bits(data_bits: str, start: int, n_bits: int, *, signed: bool = False) -> int:
    """Read ``n_bits`` from a parity-stripped data bitstring (MSB-first)."""
    chunk = data_bits[start : start + n_bits]
    value = int(chunk, 2)
    if signed and chunk[0] == "1":
        value -= 1 << n_bits
    return value


def decode_d1_subframe1(words: list[int]) -> dict[str, Any]:
    """Decode a D1 subframe 1 (clock parameters + ionospheric model).

    Parameters
    ----------
    words:
        Ten 30-bit ``int``s, one per word of the subframe. Pass the
        raw-bits form (parity included); we look at specific bit
        offsets that fall inside the data bits.

    Returns
    -------
    dict
        Subframe ID + clock + ionospheric coefficients in SI units
        per ICD-BDS-OS-200 Table 5-3.

    Raises
    ------
    ValueError
        If the leading 11 bits don't match the BeiDou preamble.
    """
    if len(words) < 10:
        raise ValueError("D1 subframe needs 10 words")
    if _native.have_decode_beidou_d1_sf1():
        try:
            return _native.decode_beidou_d1_sf1(
                np.asarray(words, dtype=np.uint32))
        except Exception as e:
            msg = str(e).lower()
            if "preamble" in msg:
                raise ValueError(
                    f"bad BeiDou preamble (native)"
                ) from None
            if "subframe" in msg:
                raise ValueError(f"expected subframe 1 (native)") from None
            raise
    data = _strip_parity(words)

    # All offsets below are into the parity-stripped data bitstring.
    # Word 1 contributes 26 data bits; words 2-10 each 22 data bits.
    # Per ICD-BDS-OS-200 §5.2.3.2:
    #   bits 0-10  Pre (11)
    #   bits 11-14 Rev (4)
    #   bits 15-22 SOW high (8)
    #   bits 23-25 FraID (3)
    #   bits 26-37 SOW low (12)            -> SOW total 20 bits
    #   bit  38    SatH1 (1)
    #   bits 39-43 AODC (5)
    #   bits 44-47 URAI (4)
    #   bits 48-60 WN (13)
    #   bits 61-77 t_oc / 8 (17)
    #   bits 78-87 TGD1 (10, signed)
    #   bits 88-97 TGD2 (10, signed)
    #   ... iono coefficients ... (offsets continue)
    pre = _bits(data, 0, 11)
    if pre != PREAMBLE:
        raise ValueError(f"bad BeiDou preamble: {pre:#05x} != {PREAMBLE:#05x}")
    fra_id = _bits(data, 23, 3)
    if fra_id != 1:
        raise ValueError(f"expected subframe 1, got {fra_id}")

    sath1 = _bits(data, 38, 1)
    aodc = _bits(data, 39, 5)
    urai = _bits(data, 44, 4)
    wn = _bits(data, 48, 13)
    toc = _bits(data, 61, 17) * 8
    tgd1 = _bits(data, 78, 10, signed=True) * 0.1e-9
    tgd2 = _bits(data, 88, 10, signed=True) * 0.1e-9

    # Iono coefficients start at bit 98 in the data stream:
    alpha0 = _bits(data, 98, 8, signed=True) * 2**-30
    alpha1 = _bits(data, 106, 8, signed=True) * 2**-27
    alpha2 = _bits(data, 114, 8, signed=True) * 2**-24
    alpha3 = _bits(data, 122, 8, signed=True) * 2**-24
    beta0 = _bits(data, 130, 8, signed=True) * 2**11
    beta1 = _bits(data, 138, 8, signed=True) * 2**14
    beta2 = _bits(data, 146, 8, signed=True) * 2**16
    beta3 = _bits(data, 154, 8, signed=True) * 2**16
    a2 = _bits(data, 162, 11, signed=True) * 2**-66
    a0 = _bits(data, 173, 24, signed=True) * 2**-33
    a1 = _bits(data, 197, 22, signed=True) * 2**-50
    aode = _bits(data, 219, 5)

    return {
        "subframe_id": fra_id,
        "satH1": sath1,
        "AODC": aodc,
        "URAI": urai,
        "week": wn,
        "t_oc_s": toc,
        "TGD1_s": tgd1,
        "TGD2_s": tgd2,
        "iono_alpha": (alpha0, alpha1, alpha2, alpha3),
        "iono_beta": (beta0, beta1, beta2, beta3),
        "a0_s": a0,
        "a1_s_per_s": a1,
        "a2_s_per_s2": a2,
        "AODE": aode,
    }


def decode_d2_page1(words: list[int]) -> dict[str, Any]:
    """Decode a D2 page 1 (clock parameters for GEO satellites).

    D2 carries the same clock parameters as D1 subframe 1 but with
    different bit offsets per ICD-BDS-OS-200 §5.3. We extract the
    most-used fields (week, toc, a0/a1/a2, TGD1/TGD2).

    Parameters
    ----------
    words:
        Ten 30-bit ``int``s, one per word.

    Returns
    -------
    dict
        Same shape as :func:`decode_d1_subframe1` but with the D2
        page-1 layout. Iono coefficients are *not* present in D2 page 1;
        they live in page 2.

    Raises
    ------
    ValueError
        If the preamble or page-id don't match.
    """
    if len(words) < 10:
        raise ValueError("D2 page needs 10 words")
    if _native.have_decode_beidou_d2_page1():
        try:
            return _native.decode_beidou_d2_page1(
                np.asarray(words, dtype=np.uint32))
        except Exception as e:
            msg = str(e).lower()
            if "preamble" in msg:
                raise ValueError(
                    f"bad BeiDou preamble (native)"
                ) from None
            if "frame" in msg or "page" in msg:
                raise ValueError(
                    f"expected D2 frame=1 page=1 (native)"
                ) from None
            raise
    data = _strip_parity(words)

    # Word 1 of D2 has the same Pre/Rev/SOW/FraID/parity layout as D1.
    # Word 2 of D2 carries the page number in bits 26-29 (4 bits) plus
    # the rest of the SOW etc. We extract from the parity-stripped
    # 224-bit data string, so:
    #   bits 0-10  Pre
    #   bits 23-25 FraID
    #   bits 38-41 page number (4 bits, after the 12-bit SOW low)
    pre = _bits(data, 0, 11)
    if pre != PREAMBLE:
        raise ValueError(f"bad BeiDou preamble: {pre:#05x} != {PREAMBLE:#05x}")
    fra_id = _bits(data, 23, 3)
    page_num = _bits(data, 38, 4)
    if fra_id != 1 or page_num != 1:
        raise ValueError(
            f"expected D2 frame=1 page=1, got frame={fra_id} page={page_num}"
        )

    sath1 = _bits(data, 42, 1)
    aodc = _bits(data, 43, 5)
    urai = _bits(data, 48, 4)
    wn = _bits(data, 52, 13)
    toc = _bits(data, 65, 17) * 8
    tgd1 = _bits(data, 82, 10, signed=True) * 0.1e-9
    tgd2 = _bits(data, 92, 10, signed=True) * 0.1e-9
    a0 = _bits(data, 102, 24, signed=True) * 2**-33
    a1 = _bits(data, 126, 22, signed=True) * 2**-50
    a2 = _bits(data, 148, 11, signed=True) * 2**-66
    aode = _bits(data, 159, 5)

    return {
        "page_num": page_num,
        "subframe_id": fra_id,
        "satH1": sath1,
        "AODC": aodc,
        "URAI": urai,
        "week": wn,
        "t_oc_s": toc,
        "TGD1_s": tgd1,
        "TGD2_s": tgd2,
        "a0_s": a0,
        "a1_s_per_s": a1,
        "a2_s_per_s2": a2,
        "AODE": aode,
    }


def encode_subframe_words(field_specs: list[tuple[int, int]]) -> list[int]:
    """Helper for tests: pack ``[(value, n_bits), ...]`` into 10 30-bit words.

    The list of (value, n_bits) is treated as a stream of *data* bits
    (parity-stripped). The total data capacity is 224 bits (26 in word 1
    plus 22 in each of words 2-10). The result is a 10-element list of
    30-bit ints with **zero** parity bits (the decoders strip parity
    before reading, so this round-trips faithfully without us having to
    compute the BCH parity).
    """
    bits = ""
    for value, n in field_specs:
        if n <= 0:
            continue
        bits += f"{value & ((1 << n) - 1):0{n}b}"
    # Pad to the 224-bit data capacity.
    pad = max(0, 224 - len(bits))
    bits = (bits + "0" * pad)[:224]

    words: list[int] = []
    # Word 1: 26 data bits + 4 parity zeros.
    word1_data = int(bits[:26], 2)
    words.append(word1_data << 4)
    # Words 2-10: 22 data bits + 8 parity zeros each.
    for w in range(9):
        chunk = bits[26 + w * 22 : 26 + (w + 1) * 22]
        words.append(int(chunk, 2) << 8)
    return words


__all__ = [
    "PREAMBLE",
    "decode_d1_subframe1",
    "decode_d2_page1",
    "encode_subframe_words",
]


# Constants kept to silence "unused" if future code references them.
_ = _C
_ = _PI
