# Raw nav subframes

GNSS satellites broadcast their orbit and clock parameters in tightly-packed
bitstreams. The receiver demodulates the bits and either applies the
parity check, decodes the structured fields, and exposes them through a
RINEX NAV file, or passes the raw bits through in messages like u-blox
RXM-SFRBX for downstream decode.

`rinexpy` includes decoders for every modernised-signal nav message in the
operational constellations. Each decoder has the same shape: feed the
bit-packed bytes for one message body, get a dict with the named ICD fields.

| Constellation | Signal | Message | Decoder module |
| --- | --- | --- | --- |
| GPS | L1 C/A | LNAV (subframes 1-3) | `rinexpy.gps_lnav` |
| GPS | L2C, L5 | CNAV (MT 10, 11) | `rinexpy.gps_cnav` |
| GPS | L1C | CNAV-2 (subframe 2) | `rinexpy.gps_cnav2` |
| Galileo | E5a | F-NAV (pages 1, 2) | `rinexpy.galileo_nav` |
| Galileo | E1B, E5b | I-NAV (words 1, 4) | `rinexpy.galileo_nav` |
| GLONASS | L1OF, L2OF | strings 1-3 | `rinexpy.glonass` |
| BeiDou | B1I (MEO) | D1 subframe 1 | `rinexpy.beidou` |
| BeiDou | B1I (GEO) | D2 page 1 | `rinexpy.beidou` |
| NavIC | L5, S | subframes 1-4 | `rinexpy.navic` |
| SBAS | L1 | MT 1, 2-5, 6, 7, 9, 17, 18, 24-26 | `rinexpy.sbas` |

For the structured RINEX 4 NAV record types (STO, EOP, ION, EPH), the
high-level reader is `rinexpy.nav4.load_nav4`.

This page walks through each module. SBAS gets its own page; see
[SBAS and Galileo HAS](sbas-and-has.md).

## GPS LNAV

LNAV is the legacy GPS broadcast nav message on L1 C/A and L2 P(Y). Each
frame is 5 subframes, each subframe is 10 words, each word is 30 bits (24
data bits plus 6 parity bits). Parity is not validated by these decoders.

The relevant subframes are 1 (clock, T_GD, week, IODC), 2 (ephemeris
part 1), and 3 (ephemeris part 2).

```python
from rinexpy.gps_lnav import (
    PREAMBLE,
    decode_lnav_subframe1,
    decode_lnav_subframe2,
    decode_lnav_subframe3,
    encode_lnav_words,
)
print(hex(PREAMBLE))      # 0x8b
```

### Decoded fields

Subframe 1:

```
{'preamble': 0x8B, 'tlm': ..., 'how': ..., 'week_mod_1024': 0,
 'L2_codes': 0, 'ura_index': 0, 'sv_health': 0, 'iodc': 0,
 'l2_p_flag': 0, 't_gd_s': -1.4e-9, 't_oc_s': 86400.0,
 'a_f2_s_s2': 0.0, 'a_f1_s_s': 1.5e-12, 'a_f0_s': 4.2e-5}
```

Subframe 2:

```
{'preamble': 0x8B, 'tlm': ..., 'how': ..., 'iode': 0,
 'c_rs_m': 8.0, 'delta_n_rad_s': 4.5e-9, 'm_0_rad': 2.1,
 'c_uc_rad': -3.2e-7, 'e': 0.012, 'c_us_rad': 1.1e-5,
 'sqrt_a_sqrt_m': 5153.7, 't_oe_s': 86400.0,
 'fit_interval_flag': 0, 'aodo': 0}
```

Subframe 3:

```
{'preamble': 0x8B, 'tlm': ..., 'how': ..., 'c_ic_rad': 1.5e-7,
 'omega_0_rad': -1.8, 'c_is_rad': -2.0e-7, 'i_0_rad': 0.95,
 'c_rc_m': 250.0, 'omega_rad': -2.6,
 'omega_dot_rad_s': -8.1e-9, 'iode': 0, 'idot_rad_s': -1.0e-10}
```

### Usage

The decoders take a list of 10 30-bit integer words (one subframe per call).
Most receivers expose subframes as 32-bit words; either right-shift by 2 or
mask off the parity bits.

```python
from rinexpy.gps_lnav import decode_lnav_subframe1
words_sf1 = [
    0x22C000B4, 0x05CE6178, 0x1C00B3B0, 0x00000000, 0x00000000,
    0x00000000, 0x000000FF, 0xFFFFFF24, 0x0000FFFF, 0xC3801FFF,
]
out = decode_lnav_subframe1(words_sf1)
print(out["t_gd_s"], out["t_oc_s"], out["a_f0_s"])
```

### Test helper

`encode_lnav_words` packs `[(value, n_bits), ...]` into the 10-word
subframe layout for test fixtures.

## GPS CNAV (L2C, L5)

CNAV is the modernised GPS broadcast message on L2C (50 sps) and L5 (100
sps). It carries the ephemeris in messages rather than subframes; each
message is 300 bits / 6 seconds.

Two of the message types carry ephemerides: MT 10 (clock plus ephemeris
part 1) and MT 11 (ephemeris part 2). The combination is the modern
equivalent of LNAV subframes 1+2+3.

```python
from rinexpy.gps_cnav import (
    PREAMBLE,
    decode_cnav_mt10,
    decode_cnav_mt11,
    decode_cnav_message,
)
```

### Decoded fields

MT 10:

```
{'header': {...}, 'week': 1234, 'ura_index': 0,
 'sv_health_l1': 0, 'sv_health_l2': 0, 'sv_health_l5': 0,
 't_op_s': 86400.0, 't_oe_s': 86400.0,
 'delta_A_m': 0.0, 'a_dot_m_s': 0.0,
 'delta_n0_rad_s': 4.5e-9, 'delta_n0_dot_rad_s2': 0.0,
 'm0_n_rad': 2.1, 'e_n': 0.012, 'omega_n_rad': -2.6}
```

MT 11:

```
{'header': {...}, 'omega_0_rad': -1.8, 'i0_rad': 0.95,
 'delta_omega_dot_rad_s': 1e-12, 'idot_rad_s': -1e-10,
 'c_ic_rad': 1.5e-7, 'c_is_rad': -2e-7,
 'c_rc_m': 250.0, 'c_rs_m': 8.0,
 'c_uc_rad': -3.2e-7, 'c_us_rad': 1.1e-5}
```

### Dispatching

```python
out = decode_cnav_message(cnav_bytes)
if out["mt"] == 10:
    print(out["t_oe_s"])
elif out["mt"] == 11:
    print(out["i0_rad"])
```

`decode_cnav_message` reads the 6-bit message type from the header and
dispatches. Unknown types come back as `{'header': ..., 'raw': bytes}`.

## GPS CNAV-2 (L1C)

CNAV-2 is the L1C civilian signal's nav message. Each frame is structured
as SF1 (TOI, 9 bits) + SF2 (1200 bits of clock plus ephemeris) + SF3
(274 bits of pages).

The full ephemeris is in SF2; `rinexpy` decodes it.

```python
from rinexpy.gps_cnav2 import decode_cnav2_subframe2

out = decode_cnav2_subframe2(payload_1200_bits)
print(out["TOW_s"], out["WN"], out["t_oe_s"])
print(out["m0_n_rad"], out["sqrt_a_n_sqrt_m"])
```

The field set is the same as CNAV MT 10 + MT 11 combined.

## Galileo F-NAV (E5a)

F-NAV is broadcast on E5a at 25 sps. Each F-NAV page is 244 bits / 10
seconds. Pages 1 and 2 carry the ephemeris.

```python
from rinexpy.galileo_nav import decode_fnav_page1, decode_fnav_page2

print(decode_fnav_page1(page_bytes))     # clock + ephemeris part 1
print(decode_fnav_page2(page_bytes))     # ephemeris part 2
```

Decoded fields:

Page 1 (clock + part 1):

```
{'iod_nav': 0, 't_0c_s': 86400.0,
 'a_f0_s': 4.2e-5, 'a_f1_s_s': 1.5e-12, 'a_f2_s_s2': 0.0,
 'sisa_index': 0, 'sv_id': 0,
 't_0e_s': 86400.0, 'm_0_rad': 2.1, 'sqrt_a_sqrt_m': 5440.6,
 'e': 0.001, 'omega_rad': -2.6, 'c_us_rad': 0.0}
```

Page 2 (part 2):

```
{'iod_nav': 0, 'omega_0_rad': -1.8, 'i_0_rad': 0.96,
 'omega_dot_rad_s': -5.7e-9, 'idot_rad_s': 0.0,
 'delta_n_rad_s': 1.5e-9, 'c_uc_rad': -7e-7, 'c_us_rad': 1e-5,
 'c_rc_m': 250.0, 'c_rs_m': -10.0,
 'c_ic_rad': 0.0, 'c_is_rad': 0.0, 't_0e_s': 86400.0}
```

## Galileo I-NAV (E1B, E5b)

I-NAV is broadcast on E1B and E5b at 250 sps. Each I-NAV word is 128 bits.
Word types 1 and 4 carry the main ephemeris and clock data.

```python
from rinexpy.galileo_nav import decode_inav_word1, decode_inav_word4

print(decode_inav_word1(word_bytes))     # ephemeris part 1
print(decode_inav_word4(word_bytes))     # clock correction
```

Word 1: `iod_nav`, `t_oe_s`, `m_0_rad`, `e`, `sqrt_a_sqrt_m`, `omega_rad`.
Word 4: `iod_nav`, `sv_id`, `t_0c_s`, `a_f0_s`, `a_f1_s_s`, `a_f2_s_s2`,
plus SISA index and signal-health flags.

## GLONASS L1OF / L2OF strings

GLONASS broadcasts at 50 bps in 5-string superframes (per ICD §4.4 of
GLONASS Edition 5.1). Strings 1 to 3 carry the ECEF position, velocity,
and acceleration components in sign-magnitude encoding.

```python
from rinexpy.glonass import (
    decode_glonass_string1,
    decode_glonass_string2,
    decode_glonass_string3,
    decode_glonass_string,
)
```

### Decoded fields

String 1:

```
{'P1': 0, 't_k_h': 0, 't_k_min': 0, 't_k_30sec': 0,
 'x_km': 7501.234, 'x_dot_km_s': -2.103, 'x_dot_dot_km_s2': 1.0e-5}
```

String 2:

```
{'B_n': 0, 'P2': 0, 't_b_15min': 56,
 'y_km': -8123.456, 'y_dot_km_s': 1.701, 'y_dot_dot_km_s2': 0.0}
```

String 3:

```
{'gamma_n': 1.2e-13, 'P3': 0, 'l_n': 0,
 'z_km': -25117.890, 'z_dot_km_s': -2.001, 'z_dot_dot_km_s2': -1.2e-6}
```

### Dispatching

```python
out = decode_glonass_string(payload)
n = out.get("string")       # 1, 2, 3, or the raw bytes for others
```

`decode_glonass_string` reads the 4-bit string number and dispatches.
Strings outside the 1-3 set come back with `{"string": m, "raw": bytes}`.

### Frequency helpers

GLONASS uses frequency-division multiple access. Each satellite transmits
on a slightly different carrier frequency indexed by a channel number
(-7 to +6).

```python
from rinexpy.glonass import (
    l1_frequency_hz,
    l2_frequency_hz,
    l1_wavelength_m,
    l2_wavelength_m,
    CHANNEL_MIN,
    CHANNEL_MAX,
)
print(CHANNEL_MIN, CHANNEL_MAX)    # -7 6
print(l1_frequency_hz(0))          # 1602000000.0
print(l1_wavelength_m(-7))         # ~0.1873 m
```

For per-SV iono-free combinations against GLONASS observations:

```python
from rinexpy.glonass import iono_free_pseudorange, iono_free_phase
import numpy as np

channels = np.array([1, -4, 2, 0])
ifp = iono_free_pseudorange(p1_m, p2_m, channels)
ifl = iono_free_phase(l1_m, l2_m, channels)
```

The functions vectorise over the SV axis using the per-SV channel
number, since GLONASS L1/L2 frequencies vary per satellite.

## BeiDou D1 / D2

BeiDou satellites broadcast nav messages in two formats. D1 is for MEO and
IGSO satellites at 50 bps (10 words × 30 bits per subframe). D2 is for GEO
satellites at 500 bps (paginated, with 120 pages per superframe).

The clock parameters and the broadcast ionospheric model (Klobuchar-style)
are in D1 subframe 1 and D2 page 1.

```python
from rinexpy.beidou import (
    PREAMBLE,
    decode_d1_subframe1,
    decode_d2_page1,
    encode_subframe_words,
)
print(hex(PREAMBLE))         # 0x712 (11 bits: 11100010010)
```

### Decoded fields

D1 subframe 1:

```
{'preamble': 0x712, 'rev': 0, 'fra_id': 1, 'sow': 0,
 'sat_h1': 0, 'aodc': 0, 'urai': 0, 'wn': 0,
 't_oc_s': 86400.0, 'tgd1_s': -1.5e-9, 'tgd2_s': 0.0,
 'alpha': (a0, a1, a2, a3), 'beta': (b0, b1, b2, b3),
 'a0_s': 4.2e-5, 'a1_s_s': 1.5e-12, 'a2_s_s2': 0.0,
 'aode': 0}
```

D2 page 1 has the same clock + iono parameters with different bit offsets
per ICD-BDS-OS-200 §5.3.

### Building test inputs

`encode_subframe_words` packs `[(value, n_bits), ...]` into the
10-word subframe layout for test fixtures.

```python
specs = [(0x712, 11), (0, 4), (1, 3), (0, 12), (0, 64), ...]
words = encode_subframe_words(specs)
```

Words come back as 30-bit integers (the parity bits are zeros).

## NavIC subframes

NavIC (IRNSS) satellites broadcast on L5 and S-band. Each subframe is 600
bits / 12 s with the layout: 16-bit sync (`0xEB90`) + 292-bit data + 24-bit
CRC + 6 tail bits. Subframes 1 and 2 carry the ephemeris; subframes 3 and 4
carry paginated almanac, iono, and UTC data.

```python
from rinexpy.navic import (
    SYNC,
    decode_navic_subframe1,
    decode_navic_subframe2,
    decode_navic_subframe34,
    decode_navic_subframe,
)
print(hex(SYNC))      # 0xeb90
```

### Decoded fields

Subframe 1 (clock + ephemeris part 1):

```
{'sf_id': 1, 'tow_count': 0, 'alert': 0, 'autonav': 0,
 'subframe_id': 0, 'sv_id': 0,
 'wn': 1234, 'a_f0_s': 4.2e-5, 'a_f1_s_s': 1.5e-12, 'a_f2_s_s2': 0.0,
 'ura_index': 0, 't_oc_s': 86400.0, 't_gd_s': -1.5e-9,
 'delta_n_rad_s': 4.5e-9, 'iodec': 0,
 'l5_health': 0, 's_health': 0,
 'c_uc_rad': -3.2e-7, 'c_us_rad': 1.1e-5,
 'c_ic_rad': 1.5e-7, 'c_is_rad': -2e-7,
 'c_rc_m': 250.0, 'c_rs_m': 8.0,
 'idot_rad_s': -1e-10}
```

Subframe 2 (ephemeris part 2):

```
{'sf_id': 2, ..., 'm_0_rad': 2.1, 't_oe_s': 86400.0,
 'e': 0.012, 'sqrt_a_sqrt_m': 5440.6,
 'omega_0_rad': -1.8, 'omega_rad': -2.6,
 'omega_dot_rad_s': -8.1e-9, 'i_0_rad': 0.95}
```

Subframes 3 and 4 come back as `{'sf_id': 3 or 4, 'message_id': N,
'raw': bytes}` so you can dispatch on the message ID for the paginated
content downstream.

### Dispatching

```python
out = decode_navic_subframe(payload)
print(out["sf_id"])      # 1, 2, 3, or 4
```

## SBAS L1

SBAS has its own page since it covers ten message types and a separate
ICD. See [SBAS and Galileo HAS](sbas-and-has.md).

## RINEX 4 NAV: STO, EOP, ION, EPH

When you do not want to walk bitstreams yourself, the RINEX 4 NAV reader
returns the structured records directly.

```python
from rinexpy.nav4 import load_nav4

out = load_nav4("BRDC00WRD_S_20231560000_01D_MN.rnx")
for sto in out["STO"]:
    print(sto["sv"], sto["message_type"], sto["A0_s"])
for ion in out["ION"]:
    if ion["model"] == "KLOB":
        print("Klobuchar alpha:", ion["alpha"], "beta:", ion["beta"])
    elif ion["model"] == "NEQG":
        print("NeQuick-G:", ion["a0"], ion["a1"], ion["a2"])
    elif ion["model"] == "BDGIM":
        print("BDGIM:", ion["coefficients"])
for eop in out["EOP"]:
    print(eop["ref_time"], eop["PM_X"], eop["PM_Y"], eop["delta_UT1"])
for eph in out["EPH"]:
    print(eph["sv"], eph["toe_s"])
```

The reader recognises the four record categories (`EPH`, `STO`, `EOP`,
`ION`) and dispatches per message type. STO records carry the polynomial
fit between two time scales (`A0`, `A1`, `A2` plus reference epoch). EOP
records carry polar motion and UT1-UTC. ION records carry the constellation-
specific ionospheric model (Klobuchar, NeQuick-G, BDGIM).

## Decoder shape

Every decoder follows the same pattern.

```python
out = decode_<system>_<message>(payload_bytes)
```

`payload_bytes` is the message body, packed into a bytes object such that
bit 0 of byte 0 is the first bit on the wire. Lengths follow the ICD; the
docstring on each decoder lists the required byte count.

The output dict is keyed by the ICD field name. Numeric fields are scaled
to SI units (seconds, metres, radians) where the wire format uses scaled
integers; the field name carries the unit suffix (`a_f0_s`, `sqrt_a_sqrt_m`,
`omega_rad`).

Fields that the ICD declares as bit-flag groups come back as integers; you
extract individual bits yourself.

## Related pages

- [RINEX navigation files](rinex-nav.md): the higher-level reader.
- [SBAS and Galileo HAS](sbas-and-has.md): SBAS L1 and HAS decoders.
- [Receiver binary formats](receiver-binary.md): how to extract subframes from a UBX log.
