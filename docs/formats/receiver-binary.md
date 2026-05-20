# Receiver binary formats

GNSS receivers stream binary logs in their own native format before any
RINEX conversion. The four common ones are u-blox UBX, Septentrio SBF,
NovAtel OEM, and UNAVCO BINEX. NMEA-0183 is an ASCII overlay that almost
every receiver also emits.

Each format has a `rinexpy` module with the same shape: a sync byte
constant, a CRC helper, and an `iter_messages` (or `iter_blocks` /
`iter_records`) generator that yields decoded dicts.

| Format | Module | Sync | CRC | Generator |
| --- | --- | --- | --- | --- |
| NMEA-0183 | `rinexpy.nmea` | `$` | XOR checksum | `iter_lines` |
| u-blox UBX | `rinexpy.ubx` | `0xB5 0x62` | 8-bit Fletcher | `iter_messages` |
| Septentrio SBF | `rinexpy.sbf` | `0x24 0x40` (`$@`) | CRC-CCITT | `iter_blocks` |
| NovAtel OEM | `rinexpy.novatel` | `0xAA 0x44 0x12` | CRC-32 IEEE-802.3 | `iter_messages` |
| UNAVCO BINEX | `rinexpy.binex` | `0xC2` | XOR / CRC-16 / CRC-32 | `iter_records` |
| Furuno GW-10 | `rinexpy.gw10` | `0x8B` | summing checksum | `iter_frames` |

The decoders cover framing for every format and the high-value records for
each: full PVT, satellite tracking, raw observations where applicable, and
broadcast ephemeris where present. Other record IDs come back with the raw
payload bytes so you can dispatch downstream.

## NMEA-0183

NMEA-0183 sentences are ASCII lines of the form
`$TALKER,field1,...,fieldN*HH` where `TALKER` is a two-letter origin (GP
for GPS, GL for GLONASS, GA for Galileo, BD or GB for BeiDou, GN for
multi-constellation), `field1..fieldN` are comma-delimited values, and
`*HH` is an XOR checksum.

### Decoder

```python
from rinexpy.nmea import iter_lines, parse_sentence, checksum

s = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
msg = parse_sentence(s)
print(msg)
```

The output dict for a GGA sentence is:

```
{
  'talker': 'GP',
  'type':   'GGA',
  'raw':    '$GPGGA,...',
  'fields': ['123519', '4807.038', 'N', '01131.000', 'E', ...],
  'time':         datetime.time(12, 35, 19),
  'lat':          48.1173,
  'lon':          11.516666666666667,
  'fix_quality':  1,
  'n_sat':        8,
  'hdop':         0.9,
  'altitude_m':   545.4,
  'geoid_sep_m':  46.9,
}
```

The decoded sentence types are:

| Type | What | Key fields |
| --- | --- | --- |
| `GGA` | fix data | `time`, `lat`, `lon`, `fix_quality`, `n_sat`, `hdop`, `altitude_m` |
| `RMC` | recommended minimum nav | `datetime`, `lat`, `lon`, `speed_kn`, `course_deg`, `status` |
| `GSA` | DOP and active SVs | `mode`, `fix`, `prns`, `pdop`, `hdop`, `vdop` |
| `GSV` | SVs in view | `total_msgs`, `msg_num`, `n_sv`, `svs` (list of `{prn, el, az, snr}`) |
| `VTG` | track and ground speed | `course_true`, `course_mag`, `speed_kn`, `speed_kmh` |

Sentences whose type is not in the list above come back with the raw
`fields` list plus the `talker` and `type` keys.

### Iterating a log

```python
with open("nmea.log") as fp:
    for msg in iter_lines(fp):
        if msg["type"] == "GGA" and msg["fix_quality"]:
            print(msg["time"], msg["lat"], msg["lon"], msg["altitude_m"])
```

Lines that fail the CRC check (or are non-NMEA noise on the same serial
line) are silently skipped. To inspect them, set `check_crc=False`.

### Building a sentence in tests

```python
from rinexpy.nmea import checksum

body = "GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"
sentence = f"${body}*{checksum(body):02X}"
```

## u-blox UBX

u-blox UBX is a little-endian binary protocol with a two-byte sync
(`0xB5 0x62`), a 1-byte class, a 1-byte ID, a 16-bit length, the payload,
and a 16-bit (8-bit Fletcher) checksum.

### Decoder

```python
from rinexpy.ubx import iter_messages, decode_message, SYNC1, SYNC2

print(hex(SYNC1), hex(SYNC2))    # 0xb5 0x62

with open("ublox.ubx", "rb") as fp:
    for msg in iter_messages(fp, check_crc=True):
        cls, mid = msg["msg_class"], msg["msg_id"]
        if (cls, mid) == (0x01, 0x07):       # NAV-PVT
            print(msg["lat_deg"], msg["lon_deg"], msg["fix_type"])
        if (cls, mid) == (0x01, 0x35):       # NAV-SAT
            for s in msg["satellites"]:
                print(s["prn"], s["cn0"], s["az"], s["el"])
        if (cls, mid) == (0x02, 0x15):       # RXM-RAWX
            print(msg["n_meas"], "raw measurements at",
                  msg["rcv_tow"], "TOW")
```

### Decoded messages

| Class / ID | Name | Key fields |
| --- | --- | --- |
| `0x01, 0x07` | NAV-PVT | `lat_deg`, `lon_deg`, `height_m`, `fix_type`, `vel_n_m_s`, `vel_e_m_s`, `vel_d_m_s`, `gdop`, `pdop` |
| `0x01, 0x35` | NAV-SAT | `n_sat`, `satellites` (list of per-SV dicts with `gnss_id`, `prn`, `cn0`, `el`, `az`, `flags`) |
| `0x01, 0x21` | NAV-TIMEUTC | `year`, `month`, `day`, `hour`, `min`, `sec`, `nano`, `valid_flags` |
| `0x01, 0x12` | NAV-VELNED | `vel_n_m_s`, `vel_e_m_s`, `vel_d_m_s`, `speed`, `heading_deg`, `s_acc`, `c_acc` |
| `0x01, 0x22` | NAV-CLOCK | `clk_bias_ns`, `clk_drift_ns_s`, `t_acc`, `f_acc` |
| `0x01, 0x04` | NAV-DOP | `gdop`, `pdop`, `tdop`, `vdop`, `hdop`, `ndop`, `edop` |
| `0x02, 0x15` | RXM-RAWX | `rcv_tow`, `week`, `leap_s`, `n_meas`, `measurements` (list of per-SV pseudorange / phase / Doppler / cn0 / locktime / SV flags) |
| `0x02, 0x13` | RXM-SFRBX | `gnss_id`, `prn`, `freq_id`, `n_words`, `words` (list of 32-bit nav-message words) |

Unknown class/ID pairs come back as `{"msg_class", "msg_id", "length",
"payload_bytes"}`.

### Sync hunting

The decoder is resync-safe. If you point it at a binary log that contains
non-UBX bytes (a NMEA preamble, junk from a tracking lock failure), it
scans for the next `0xB5 0x62` and continues. The cost is a one-byte-at-a-
time scan during the resync window.

## Septentrio SBF

Septentrio SBF blocks start with the 2-byte sync `$@`, a 16-bit CRC (CCITT
polynomial 0x1021), a 16-bit block ID, a 16-bit length, then the body
which contains a TOW (32-bit) and a WNc (16-bit) plus block-specific
fields.

### Decoder

```python
from rinexpy.sbf import iter_blocks, SYNC, crc_ccitt

print(SYNC)            # b'$@'

with open("septentrio.sbf", "rb") as fp:
    for blk in iter_blocks(fp, check_crc=True):
        if blk["block_id"] == 4007:                  # PVTGeodetic
            print(blk["lat_rad"], blk["lon_rad"], blk["height_m"])
        if blk["block_id"] == 4027:                  # MeasEpoch
            print(blk["n_satellites"], "satellites at",
                  blk["tow_ms"], "TOW")
        if blk["block_id"] == 5891:                  # GPSNav
            print(blk["prn"], blk["t_oc"], blk["af0"])
```

### Decoded blocks

| Block ID | Name | Key fields |
| --- | --- | --- |
| 4007 | PVTGeodetic | `lat_rad`, `lon_rad`, `height_m`, `mode`, `error`, `dop_factors` |
| 4027 | MeasEpoch | `tow_ms`, `wnc`, `n_satellites`, `cn0_per_sat`, `pseudorange_per_sat`, ... |
| 5891 | GPSNav | `prn`, `t_oc`, `t_oe`, `af0`, `af1`, `af2`, `m0`, `sqrt_a`, ... |

Other block IDs return with `payload_bytes` and the parsed header fields.

## NovAtel OEM

NovAtel OEM logs use a 3-byte sync `0xAA 0x44 0x12`, a 28-byte long header,
the body, and an IEEE-802.3 32-bit CRC.

### Decoder

```python
from rinexpy.novatel import iter_messages, SYNC, crc32

print(SYNC.hex())             # 'aa4412'

with open("novatel.bin", "rb") as fp:
    for msg in iter_messages(fp, check_crc=True):
        if msg["msg_id"] == 42:        # BESTPOS
            print(msg["lat_deg"], msg["lon_deg"], msg["height_m"], msg["lat_sigma"])
        if msg["msg_id"] == 241:       # BESTXYZ
            print(msg["x"], msg["y"], msg["z"])
        if msg["msg_id"] == 41:        # RAWEPHEM
            print(msg["prn"], msg["sf1"][:4], "...")
```

### Decoded messages

| Msg ID | Name | Key fields |
| --- | --- | --- |
| 42 | BESTPOS | `lat_deg`, `lon_deg`, `height_m`, `undulation`, `lat_sigma`, `lon_sigma`, `n_svs`, `pos_type` |
| 241 | BESTXYZ | `x`, `y`, `z`, `vx`, `vy`, `vz`, `sol_status`, `x_sigma`, ... |
| 41 | RAWEPHEM | `prn`, `ref_week`, `ref_secs`, `sf1`, `sf2`, `sf3` (the 30-byte subframe blocks) |

## UNAVCO BINEX

BINEX (Binary EXchange) is UNAVCO's container for GNSS data archival. The
format is structured as variable-length records with a `0xC2` forward
sync byte, a `ubnxi` (variable-length integer) record ID, another `ubnxi`
length, the body, and a checksum whose width depends on the body length.

| Body length | Checksum |
| --- | --- |
| 0..127 bytes | 1-byte XOR |
| 128..4095 | CRC-16/CCITT |
| 4096+ | CRC-32 |

### Decoder

```python
from rinexpy.binex import iter_records, SYNC, read_ubnxi, encode_ubnxi

print(hex(SYNC))                  # 0xc2

with open("archive.bnx", "rb") as fp:
    for rec in iter_records(fp, check_crc=True):
        print(f"record {rec['record_id']:#04x}, "
              f"{rec['length']} bytes")
```

Each yielded record dict carries `record_id`, `length`, and `body_bytes`.
Record-body decoding is not currently implemented; the bytes come back raw
so you can dispatch to a downstream decoder.

### ubnxi

The variable-length integer encoder/decoder is exposed for tests.

```python
print(encode_ubnxi(127))          # b'\x7f'
print(encode_ubnxi(200))          # b'\xc8\x01'
import io
print(read_ubnxi(io.BytesIO(b'\x7f')))   # 127
```

## Furuno GW-10

The GW-10 receiver from Furuno wraps its messages in a simple framed
binary format: `0x8B (sync) | ID | payload (length by ID) | checksum`.
The `rinexpy.gw10` module exposes the framer plus a dedicated SBAS L1
extractor (the receiver's main feature is its SBAS L1 dump).

### Decoder

```python
from rinexpy.gw10 import iter_frames, decode_sbas, iter_sbas_messages, SYNC

print(hex(SYNC))                  # 0x8b

with open("gw10.log", "rb") as fp:
    for frame in iter_sbas_messages(fp):
        # frame['prn'] is the SBAS PRN
        # frame['sbas_l1_bytes'] is the 29-byte SBAS L1 message body
        print(frame["prn"], frame["sbas_l1_bytes"][:6].hex())
```

The `iter_sbas_messages` helper filters down to ID 0x03 (SBAS L1) frames
and pre-parses the timestamp and PRN. If you want every frame, use
`iter_frames` and dispatch on `frame["id"]`.

The decoded SBAS L1 payload can be fed straight to
`rinexpy.sbas.decode_sbas_message`.

```python
from rinexpy.gw10 import iter_sbas_messages
from rinexpy.sbas import decode_sbas_message

with open("gw10.log", "rb") as fp:
    for frame in iter_sbas_messages(fp):
        out = decode_sbas_message(frame["sbas_l1_bytes"])
        if out["msg_type"] == 9:
            print(out["x_m"], out["y_m"], out["z_m"])
```

## Performance

Decoder throughput on a modern laptop:

- NMEA-0183: roughly 100 000 sentences per second.
- UBX, SBF, NovAtel: roughly 50 000 messages per second.
- BINEX: framing-only at roughly 200 000 records per second.

The bottleneck is the bit-unpacking step. For binary streams without
intermediate decoding work, the framer alone runs faster than the
decoded-dict path.

## Related pages

- [RTCM and NTRIP](rtcm.md): the higher-level stream.
- [Raw nav subframes](nav-subframes.md): walking the bit-level GPS / Galileo / GLONASS / BeiDou nav messages.
- [SBAS and Galileo HAS](sbas-and-has.md): SBAS L1 and HAS decode.
