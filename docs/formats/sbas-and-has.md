# SBAS L1 and Galileo HAS

Two augmentation services broadcast corrections that improve a single-frequency
fix beyond standard SPP: SBAS over L1 and Galileo HAS over E6-B.

SBAS (Satellite-Based Augmentation System) is the family of geostationary
satellites that broadcast wide-area integrity and correction messages on
the GPS L1 frequency. The published deployments are WAAS in North America,
EGNOS in Europe, MSAS in Japan, GAGAN in India, and SDCM in Russia. Every
SBAS satellite transmits a 250 bit message every second.

Galileo HAS (High Accuracy Service) is Europe's free-to-air precise
correction service. It broadcasts SSR-style corrections on the Galileo E6-B
data signal and (since 2023) over the network through the RTCM3 message
4076 wrapper. The published target accuracy is 20 cm horizontal,
40 cm vertical, after one minute of convergence.

## SBAS L1 decoder

The decoder is `rinexpy.sbas`. It exposes one function per message type
plus a `decode_sbas_message` dispatcher.

### Framing

Each SBAS L1 message is 250 bits, transmitted at one message per second.
The wire layout is an 8-bit preamble that cycles through `0x53`, `0x9A`,
`0xC6`, followed by a 6-bit message type, then a 212-bit payload, then a
24-bit CRC-24Q.

```python
from rinexpy.sbas import PREAMBLES, decode_sbas_message
print(PREAMBLES)        # (0x53, 0x9A, 0xC6)
```

### Decoded message types

The decoder covers the operational set per RTCA DO-229E §A.4.

| MT | Purpose | Decoder | Key fields |
| --- | --- | --- | --- |
| 1 | PRN mask | `decode_sbas_mt1` | `prn_mask`, `iodp` |
| 2-5 | Fast pseudorange corrections | `decode_sbas_mt2_5` | `iodf`, `iodp`, `prc_m` (13), `udrei` (13) |
| 6 | Integrity | `decode_sbas_mt6` | `iodf` (4), `udrei` (51) |
| 7 | Fast-correction degradation | `decode_sbas_mt7` | `system_latency`, `iodp`, `a_i_index` (51) |
| 9 | GEO navigation | `decode_sbas_mt9` | `t_0`, `ura`, `x_m`, `y_m`, `z_m`, `x_dot_m_s`, `y_dot_m_s`, `z_dot_m_s`, `a_Gf0_s`, `a_Gf1_s_s` |
| 17 | GEO almanacs | `decode_sbas_mt17` | `t_0`, `almanacs` (list of up to 3 entries) |
| 18 | Iono grid mask | `decode_sbas_mt18` | `n_bands`, `band_number`, `iodi`, `igp_mask` |
| 24 | Mixed fast + long-term | `decode_sbas_mt24` | first 6 PRCs of fast set + long-term half |
| 25 | Long-term corrections | `decode_sbas_mt25` | two 106-bit halves, each decoded into per-PRN deltas |
| 26 | Iono delays | `decode_sbas_mt26` | `iodi`, `band_number`, `block_id`, `delays` (15 × `{prn_offset, vertical_delay_m, give_index}`) |

Unknown or unimplemented MTs come back as
`{"preamble", "msg_type", "raw"}`.

### Using it

```python
from rinexpy.sbas import decode_sbas_message

for msg_bytes in iter_sbas_messages_from_capture():
    out = decode_sbas_message(msg_bytes)
    mt = out["msg_type"]
    if mt == 1:
        prn_mask = out["prn_mask"]
        iodp = out["iodp"]
    elif mt in (2, 3, 4, 5):
        for prc in out["prc_m"]:
            print(prc)
    elif mt == 9:
        # The GEO ranging signal. Use the broadcast ECEF to range to the
        # SBAS satellite itself.
        print(out["x_m"], out["y_m"], out["z_m"])
    elif mt == 26:
        for d in out["delays"]:
            print(d["prn_offset"], d["vertical_delay_m"])
```

Each message dict comes back with its 6-bit MT under `msg_type` and an
extra `preamble` field with the byte from the wire.

### Source: a Furuno GW-10 capture

Furuno's GW-10 receiver dumps SBAS L1 messages in a 38-byte framed format.
The `rinexpy.gw10` module unwraps them and the `rinexpy.sbas` decoder
takes the 29-byte body.

```python
from rinexpy.gw10 import iter_sbas_messages
from rinexpy.sbas import decode_sbas_message

with open("gw10.log", "rb") as fp:
    for frame in iter_sbas_messages(fp):
        out = decode_sbas_message(frame["sbas_l1_bytes"])
        print(frame["prn"], out["msg_type"])
```

## Galileo HAS decoder

The HAS decoder is `rinexpy.has`. It handles the seven HAS message types
that wrap the SSR-style correction set.

### Framing

HAS messages have a 32-bit header (status, sub-mask, page count, ...) and a
variable-length body. The dispatcher reads the header first, then routes
the body to the right per-MT handler.

The seven message types are:

| MT | Purpose | Decoder |
| --- | --- | --- |
| 1 | Mask | `decode_has_mask` |
| 2 | Orbit corrections | `decode_has_orbit` |
| 3 | Clock full-set | `decode_has_clock_full` |
| 4 | Clock subset | `decode_has_clock_subset` |
| 5 | Code biases | `decode_has_code_bias` |
| 6 | Phase biases | `decode_has_phase_bias` |
| 7 | URA | `decode_has_ura` |

### Using it

The high-level dispatcher needs the most-recent Mask (MT 1) so it can
interpret the per-PRN slots in the other MTs.

```python
from rinexpy.has import decode_has_header, decode_has_mask, decode_has_message

# Walk a HAS message body:
mask = None
for body in has_message_bodies:
    header, bit_cursor = decode_has_header(body)
    if header["mt"] == 1:
        mask = decode_has_mask(body, bit_cursor)
        continue
    if mask is None:
        continue                       # not yet bootstrapped
    out = decode_has_message(body, mask=mask)
    if out["mt"] == 2:
        for sv, delta in out["orbits"].items():
            print(sv, delta["delta_radial_m"], delta["delta_along_m"])
    if out["mt"] == 3:
        for sv, clk in out["clocks"].items():
            print(sv, clk["c0_m"])
```

### Constants

The module also exposes two reference tables:

```python
from rinexpy.has import HAS_GNSS_NAMES, HAS_VALIDITY_S
print(HAS_GNSS_NAMES)
# {0: 'GPS', 1: 'GLONASS', 2: 'Galileo', 3: 'BDS', 4: 'SBAS', 5: 'QZSS', 6: 'NavIC'}
print(HAS_VALIDITY_S[3])    # 20 -- validity index 3 = 20 seconds
```

The `HAS_VALIDITY_S` table maps the validity-index field in the HAS
header to a duration in seconds.

### HAS over RTCM3 (MT 4076)

When HAS is delivered over a network instead of E6-B, RTCM3 message 4076
wraps the same payload. The RTCM3 decoder routes 4076 messages to the
HAS handlers automatically.

```python
from rinexpy.rtcm3 import iter_messages

with open("rtcm-stream.rtcm3", "rb") as fp:
    for msg in iter_messages(fp):
        if msg["msg_id"] == 4076:
            # msg already has has-style decoded fields
            print(msg["has_mt"], msg.get("orbits"))
```

## Storing the corrections live

The `RealtimeOrbitClock` cache in `rinexpy.realtime` takes in SBAS, SSR,
and HAS messages together, plus broadcast ephemerides from RINEX 3 NAV.
The cache picks the freshest valid correction per satellite when
queried.

```python
from rinexpy.realtime import RealtimeOrbitClock

cache = RealtimeOrbitClock(ssr_validity_s=10.0)
for msg in messages:
    cache.ingest(msg)

# When ranging to G05 with broadcast ECEF:
corrected = cache.apply_orbit_correction(
    prn=5,
    sv_ecef_broadcast=broadcast_pos,
    sv_velocity_ecef=broadcast_vel,
    elapsed_s=0.0,
)
clock_s = cache.apply_clock_correction(prn=5, broadcast_clock_s=clock0_s)
```

The page on [real-time PPP](../positioning/realtime.md) walks through the
end-to-end NTRIP pipeline that powers this.

## Related pages

- [RTCM and NTRIP](rtcm.md): the RTCM3 framer that wraps HAS as MT 4076.
- [Raw nav subframes](nav-subframes.md): the underlying broadcast decoders.
- [Real-time PPP](../positioning/realtime.md): NTRIP + correction cache + PPP.
- [Receiver binary formats](receiver-binary.md): how to extract SBAS L1 from a Furuno log.
