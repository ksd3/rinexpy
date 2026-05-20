# RTCM and NTRIP

RTCM (Radio Technical Commission for Maritime Services) defines the wire
formats that GNSS reference networks use to broadcast corrections to
roving receivers. There are two generations, both still in active use.

RTCM 2.x is the older DGPS / RTK format with a 6-of-8 wire encoding
inherited from the GPS L1 C/A navigation message style. It carries
pseudorange corrections, reference-station ECEF, and a small set of other
record types.

RTCM 3.x is the newer binary format. Every message is a CRC-24Q-protected
frame: an 8-bit preamble, 6 reserved bits, a 10-bit length, the payload,
and a 24-bit CRC. The payload's first 12 bits give the message number;
the rest is decoded per ICD by the matching handler. The format covers
everything from the legacy base-station-correction types to the full SSR
family used by IGS-SSR services.

NTRIP (Networked Transport of RTCM via Internet Protocol) is an
HTTP-styled streaming protocol for delivering RTCM3 frames over TCP. A
caster aggregates streams from N reference stations and serves them on
named mountpoints.

## RTCM 3.x decoder

The decoder is `rinexpy.rtcm3`. It exposes the framing helpers and a
high-level `iter_messages` generator that yields decoded message dicts.

### Framing

```python
from rinexpy.rtcm3 import PREAMBLE, crc24q, iter_messages
print(hex(PREAMBLE))      # 0xd3
```

The CRC-24Q checksum is also exposed; it is what `iter_messages` uses
when `check_crc=True`.

```python
crc = crc24q(header + payload)
```

### Iterating messages

```python
import io
from rinexpy.rtcm3 import iter_messages

with open("capture.bin", "rb") as fp:
    for msg in iter_messages(fp, check_crc=True):
        print(msg["msg_id"], msg.get("station_id"))
```

Each yielded `msg` is a dict. Every dict has at least `msg_id`. Messages
with a decoded structure carry the relevant ICD fields. Messages whose ID
is recognised but whose body is not yet decoded come back with
`payload_bytes` so you can walk the bits manually.

### Supported message types

The decoder ships fully-typed decoders for the following messages, grouped
by purpose.

**Observation messages.** 1004 (extended L1/L2 RTK observations), and the
MSM (Multiple Signal Message) family: MSM1 through MSM7 across all seven
constellations. MSM messages cover pseudorange-only, phase-only, mixed,
and signal-extended variants:

| Family | GPS | GLONASS | Galileo | SBAS | QZSS | BeiDou | NavIC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSM1 | 1071 | 1081 | 1091 | 1101 | 1111 | 1121 | 1131 |
| MSM2 | 1072 | 1082 | 1092 | 1102 | 1112 | 1122 | 1132 |
| MSM3 | 1073 | 1083 | 1093 | 1103 | 1113 | 1123 | 1133 |
| MSM4 | 1074 | 1084 | 1094 | 1104 | 1114 | 1124 | 1134 |
| MSM5 | 1075 | 1085 | 1095 | 1105 | 1115 | 1125 | 1135 |
| MSM6 | 1076 | 1086 | 1096 | 1106 | 1116 | 1126 | 1136 |
| MSM7 | 1077 | 1087 | 1097 | 1107 | 1117 | 1127 | 1137 |

Each MSM message carries the satellite mask, the signal mask, the cell
mask, and per-cell pseudorange / phase / DopplerCount / lock-time fields.
MSM5 and MSM7 add Doppler and signal-strength columns.

**Base station messages.** 1005 (stationary reference station, ECEF only),
1006 (same plus antenna height). Both come back as
`{"msg_id", "station_id", "x_m", "y_m", "z_m", ...}`.

**Antenna metadata.** 1029 (Unicode text string), 1033 (antenna and
receiver descriptors), 1230 (GLONASS L1/L2 C/A + P code-phase biases).

**Broadcast ephemeris.** 1019 (GPS LNAV subset: clock + ephemeris), 1020
(GLONASS slot, frequency channel, broadcast clock and ephemeris).

**State-Space Representation (SSR) for orbits and clocks.** This is the
family that real-time PPP services use. rinexpy decodes the full set per
the RTCM 10403.3 §3.5.10 specification.

| Message ID | What | System |
| --- | --- | --- |
| 1057 / 1058 / 1059 / 1060 | GPS orbit / clock / code-bias / combined | GPS |
| 1061 / 1062 / 1063 | GPS URA / high-rate clock / GLONASS orbit | mixed |
| 1064 / 1065 / 1066 | GLONASS clock / code-bias / combined | GLONASS |
| 1067 / 1068 | GLONASS URA / high-rate clock | GLONASS |
| 1240..1245 | Galileo orbit / clock / code-bias / combined / URA / HR-clock | Galileo |
| 1246..1251 | QZSS orbit / clock / code-bias / combined / URA / HR-clock | QZSS |
| 1252..1257 | SBAS orbit / clock / code-bias / combined / URA / HR-clock | SBAS |
| 1258..1263 | BeiDou orbit / clock / code-bias / combined / URA / HR-clock | BeiDou |

The decoded fields for an orbit message include `epoch_seconds_of_week`,
`update_interval_index`, `iod_ssr`, `provider_id`, `solution_id`, and an
`orbits` list of per-satellite corrections:

```python
{
  "sv": "G05",
  "iode": 12,
  "delta_radial_m": -0.18,
  "delta_along_m":   0.04,
  "delta_cross_m":  -0.02,
  "dot_radial_m_s":  0.00012,
  "dot_along_m_s":   0.00018,
  "dot_cross_m_s":  -0.00005,
}
```

Clock messages carry `c0_m`, `c1_m_s`, `c2_m_s2` polynomial coefficients
per satellite (in metres for the constant term and metres-per-second for
the rate). Code-bias messages carry a `biases` list of
`{obs_code, value_m}` per satellite.

**IGS-SSR.** Message 4076 wraps a subtype-dispatched SSR payload. The
decoder reads the IGS-SSR subtype field and delegates to the matching
handler.

### Round-trip with the framer

The decoder's framer is symmetric: you can build valid frames in tests
with the public `PREAMBLE` and `crc24q` helpers.

```python
import io
from rinexpy.rtcm3 import PREAMBLE, crc24q, iter_messages

body = bytes([0x3E, 0x80] + [0]*11)   # msg 1000ish, mostly zero payload
head = bytes([PREAMBLE, 0, len(body)])
crc  = crc24q(head + body)
frame = head + body + bytes([crc >> 16, (crc >> 8) & 0xFF, crc & 0xFF])

buf = io.BytesIO(frame * 3)            # 3 identical frames
n = 0
for msg in iter_messages(buf):
    n += 1
print(f"decoded {n} frames")           # 3
```

The frame above is not a real RTCM3 message; it is the smallest possible
well-formed CRC-passing frame for parser tests.

## RTCM 2.x decoder

The legacy DGPS / RTK format. The decoder is `rinexpy.rtcm2`. It strips the
6-of-8 wire encoding from each 30-bit word and dispatches on the message
type.

```python
from rinexpy.rtcm2 import iter_messages, PREAMBLE
print(hex(PREAMBLE))     # 0x66

with open("dgps.rtcm2", "rb") as fp:
    for msg in iter_messages(fp):
        print(msg["msg_type"])
```

Each yielded dict carries:

```
msg_type, station_id, z_count, sequence, n_words, health, data_words
```

For decoded types (1, 3, 9), the dict adds type-specific fields:

- **Type 1 (pseudorange corrections):** a `corrections` list with
  `sat_id`, `prc_m`, `rrc_m_s`, `iode` per satellite, plus a UDRE field.
- **Type 3 (reference station ECEF):** `x_m`, `y_m`, `z_m`.
- **Type 9 (high-rate pseudorange corrections):** same payload shape as
  type 1.

The Hamming parity bits are NOT validated; the framing assumes the
upstream link is reliable, which is the usual deployment.

## NTRIP client

The NTRIP module exposes a synchronous and an asynchronous client. Both
fetch the sourcetable and stream raw bytes from one mountpoint.

### Sourcetable

```python
from rinexpy.ntrip import fetch_sourcetable

entries = fetch_sourcetable("rtk2go.com", port=2101)
for e in entries:
    if e["type"] == "STR":
        print(e["mountpoint"], e["format"], e.get("location"))
```

The sourcetable is the caster's catalogue. Three record types appear:
`STR;` lines (one per mountpoint), `CAS;` lines (one per caster), and
`NET;` lines (one per network). The reader parses them all.

### Streaming bytes

```python
from rinexpy.ntrip import stream
from rinexpy.rtcm3 import iter_messages
import io

bytes_iter = stream(
    "rtk2go.com", "MOUNT01",
    user="you@example.com", password="x",
    port=2101,
)

buf = io.BytesIO()
for chunk in bytes_iter:
    buf.write(chunk)
    if buf.tell() > 4096:
        break
buf.seek(0)

for msg in iter_messages(buf):
    print(msg["msg_id"])
```

`stream` is a generator that yields raw `bytes` chunks indefinitely. You
glue the chunks into an `io.BytesIO` (or any seekable byte stream) and
feed the result into `iter_messages` for decoding.

The HTTP-over-TLS variant comes for free if you pass `port=443`; rinexpy
delegates to `ssl.create_default_context()` for the TLS handshake.

### asyncio variant

For `asyncio` apps, the `astream` coroutine is the equivalent. Same
arguments, same byte chunks.

```python
import asyncio
from rinexpy.ntrip import astream

async def main():
    n = 0
    async for chunk in astream("rtk2go.com", "MOUNT01", port=2101):
        n += len(chunk)
        if n > 4096:
            break

asyncio.run(main())
```

`afetch_sourcetable` is the async equivalent of `fetch_sourcetable`.

### Authentication

NTRIP v1 uses HTTP Basic auth with no headers beyond `User-Agent`. The
caster validates `user:password` against its access control list. NTRIP v2
adds a `Ntrip-Version: Ntrip/2.0` header and may add a server-issued
session cookie. The client handles both.

For public crowd-sourced casters (like `rtk2go.com`), `user=` is an email
address and `password=` is anything non-empty.

## Real-time PPP

The `rinexpy.realtime` module wires the NTRIP byte stream through the
RTCM3 framer into a `RealtimeOrbitClock` cache. The cache tracks live
broadcast ephemerides plus SSR corrections plus HAS messages, and lets
PPP-style code apply orbit / clock corrections in metres or seconds.

```python
from rinexpy.realtime import RealtimeOrbitClock, ntrip_message_loop

cache = RealtimeOrbitClock(ssr_validity_s=10.0)
for msg in ntrip_message_loop(
    caster="igs-ip.net",
    port=2101,
    mountpoint="SSRA00BKG0",
    user="you@example.com",
    password="anonymous",
):
    cache.ingest(msg)
    # later, for an SV from broadcast ephemeris:
    pos_corrected = cache.apply_orbit_correction(
        prn=5,
        sv_ecef_broadcast=broadcast_pos,
    )
    clock_corrected = cache.apply_clock_correction(prn=5, broadcast_clock_s=clk0)
```

The cache routes each decoded message to the right slot internally:
broadcast ephemerides (1019 / 1020) go to `broadcast`, SSR orbit /
clock go to `ssr_orbit` / `ssr_clock`, HAS mask / orbit / clock go to
`has_mask` / `has_orbit` / `has_clock`. See
[Real-time PPP](../positioning/realtime.md).

## SSR corrections

For PPP-style use of the SSR family, `rinexpy.ssr.SSRCorrections` is the
high-level composer. It takes an iterable of decoded RTCM3 messages and
exposes per-`(sv, epoch)` orbit corrections, per-`(sv, epoch)` clock
corrections, and per-`(sv, obs_code)` code biases.

```python
from rinexpy.ssr import SSRCorrections
from rinexpy.rtcm3 import iter_messages

with open("ssr-stream.rtcm", "rb") as fp:
    ssr = SSRCorrections(iter_messages(fp))

ssr.known_satellites()         # ['G05', 'G10', 'E09', 'E18', ...]
ssr.clock_correction_s("G05", epoch_seconds_of_week=86400.0)
ssr.code_bias_m("G05", "C1W")
```

The orbit correction is the radial / along-track / cross-track delta from
the satellite's broadcast ephemeris, evaluated at a query epoch and
rotated into ECEF using the satellite's instantaneous frame at that epoch.
The clock correction is the polynomial `c0 + c1*dt + c2*dt²` evaluated at
the query epoch. The code biases are direct per-`(sv, code)` lookups.

Pass `ssr=` to `ppp_solve` to use SSR in place of the `.clk` file. See
[SSR corrections](../corrections/ssr.md).

## Performance notes

The RTCM3 framer is bit-level Python and decodes a few thousand messages
per second on a modern laptop. For higher throughput, the `check_crc=False`
mode skips the CRC validation and is roughly twice as fast at the cost of
silently dropping bit errors.

The MSM7 decoder is the heaviest per-message; a full mixed-constellation
MSM7 frame can carry several hundred per-cell rows.

## Bundled fixtures

There are no RTCM captures in `tests/data/` because RTCM streams are
operator-specific and large. Every example in the documentation that uses
RTCM3 either constructs the frame in-line with the `PREAMBLE` and `crc24q`
helpers, or replays from an `example/_rtcm3_sample.bin` file that the
example builds on first run.

```sh
uv run python examples/08_ntrip_to_rtcm.py --offline
```

The above command writes a 5-frame synthetic capture to
`examples/_rtcm3_sample.bin` and then replays it through `iter_messages`.

## Related pages

- [SSR corrections](../corrections/ssr.md): the composer.
- [Receiver binary formats](receiver-binary.md): the other streaming protocols.
- [Real-time PPP](../positioning/realtime.md): NTRIP + RTCM + cache in one place.
- [Async loading](../tooling/async.md): `astream` and the asyncio variants.
