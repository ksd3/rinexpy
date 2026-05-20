# SSR corrections

State-Space Representation (SSR) is the modern way to deliver precise
satellite corrections over a low-bandwidth link. Instead of broadcasting
full orbit and clock products, the operator transmits deltas relative to
the broadcast ephemeris: a radial / along-track / cross-track offset, a
clock polynomial, and per-(SV, observation code) code biases. A typical
SSR mountpoint streams these at a few hundred bytes per second.

rinexpy's `SSRCorrections` class absorbs the decoded RTCM3 SSR messages
and exposes per-`(sv, epoch)` corrections to the positioning layer.

## Construction

```python
from rinexpy.ssr import SSRCorrections
from rinexpy.rtcm3 import iter_messages

# From a captured SSR feed:
with open("ssr-stream.rtcm", "rb") as fp:
    ssr = SSRCorrections(iter_messages(fp))
```

Or build the composer empty and feed messages in incrementally:

```python
ssr = SSRCorrections()
for msg in iter_messages(stream):
    ssr.add_message(msg)
```

## Querying corrections

### Orbit

`orbit_correction_ecef` returns the ECEF correction at a given epoch for a
satellite whose broadcast or SP3 position and velocity you have.

```python
import numpy as np

correction_m = ssr.orbit_correction_ecef(
    sv="G05",
    sat_pos_ecef=np.array([sx, sy, sz]),
    sat_vel_ecef=np.array([vx, vy, vz]),
    epoch_seconds_of_week=86400.0,
)
# (3,) ECEF correction in metres; add to the broadcast/SP3 position.
```

The function transforms the radial / along-track / cross-track delta into
ECEF using the satellite's instantaneous frame. The radial direction is
the unit vector from the geocentre to the satellite; the along-track is
along the velocity vector; the cross-track is the cross product.

### Clock

`clock_correction_s` returns the polynomial-evaluated clock correction in
seconds.

```python
delta_clock_s = ssr.clock_correction_s(
    sv="G05",
    epoch_seconds_of_week=86400.0,
)
```

The polynomial is `c0 + c1*dt + c2*dt²` where `dt` is `(epoch -
correction_reference_epoch)`. The result is in seconds.

### Code bias

`code_bias_m` returns the per-(SV, code) bias in metres.

```python
bias_m = ssr.code_bias_m(sv="G05", obs_code="C1W")
```

The default RINEX 3 observation code mapping for the SSR signal IDs is
built in. Common mappings:

| System | Signal ID | RINEX 3 code |
| --- | --- | --- |
| GPS | 0 | C1C |
| GPS | 3 | C1W |
| GPS | 16 | C2W |
| GPS | 18 | C2L |
| Galileo | 0 | C1B |
| Galileo | 5 | C5Q |
| Galileo | 8 | C7Q |
| GLONASS | 0 | C1C |
| GLONASS | 1 | C1P |
| GLONASS | 3 | C2C |
| BeiDou | 0 | C2I |
| BeiDou | 14 | C7I |

If your SSR feed uses a non-default mapping, pass a custom mapping when
constructing `SSRCorrections`.

### Inventory

```python
ssr.known_satellites()      # list of SVs with any kind of correction
ssr.has_orbit("G05")        # bool
ssr.has_clock("G05")        # bool
```

## Validity windows

SSR corrections carry an `update_interval_index` field that maps to a
maximum validity duration. The composer honours this; corrections older
than their validity bound are treated as unavailable.

The full update-interval table is in the RTCM 10403.3 spec. Typical
values are 5 seconds for orbit, 1 second for clock, 30 seconds for code
bias.

For real-time PPP with a 1 Hz observation cadence, the SSR clock
update is the binding latency.

## Use in PPP

The PPP driver accepts an SSR object directly via `ppp_solve(ssr=...)`.

```python
from rinexpy.ppp import ppp_solve

out = ppp_solve(obs, sp3, clk=None, ssr=ssr,
                initial_position_ecef=tuple(approx_xyz))
```

When `clk=None` and `ssr` is supplied, the driver substitutes the SSR
clock correction for the (missing) CLK lookup per satellite. The orbit
correction is applied as a delta to the SP3-interpolated position. Code
biases override the DCB lookups when both are supplied.

You can also pass `clk=` and `ssr=` together; the SSR clock takes
precedence when both have an entry for the same `(sv, epoch)`. This is
the recommended path for PPP applications that want a real-time clock on
top of a published orbit (post-processed SP3 with live clock corrections).

## Live ingest

For a live feed, wire the cache to an `iter_messages` generator from an
NTRIP stream.

```python
import threading
from rinexpy.ntrip import stream
from rinexpy.rtcm3 import iter_messages
from rinexpy.ssr import SSRCorrections
import io

ssr = SSRCorrections()

def feed_ssr():
    buf = io.BytesIO()
    for chunk in stream("igs-ip.net", "SSRA00BKG0",
                        user="me@example.com", password="anonymous",
                        port=2101):
        buf.write(chunk)
        buf.seek(0)
        try:
            for msg in iter_messages(buf):
                ssr.add_message(msg)
        finally:
            # Truncate the buffer; iter_messages stops at the last
            # complete frame so the remainder is the next partial frame.
            remaining = buf.read()
            buf.seek(0); buf.truncate()
            buf.write(remaining)

threading.Thread(target=feed_ssr, daemon=True).start()
```

For a cleaner pipeline, use `RealtimeOrbitClock` in
`rinexpy.realtime` which absorbs broadcast nav + SSR + HAS together. See
[Real-time PPP](../positioning/realtime.md).

## Worked example

Below is a synthetic walkthrough: build an SSR orbit message, feed it
into the composer, and apply the correction.

```python
import numpy as np
from rinexpy.ssr import SSRCorrections

ssr = SSRCorrections()

# Pretend we got this from iter_messages decoding a 1057 message:
ssr.add_message({
    "msg_id": 1057,
    "epoch_seconds_of_week": 86400.0,
    "update_interval_index": 4,             # 5 seconds
    "iod_ssr": 0,
    "provider_id": 0,
    "solution_id": 0,
    "system": "G",
    "orbits": [
        {
            "sv": "G05",
            "iode": 12,
            "delta_radial_m": -0.18,
            "delta_along_m":   0.04,
            "delta_cross_m":  -0.02,
            "dot_radial_m_s":  0.00012,
            "dot_along_m_s":   0.00018,
            "dot_cross_m_s":  -0.00005,
        },
    ],
})

# Broadcast/SP3 satellite position and velocity:
sat_pos = np.array([15000e3, 12000e3, 18000e3])
sat_vel = np.array([1200.0, -1500.0, 2000.0])

# Apply the SSR correction at a query epoch:
correction = ssr.orbit_correction_ecef(
    sv="G05",
    sat_pos_ecef=sat_pos,
    sat_vel_ecef=sat_vel,
    epoch_seconds_of_week=86400.0,
)
corrected_pos = sat_pos + correction
```

## Message-type coverage

The composer recognises every SSR message in the RTCM 10403.3 family:

- **GPS:** 1057 (orbit), 1058 (clock), 1059 (code bias), 1060 (combined),
  1061 (URA), 1062 (high-rate clock).
- **GLONASS:** 1063, 1064, 1065, 1066, 1067, 1068.
- **Galileo:** 1240, 1241, 1242, 1243, 1244, 1245.
- **QZSS:** 1246, 1247, 1248, 1249, 1250, 1251.
- **SBAS:** 1252, 1253, 1254, 1255, 1256, 1257.
- **BeiDou:** 1258, 1259, 1260, 1261, 1262, 1263.
- **IGS-SSR (MT 4076):** subtype-dispatched.

For Galileo HAS, the dedicated HAS decoder feeds the same composer
through a thin shim; see [SBAS and Galileo HAS](../formats/sbas-and-has.md).

## Related pages

- [RTCM and NTRIP](../formats/rtcm.md): the decoder family.
- [SBAS and Galileo HAS](../formats/sbas-and-has.md): the HAS path.
- [Real-time PPP](../positioning/realtime.md): NTRIP + cache + filter.
- [Precise point positioning](../positioning/ppp.md): the PPP driver.
- [DCB and code biases](dcb.md): the offline alternative to SSR code biases.
