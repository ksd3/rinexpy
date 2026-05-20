# Real-time PPP

The real-time PPP workflow combines an NTRIP byte stream, an RTCM3 framer,
the SSR correction set, and a stationary or kinematic PPP filter into one
pipeline. The endpoint is a sub-decimetre absolute position, updated every
second, while the connection holds.

rinexpy ships two layers for this. `rinexpy.realtime.RealtimeOrbitClock`
is the orbit/clock cache that ingests every kind of correction message
and reports the live correction per satellite. The `ppp_solve` driver in
`rinexpy.ppp` accepts an `ssr=` argument that wraps the same data path.

## The cache

`RealtimeOrbitClock` is a dataclass that holds:

- `ssr_orbit`: live SSR orbit corrections per PRN.
- `ssr_clock`: live SSR clock corrections per PRN.
- `has_mask`: the most recent Galileo HAS mask.
- `has_orbit`, `has_clock`: live HAS corrections.
- `broadcast`: broadcast ephemerides per `(system, prn)`, indexed by `t_oe`.
- `ssr_validity_s`: how long an SSR correction is considered valid past
  its reception time.

```python
from rinexpy.realtime import RealtimeOrbitClock

cache = RealtimeOrbitClock(ssr_validity_s=10.0)
```

Each decoded RTCM3 / HAS / NAV message gets fed in via `ingest`:

```python
for msg in decoded_messages:
    cache.ingest(msg)
```

The cache decides where each message goes by message ID. Broadcast nav
records (RTCM3 1019/1020) go to `broadcast`. SSR orbit messages
(1057/1063/1240/...) go to `ssr_orbit`. SSR clock messages go to
`ssr_clock`. HAS messages go to `has_mask`, `has_orbit`, `has_clock`.

## Querying the cache

When ranging to a satellite, you ask the cache for the live correction.
Orbit corrections come back as ECEF deltas; clock corrections come back
as seconds.

```python
# Apply the live SSR orbit correction to a broadcast ECEF:
corrected_pos = cache.apply_orbit_correction(
    prn=5,
    sv_ecef_broadcast=broadcast_pos,
    sv_velocity_ecef=broadcast_vel,
    elapsed_s=elapsed_seconds_since_correction_epoch,
)

# Apply the live SSR clock correction:
corrected_clock_s = cache.apply_clock_correction(
    prn=5,
    broadcast_clock_s=clock_from_broadcast,
    elapsed_s=elapsed_seconds_since_correction_epoch,
)
```

The orbit correction transforms the radial / along-track / cross-track
delta into ECEF using the SV's instantaneous radial direction. The clock
correction is the polynomial `c0 + c1*dt + c2*dt²` evaluated at `dt =
elapsed_s`. Corrections older than `ssr_validity_s` are treated as
unavailable (`None`).

## End-to-end pipeline

The headline NTRIP loop is the recommended entry point.

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
    # When you have a broadcast or SP3 satellite position, refine it:
    if msg.get("msg_id") == 1057:           # SSR1 GPS Orbit
        for entry in msg["orbits"]:
            sv_label = f"G{entry['sv']:02d}"
            # do something with the live correction
```

`ntrip_message_loop` is a thin wrapper that connects to the caster, runs
the byte stream through `iter_messages`, and yields each decoded
message. It is the same path as

```python
from rinexpy.ntrip import stream
from rinexpy.rtcm3 import iter_messages
import io

bytes_iter = stream("igs-ip.net", "SSRA00BKG0",
                    user="you@example.com", password="anonymous", port=2101)
# Glue bytes into a stream; iter_messages framer takes a file-like object.
```

just packaged with cleaner reconnection semantics.

## Driving PPP from the cache

For a full real-time PPP, the typical pattern is:

1. Open an NTRIP connection to an SSR mountpoint in one thread.
2. Open an NTRIP connection to your own RTCM3 base stream (or a local
   observation source) in another.
3. Apply the cache's corrections to your broadcast or SP3-interpolated
   satellite positions per epoch.
4. Feed the corrected positions and clocks into `ppp_solve` (or a
   long-running `StaticPPPFilter`).

```python
import threading
from rinexpy.realtime import RealtimeOrbitClock, ntrip_message_loop
from rinexpy.kalman import StaticPPPFilter

cache = RealtimeOrbitClock()
ekf = StaticPPPFilter(n_sv=12, initial_position=(x0, y0, z0))

def feed_ssr():
    for msg in ntrip_message_loop("igs-ip.net", 2101, "SSRA00BKG0",
                                  user="you@example.com", password="anonymous"):
        cache.ingest(msg)

threading.Thread(target=feed_ssr, daemon=True).start()

# Main loop reads observations from your receiver:
while True:
    obs_epoch = wait_for_observation_epoch()
    # ... compute satellite ECEF from broadcast or SP3
    # ... apply cache corrections per satellite
    # ... compute iono-free combinations
    ekf.predict(dt)
    ekf.update(sv_ecef_corrected, sat_clock_s_corrected, pr_if, phase_if)
```

## SSR mountpoints to try

The IGS RTS (Real-time Service) publishes free SSR streams from a few
public casters. Common mountpoints:

| Caster | Mountpoint | What |
| --- | --- | --- |
| `products.igs-ip.net` | `SSRA00BKG0` | combined IGS SSR (orbit + clock + code-bias) |
| `products.igs-ip.net` | `IGS00BKG1` | broadcast ephemerides |
| `products.igs-ip.net` | `BCEP00BKG0` | broadcast ephemerides (alternate) |

The IGS RTS terms of service require registration; check the
[IGS RTS site](http://rts.igs.org/) for current credentials and
mountpoints.

## Validity windows

`ssr_validity_s` controls how long the cache treats a correction as
fresh. The default of 10 seconds is a reasonable choice for an SSR feed
that updates every 1-5 seconds. Setting it too low means short
disconnections drop all corrections; too high means stale corrections
are applied during outages.

For HAS-based workflows the validity is implicit in the HAS message
header (`HAS_VALIDITY_S` table in `rinexpy.has`). The cache honours that
internally.

## Latency budget

For 1 Hz PPP the latency budget breaks down like this:

- NTRIP TCP latency: 50-150 ms typical to a regional caster.
- RTCM3 + SSR generation latency at the network operations centre:
  500-2000 ms (operator-dependent).
- Receiver observation epoch: 0-1000 ms in the worst case (1 Hz
  receiver).
- Filter update: a few milliseconds.

Total end-to-end latency is typically 1-3 seconds. That is fine for
static PPP and for most kinematic applications below highway speeds.

## Related pages

- [SSR corrections](../corrections/ssr.md): the composer in depth.
- [RTCM and NTRIP](../formats/rtcm.md): the underlying wire formats.
- [SBAS and Galileo HAS](../formats/sbas-and-has.md): the HAS decoder.
- [Precise point positioning](ppp.md): the offline PPP driver.
- [Async loading](../tooling/async.md): the asyncio NTRIP variant.
