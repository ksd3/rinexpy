# Snapshot positioning

Snapshot positioning is what assisted GPS (A-GPS) receivers do when they
only see a fraction of a millisecond of raw signal. The receiver has a
coarse position prior (typically from cellular tower geolocation, accurate
to a few tens of kilometres), and the captured signal gives only the
fractional code phase per satellite (0 to 1023 chips). The integer
millisecond ambiguity (which integer number of code periods the signal
travelled) cannot be measured directly because the receiver did not see a
full bit boundary.

Van Diggelen's "A-GPS" pattern (Van Diggelen, 2009) recovers the position
anyway: assume a coarse position, predict the integer number of code
periods to each satellite, then solve four unknowns (position + clock
bias) by least squares. As the position estimate updates, the integer
predictions update, and the loop converges.

rinexpy's snapshot solver is `rinexpy.snapshot.snapshot_positioning`.

## The model

For each satellite, the receiver records the fractional code phase
`p_i` ∈ [0, 1023]. The full pseudorange is

```
PR_i = (K_i * 1023 + p_i) * (chip_length_m)
```

where `K_i` is the unknown integer code-period count. The chip length is
about 293 m for GPS C/A.

The receiver also has an approximate position `x_0` and an approximate
clock bias `dt_0`. It computes the predicted pseudorange from `x_0` to
the satellite, divides by the code period, and rounds to the nearest
integer. That gives an initial `K_i`. Substituting the recovered
pseudoranges into a standard SPP linearisation gives an update to
`(x, dt)`. Iterate until convergence.

The recipe works when the prior is within roughly 150 km of the truth,
which is enough for cellular-tower-class assistance.

## Calling the solver

```python
import numpy as np
from rinexpy.snapshot import snapshot_positioning

out = snapshot_positioning(
    code_phase_chips=np.array([523.1, 102.8, 819.4, 281.6, 1015.2]),
    sv_positions_ecef=sv_ecef_at_emission,        # (n_sv, 3)
    initial_position_ecef=(x_prior, y_prior, z_prior),
    max_iter=20,
    tol=1.0,
)

print("ECEF:", out["position_ecef"])
print("LLA: ", out["lla"])
print("time bias:", out["time_bias_s"], "s")
print("integer K:", out["K_integer_ms"])
print("pseudoranges:", out["pseudoranges_m"])
print("n_iter:", out["n_iter"])
```

The return dict has:

| Key | Meaning |
| --- | --- |
| `position_ecef` | resolved ECEF in metres |
| `lla` | latitude (deg), longitude (deg), altitude (m) |
| `time_bias_s` | receiver clock bias in seconds |
| `pseudoranges_m` | per-SV resolved pseudoranges (with integer K applied) |
| `K_integer_ms` | resolved integer-millisecond ambiguities |
| `n_iter` | iterations to converge |

`tol=1.0` is the iteration tolerance in metres. The default is plenty for
A-GPS workflows. `max_iter=20` is a conservative cap; convergence is
usually 5-10 iterations.

## Where the prior comes from

The prior position can come from:

- A cellular tower or Wi-Fi network's database lookup.
- A previous position fix that has not aged past the prior's uncertainty.
- A coarse map of the user's expected location.

If the prior is wrong by more than ~150 km, the integer K predictions are
wrong by 1 or more, and the algorithm converges on the wrong position.
The 150 km horizon comes from the GPS C/A code period: at 293 m per chip
and 1023 chips per code period, the period is 300 km of free-space
propagation, so the half-period is 150 km. Past that, the integer
boundary flips.

For Galileo E1 the period is roughly the same. For BeiDou B1I the period
is about 100 km, so the prior must be tighter.

## Where the SV positions come from

The snapshot pattern needs SV positions at the *signal emission time*,
not the receive time. In practice the receiver does not know the receive
time with high accuracy either (because the clock bias is one of the
unknowns), but it knows it within a few seconds. That is good enough for
a coarse SV position, since the satellite moves about 4 km in 1 second.

For an A-GPS deployment, the assistance server typically provides:

- The approximate position prior.
- Broadcast or interpolated SP3 satellite positions at the assumed receive
  time.
- A coarse satellite clock estimate.

rinexpy does not bundle the assistance server. The snapshot solver only
takes the pre-computed satellite positions.

## Use cases

The snapshot fix is what makes a one-second cold start possible. Without
assistance, a GPS receiver needs to demodulate at least one full
navigation subframe (6 seconds) before it has the satellite ephemeris.
The snapshot pattern bypasses that wait when assistance is available.

Modern smartphones use snapshot positioning at every cold start. Their
GPS receivers stream raw code-phase samples to the cellular network, which
returns the assistance data and (in some networks) the actual position
solve.

## Implementation notes

`snapshot_positioning` is a clean four-unknown LSQ wrapped around the
integer K disambiguation. The cost is dominated by the per-iteration
3×3 normal equations system; the integer step is just a `np.round`.

The accuracy depends on the prior quality. A 30 km prior typically lands
within a kilometre after one iteration and within ten metres after
convergence. A 100 km prior takes a few extra iterations but still
converges.

## Related pages

- [Single-point positioning](spp.md): the full-precision SPP that comes after the snapshot has bootstrapped.
- [Real-time PPP](realtime.md): the modern alternative for accuracy-class A-GPS.
