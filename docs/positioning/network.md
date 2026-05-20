# Network RTK and VRS

A single-baseline RTK fix fails past roughly ten kilometres. Beyond
that, the residual ionospheric and tropospheric errors do not cancel in
the double-difference, and the integer ambiguity becomes biased. Network
RTK is the standard fix: a network of reference stations characterises
the spatial gradient of the atmosphere, and the rover gets a tailored
correction at its own position.

`rinexpy` includes two paths into network RTK. `synthesize_vrs` builds a
Virtual Reference Station (VRS) at the rover's approximate position
from a network of bases. `network_dd_solve` and `network_dd_solve_ar`
do a joint multi-baseline solve directly.

## VRS synthesis

The Virtual Reference Station approach pretends that there is a real
reference receiver co-located with the rover. The synthesized
observations are the rover's geometric range plus a plane fit of the
per-base residuals across the network. The rover then runs ordinary
single-baseline RTK against the synthesized base.

```python
from rinexpy.vrs import synthesize_vrs
from rinexpy.rtk import rtk_fix
from rinexpy.multifreq import LAMBDA_L1

bases = [
    {
        "station_ecef": (x1, y1, z1),
        "sv_positions":  sv1_ecef,           # (n_sv, 3)
        "pr":            pr1_m,              # (n_sv,)
        "phase":         phase1_cycles,      # (n_sv,)
    },
    {
        "station_ecef": (x2, y2, z2),
        "sv_positions":  sv2_ecef,
        "pr":            pr2_m,
        "phase":         phase2_cycles,
    },
    # ... at least three for a meaningful plane fit
]

vrs = synthesize_vrs(bases, rover_approx_pos, wavelength=LAMBDA_L1)

sol = rtk_fix(
    rover_pr, vrs["pr"],
    rover_phase, vrs["phase"],
    vrs["sv_positions"], vrs["base_position"],
    wavelength=LAMBDA_L1,
)
```

The `vrs` dict has:

| Key | Type | Meaning |
| --- | --- | --- |
| `base_position` | `tuple[float, float, float]` | the VRS ECEF (same as `rover_approx_pos`) |
| `sv_positions` | `ndarray` | shared (n_sv, 3) ECEF |
| `pr` | `ndarray` | synthesized per-SV pseudoranges in metres |
| `phase` | `ndarray` | synthesized per-SV phases in cycles |

The synthesiser walks the satellite list. For each satellite, it computes
the per-base double-difference residual (relative to the network's first
station), fits a plane in the network coordinates, and evaluates the plane
at the rover's approximate position. The plane fit takes in the spatial
gradient of the atmosphere; once subtracted, the rover's single-baseline
RTK against the synthesised base sees an "short-baseline"
atmosphere.

`synthesize_vrs` needs at least three reference stations. With more, the
plane is over-determined and the least-squares fit is robust to noise.

For the taken in-error model and Wanninger's original paper, see Wanninger
(2002).

## Network double-difference

The DD solver generalises to multiple bases. Every base contributes its
own DD observations and its own per-satellite ambiguities, but they share
one rover position.

```python
from rinexpy.network_dd import network_dd_solve

baselines = [
    {
        "base_ecef": (xb1, yb1, zb1),
        "sv_positions": sv1_ecef,
        "rover_pr": rover_pr1,
        "base_pr":  base_pr1,
        "rover_phase": rover_phase1,
        "base_phase":  base_phase1,
    },
    # ... one entry per reference station
]

sol = network_dd_solve(
    baselines,
    wavelength=LAMBDA_L1,
    initial_rover=(x0, y0, z0),
)
print(sol["rover_position"])
print(sol["ambiguities"])     # per-baseline float ambiguities
print(sol["residuals"])
print(sol["cov_rover"])
```

The float solver is sufficient when the residual atmosphere is small
(short baselines, calm ionosphere). For centimetre-class network RTK,
add LAMBDA per baseline.

```python
from rinexpy.network_dd import network_dd_solve_ar

sol = network_dd_solve_ar(
    baselines,
    wavelength=LAMBDA_L1,
    sigma_pr_m=1.0,
    sigma_phase_cycles=0.005,
    ratio_threshold=3.0,
)
print("fixed accepted:", sol["fixed_accepted"])
print("ratio:", sol["ratio"])
print("rover position:", sol["rover_position"])
print("per-baseline fixes:", sol["per_baseline_fixed"])
```

The ratio test is over the joint integer set, so passing it implies every
per-baseline ambiguity is fixed consistently.

## When to use which

`synthesize_vrs` + `rtk_fix` is the lighter path. The VRS observations
are easy to debug, and the single-baseline RTK code is identical to the
short-baseline case.

`network_dd_solve_ar` is the heavier path. It solves the network jointly,
which gives a slightly better position than the VRS path (the VRS step
itself adds noise from the plane fit), but the code is more involved and
the integer-ambiguity bookkeeping is less obvious.

In production, VRS is what commercial network RTK services (Trimble VRS,
Topcon TopNET, Hexagon Smartnet, ...) deploy. The joint DD solver lives
mostly in research code.

## Real-time network RTK over NTRIP

Most operational network RTK services deliver VRS observations through an
NTRIP mountpoint. The receiver sends a `$GGA` NMEA sentence with its
approximate position, the caster computes the VRS observations server-side,
and the receiver sees what looks like a short-baseline single reference.

`rinexpy`'s NTRIP client supports this pattern. After opening the stream,
push a NMEA `$GPGGA` sentence with the rover's approximate position to
the caster every couple of seconds.

```python
from rinexpy.ntrip import stream

bytes_iter = stream(
    "vrs.network.example.com", "VRS_MOUNT01",
    user="me", password="x", port=2101,
)
# Then poll bytes_iter and emit your own $GPGGA periodically...
```

The actual NMEA out-of-band channel is handled by the caster; the bytes
you stream back are standard RTCM3 single-baseline RTK from the
synthesized VRS.

## Long-baseline atmospheric modelling

For research workflows that want to break the baseline-length limit
without a network, the alternative is to estimate the residual
atmospheric delay as part of the filter state. The
`StaticPPPFilterMultiGNSS` and `StaticPPPFilterZTD` filters do that for
PPP; the equivalent for DD is an ambiguity-+- ZWD joint solve, which
`rinexpy` does not currently expose as a one-call helper.

For most users the VRS path is the standard choice until the baselines
exceed roughly 70 km.

## Related pages

- [RTK and integer fixing](rtk.md): the single-baseline solver underneath VRS.
- [Precise point positioning](ppp.md): the absolute-position alternative.
- [Atmospheric models](../corrections/atmosphere.md): the ZWD and ionospheric chains.
- [RTCM and NTRIP](../formats/rtcm.md): the wire format VRS travels on.
