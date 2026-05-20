# Precise point positioning

Precise point positioning (PPP) is the absolute-position cousin of RTK. It
uses the IGS precise satellite orbit (SP3) and clock (CLK) products in
place of a nearby reference station, plus a sequential Kalman filter that
estimates the receiver position, the receiver clock, the wet zenith delay,
and the carrier-phase ambiguities together.

A static receiver converges to centimetre-class accuracy in about thirty
minutes of dual-frequency observations. A kinematic receiver tracks to a
few centimetres in real time after convergence. With integer ambiguity
resolution (PPP-AR) the convergence drops to roughly fifteen minutes.

The main PPP driver in `rinexpy.ppp.ppp_solve` combines every layer
into one call. Underneath, the `kalman` module exposes the EKF directly
for custom workflows.

## The high-level driver

```python
from rinexpy.ppp import ppp_solve
import rinexpy as rp
from rinexpy.clk import load_clk

obs = rp.load("data/STAT00BRA_R_20231560000_24H_01S_MO.rnx.gz")
sp3 = rp.load_sp3("data/IGS0OPSFIN_20231560000_01D_15M_ORB.SP3")
clk = load_clk("data/IGS0OPSFIN_20231560000_01D_30S_CLK.CLK")

approx_xyz = obs.attrs["position"]
out = ppp_solve(
    obs, sp3, clk,
    initial_position_ecef=tuple(approx_xyz),
    elevation_mask_deg=7.0,
)
print("position:", out["position"])
print("1-sigma: ", out["position_sigma_m"])
print("epochs:  ", out["n_epochs"])
```

The return dict has:

| Key | Type | Meaning |
| --- | --- | --- |
| `position` | `tuple[float, float, float]` | final ECEF estimate in metres |
| `lla` | `tuple[float, float, float]` | (lat_deg, lon_deg, alt_m) |
| `clock_bias_s` | `float` | final receiver clock bias in seconds |
| `position_sigma_m` | `tuple[float, float, float]` | per-axis 1-sigma |
| `n_epochs` | `int` | epochs used (after elevation mask, NaN drop, slip drop) |
| `trace` | `list[dict]` | per-epoch `{epoch, position, clock_bias_s}` history |
| `obs_codes` | `tuple[str, str, str, str]` | chosen (C1, C2, L1, L2) labels |
| `filter` | `StaticPPPFilter` | the filter object at the end of the run |

The driver picks an L1/L2 observable code quadruple automatically from the
dataset, falling back through a documented priority list
(`C1C/C2W/L1C/L2W`, `C1W/C2W/L1W/L2W`, ...). Override with
`obs_codes=("C1W", "C2W", "L1W", "L2W")` if you want.

## The full correction stack

For sub-centimetre PPP, supply the antenna, the troposphere grid, the DCB
records, and turn on wind-up.

```python
from rinexpy.antex import find_antenna, load_antex
from rinexpy.gpt2w import load_gpt2w_grid
from rinexpy.dcb_download import auto_load_dcb

ant = find_antenna(load_antex("igs20.atx"), "TRM59800.00     NONE")
gpt = load_gpt2w_grid("/path/to/gpt2_5w.grd")
dcb = auto_load_dcb(obs.time.values[0].astype("datetime64[D]").astype(object))

out = ppp_solve(
    obs, sp3, clk,
    initial_position_ecef=tuple(approx_xyz),
    antenna=ant,
    gpt2w_grid=gpt,
    dcb_records=dcb,
    station_id="STAT00BRA",
    apply_wind_up=True,
)
```

The corrections compose like this inside the driver:

1. Per epoch and per satellite, interpolate the SP3 orbit and the CLK
 clock at the signal-emission time (with light-time + Earth-rotation
 correction).
2. If `antenna=` is supplied, apply the ANTEX PCV correction per
 satellite per band.
3. If `dcb_records=` is supplied, look up the satellite (and receiver, if
 `station_id` is set) observable-specific bias for each pseudorange
 and subtract it.
4. If `gpt2w_grid=` is supplied, evaluate GPT2w at the receiver's
 approximate position to get `(P, T, e, a_h, a_w)`. Use Saastamoinen
 for the zenith hydrostatic delay and VMF1 for the mapping factor.
 Otherwise fall back to Saastamoinen with standard atmosphere and
 Niell mapping.
5. If `apply_wind_up=True`, accumulate the Wu et al. 1993 carrier-phase
 wind-up correction per satellite.
6. Form the iono-free combinations of the corrected pseudorange and the
 carrier phase.
7. Feed the per-epoch measurements into `StaticPPPFilter` (the EKF; see
 below).

## Real-time PPP with SSR

When you have a real-time SSR feed from an NTRIP caster instead of a CLK
file, swap them.

```python
from rinexpy.ssr import SSRCorrections
from rinexpy.rtcm3 import iter_messages

with open("ssr-stream.rtcm", "rb") as fp:
    ssr = SSRCorrections(iter_messages(fp))

out = ppp_solve(obs, sp3, clk=None, ssr=ssr)
```

The driver substitutes the SSR clock correction for the (missing) CLK
lookup per satellite. SSR orbit corrections are applied as a delta to the
broadcast or SP3-interpolated position. SSR code biases replace the DCB
lookups when both are supplied.

See [SSR corrections](../corrections/ssr.md) for the composer details.

## The Kalman filter

`StaticPPPFilter` is the EKF the driver runs internally. State is
`[px, py, pz, c*dt_rx, N_1, ..., N_n_sv]`. You can drive it directly if
you want full control over the measurement update.

```python
import numpy as np
from rinexpy.kalman import StaticPPPFilter

ekf = StaticPPPFilter(
    n_sv=8,
    initial_position=(x0, y0, z0),
    sigma_code=1.0,                  # 1-sigma pseudorange noise in metres
    sigma_phase=0.005,                # 1-sigma carrier-phase noise in metres
    sigma_position_init=10.0,         # initial position uncertainty
    sigma_clock_init=300.0,           # initial clock uncertainty
    sigma_clock_rate_m=10.0,          # random-walk clock per sqrt(s)
    sigma_position_rate_m=0.0,        # 0 means static; > 0 is kinematic
    sigma_ambig_init_m=1000.0,        # initial ambiguity uncertainty
)

for dt, sv_ecef, sat_clock, pr_if, phase_if, tropo in epochs:
    ekf.predict(dt)
    ekf.update(sv_ecef, sat_clock, pr_if, phase_if, tropo_m=tropo)

print(ekf.position)
print(ekf.clock_bias_s)
print(ekf.position_sigma)
```

The methods are:

| Method | Purpose |
| --- | --- |
| `predict(dt)` | time update, advances the state by `dt` seconds |
| `update(sv_ecef, sat_clock_s, pr_if, phase_if, tropo_m=None)` | one-epoch measurement update |
| `update_with_slip_check(sv_ecef, sat_clock, p1_m, p2_m, phi1, phi2, ...)` | slip-aware measurement update, returns indices of slipped SVs |
| `reset_ambiguity(sv_index)` | wipe one SV's ambiguity state |
| `reset_ambiguities(sv_indices)` | batch-wipe |

Properties: `position`, `clock_bias_s`, `ambiguities_m`, `position_sigma`.

The alias `GNSSFilter = StaticPPPFilter` is re-exported for symmetry with
older code.

## Slip-aware updates

`update_with_slip_check` is the standard recipe for kinematic PPP: detect
cycle slips via the inter-epoch geometry-free combination, drop the
ambiguity state for any slipped satellite, then apply the normal update.

```python
slipped = ekf.update_with_slip_check(
    sv_ecef, sat_clock_s,
    p1_m, p2_m, phi1_cycles, phi2_cycles,
    tropo_m=tropo,
    slip_threshold_cycles=2.0,
    f1=1575.42e6, f2=1227.60e6,
)
if slipped.size:
    print(f"slipped at this epoch: {slipped}")
```

The slip threshold is in cycles of the geometry-free phase. A real
multipath spike rarely exceeds 0.5 cycles; a true cycle slip typically
jumps by 1+ cycles. The default of 2 cycles is a conservative bound that
favours fewer false positives.

## Multi-constellation filter

The standard `StaticPPPFilter` lumps all satellites into one state vector
and uses one receiver-clock bias. For multi-constellation work, each
constellation has its own inter-system bias (ISB), and the standard choice is
to estimate one ISB per non-GPS constellation.

```python
import numpy as np
from rinexpy.kalman_multignss import StaticPPPFilterMultiGNSS

ekf = StaticPPPFilterMultiGNSS(
    n_sv=12,
    constellations=["G", "G", "G", "G", "G", "E", "E", "E", "E", "R", "R", "R"],
    initial_position=(x0, y0, z0),
    initial_zwd_m=0.1,
    sigma_isb_init=100.0,
)
ekf.predict(dt)
ekf.update(sv_ecef, sat_clock, pr_if, phase_if, wet_mapping_per_sv)
print(ekf.isb_m("E"))     # Galileo ISB in metres
print(ekf.isb_m("R"))     # GLONASS ISB
print(ekf.zwd_m)           # current zenith wet delay
```

The state adds one ZWD scalar and one ISB per non-GPS constellation.

## ZWD-augmented filter

`StaticPPPFilterZTD` (also exported as a sibling) carries the zenith wet
delay as part of the state vector. The driver passes wet mapping factors
per satellite per epoch, and the filter blends them with the per-SV
elevation to back out a ZWD estimate.

```python
from rinexpy.kalman_ztd import StaticPPPFilterZTD

ekf = StaticPPPFilterZTD(
    n_sv=8,
    initial_position=(x0, y0, z0),
    initial_zwd_m=0.1,
    sigma_zwd_rate_m_per_sqrt_hr=0.01,    # 1 cm per sqrt(hr) random walk
)
ekf.predict(dt)
ekf.update(sv_ecef, sat_clock, pr_if, phase_if, wet_mapping_per_sv)
print(ekf.zwd_m, "+/-", ekf.zwd_sigma_m)
```

ZWD estimation is the standard move for stationary geodetic-class deployments
that need the meteorological observable, or for any kinematic platform
flying through a non-uniform troposphere.

## PPP-RTK fusion

`PPPRTKFusion` runs a PPP filter and a single-baseline RTK solver in
parallel, then inverse-variance-blends the two position estimates. Short
baselines lean toward RTK; long baselines lean toward PPP.

```python
from rinexpy.ppp_rtk import PPPRTKFusion
from rinexpy.multifreq import LAMBDA_L1

fusion = PPPRTKFusion(
    n_sv=12,
    initial_position=(x0, y0, z0),
    base_position=(base_x, base_y, base_z),
    rtk_sigma_floor_m=0.01,
    rtk_sigma_ppm_per_km=1.0,
)

# Per epoch, run PPP and RTK updates separately:
fusion.update_ppp(sv_ecef, sat_clock, pr_if, phase_if, wet_mapping)
rtk_result = fusion.update_rtk(
    rover_pr, base_pr, rover_phase, base_phase,
    sv_positions_ecef, wavelength=LAMBDA_L1,
)

print("PPP only:  ", fusion.ppp_position, "+/-", fusion.ppp_sigma)
print("Fused:     ", fusion.fused_position, "+/-", fusion.fused_sigma)
print("RTK weight:", fusion.rtk_weight)
print("baseline:  ", fusion.baseline_km, "km")
```

The RTK weight goes to zero past a few tens of kilometres; the fusion
estimate is then PPP.

## Static-batch PPP with integer fixing

For applications that need a single static fix over a session (geodetic
benchmarks, antenna calibrations), the static-batch solver does the whole
session in one solve.

```python
from rinexpy.positioning import (
    ppp_solve_static_batch,
    ppp_solve_static_batch_with_ar,
)

sol = ppp_solve_static_batch(
    pr_if=pr_if_m,                # (n_epoch, n_sv) iono-free pseudorange
    phase_if=phase_if_m,          # (n_epoch, n_sv) iono-free phase
    sv_ecef=sv,                   # (n_epoch, n_sv, 3) ECEF at signal-emission
    sat_clock_s=sat_clock,        # (n_epoch, n_sv)
    tropo=tropo,                  # (n_epoch, n_sv), optional
    initial_position=(x0, y0, z0),
    sigma_code=1.0,
    sigma_phase=0.005,
)
```

The closed-loop AR variant resolves the iono-free ambiguity through the
WL+NL chain.

```python
sol = ppp_solve_static_batch_with_ar(
    pr_if=pr_if_m, phase_if=phase_if_m,
    sv_ecef=sv, sat_clock_s=sat_clock,
    p1_m=p1, p2_m=p2,                 # individual frequency observables
    phi1_cycles=phi1, phi2_cycles=phi2,
    tropo=tropo,
    initial_position=(x0, y0, z0),
)
```

The output dict carries `position`, `position_cov`, `ambiguities_m` (float
or integer-fixed depending on which variant), and a per-iteration trace
of the residual norm.

## PPP convergence

The main timeline for a static receiver tracking GPS + Galileo at 1 Hz:

| Time | Float position accuracy | Notes |
| --- | --- | --- |
| 0 s | 5 m | bootstrap from broadcast nav |
| 30 s | 1 m | first decent code-only fix |
| 5 min | 30 cm | carrier-phase starts to bind ambiguities |
| 15 min | 10 cm | mature float ambiguities |
| 30 min | 3 cm | classic PPP convergence |
| 1 hr | 2 cm | typical asymptote |
| 24 hr | 1 cm | overnight observation session |

With integer ambiguity resolution (`ppp_solve_static_batch_with_ar`),
times shrink by roughly a factor of two.

A kinematic receiver tracks to a few cm after the static convergence
finishes; if the kinematic mode starts cold, expect the same convergence
timeline.

## CLI

```sh
uv run rinexpy ppp obs.rnx sp3.sp3 clk.clk
```

The CLI reads each input, calls `ppp_solve` with default options, and
prints the final position plus sigma. It is mostly useful for one-shot
spot checks; for serious work, drive the solver from Python so you can
configure the corrections.

## Related pages

- [SP3 and clock products](../formats/sp3-clk.md): the precise-products reader.
- [Atmosphere products](../formats/atmosphere-products.md): the ANTEX / GPT2w / DCB readers.
- [Atmospheric models](../corrections/atmosphere.md): the Saastamoinen / VMF1 helpers.
- [SSR corrections](../corrections/ssr.md): real-time alternative to CLK.
- [LAMBDA and ambiguity resolution](lambda.md): the integer fix for PPP-AR.
- [Kalman filters](kalman.md): the filter internals.
- [Time transfer](time-transfer.md): a related use of the iono-free combination.
