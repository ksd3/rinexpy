# Single-point positioning

Single-point positioning (SPP) is the simplest GNSS fix. The receiver
measures a pseudorange to four or more satellites whose ECEF positions are
known (from broadcast NAV or SP3), and solves four unknowns: its own ECEF
position `(x, y, z)` and clock bias `dt_rx`. The solver in
`rinexpy.spp_solve` is the iterative weighted least-squares algorithm that
every textbook covers, plus a handful of extras.

## The model

The pseudorange to satellite `i` is

```
PR_i = ||sv_i - rx|| + c*dt_rx + (other delays)
```

`(other delays)` include the satellite clock bias, the ionospheric delay,
the tropospheric delay, the satellite hardware bias (DCB), and the receiver
hardware bias. In bare SPP you ignore everything except the satellite-clock
bias (which is corrected from the broadcast ephemeris before the call) and
get a 5-10 m horizontal fix on a noisy single-frequency receiver, or 1-2 m
on a dual-frequency receiver with the iono-free combination.

The solver linearises the geometric distance around an initial guess and
iterates until the position update drops below `tol`.

## The basic call

```python
import numpy as np
import rinexpy as rp

# In a real workflow:
#   sv_ecef is (n_sv, 3) array of satellite ECEF positions at signal-emission time
#   pseudoranges is (n_sv,) array of measured pseudoranges in metres
sol = rp.spp_solve(sv_ecef, pseudoranges, max_iter=20)
print(sol)
```

The return dict has:

| Key | Type | Meaning |
| --- | --- | --- |
| `position` | `tuple[float, float, float]` | ECEF in metres |
| `lla` | `tuple[float, float, float]` | latitude (deg), longitude (deg), altitude (m) |
| `clock_bias` | `float` | receiver clock bias in seconds |
| `n_iter` | `int` | number of iterations to converge |
| `residuals` | `ndarray` | per-SV post-fit residual in metres |

For noise-free synthetic input, the solver converges in 5-6 iterations
from the `(0, 0, 0)` initial guess at Earth's centre. With real noisy
pseudoranges and a strong geometry, convergence is similar; with weak
geometry it may take 10-15 iterations or fail.

## A worked synthetic example

The snippet below builds six well-spread satellites at GPS altitude around
a receiver at (40°N, 3°W, 100 m), applies a 100-microsecond clock bias to
the pseudoranges, and recovers the truth to floating-point precision.

```python
import numpy as np
import rinexpy as rp
from rinexpy.geodesy import lla_to_ecef

C_M_PER_S = 299_792_458.0

truth = np.array(lla_to_ecef(40.0, -3.0, 100.0))

sv_radius = 2.66e7
az_el = [(0, 70), (60, 30), (120, 50), (200, 20), (260, 60), (320, 40)]
sv = []
lat, lon = np.radians(40.0), np.radians(-3.0)
sl, cl = np.sin(lon), np.cos(lon)
sp, cp = np.sin(lat), np.cos(lat)
for az, el in az_el:
    a = np.radians(az)
    elr = np.radians(el)
    e = np.cos(elr) * np.sin(a)
    n = np.cos(elr) * np.cos(a)
    u = np.sin(elr)
    x = -sl * e - sp * cl * n + cp * cl * u
    y = cl * e - sp * sl * n + cp * sl * u
    z = cp * n + sp * u
    sv.append(truth + sv_radius * np.array([x, y, z]))
sv = np.array(sv)

bias_s = 1e-4
pr = np.linalg.norm(sv - truth, axis=1) + C_M_PER_S * bias_s

sol = rp.spp_solve(sv, pr, max_iter=20)
print("ECEF:", sol["position"])
print("LLA: ", sol["lla"])
print("bias:", sol["clock_bias"], "s")
print("iter:", sol["n_iter"])
```

## RAIM (Receiver Autonomous Integrity Monitoring)

Real pseudoranges sometimes contain outliers from multipath or temporary
loss of lock. `spp_solve(raim=True)` runs a chi-squared residual test
after the fit. If the test fails, the satellite with the worst residual is
dropped and the LSQ re-runs. The process repeats up to `max_exclusions`
times.

```python
sol = rp.spp_solve(
    sv_ecef, pseudoranges,
    raim=True,
    sigma_pr=5.0,             # 1-sigma pseudorange noise in metres
    p_fa=1e-4,                # acceptable false-alarm rate
    max_exclusions=2,
)
if sol["fault_detected"]:
    print("excluded:", sol["excluded_svs"])
```

Extra return keys when `raim=True`:

| Key | Meaning |
| --- | --- |
| `raim_test` | chi-squared statistic of the final fix |
| `raim_threshold` | rejection threshold for `p_fa` and the current geometry |
| `fault_detected` | bool: True if any SV was excluded |
| `excluded_svs` | list of indices that were dropped |
| `raim_failed` | bool: True if `max_exclusions` ran out without passing |

The chi-squared threshold scales with the number of redundant satellites,
so a fix with only 5 SVs (one redundant beyond the 4 unknowns) has a high
threshold and a noisy fault detection. With 8+ SVs the false-alarm rate
converges on `p_fa`.

The matching direct entry point is `rinexpy.positioning.spp_solve_raim`.

## DCB correction

For PPP-class single-frequency work the satellite hardware bias on C1W
versus C1C matters at the metre level. Pull a SINEX-BIAS DCB record set
and pass it to the solver.

```python
from rinexpy.dcb import read_bsx

records = read_bsx("CAS0MGXRAP_20231560000_01D_01D_DCB.BSX")
sol = rp.spp_solve(
    sv_ecef, pseudoranges_m,
    sv_labels=["G05", "G10", "G12", ...],
    dcb_records=records,
    dcb_obs_code="C1W",
    dcb_epoch=epoch,
)
```

The solver looks up the satellite OSB (observable-specific signal bias) for
each SV at the given epoch and subtracts `c * value` from the matching
pseudorange.

For a station with its own receiver biases, add `dcb_station="STAT00BRA"`.

## Broadcast TGD correction

When you do not have a SINEX-BIAS file but you do have a NAV file, the
broadcast TGD (timing group delay) is a coarser correction that still
removes most of the satellite-dependent bias.

```python
from rinexpy.positioning import tgd_from_nav

nav = rp.load("brdc2800.15n")
tgd_map = tgd_from_nav(nav, epoch)
sol = rp.spp_solve(sv_ecef, pseudoranges_m,
                   sv_labels=svs,
                   tgd_map=tgd_map,
                   tgd_gamma=1.0)            # L1 scaling
```

The `tgd_gamma` argument is the frequency scaling factor:

| Observation | `tgd_gamma` |
| --- | --- |
| L1 | 1.0 |
| L2 | `(f1/f2)**2 ≈ 1.647` |
| Iono-free combination | 0.0 |

For BeiDou the broadcast TGD is in two parts (`TGD1` and `TGD2`). Pass
`field="TGD1"` to `tgd_from_nav` to pick which one.

## Ionospheric correction

The L1 ionospheric delay is several metres at quiet times and tens of
metres during solar maxima. SPP has three options for it.

### Klobuchar broadcast

The eight alpha/beta coefficients in the GPS NAV header drive the
Klobuchar L1 ionospheric correction.

```python
from rinexpy.geodesy import klobuchar
from rinexpy.positioning import apply_klobuchar_correction

# Alpha and beta come from the NAV header.
nav_hdr = rp.rinexheader("brdc2800.15n")
alpha = nav_hdr["ION ALPHA"]
beta = nav_hdr["ION BETA"]

# Apply to one pseudorange per SV:
corrected = apply_klobuchar_correction(
    pseudoranges_m, sv_ecef, rx_ecef=rx,
    iono_alpha=alpha, iono_beta=beta,
    epoch=datetime(2015, 10, 7, 12, 0, 0),
)
```

The function evaluates the Klobuchar model at each SV's pierce point in
the ionospheric thin shell (350 km), maps to slant, and subtracts the
slant delay from the pseudorange.

### IONEX

For higher precision than Klobuchar, an IONEX file gives a global TEC map.

```python
from rinexpy.ionex import load_ionex, interp_tec, slant_tec

ionex = load_ionex("CODG2860.24I")
for i, sv_pos in enumerate(sv_ecef):
    az, el = azimuth_elevation(rx, sv_pos)
    # Pierce point ignored for brevity; in real code use the geodetic
    # of the pierce point at the 350 km thin shell.
    vtec = interp_tec(ionex, lat_deg=40.0, lon_deg=-3.0, epoch=epoch)
    delay_l1_m = slant_tec(vtec, el) * 0.16   # 0.16 m / TECU on L1
    pseudoranges_m[i] -= delay_l1_m
```

### Iono-free combination

For dual-frequency observations, the iono-free combination removes the
first-order ionospheric delay analytically.

```python
from rinexpy.positioning import iono_free_pseudorange

p_if = iono_free_pseudorange(p1_m, p2_m)
sol = rp.spp_solve(sv_ecef, p_if, max_iter=20)
```

The iono-free combination is the workhorse for PPP. See
[Precise point positioning](ppp.md).

## Tropospheric correction

Standard-atmosphere Saastamoinen is the right default for SPP. At
elevations above 15° it is accurate to about a centimetre.

```python
from rinexpy.geodesy import azimuth_elevation, saastamoinen

az, el = azimuth_elevation(rx, sv_ecef)
tropo_m = np.array([saastamoinen(el_i, alt_m) for el_i in el])
corrected = pseudoranges_m - tropo_m
```

For PPP-class corrections, swap in GPT2w + VMF1. See
[Atmospheric models](../corrections/atmosphere.md).

## Earth rotation and light-time

The satellite transmits the signal at one instant; the receiver records it
about 70 ms later, by which point Earth has rotated by about 1.5
microradians (around 30 metres at the equator). For most workflows you
correct for this by evaluating the SV position at the signal-emission
time, not the receive time.

`apply_light_time_and_earth_rotation` does this for an SP3 dataset:

```python
from rinexpy.positioning import apply_light_time_and_earth_rotation

sv_emission_pos = apply_light_time_and_earth_rotation(
    sp3, receive_time=t, rx_ecef=rx, sv_label="G05",
    order=10, max_iter=3,
)
```

The function fixed-point iterates: estimate `t_tx = t - rho/c`, evaluate
the SV at `t_tx`, recompute `rho`, iterate. Three iterations is enough for
millimetre accuracy.

## Code-only PPP

For PPP-class accuracy without the full filter, the iono-free pseudorange
combination plus a tropospheric correction is the workhorse.

```python
from rinexpy.positioning import ppp_solve_code_only

sol = ppp_solve_code_only(
    pseudoranges_if=p_if_m,
    sv_ecef=sv,
    sat_clock_s=sat_clock,
    tropospheric_delay_m=tropo_per_sv,
    initial_guess=(x0, y0, z0),
)
```

This is float code-only; expect 30 cm horizontal at convergence, after the
sub-metre satellite clocks from SP3 + CLK feed in. For sub-centimetre
results, use the carrier-phase filter (`StaticPPPFilter` / `ppp_solve`).

## Static-batch carrier-phase PPP

For a static receiver and dual-frequency data, the static-batch solver
adds carrier-phase observations and solves for ambiguities in one pass.

```python
from rinexpy.positioning import ppp_solve_static_batch

sol = ppp_solve_static_batch(
    pr_if=pr_if_m,                      # (n_epoch, n_sv)
    phase_if=phase_if_m,                # (n_epoch, n_sv)
    sv_ecef=sv,                         # (n_epoch, n_sv, 3)
    sat_clock_s=sat_clock,
    tropo=tropo_per_epoch_per_sv,
    initial_position=(x0, y0, z0),
    sigma_code=1.0,
    sigma_phase=0.005,
)
```

The float-ambiguity result is centimetre-class after about 30 minutes of
data; integer ambiguity resolution lifts it to sub-centimetre. See
`ppp_solve_static_batch_with_ar` for the closed-loop version.

## Performance

A 20-satellite SPP fix runs in about 200 microseconds on a modern CPU.
The cost is dominated by the per-iteration linear algebra (a 4×4 normal
equations system), not the residual evaluation. Geometry-strong epochs
converge in 4-5 iterations; weak geometry can need 10-15.

## CLI

The CLI subcommand combines an OBS file and a NAV file into a single fix.

```sh
uv run rinexpy spp file.18o file.18n
```

The CLI reads the OBS file, picks the C1 or C1C pseudoranges, evaluates
the satellite positions from the NAV file at every epoch, and calls
`spp_solve` per epoch. Output is one position per epoch.

## Related pages

- [RTK and integer fixing](rtk.md): centimetre-class baseline solver.
- [Precise point positioning](ppp.md): centimetre-class absolute fix.
- [LAMBDA and ambiguity resolution](lambda.md): the integer fix algorithm.
- [Snapshot positioning](snapshot.md): SPP with sub-second data.
- [Atmospheric models](../corrections/atmosphere.md): the underlying tropo / iono helpers.
