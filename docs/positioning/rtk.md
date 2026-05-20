# RTK and integer fixing

Real-time kinematic (RTK) positioning solves the rover position relative
to a reference base by double-differencing the carrier-phase observations.
The double difference cancels the satellite and receiver clocks, the
ionospheric delay (to first order), and the tropospheric delay (when the
baseline is short). What is left is the geometry plus an integer ambiguity
per satellite pair.

When the ambiguity is fixed to its true integer value, the baseline is
centimetre-class. When it cannot be fixed (weak geometry, high noise,
recently slipped lock), the float-ambiguity solution is decimetre-class.

`rinexpy` includes three layers: `double_difference_solve` for the float
solution, `rtk_fix` for one-shot LAMBDA integer fixing, and
`SequentialRTK` for multi-epoch RTK with ambiguity carry-over.

## Float baseline

`double_difference_solve` returns the float-ambiguity DD solution. Use it
when you want to inspect the float baseline, or as the input to LAMBDA.

```python
import numpy as np
from rinexpy.rtk import double_difference_solve
from rinexpy.multifreq import LAMBDA_L1

sol = double_difference_solve(
    rover_pr_m,
    base_pr_m,
    rover_phase_cycles,
    base_phase_cycles,
    sv_positions_ecef,           # (n_sv, 3)
    base_position_ecef,          # (3,) tuple or ndarray
    wavelength=LAMBDA_L1,
    initial_baseline=(0.0, 0.0, 0.0),
    max_iter=8,
    tol=1e-3,
)
print(sol["baseline"])           # (b_x, b_y, b_z) in metres
print(sol["rover_position"])     # base + baseline in ECEF
print(sol["ambiguities"])        # float DD ambiguities, in cycles
print(sol["residuals"])          # post-fit residuals
print(sol["reference_sv_index"]) # the SV that anchors the DD
print(sol["dd_pseudorange"])     # DD pseudoranges fed to the LSQ
```

The solver picks the highest-elevation satellite as the reference and
forms double differences against the remaining satellites.

## LAMBDA integer fix

`rtk_fix` runs the joint baseline-and-ambiguity LSQ, fixes the integers
via LAMBDA, and re-solves the baseline with the fixed integers held in.

```python
from rinexpy.rtk import rtk_fix

sol = rtk_fix(
    rover_pr_m, base_pr_m,
    rover_phase_cycles, base_phase_cycles,
    sv_positions_ecef,
    base_position_ecef,
    wavelength=LAMBDA_L1,
    sigma_pr=1.0,                # m
    sigma_phase=0.005,           # m
    ratio_threshold=3.0,
)
print("float:    ", sol["float"]["baseline"])
print("fixed:    ", sol["fixed"]["baseline"])
print("accepted: ", sol["fixed_accepted"])
print("ratio:    ", sol["lambda"]["ratio"])
```

The return dict has three sub-dicts: `float`, `fixed`, and `lambda`,
plus the top-level `fixed_accepted` bool and `reference_sv_index`.

```
float:    {'baseline', 'rover_position', 'ambiguities', 'n_iter', 'ambiguity_covariance'}
fixed:    {'baseline', 'rover_position', 'ambiguities'}     # integer-fixed
lambda:   {'a_int', 'ratio', 'accepted', 'candidates', 'sq_errors', 'aborted'}
```

The top-level `fixed_accepted` is True when the LAMBDA ratio test exceeds
`ratio_threshold`. The conservative default is 3.0; for high-confidence
applications use 5.0 or higher.

## Worked synthetic example

Below is the full path from "two receivers tracking the same satellites" to
a centimetre-class baseline.

```python
import numpy as np
from rinexpy.geodesy import lla_to_ecef
from rinexpy.multifreq import LAMBDA_L1
from rinexpy.rtk import rtk_fix

rng = np.random.default_rng(2026)
base = np.array(lla_to_ecef(40.0, -3.0, 0.0))
truth_baseline = np.array([5.4, -2.1, 0.7])
rover = base + truth_baseline

sv_radius = 2.66e7
az_el = [(10, 70), (70, 30), (130, 55), (190, 20), (250, 50), (310, 40)]
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
    sv.append(base + sv_radius * np.array([x, y, z]))
sv = np.array(sv)

rho_r = np.linalg.norm(sv - rover, axis=1)
rho_b = np.linalg.norm(sv - base, axis=1)
true_amb = rng.integers(-200, 200, size=sv.shape[0])
pr_r = rho_r
pr_b = rho_b
phase_r = rho_r / LAMBDA_L1 + true_amb
phase_b = rho_b / LAMBDA_L1 + true_amb

sol = rtk_fix(
    pr_r, pr_b, phase_r, phase_b,
    sv, tuple(base),
    wavelength=LAMBDA_L1,
    sigma_pr=1.0,
    sigma_phase=0.005,
    ratio_threshold=3.0,
)
print(f"truth baseline: {truth_baseline}")
print(f"float baseline: {sol['float']['baseline']}")
print(f"fixed accepted: {sol['fixed_accepted']}")
if sol['fixed_accepted']:
    print(f"fixed baseline: {sol['fixed']['baseline']}")
print(f"LAMBDA ratio:   {sol['lambda']['ratio']:.2f}")
```

A noise-free input lands within micrometres of the truth baseline because
the carrier phase resolves all of the ambiguity once the integers are
fixed.

## Sequential RTK

`rtk_fix` solves one epoch at a time. For a moving rover, the integer fix
should carry between epochs while the per-SV lock holds, and only
re-bootstrap on slip events. `SequentialRTK` provides that.

```python
from rinexpy.rtk import SequentialRTK
from rinexpy.multifreq import LAMBDA_L1

rtk = SequentialRTK(
    tuple(base),
    wavelength=LAMBDA_L1,
    ratio_threshold=3.0,
    slip_threshold_cycles=0.5,
    sigma_pr=1.0,
    sigma_phase=0.005,
    min_lock_to_fix=2,
)

for svs, pr_r, pr_b, phase_r, phase_b, sv_pos in epochs:
    out = rtk.update(svs, pr_r, pr_b, phase_r, phase_b, sv_pos)
    print(f"baseline: {out['baseline']}")
    print(f"fixed accepted: {out['fixed_accepted']}, ratio: {out['ratio']:.2f}")
    print(f"carry-over: {out['carry_over_count']}, slipped: {out['slipped_svs']}")
```

The per-epoch return dict carries:

| Key | Meaning |
| --- | --- |
| `baseline` | the current best baseline (fixed if accepted, float otherwise) |
| `rover_position` | base + baseline in ECEF |
| `n_total` | satellites tracked this epoch |
| `n_fixed` | satellites whose ambiguity is integer-fixed |
| `fixed_accepted` | bool: the full ratio test passed |
| `ratio` | LAMBDA ratio of the joint fix |
| `carry_over_count` | satellites whose integer fix survived from the previous epoch |
| `slipped_svs` | satellites whose lock dropped this epoch |

Per-SV cycle slips are detected by comparing the inter-epoch
single-difference phase against the single-difference code. When the SD
phase jumps more than `slip_threshold_cycles` while the SD code is within
its noise floor, the satellite's lock is dropped.

If the full ratio test fails, the solver runs partial AR on the
highest-confidence subset of satellites. The result is a fixed baseline
constrained to the partial integer set when possible, and a float fallback
when not.

To wipe all state (after a long gap, or to reseed the filter), call
`rtk.reset()`.

## Wide-lane resolution

For dual-frequency observations, fixing the wide-lane ambiguity first
makes the subsequent L1/L2 fix easier. The Melbourne-Wuebbena combination
removes geometry and ionospheric delay, leaving only the WL integer plus
noise.

```python
from rinexpy.multifreq import (
    LAMBDA_L1, LAMBDA_L2, LAMBDA_WL, LAMBDA_NL,
    wide_lane_phase,
    narrow_lane_phase,
    melbourne_wubbena,
    resolve_wide_lane,
    split_wl_into_l1_l2,
    lambda_dual_freq,
)

# Wide-lane resolution from MW gate:
out = resolve_wide_lane(phi1_cycles, phi2_cycles, p1_m, p2_m,
                         sigma_threshold=0.25)
print("N_WL:", out["N_WL"])
print("float WL:", out["float_WL"])
print("per-SV fixed:", out["fixed_mask"])
print("fraction fixed:", out["fraction_fixed"])

# Joint L1+L2 fix via WL+NL decomposition:
out = lambda_dual_freq(a_l1_float, a_l2_float,
                       p1_m=pr_l1, p2_m=pr_l2,
                       sigma_threshold=0.25)
print("N_L1:", out["N_L1"])
print("N_L2:", out["N_L2"])
print("Fraction fixed:", out["fraction_fixed"])
```

The wavelengths are exposed as constants:

| Constant | Value (m) | Notes |
| --- | --- | --- |
| `LAMBDA_L1` | 0.19029 | GPS L1 |
| `LAMBDA_L2` | 0.24421 | GPS L2 |
| `LAMBDA_L5` | 0.25483 | GPS L5 |
| `LAMBDA_WL` | 0.86192 | L1 - L2 wide-lane |
| `LAMBDA_NL` | 0.10695 | L1 + L2 narrow-lane |
| `LAMBDA_EWL_15` | 0.75148 | L1 - L5 extra-wide-lane |
| `LAMBDA_EWL_25` | 5.86103 | L2 - L5 extra-wide-lane |

## Three-Carrier Ambiguity Resolution

Modern receivers tracking L1 + L2 + L5 (or E1 + E5a + E5b for Galileo) can
use Three-Carrier Ambiguity Resolution (TCAR). The extra-wide-lane
combination on L2 - L5 has a wavelength of nearly 6 metres, so the integer
ambiguity is noise-free.

```python
from rinexpy.multifreq import tcar_resolve

out = tcar_resolve(
    phi1_cycles, phi2_cycles, phi5_cycles,
    p1_m, p2_m, p5_m,
)
print("N_EWL:", out["N_EWL"])
print("N_WL:", out["N_WL"])
print("N_L1:", out["N_L1"])
print("fixed:", out["fixed_mask"])
```

The output is the chain `EWL → WL → L1` of integer fixes, each constraining
the next.

## Network RTK

For long baselines (more than ~10 km), the residual ionospheric and
tropospheric errors break the double-difference cancellation. Network RTK
uses observations from N reference stations to model the spatial gradient
of those errors and synthesise a virtual base co-located with the rover.

```python
from rinexpy.vrs import synthesize_vrs
from rinexpy.rtk import rtk_fix

vrs = synthesize_vrs(bases, rover_approx_pos, wavelength=LAMBDA_L1)
sol = rtk_fix(
    rover_pr, vrs["pr"],
    rover_phase, vrs["phase"],
    vrs["sv_positions"], vrs["base_position"],
    wavelength=LAMBDA_L1,
)
```

`bases` is a list of dicts, one per physical reference station, with
`station_ecef`, `sv_positions`, `pr`, `phase`. The synthesizer fits a plane
to the per-satellite residuals across the network and evaluates the plane
at the rover's approximate position.

For more network RTK details, see [Network RTK and VRS](network.md).

## LAMBDA internals

The LAMBDA algorithm runs three steps. First, it decorrelates the float
ambiguity covariance via Z-transform. Second, it bootstraps a single
integer candidate from the decorrelated LDL factorisation. Third, it
searches the integer lattice around the bootstrap with a branch-and-bound
that finds the two best candidates by squared-residual norm.

`rinexpy.lambda_ar.lambda_resolve` is the public entry point.

```python
from rinexpy.lambda_ar import lambda_resolve
import numpy as np

a_float = np.array([5.1, 10.4])
Q = np.array([[1.0, 0.5], [0.5, 1.0]])
out = lambda_resolve(a_float, Q)
print(out["a_int"])            # [5, 10]
print(out["ratio"])            # second-best / best squared residual
print(out["accepted"])         # bool: passed the ratio test
print(out["candidates"])       # the two best integer candidates
print(out["sq_errors"])        # their squared residuals
```

`ratio_threshold=3.0` is the conservative default. Lower values accept more
fixes but increase the chance of a wrong-integer fix.

Internally `lambda_resolve` uses three building blocks:

| Function | Purpose |
| --- | --- |
| `ldl(Q)` | LDL decomposition of the symmetric positive-definite covariance |
| `bootstrap(L, a_float)` | quick integer estimate from the decorrelated float ambiguities |
| `integer_least_squares(a_float, Q, n_cands)` | full ILS branch-and-bound for the `n_cands` best integers |

For LAMBDA exit conditions, `integer_least_squares` raises `ILSAborted`
when the search exceeds `max_nodes` or `max_seconds`. The exception
carries the best partial candidates so the caller can apply a soft ratio
test instead of failing outright.

## CLI

```sh
uv run rinexpy rtk rover.obs base.obs nav.nav
```

The CLI reads both observation files, evaluates the satellite positions
from the NAV file at every common epoch, calls `rtk_fix` per epoch, and
prints one baseline per accepted epoch.

## Related pages

- [LAMBDA and ambiguity resolution](lambda.md): the LAMBDA decoder in depth.
- [Network RTK and VRS](network.md): joint multi-base solver.
- [Precise point positioning](ppp.md): the absolute-position alternative.
- [Single-point positioning](spp.md): the code-only fallback.
- [QC and cycle slips](../quality/qc.md): slip detection that feeds `SequentialRTK`.
