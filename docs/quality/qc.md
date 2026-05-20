# QC and cycle slips

For PPP, RTK, and any other carrier-phase processing, the per-satellite
phase observation must be continuous through the session. A cycle slip
(a sudden jump in the integer number of cycles the receiver is tracking)
breaks that continuity, and any downstream filter that does not detect
the slip will take in it as a position error or as a biased ambiguity.

`rinexpy`'s QC module is released three independent slip detectors, multipath
combinations, a Hatch filter for carrier-smoothed code, and repair
helpers. All of them live in `rinexpy.qc`.

## Cycle slip detectors

Three detectors, each appropriate for different observations:

| Detector | Inputs | Detects |
| --- | --- | --- |
| `detect_slips_phase_only` | one phase series | jumps in single-frequency phase |
| `detect_slips_geometry_free` | dual-frequency phase | iono + cycle slip combined |
| `detect_slips_mw` | dual-frequency phase + code | wide-lane jumps |

### Single-frequency phase-only

The simplest detector. Look at the first difference of the phase series;
flag any jump above a threshold.

```python
import numpy as np
from rinexpy.qc import detect_slips_phase_only

phi = np.array([100.0, 101.0, 102.0, 200.0, 201.0])  # cycles
slips = detect_slips_phase_only(phi, threshold_cycles=1.0)
print(np.where(slips)[0])     # array([3])
```

The output is a boolean array of the same length, with `True` at every
epoch where a slip is detected. The detector is straightforward and
catches obvious large slips, but it cannot distinguish a cycle slip from
a multipath-induced phase jump, and it cannot detect a slip whose size
is below the threshold.

For continuous tracking with clean L1 data, threshold = 1 cycle (about 19
cm) is a sensible default.

### Geometry-free phase

For dual-frequency receivers, the geometry-free combination `L1 - L2`
cancels the geometry and the receiver clock, leaving only the
ionospheric delay plus the ambiguity difference. A cycle slip shows up
as a discontinuity in the combination.

```python
from rinexpy.qc import detect_slips_geometry_free

slips = detect_slips_geometry_free(
    phi1_cycles, phi2_cycles,
    lambda1=0.19029367,
    lambda2=0.24421021,
    threshold_m=0.05,                   # 5 cm
)
print(np.where(slips)[0])
```

The threshold is in metres of geometry-free phase. A typical ionosphere
varies smoothly at the cm-per-minute level, so a 5 cm threshold catches
slips reliably while letting the iono drift through.

### Melbourne-Wuebbena

The MW combination cancels both geometry and ionosphere. What is left is
the wide-lane integer ambiguity plus noise. A wide-lane cycle slip jumps
the combination by exactly one cycle of the WL wavelength (about 86 cm).

```python
from rinexpy.qc import detect_slips_mw

slips = detect_slips_mw(
    phi1_cycles, phi2_cycles, p1_m, p2_m,
    threshold_cycles=0.5,
)
```

The threshold is in cycles of the WL combination. The MW combination is
noisy (about 0.7 cycles 1-sigma) because it depends on the pseudorange,
so the threshold needs to be a few sigma above the noise floor.

### Combined dispatch

For an OBS dataset, `detect_slips` dispatches the right detector per
satellite based on which observables are present.

```python
import rinexpy as rp
from rinexpy.qc import detect_slips

obs = rp.load("tests/data/obs3.01gage.10o")
out = detect_slips(
    obs,
    threshold_cycles_mw=0.5,
    threshold_m_gf=0.05,
    threshold_cycles_phase=1.0,
)
for sv, mask in out["slips_by_sv"].items():
    if mask.any():
        print(f"{sv}: slips at {np.where(mask)[0]} (method: {out['methods_by_sv'][sv]})")
```

The function returns a dict with two keys: `slips_by_sv` (per-SV
boolean mask of slip epochs) and `methods_by_sv` (which detector was
used per SV). The dispatcher picks the best available method per SV.
If both L1 and L2 phase plus pseudoranges are present, MW is used. If
L1 and L2 phase are present but pseudoranges are missing,
geometry-free is used. Otherwise it falls back to phase-only on L1.

## Repairing slips

For phase-only series, the repair is a polynomial fit on the pre-slip
window, extrapolated forward, and the gap rounded to the nearest integer.

```python
from rinexpy.qc import repair_slips

repaired = repair_slips(
    phase_cycles, slips, fit_window=5,
)
```

The window is the number of post-slip samples used to fit the trend; 5
is a sensible default for 1 Hz data.

For dual-frequency receivers, `repair_slips_dual` uses both the MW and the
geometry-free combinations to decide the exact integer jump on L1 and L2
separately.

```python
from rinexpy.qc import repair_slips_dual

phi1_repaired, phi2_repaired = repair_slips_dual(
    phi1_cycles, phi2_cycles, p1_m, p2_m, slips,
)
```

The dual repair is more accurate than the single-frequency one. It uses
the wide-lane jump from MW to constrain the L1-L2 integer difference,
then the geometry-free jump to constrain the absolute L1 jump.

## Multipath combinations

The TEQC-style multipath combinations isolate the multipath signature on
the pseudorange. The MP1 combination is

```
MP1 = P1 - (1 + 2/(γ-1)) L1 + (2/(γ-1)) L2
```

where `γ = (f1/f2)²`. MP2 has the corresponding form for the P2 / L2
pseudorange.

The combination removes geometry, ionosphere, and tropospheric delay.
What is left is the multipath plus the satellite-station hardware bias
plus noise.

```python
from rinexpy.qc import mp1, mp2, multipath_rms

mp1_m = mp1(p1_m, l1_m, l2_m)
mp2_m = mp2(p2_m, l1_m, l2_m)

rms1 = multipath_rms(mp1_m)            # per-arc RMS, averaged across arcs
rms2 = multipath_rms(mp2_m)
print(f"MP1 RMS: {rms1*100:.1f} cm")
print(f"MP2 RMS: {rms2*100:.1f} cm")
```

For a clean site, MP1 RMS is below 10 cm and MP2 RMS is below 15 cm.
Higher numbers point to a multipath-prone environment (buildings,
foliage, ground reflectors).

When you pass `slips=` to `multipath_rms`, the function recomputes the
arc-mean within each slip-bounded arc rather than across the whole
session. Without this, a single slip in the middle of a clean arc
inflates the RMS.

```python
rms1 = multipath_rms(mp1_m, slips=slips_mask)
```

## Hatch carrier-smoothing

The Hatch filter blends pseudorange and carrier phase. The carrier is
noisier in absolute scale (the integer ambiguity is unknown) but very
smooth in time; the pseudorange is unbiased but noisy. The Hatch filter
takes the pseudorange's bias and the carrier phase's smoothness.

```python
from rinexpy.qc import hatch_filter

smoothed = hatch_filter(
    pr_m,
    phase_m,
    window=100,
    slips=None,
)
```

The `window` is the smoothing length in epochs. With 100-epoch smoothing
on 1 Hz data, the output is as smooth as the carrier with
the (long-term) bias of the pseudorange. Typical use: SPP with
carrier-smoothed pseudorange has a 5-10x lower noise than raw SPP.

When you pass `slips=` (a boolean per-epoch mask of cycle slips), the
filter restarts at each slip. Without this, a slip propagates through
the smoothing window and biases the output for the next `window` epochs.

## End-to-end QC flow

A reasonable QC pass looks like this:

```python
import rinexpy as rp
from rinexpy.qc import (
    detect_slips,
    mp1,
    mp2,
    multipath_rms,
    hatch_filter,
)

obs = rp.load("station.rnx.gz")

# 1. Detect slips per SV.
slip_report = detect_slips(obs)
slip_map = slip_report["slips_by_sv"]

# 2. Multipath metrics per SV.
multipath = {}
for sv in obs.sv.values:
    if sv[0] == "G":   # only the example signal set
        p1 = obs.C1W.sel(sv=sv).values
        p2 = obs.C2W.sel(sv=sv).values
        l1 = obs.L1C.sel(sv=sv).values
        l2 = obs.L2W.sel(sv=sv).values
        mp1_arc = mp1(p1, l1, l2)
        mp2_arc = mp2(p2, l1, l2)
        multipath[sv] = {
            "mp1_rms_m": multipath_rms(mp1_arc, slips=slip_map.get(sv)),
            "mp2_rms_m": multipath_rms(mp2_arc, slips=slip_map.get(sv)),
        }

print(multipath)
```

The QC dict, paired with `tools.validate_file`, gives you enough to
triage a directory of archive files.

## Higher-level validation

For a one-line QC report, `rinexpy.tools.validate_file` is the high-level
entry point. It walks the header and the first few epochs.

```python
from rinexpy.tools import validate_file

rep = validate_file("tests/data/demo.10o")
print(rep)
```

Output:

```
{
  'ok': True,
  'warnings': [],
  'info': {'version': 2.11, 'filetype': 'O', 'rinextype': 'obs',
           'systems': 'M'},
  'n_epochs': 2,
  'n_sv': 14,
  'time_first': '2010-03-05T00:00:00.000000000',
  'time_last':  '2010-03-05T00:00:30.000000000',
  'interval_seconds': 30.0,
  'gap_count': 0,
}
```

Warnings include header inconsistencies, gap counts beyond the nominal
interval, malformed observations, and any parsing-stage error that did
not stop the read. The function does not run cycle slip detection or
multipath analysis; for that, use the `qc` module directly.

## Walking a directory

```python
from pathlib import Path
from rinexpy.tools import validate_file

for p in sorted(Path("data/2024").glob("*.rnx.gz")):
    rep = validate_file(p)
    if not rep["ok"]:
        print(p.name, rep["warnings"])
```

## Related pages

- [RINEX observation files](../formats/rinex-obs.md): the file reader.
- [RTK and integer fixing](../positioning/rtk.md): how `SequentialRTK` uses slips.
- [Precise point positioning](../positioning/ppp.md): the EKF slip-aware update.
- [Multi-file tools](../tooling/multi-file.md): `validate_file`, `concat_files`, `diff_datasets`.
- [Spoofing and jamming heuristics](spoofing.md): a different integrity layer.
