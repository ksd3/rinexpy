# Time transfer

GNSS time transfer compares the clocks of two distant receivers. The
canonical recipe is common-view: both receivers observe the same satellite
at the same time, and the difference of their pseudorange measurements
cancels the satellite clock. After subtracting the geometric ranges
(known from precise satellite ECEF), what is left is the
receiver-clock-difference plus residual atmosphere and multipath.

rinexpy ships three helpers in `rinexpy.time_transfer`.

| Function | Purpose |
| --- | --- |
| `p3_combination(p1_m, p2_m)` | iono-free P3 combination per satellite |
| `common_view_difference(...)` | per-SV common-view clock difference |
| `estimate_clock_difference_s(...)` | robust median over SVs |

## The P3 combination

The P3 (iono-free) combination is

```
P3 = (f1^2 * P1 - f2^2 * P2) / (f1^2 - f2^2)
```

It removes the first-order ionospheric delay from a pair of pseudoranges,
at the cost of amplifying the noise by a factor of about 3.

```python
import numpy as np
from rinexpy.time_transfer import p3_combination

p1 = np.array([24123456.7, 24456789.1, ...])    # L1 pseudoranges in metres
p2 = np.array([24123456.9, 24456790.0, ...])    # L2 pseudoranges in metres
p3 = p3_combination(p1, p2)
```

For Galileo or other constellations with non-default frequencies, pass the
band frequencies explicitly:

```python
p3 = p3_combination(p1, p2, f1=1.57542e9, f2=1.20714e9)
```

## Common-view difference

The standard common-view recipe is:

1. Both stations observe the same satellite at the same epoch.
2. Each station subtracts the geometric range from the pseudorange.
3. The two corrected pseudoranges differ by the receiver-clock difference
   (plus residual atmosphere, multipath, and SV ephemeris error, which
   cancel to first order).

```python
from rinexpy.time_transfer import common_view_difference

# pr_A and pr_B are P3-combined pseudoranges at stations A and B,
# in metres, on the same satellites at the same epoch.
# sv_ecef is the (n_sv, 3) satellite positions.
# station_A_ecef, station_B_ecef are the two known station coordinates.

per_sv_delta_m = common_view_difference(
    pr_A_m, pr_B_m,
    sv_ecef,
    station_a_ecef=(x_a, y_a, z_a),
    station_b_ecef=(x_b, y_b, z_b),
)
print(per_sv_delta_m)        # (n_sv,) per-SV estimates of c*(dt_A - dt_B)
```

The per-SV estimates have scatter from multipath, residual ionosphere
(if not using P3), residual troposphere, and ephemeris error. The
right move is to aggregate over SVs.

## Estimating the clock difference

`estimate_clock_difference_s` runs `common_view_difference` and then
takes a robust estimator (median by default) across the SVs.

```python
from rinexpy.time_transfer import estimate_clock_difference_s

delta_dt_s = estimate_clock_difference_s(
    pr_A_m, pr_B_m,
    sv_ecef,
    station_a_ecef=(x_a, y_a, z_a),
    station_b_ecef=(x_b, y_b, z_b),
    estimator="median",
)
print(f"dt_A - dt_B = {delta_dt_s * 1e9:.1f} ns")
```

The estimator argument accepts `"median"` (default), `"mean"`, or
`"trimmed_mean"`. The median is robust to a couple of bad SVs (e.g. one
with severe multipath); the trimmed mean drops the worst 10% on each
side.

For nanosecond-class clock comparisons, aggregate over many epochs in
addition to many SVs. A 30-minute common-view session with 5+ shared
satellites typically resolves the clock difference to a few nanoseconds.

## Worked example

A short synthetic where two stations are 100 km apart and station B has a
1 microsecond clock offset relative to station A.

```python
import numpy as np
from rinexpy.geodesy import lla_to_ecef
from rinexpy.time_transfer import (
    p3_combination,
    common_view_difference,
    estimate_clock_difference_s,
)

C_M_PER_S = 299792458.0

station_A = np.array(lla_to_ecef(40.0, -3.0, 0.0))
station_B = np.array(lla_to_ecef(40.5, -3.5, 0.0))
dt_A = 0.0
dt_B = 1e-6                      # 1 microsecond offset

# Synthetic satellite positions, 6 well-spread SVs.
sv = []
sv_radius = 2.66e7
for az, el in [(0, 70), (60, 30), (120, 50), (200, 20), (260, 60), (320, 40)]:
    a = np.radians(az); elr = np.radians(el)
    e = np.cos(elr) * np.sin(a)
    n = np.cos(elr) * np.cos(a)
    u = np.sin(elr)
    lat, lon = np.radians(40.25), np.radians(-3.25)
    sl, cl = np.sin(lon), np.cos(lon); sp, cp = np.sin(lat), np.cos(lat)
    x = -sl * e - sp * cl * n + cp * cl * u
    y = cl * e - sp * sl * n + cp * sl * u
    z = cp * n + sp * u
    mid = np.array(lla_to_ecef(40.25, -3.25, 0.0))
    sv.append(mid + sv_radius * np.array([x, y, z]))
sv = np.array(sv)

# Geometric pseudoranges + clock biases. Skipping atmosphere for clarity.
pr_A = np.linalg.norm(sv - station_A, axis=1) + C_M_PER_S * dt_A
pr_B = np.linalg.norm(sv - station_B, axis=1) + C_M_PER_S * dt_B

# Common-view per SV:
delta_per_sv = common_view_difference(
    pr_A, pr_B, sv,
    station_a_ecef=tuple(station_A),
    station_b_ecef=tuple(station_B),
)
print("per SV (ns):", (delta_per_sv / C_M_PER_S * 1e9).round(1))

# Aggregated:
delta_dt_s = estimate_clock_difference_s(
    pr_A, pr_B, sv,
    station_a_ecef=tuple(station_A),
    station_b_ecef=tuple(station_B),
)
print(f"dt_A - dt_B: {delta_dt_s * 1e9:.1f} ns")    # ~ -1000 ns
```

The recovered difference matches `-(dt_B - dt_A) = -1 microsecond` to
floating-point precision (the noise-free case).

## In production

Real common-view time transfer adds:

- **P3 combination** to cancel the ionosphere. Always use P3 unless your
  link is < 50 km and you can ignore ionospheric mismatch.
- **Troposphere correction** at both stations. Saastamoinen at standard
  atmosphere is enough for most links.
- **Precise satellite positions and clocks** from SP3 + CLK. Broadcast
  SV clocks have multi-nanosecond error.
- **Multipath rejection** via SNR mask and elevation mask. A 15° mask is
  the conventional choice.
- **Aggregation over time.** Average over 5-minute windows, or apply a
  Kalman filter to the per-epoch differences for smoothing.

For UTC traceability the standard is BIPM's CCTF Common-View time
transfer, with a daily averaging window.

## Related pages

- [RINEX observation files](../formats/rinex-obs.md): the source of dual-frequency pseudoranges.
- [SP3 and clock products](../formats/sp3-clk.md): precise satellite clocks.
- [Single-point positioning](spp.md): the iono-free pseudorange combination.
