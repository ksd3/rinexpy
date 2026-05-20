# Kalman filters

`rinexpy` includes three closely-related extended Kalman filters for GNSS
positioning. All three are static-receiver designs (with optional kinematic
mode) that estimate the receiver state from iono-free pseudorange and
carrier-phase observations. The difference between them is the state
vector.

| Filter | State vector | Use case |
| --- | --- | --- |
| `StaticPPPFilter` (alias `GNSSFilter`) | position + clock + ambiguities | classic single-system PPP |
| `StaticPPPFilterZTD` | position + clock + ZWD + ambiguities | PPP with tropospheric ZWD as a measurand |
| `StaticPPPFilterMultiGNSS` | position + clock + ZWD + per-constellation ISBs + ambiguities | multi-constellation PPP |

Each filter is independently usable. The `ppp_solve` driver in
`rinexpy.ppp` wraps `StaticPPPFilter`; for ZWD and multi-constellation
workflows, instantiate the corresponding filter directly.

## StaticPPPFilter

The basic PPP filter. State is `[px, py, pz, c*dt_rx, N_1, ..., N_n_sv]`,
where the ambiguities are the float iono-free ambiguities in metres.

### Construction

```python
from rinexpy.kalman import StaticPPPFilter

ekf = StaticPPPFilter(
    n_sv=8,                          # number of tracked SVs
    initial_position=(x0, y0, z0),   # ECEF initial guess in metres
    sigma_code=1.0,                  # 1-sigma code noise (metres)
    sigma_phase=0.005,               # 1-sigma phase noise (metres)
    sigma_position_init=10.0,        # initial position uncertainty (metres)
    sigma_clock_init=300.0,          # initial clock uncertainty (metres)
    sigma_clock_rate_m=10.0,         # random-walk clock per sqrt(s)
    sigma_position_rate_m=0.0,       # 0 for static, > 0 for kinematic
    sigma_ambig_init_m=1000.0,       # initial ambiguity uncertainty (metres)
)
```

For a kinematic receiver, set `sigma_position_rate_m` to the expected
random-walk per `sqrt(s)`. A walking pedestrian carries roughly
0.5 m/sqrt(s); a car at constant heading is closer to 0.05 m/sqrt(s).

### Time and measurement update

```python
ekf.predict(dt)
ekf.update(sv_ecef, sat_clock_s, pr_if, phase_if, tropo_m=None)
```

`predict(dt)` advances the state by `dt` seconds. The position random walk
grows the position variance by `(sigma_position_rate_m**2 * dt)` per axis.
The clock random walk grows the clock variance by `(sigma_clock_rate_m**2
* dt)`.

`update(...)` applies one epoch's measurements:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `sv_ecef` | `(n_sv, 3)` | satellite ECEF at signal-emission time |
| `sat_clock_s` | `(n_sv,)` | per-SV clock bias in seconds (from CLK or SSR) |
| `pr_if` | `(n_sv,)` | iono-free pseudorange in metres |
| `phase_if` | `(n_sv,)` | iono-free carrier phase in metres |
| `tropo_m` | `(n_sv,)` or None | tropospheric correction per SV in metres |

The measurement model is

```
y_pr_i    = ||sv_i - rx|| + c*dt_rx + tropo_i
y_phase_i = ||sv_i - rx|| + c*dt_rx + tropo_i + N_i
```

The filter linearises around the current state and applies the standard
Kalman gain.

### Slip-aware update

```python
slipped = ekf.update_with_slip_check(
    sv_ecef, sat_clock_s,
    p1_m, p2_m, phi1_cycles, phi2_cycles,
    tropo_m=None,
    slip_threshold_cycles=2.0,
    f1=1575.42e6, f2=1227.60e6,
)
```

Computes the geometry-free phase combination per satellite, compares
against the previous epoch, and drops any satellite whose change exceeds
`slip_threshold_cycles`. Returns the indices of the slipped satellites so
the caller can log or trigger a re-bootstrap.

### Resetting ambiguities

When a cycle slip happens or an SV reacquires lock, you wipe its
ambiguity so the filter can re-converge.

```python
ekf.reset_ambiguity(sv_index=3)              # one SV
ekf.reset_ambiguities([0, 2, 5])             # batch
```

Each call resets the per-SV ambiguity to zero and inflates its variance to
`sigma_ambig_init_m**2`.

### Properties

```python
ekf.position                # tuple (px, py, pz) in metres
ekf.clock_bias_s            # float, seconds
ekf.ambiguities_m           # ndarray of iono-free ambiguities in metres
ekf.position_sigma          # tuple of per-axis 1-sigma in metres
```

## StaticPPPFilterZTD

Same shape as the basic filter, plus a ZWD state. The state vector is
`[px, py, pz, c*dt_rx, ZWD, N_1, ..., N_n_sv]`.

The wet zenith delay is modelled as a random walk with rate
`sigma_zwd_rate_m_per_sqrt_hr`. The measurement update needs a per-SV
wet mapping factor (typically VMF1).

### Construction

```python
from rinexpy.kalman_ztd import StaticPPPFilterZTD

ekf = StaticPPPFilterZTD(
    n_sv=8,
    initial_position=(x0, y0, z0),
    initial_zwd_m=0.1,                          # 10 cm starting guess
    sigma_zwd_init=0.5,                          # initial uncertainty
    sigma_zwd_rate_m_per_sqrt_hr=0.01,           # 1 cm per sqrt(hr)
    sigma_position_rate_m=0.0,                   # static
)
```

### Update

```python
ekf.predict(dt)
ekf.update(
    sv_ecef, sat_clock_s,
    pr_if, phase_if,
    wet_mapping,                  # (n_sv,) wet mapping factor per SV
    tropo_apriori_m=None,         # optional ZHD a priori per SV
)
```

The `wet_mapping` argument is the elevation-dependent wet mapping factor
from VMF1 or Niell. `tropo_apriori_m`, when supplied, is the hydrostatic
zenith delay times the hydrostatic mapping factor; the filter subtracts
it before estimating ZWD.

### Properties

```python
ekf.position
ekf.clock_bias_s
ekf.zwd_m
ekf.zwd_sigma_m
ekf.ambiguities_m
ekf.position_sigma
```

## StaticPPPFilterMultiGNSS

The multi-constellation filter. State vector is
`[px, py, pz, c*dt_rx, ZWD, ISB_1, ..., ISB_k, N_1, ..., N_n_sv]`.

The GPS clock bias is the reference; one ISB scalar is estimated per
non-GPS constellation present in the dataset.

### Construction

```python
from rinexpy.kalman_multignss import StaticPPPFilterMultiGNSS

constellations = ["G", "G", "G", "G", "G", "E", "E", "E", "E", "R", "R", "R"]
ekf = StaticPPPFilterMultiGNSS(
    n_sv=12,
    constellations=constellations,
    initial_position=(x0, y0, z0),
    initial_zwd_m=0.1,
    sigma_isb_init=100.0,
    sigma_isb_rate_m_per_sqrt_hr=0.0,    # ISBs are usually constant in a session
)
```

The `constellations` list must match the SV order in the measurements. For
each unique non-GPS letter, the filter adds one ISB state.

### Update

```python
ekf.predict(dt)
ekf.update(
    sv_ecef, sat_clock_s,
    pr_if, phase_if,
    wet_mapping,
    tropo_apriori_m=None,
)
```

### Properties

```python
ekf.position
ekf.clock_bias_s
ekf.zwd_m
ekf.isb_m("E")     # Galileo inter-system bias in metres
ekf.isb_m("R")     # GLONASS inter-system bias
ekf.ambiguities_m
ekf.position_sigma
```

## Picking the right filter

| Filter | Use when |
| --- | --- |
| `StaticPPPFilter` | single constellation, no ZTD estimate, short session |
| `StaticPPPFilterZTD` | long session, need ZWD as a measurand, GPS-only or fixed ISBs |
| `StaticPPPFilterMultiGNSS` | multi-constellation, long session, ISBs estimated |

The default `ppp_solve` driver uses `StaticPPPFilter` because that is what
fits the widest range of users without surprise. If you have a 24-hour
session and care about the troposphere or about multi-system bookkeeping,
spawn the filter directly.

## Tuning the noise

The four main sigmas are:

`sigma_code` — pseudorange 1-sigma in metres. A modern receiver tracking
in clear sky records around 30 cm of noise on the iono-free combination.
A poorly-sited or noisy receiver may be at 1 m or more.

`sigma_phase` — carrier-phase 1-sigma in metres. Typically 1-3 mm in clear
sky; the iono-free combination amplifies this by about 3.

`sigma_clock_rate_m` — random-walk clock noise per sqrt(s). For a
TCXO/OCXO this is 1-10 m/sqrt(s); for a Rb maser it is closer to 0.1.

`sigma_zwd_rate_m_per_sqrt_hr` — wet zenith delay random walk per
sqrt(hr). Typical values are 5-30 mm/sqrt(hr).

The filter's covariance matrix grows over time when `predict` runs without
a corresponding update; the standard Kalman drift.

## Process noise

The full process-noise matrix Q used by `predict(dt)` is:

```
Q_position = sigma_position_rate_m^2 * dt * eye(3)
Q_clock    = sigma_clock_rate_m^2 * dt
Q_zwd      = (sigma_zwd_rate_m_per_sqrt_hr / sqrt(3600))^2 * dt        # for ZTD filter
Q_isb      = (sigma_isb_rate_m_per_sqrt_hr / sqrt(3600))^2 * dt        # for multi-GNSS filter
Q_ambig    = 0                                                            # ambiguities are constants
```

The random-walk units feed in via Q. The ambiguity states have zero
process noise; their variance only changes through measurement updates,
or when explicitly reset.

## Related pages

- [Precise point positioning](ppp.md): the high-level driver.
- [LAMBDA and ambiguity resolution](lambda.md): integer fixing for PPP-AR.
- [Atmospheric models](../corrections/atmosphere.md): the troposphere chain feeding the filter.
- [SSR corrections](../corrections/ssr.md): real-time orbit/clock for PPP.
