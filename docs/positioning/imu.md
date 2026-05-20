# IMU and INS fusion

For platforms that need a position fix faster than GNSS alone can deliver,
or that need to bridge gaps in GNSS coverage (urban canyons, tunnels,
indoor stretches), an inertial measurement unit (IMU) fills in. The IMU
samples specific force (accelerometer) and angular rate (gyro) at hundreds
of Hertz; integrating these gives a high-rate position estimate that
drifts over time but is locally smooth. GNSS provides the absolute
reference that bounds the drift.

rinexpy ships two extended Kalman filters for GNSS/IMU fusion.

| Filter | Coupling | State | Use case |
| --- | --- | --- | --- |
| `LooseINS15` | loose | 15: position + velocity + attitude + accel bias + gyro bias | accepts pre-computed GNSS position fixes |
| `TightINS16` | tight | 16: above + receiver clock bias | accepts raw pseudoranges from any number of satellites |

The 15-state and 16-state design follows the standard error-state EKF
recipe with a Hamilton-style multiplicative attitude update (MEKF, Lefferts
et al. 1982).

## LooseINS15

The loose filter takes a pre-computed GNSS position fix as its measurement.
Use this when your GNSS receiver already computes a PVT solution and you
just want to fuse it with an IMU.

### Construction

```python
import numpy as np
from rinexpy.imu import LooseINS15, GRAVITY_M_PER_S2

ins = LooseINS15(
    position=np.array([x0, y0, z0]),       # ECEF metres
    velocity=np.array([0.0, 0.0, 0.0]),    # ECEF m/s
    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # identity (body == ECEF)
    sigma_accel=0.05,                       # 1-sigma accel noise, m/s^2
    sigma_gyro=0.005,                       # 1-sigma gyro noise, rad/s
    sigma_accel_bias_rw=1e-4,               # accel bias random walk
    sigma_gyro_bias_rw=1e-6,                # gyro bias random walk
    gravity=np.array([0.0, 0.0, -GRAVITY_M_PER_S2]),
)
```

The gravity vector is in the ECEF frame. The default points down at the
North Pole; for general latitudes, point it along the local "down".

### Predict

Per IMU sample (typically 100-1000 Hz):

```python
ins.predict(
    accel_body=np.array([a_x, a_y, a_z]),   # specific force in body frame
    gyro_body=np.array([w_x, w_y, w_z]),     # angular rate in body frame
    dt=0.01,                                  # seconds since last predict
)
```

The strapdown step rotates the body-frame specific force into ECEF, adds
gravity, integrates to get velocity and position, and updates the
attitude quaternion. The error-state covariance grows by the process
noise matrix times `dt`.

### Measurement update from GNSS

Per GNSS epoch (typically 1-10 Hz):

```python
ins.update_gnss(
    position_ecef=np.array([gnss_x, gnss_y, gnss_z]),
    R_gnss=1.0,                               # 1-sigma in metres, isotropic
)
```

`R_gnss` can be a scalar (isotropic noise), a length-3 array (per-axis
sigmas), or a 3x3 matrix (full covariance). The filter applies the
standard Kalman gain to update both the nominal state and the
error-state covariance.

### Reading the state

```python
print(ins.position, ins.velocity)
print(ins.quaternion)
print(ins.accel_bias, ins.gyro_bias)
print(np.sqrt(np.diag(ins.P)[:3]))    # per-axis position 1-sigma
```

## TightINS16

The tight filter accepts raw pseudoranges directly. Use this when your
receiver streams individual pseudorange measurements (e.g. via UBX
RXM-RAWX or RINEX OBS).

### Construction

```python
import numpy as np
from rinexpy.imu_tight import TightINS16

ins = TightINS16(
    position=np.array([x0, y0, z0]),
    velocity=np.array([0.0, 0.0, 0.0]),
    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    clock_bias_m=0.0,                          # c * dt_rx, metres
    sigma_clock_drift_m_per_s=1.0,             # 1 m/s random walk on c*dt_rx
)
```

The 16th state is the receiver clock bias times the speed of light, in
metres. This makes the units in the measurement update match the
pseudorange units.

### Predict

Same shape as `LooseINS15.predict`.

```python
ins.predict(accel_body, gyro_body, dt=0.01)
```

The clock-bias state is propagated as a random walk with rate
`sigma_clock_drift_m_per_s` per `sqrt(s)`.

### Pseudorange update

Per GNSS epoch:

```python
result = ins.update_pseudoranges(
    sv_ecef=sv_ecef_at_emission,             # (n_sv, 3)
    pseudoranges=pr_m,                        # (n_sv,) in metres
    sigma_pr=5.0,                             # 1-sigma noise, m, scalar or (n_sv,)
)
print(result["residuals"])                    # per-SV post-fit residual
print(result["n_obs"])                        # number of observations used
```

The measurement model per satellite is

```
PR_i = ||sv_i - rx|| + c*dt_rx + noise
```

Same as the SPP linearisation, applied through the EKF update. The
benefit over loose coupling is that even one or two satellites can update
the filter (whereas loose coupling needs four for a position fix).

## Quaternion helpers

For tests and custom workflows, the quaternion utilities are exposed.

```python
from rinexpy.imu import (
    quat_normalize,
    quat_to_matrix,
    quat_mul,
    quat_from_axis_angle,
)

q = np.array([1.0, 0.0, 0.0, 0.0])
print(quat_to_matrix(q))                # 3x3 identity

q_rot = quat_from_axis_angle(np.array([0.0, 0.0, 0.1]))     # 0.1 rad about z
q_combined = quat_mul(q, q_rot)
q_normalised = quat_normalize(q_combined)
```

Quaternions are Hamilton-style 4-vectors `[w, x, y, z]`. Rotation
matrices are 3x3 ECEF transforms.

## Initialising the attitude

The hard part of an INS is initialising the attitude. The two standard
recipes:

**Coarse alignment.** With the platform stationary, average the gyro
output over several seconds to estimate the gyro bias, then use the
gravity vector and the Earth-rate vector (from the IMU outputs) to solve
for the body-to-ECEF rotation. Works for tactical-grade IMUs (gyro bias
~10°/hr).

**Fine alignment.** Run the filter through a calibration trajectory
(figure-eight) for a few minutes. The filter observability improves and
the attitude converges. Required for consumer-grade IMUs.

rinexpy does not bundle initialisation helpers. Pass in your best estimate
of the attitude and let the filter converge through the trajectory.

## Tuning the noise

The dominant tuning parameters are the IMU noise levels.

`sigma_accel` — accelerometer white noise in m/s². Consumer IMUs are
around 0.1; tactical IMUs are around 0.005; navigation grade is 0.001.

`sigma_gyro` — gyro white noise in rad/s. Consumer IMUs are around 0.01;
tactical IMUs are around 0.0001; navigation grade is 0.00001.

`sigma_accel_bias_rw` — accel bias random walk in m/s² per sqrt(s).
Typically 1e-4 to 1e-5.

`sigma_gyro_bias_rw` — gyro bias random walk in rad/s per sqrt(s).
Typically 1e-6 to 1e-7.

The right values depend on the IMU datasheet's noise spectral density.
For consumer IMUs (e.g. ICM-20948), the values above are a sensible
starting point. For tactical or navigation-grade IMUs (KVH, Honeywell,
Northrop Grumman) the numbers can be ten or a hundred times smaller.

## When to use which

`LooseINS15` is the right choice when your GNSS receiver is a black-box
producing PVT fixes. It also avoids ambiguity bookkeeping; you do not need
to maintain a state per satellite, only per IMU sample.

`TightINS16` is the right choice when you have raw pseudoranges and the
geometry is sometimes weak (urban canyons, fewer than 4 satellites).
Tight coupling lets even 1-2 satellites bound the drift, which can mean
the difference between a working fix and a lost lock.

Neither filter handles carrier phase. For carrier-phase-tight coupling
you would extend the state with a per-satellite ambiguity term (see the
PPP filters). The full carrier-phase-tight design is on the roadmap.

## Output examples

### A static recording

```python
# 1 minute of static IMU + GNSS at 100 Hz IMU, 1 Hz GNSS
for i in range(60 * 100):
    ins.predict(accel_body=imu_samples[i, :3], gyro_body=imu_samples[i, 3:6], dt=0.01)
    if i % 100 == 0:                              # GNSS epoch
        ins.update_gnss(position_ecef=gnss_fixes[i // 100], R_gnss=2.0)

print("final position:", ins.position)
print("position sigma:", np.sqrt(np.diag(ins.P)[:3]), "metres")
```

### A short GNSS outage

```python
gnss_available = True
for i in range(N):
    ins.predict(accel_body, gyro_body, dt=0.01)
    if gnss_available and gnss_present(i):
        ins.update_gnss(position_ecef=gnss[i], R_gnss=2.0)

# After 30 s of GNSS outage with tactical-grade IMU,
# horizontal position has drifted ~5 m, velocity ~0.5 m/s.
```

## Related pages

- [Single-point positioning](spp.md): the source of the GNSS fix for the loose filter.
- [Kalman filters](kalman.md): the underlying EKF machinery.
