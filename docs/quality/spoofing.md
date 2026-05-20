# Spoofing and jamming heuristics

GNSS signals are weak by the time they reach the receiver: about
-130 dBm at the antenna. A nearby transmitter at the same frequency
overwhelms the real signal. Spoofing in particular is the deliberate
broadcast of fake GNSS signals, designed to make the receiver believe it
is somewhere it is not.

`rinexpy`'s spoofing module is a set of four short heuristics. None of them
constitute a definitive spoofing detector; together they raise enough
suspicion to demand a closer look. The module is in
`rinexpy.spoofing`.

| Function | What | When to use |
| --- | --- | --- |
| `check_snr_uniformity` | suspicious SNR uniformity across SVs | spoofer transmitter is co-located |
| `check_position_jumps` | impossible position changes between epochs | spoofer is dragging the position |
| `check_clock_drift` | impossible clock drift or jump | spoofer time alignment is off |
| `check_agc` | sudden AGC change | jammer just turned on |

## SNR uniformity

A real GNSS receiver sees a range of SNR values across satellites
because the satellites are at different elevations, the antenna pattern
favours some directions, and atmospheric and multipath losses vary. A
single-source spoofer broadcasts all the satellites at roughly the same
power, so the recorded SNR distribution across SVs is suspiciously flat.

```python
from rinexpy.spoofing import check_snr_uniformity
import numpy as np

# snr is a (n_epoch, n_sv) array of SNR values in dB-Hz.
# elevation is the matching (n_epoch, n_sv) array of elevations in deg.
snr = np.array([[45.0, 44.8, 45.1, 44.9, 45.0]])
el  = np.array([[15.0, 35.0, 55.0, 75.0, 10.0]])

out = check_snr_uniformity(snr, el, sigma_threshold=1.5)
print(out)
# {'flagged_epochs': array([False]), 'residual_std': array([...])}
```

The function returns:

| Key | Meaning |
| --- | --- |
| `flagged_epochs` | boolean array per epoch, True when uniformity test fires |
| `residual_std` | observed SNR residual standard deviation per epoch |

The residual is the deviation of the per-SV SNR from a linear fit
against `sin(el)`. A real receiver shows a strong elevation dependence
(roughly 8 dB from zenith to 10° elevation); a spoofer that transmits
all satellites at the same power lands well below the expected
residual.

## Position jumps

A real receiver moves smoothly. A spoofer dragging the position around
shows up as impossibly fast jumps between epochs.

```python
from rinexpy.spoofing import check_position_jumps

positions_ecef = np.array([
    [4789028.0, 176610.0, 4195017.0],
    [4789030.0, 176611.0, 4195017.0],     # 2.2 m, fine
    [4801028.0, 176610.0, 4195017.0],     # 12 km jump in one second!
])
times_s = np.array([0.0, 1.0, 2.0])

out = check_position_jumps(
    positions_ecef, times_s,
    max_speed_m_per_s=300.0,             # 1000 km/h, faster than a jet
)
print(out)
```

The return dict has two keys: `flagged_epochs` (boolean array per
epoch) and `speed_mps` (per-epoch instantaneous speed in metres per
second). Any epoch where `speed_mps > max_speed_m_per_s` is flagged.

The default 300 m/s is for a high-speed vehicle. For a stationary
receiver, drop it to 5 m/s; the test then catches small jumps.

## Clock drift

A real receiver clock drifts smoothly at a rate determined by its
oscillator (a TCXO drifts at about 1 µs/s; an OCXO at about 0.1 µs/s; a
caesium beam at about 1e-12 s/s). A spoofer's induced clock often jumps,
since the spoofer has to roll back its transmitted time to land the
receiver at the spoofed location.

```python
from rinexpy.spoofing import check_clock_drift

clock_bias_s = np.array([1.0e-7, 1.1e-7, 1.05e-7, 5.2e-6])  # last is suspect
times_s = np.array([0.0, 1.0, 2.0, 3.0])

out = check_clock_drift(
    clock_bias_s, times_s,
    max_drift_rate=1e-6,       # 1 µs/s
    max_jump_s=1e-5,            # 10 µs
)
print(out)
```

The return dict has:

- `flagged_epochs`: boolean array per epoch.
- `drift_rate`: per-epoch instantaneous drift rate (s/s).
- `jumps`: per-epoch inter-epoch clock-bias change (s).

An epoch is flagged if `drift_rate` exceeds `max_drift_rate` or `jumps`
exceeds `max_jump_s` in absolute value.

## AGC

The automatic gain control (AGC) on a GNSS receiver is fixed when the
receiver is in clear sky with no interference. When a jammer turns on, the
AGC drops sharply to avoid saturation. The AGC level by itself does not
prove jamming, but a sudden change is a strong indicator.

```python
from rinexpy.spoofing import check_agc

agc_db = np.array([60, 60.1, 59.9, 60.0, 40.0, 39.8])   # last 2 are jammed
out = check_agc(agc_db, max_jump_db=6.0)
print(out)
# {'flagged_epochs': array([False, False, False, False, True, False]),
#  'jumps':          array([0.0, 0.1, -0.2, 0.1, -20.0, -0.2])}
```

The return dict has `flagged_epochs` (boolean per epoch) and `jumps`
(per-epoch inter-epoch AGC change in dB). Any epoch where
`|jump|` exceeds `max_jump_db` is flagged.

## What these heuristics are good for

The heuristics are designed for batch QC, not for live arming an alert.
The typical workflow:

1. Run a real position solver on the receiver log.
2. Apply the four heuristics to the per-epoch outputs.
3. If any heuristic fires in more than, say, 5% of epochs, flag the log
 for manual review.
4. The actual decision to reject the log is human-in-the-loop.

For a real anti-spoofing deployment, the standard answer is multi-sensor
fusion (inertial, optical, cellular) with formal hypothesis testing.
GNSS-only heuristics catch the dumb spoofers, not the sophisticated ones.

## What these heuristics are not

These are not cryptographic anti-spoofing. They do not validate the GPS
authenticated signal (Chimera, M-code) or Galileo OS-NMA. The
authenticated-signal verifiers live below the RINEX layer; `rinexpy` does
not implement them. For OS-NMA, see the
[Galileo OS-NMA test bench](https://www.gsc-europa.eu/galileo/services/os-nma).

## Related pages

- [QC and cycle slips](qc.md): the data-quality side of the integrity story.
- [Single-point positioning](../positioning/spp.md): where these heuristics typically apply.
- [Receiver binary formats](../formats/receiver-binary.md): u-blox UBX has explicit AGC fields.
