# GNSS reflectometry

GNSS reflectometry (GNSS-R) is the practice of extracting environmental
parameters from GNSS signals that have bounced off a reflecting surface
(ocean, lake, ice, snow) before reaching the antenna. The signature is
visible in the antenna's signal-to-noise (SNR) trace as a slow oscillation
versus elevation; the oscillation frequency maps to the reflector height
through a simple two-ray model.

rinexpy implements the SNR-based altimetric retrieval from Larson (2008,
2013).

## The model

A GNSS antenna receives the direct line-of-sight signal plus a delayed
reflected copy. The two interfere at the antenna, producing an SNR
oscillation against `sin(elev)` with frequency

```
f = 2 H / lambda
```

where `H` is the reflector height above the antenna phase centre and
`lambda` is the GNSS carrier wavelength. Differentiating gives the
analyst's textbook recipe: detrend the SNR-versus-sin(el) series, run a
Lomb-Scargle periodogram, pick the peak frequency, and recover `H`.

## Detrending the SNR series

The function `detrend_snr` subtracts a polynomial fit in `sin(el)` from
the SNR trace. The default order 4 polynomial removes the dominant
elevation-dependent gain pattern without erasing the interferometric
oscillation.

```python
import numpy as np
from rinexpy.gnssr import detrend_snr

# snr_db is the SNR in dB-Hz from the RINEX observation file or receiver log.
# elevation_rad is the per-sample elevation angle in radians.

snr_db = np.array([...])
elevation_rad = np.radians(np.array([...]))

detrended = detrend_snr(snr_db, elevation_rad, order=4)
```

The detrended signal has zero mean and shows the interferometric
oscillation cleanly.

## Recovering the reflector height

The high-level function does the whole pipeline.

```python
from rinexpy.gnssr import snr_to_sea_height
from rinexpy.multifreq import LAMBDA_L1

out = snr_to_sea_height(
    snr_db,
    elevation_rad,
    wavelength_m=LAMBDA_L1,
    height_search_m=(0.5, 50.0),     # bounds on the periodogram search
    n_freqs=1024,                     # number of frequency bins
    detrend_order=4,
)
print("reflector height:", out["height_m"], "m")
print("peak power:      ", out["peak_power"])
print("freqs:           ", out["frequencies_per_sin_elev"])
print("power spectrum:  ", out["power"])
print("detrended SNR:   ", out["detrended"])
```

The returned dict carries:

| Key | Type | Meaning |
| --- | --- | --- |
| `height_m` | `float` | reflector height in metres |
| `peak_power` | `float` | normalised Lomb-Scargle peak power |
| `frequencies_per_sin_elev` | `ndarray` | searched frequency bins |
| `power` | `ndarray` | Lomb-Scargle power at each bin |
| `detrended` | `ndarray` | the input SNR with the polynomial trend removed |

The peak power is unitless; a clean ocean retrieval typically has
power > 0.3, while a noisy land retrieval is below 0.1.

## Choosing the inputs

GNSS-R retrievals work best on satellites whose elevation is sweeping
through a few-degree window (typically 5° to 25°). The SNR oscillation has
the highest amplitude there because the antenna gain favours the LOS over
the reflected signal at high elevations.

For a typical ocean retrieval:

- Pick a single satellite arc, ideally with at least 30 minutes of
  data sampling.
- Restrict to elevations between 5° and 25°.
- Use the GPS L1 SNR (`S1` for RINEX 2, `S1C` or `S1P` for RINEX 3).
- Use the L1 wavelength.

```python
import rinexpy as rp
import numpy as np

obs = rp.load("station.rnx")
sv = obs.sel(sv="G07")
mask = (sv.S1C > 0) & ... # apply elevation mask via geometry
snr = sv.S1C.values[mask]
```

## What can be measured

The Larson recipe gives reflector height. With additional processing the
same SNR trace can yield:

- **Significant wave height.** From the amplitude decay of the
  interferometric pattern as elevation drops.
- **Soil moisture.** Over land, the dielectric contrast between dry and
  wet soil changes the reflection coefficient, which changes the
  oscillation amplitude.
- **Sea ice / snow depth.** Same idea, with the reflector at the snow or
  ice surface rather than the bare ground.

rinexpy currently exposes the reflector-height retrieval. The other
products are downstream analyses on top of the same `power` spectrum.

## Performance

The Lomb-Scargle periodogram is the heavy step. With the default 1024
frequency bins and a 1000-sample arc, it takes a few tens of milliseconds.
For batch retrieval over a full day of data, parallelise across satellites.

## Related pages

- [RINEX observation files](../formats/rinex-obs.md): where the SNR series comes from.
- [QC and cycle slips](../quality/qc.md): SNR-based slip detection.
- [Time transfer](time-transfer.md): a sibling specialised positioning use.
