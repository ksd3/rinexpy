# Atmospheric models

GNSS signals propagate through the Earth's neutral atmosphere (the
troposphere, mostly nitrogen and water vapour) and through the upper
atmosphere (the ionosphere, mostly free electrons). Both delay the signal.
The neutral delay is non-dispersive and adds 2-10 metres at zenith. The
ionospheric delay is dispersive (frequency-dependent) and ranges from
1 metre at quiet times to 50 metres during a solar storm.

rinexpy ships the standard four-model family for the troposphere and
three independent tools for the ionosphere.

## Troposphere

### Saastamoinen zenith delay

The Saastamoinen formula is a closed-form estimate of the zenith
hydrostatic delay (and a smaller wet term) from surface pressure,
temperature, and water-vapour pressure.

```python
from rinexpy.geodesy import saastamoinen, standard_atmosphere

# Without met data, the function picks the ICAO standard atmosphere
# at the receiver altitude:
slant_m = saastamoinen(el_deg=15.0, altitude_m=100.0)
print(f"slant delay at 15 deg: {slant_m:.3f} m")
# ~ 9.0 m, dominated by the hydrostatic part

# With real met data:
slant_m = saastamoinen(
    el_deg=15.0,
    altitude_m=100.0,
    pressure_hpa=1013.25,
    temperature_k=288.15,
    humidity_e_hpa=11.7,        # water vapour partial pressure
)
print(f"slant delay at 15 deg (real met): {slant_m:.3f} m")

# Standard atmosphere at altitude:
T_k, P_hpa, e_hpa = standard_atmosphere(altitude_m=100.0)
```

The function returns the slant delay (in metres). For the zenith delay
alone, pass `el_deg=90.0`.

Accuracy is roughly 1 cm at elevations above 15° when fed real
meteorological data, or roughly 5-10 cm with the standard atmosphere
fallback. For PPP-class work, use the GPT2w grid plus VMF1 mapping.

### Niell mapping function

The Niell 1996 mapping function (NMF) extends a zenith delay to an
elevation-dependent slant delay using the hydrostatic and wet mappings.
It is widely used in geodetic-quality processing.

```python
from rinexpy.geodesy import niell_mapping

m_h, m_w = niell_mapping(
    el_deg=15.0,
    lat_deg=40.0,
    altitude_m=100.0,
    doy=74,             # day of year, e.g. 74 = March 14
)
print(f"hydrostatic mapping: {m_h:.3f}")
print(f"wet mapping:          {m_w:.3f}")
```

At zenith both `m_h` and `m_w` are 1. They grow rapidly with decreasing
elevation, reaching about 4 at el=15° and about 10 at el=5°.

### VMF1 mapping function

The Vienna Mapping Function 1 (Böhm et al., 2006) is the IGS-recommended
mapping function. It needs site-specific coefficients `a_h` and `a_w`
from a grid evaluation.

```python
from rinexpy.gpt2w import load_gpt2w_grid, gpt2w
from rinexpy.geodesy import vmf1
from datetime import datetime

grid = load_gpt2w_grid("/path/to/gpt2_5w.grd")
met = gpt2w(grid, lat_deg=40.0, lon_deg=-3.0,
            epoch=datetime(2024, 3, 14), altitude_m=100.0)

m_h, m_w = vmf1(
    a_h=met["a_h"],
    a_w=met["a_w"],
    el_deg=15.0,
    lat_deg=40.0,
    altitude_m=100.0,
    doy=74,
)
```

VMF1 is more accurate than Niell, especially at low elevations and in
non-standard atmospheres (polar regions, monsoonal weather). The PPP
driver uses VMF1 when you pass `gpt2w_grid=` and falls back to NMF
otherwise.

### GPT2w empirical met

The GPT2w model is an empirical surface meteorology grid published by the
Vienna group. Given a `(lat, lon, day-of-year, altitude)` query, it
returns:

| Field | Units | Notes |
| --- | --- | --- |
| `pressure_hpa` | hPa | dry surface pressure |
| `temperature_k` | K | dry surface temperature |
| `e_hpa` | hPa | water vapour partial pressure |
| `a_h` | dimensionless | VMF1 hydrostatic coefficient |
| `a_w` | dimensionless | VMF1 wet coefficient |
| `T_lapse` | K/km | tropospheric lapse rate |
| `undulation_m` | metres | geoid undulation above WGS-84 |

The grid is roughly 2 MB and lives at the
[Vienna VMF Data Server](https://vmf.geo.tuwien.ac.at/codes/). rinexpy
does not ship the grid; the user downloads it once.

```python
from rinexpy.gpt2w import load_gpt2w_grid, gpt2w
from datetime import datetime

grid = load_gpt2w_grid("/path/to/gpt2_5w.grd")
met = gpt2w(grid,
            lat_deg=40.0,
            lon_deg=-3.0,
            epoch=datetime(2024, 3, 14),
            altitude_m=100.0)
```

### Putting it together

The classic PPP tropospheric chain is:

1. Evaluate GPT2w at the receiver position to get `(P, T, e, a_h, a_w)`.
2. Compute the zenith hydrostatic delay with Saastamoinen.
3. Compute the wet mapping factor with VMF1.
4. The total slant delay is `ZHD * m_h + ZWD * m_w`, where ZWD is either
   estimated by the Kalman filter or supplied externally.

```python
from rinexpy.gpt2w import load_gpt2w_grid, gpt2w
from rinexpy.geodesy import saastamoinen, vmf1

grid = load_gpt2w_grid("/path/to/gpt2_5w.grd")
met = gpt2w(grid, lat_deg=40.0, lon_deg=-3.0,
            epoch=datetime(2024, 3, 14), altitude_m=100.0)

zhd_m = saastamoinen(
    el_deg=90.0, altitude_m=100.0,
    pressure_hpa=met["pressure_hpa"],
    temperature_k=met["temperature_k"],
    humidity_e_hpa=met["e_hpa"],
)
m_h, m_w = vmf1(met["a_h"], met["a_w"],
                el_deg=15.0, lat_deg=40.0, altitude_m=100.0, doy=74)

slant_dry = zhd_m * m_h
# slant_wet from the filter ZWD state, or skip for SPP-class accuracy
```

For SPP this is overkill; `saastamoinen(el_deg, altitude_m)` with
standard-atmosphere defaults is good enough. For PPP and geodesy the full
chain matters.

## Ionosphere

### Klobuchar broadcast

The 8-coefficient Klobuchar model is what the GPS broadcast nav message
publishes. It is the model for L1 single-frequency receivers that have no
better information.

```python
from rinexpy.geodesy import klobuchar
from datetime import datetime
import rinexpy as rp

# Load broadcast NAV; alpha and beta come from the header.
nav = rp.load("tests/data/brdc2800.15n")
hdr = rp.rinexheader("tests/data/brdc2800.15n")
alpha = hdr.get("ION ALPHA")   # 8 coefficients in 2 sets of 4
beta  = hdr.get("ION BETA")

if alpha and beta:
    delay_m = klobuchar(
        alpha=alpha, beta=beta,
        rx_lla=(40.0, -3.0, 100.0),
        sv_az_deg=180.0,
        sv_el_deg=30.0,
        gps_sec=43200.0,
    )
    print(f"Klobuchar L1 delay: {delay_m:.2f} m")
```

The model evaluates the iono delay at the satellite's pierce point in a
350 km thin-shell ionosphere. The output is the slant delay on the L1
frequency, in metres. For L2 multiply by `(f1/f2)**2`.

Accuracy is roughly 50% of the actual ionospheric delay: typically 1-2 m
at quiet times, 5+ m during solar storms.

### IONEX maps

For an order of magnitude more accuracy than Klobuchar, the IGS publishes
daily IONEX TEC maps with hourly or two-hourly cadence on a global grid.

```python
from rinexpy.ionex import load_ionex, interp_tec, slant_tec
from datetime import datetime

ionex = load_ionex("/path/to/igsg0860.24i")

vtec_tecu = interp_tec(
    ionex,
    lat_deg=40.0, lon_deg=-3.0,
    epoch=datetime(2024, 3, 14, 12, 0, 0),
)
stec_tecu = slant_tec(vtec_tecu, el_deg=30.0)
delay_l1_m = stec_tecu * 0.16          # 0.16 m / TECU on L1
```

The slant TEC mapping uses a 350 km thin shell by default. The IGS final
IONEX products are accurate to roughly 2 TECU (~30 cm on L1).

### Iono-free combination

For dual-frequency receivers the iono-free combination removes the
first-order ionospheric delay analytically. This is the foundation of PPP
and geodetic-class processing.

```python
from rinexpy.positioning import iono_free_pseudorange, iono_free_phase

p_if = iono_free_pseudorange(p1_m, p2_m)
l_if = iono_free_phase(l1_m, l2_m)
```

The combination amplifies the input noise by roughly 3x. The
second-order ionospheric delay (which the combination does NOT remove)
is below the centimetre level at GPS frequencies and is typically
ignored.

For Galileo or other constellations with non-default frequencies, pass
the band frequencies explicitly:

```python
p_if = iono_free_pseudorange(p1_m, p2_m, f1=1.57542e9, f2=1.17645e9)
```

The `multifreq` module has wavelength constants for L1, L2, L5, and the
wide-lane / narrow-lane / extra-wide-lane combinations. See
[LAMBDA and ambiguity resolution](../positioning/lambda.md).

## Picking a model

| Application | Troposphere | Ionosphere |
| --- | --- | --- |
| Casual SPP | Saastamoinen std-atm | Klobuchar |
| Single-freq with NAV | Saastamoinen std-atm | Klobuchar from NAV |
| Single-freq with IONEX | Saastamoinen std-atm | IONEX |
| Geodesy or PPP | GPT2w + VMF1 | Iono-free |
| Real-time PPP | GPT2w + VMF1 | Iono-free + SSR iono |

The PPP driver in `rinexpy.ppp.ppp_solve` follows the geodesy row by
default. If you pass `gpt2w_grid=`, the driver uses the VMF1 chain;
otherwise it falls back to Saastamoinen + Niell with standard atmosphere.

## Related pages

- [Atmosphere products](../formats/atmosphere-products.md): the IONEX, ANTEX, GPT2w, MET, EOP readers.
- [DCB and code biases](dcb.md): the matching hardware-bias chain.
- [Precise point positioning](../positioning/ppp.md): the high-level driver.
- [Single-point positioning](../positioning/spp.md): where Klobuchar is applied.
- [RINEX navigation files](../formats/rinex-nav.md): broadcast iono coefficients.
