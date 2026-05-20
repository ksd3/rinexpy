# DCB and code biases

Every GNSS satellite has hardware delays on its different signals.
Pseudorange measurements on different observation codes (C1C vs C1W, L1 vs
L2, etc.) carry per-satellite, per-code biases that the satellite itself
imposes. Without correction, the iono-free combination has a 1-2 metre
satellite-dependent bias that propagates straight into the position
solution.

There are three sources of differential and observable-specific code bias
data:

| Source | Format | Coverage |
| --- | --- | --- |
| GPS broadcast nav message | TGD field per SV per ephemeris record | L1 only |
| Daily IGS / MGEX SINEX-BIAS | per-(SV, code), per-day | post-2017 |
| Monthly CODE DCB | per-(SV, code), per-month | 1994 onwards (mostly pre-2017) |

rinexpy reads all three and exposes a unified application API.

## Broadcast TGD

The Timing Group Delay (TGD) field in the GPS NAV record is the broadcast
correction for the C1 (single-frequency) pseudorange.

```python
import rinexpy as rp
from rinexpy.positioning import tgd_from_nav, apply_tgd_correction
from datetime import datetime

nav = rp.load("tests/data/brdc2800.15n")
tgd_map = tgd_from_nav(nav, epoch=datetime(2015, 10, 7, 12))
# {'G01': 8.4e-9, 'G02': -1.2e-8, ...}

corrected = apply_tgd_correction(
    pseudoranges_m, sv_labels=["G01", "G02", "G05", ...], tgd_by_sv=tgd_map,
    gamma=1.0,
)
```

The TGD is the satellite hardware delay on the L1 P(Y) code, used as a
proxy for the L1 C/A bias. For L2 measurements, `gamma=(f1/f2)**2`. For
the iono-free combination, `gamma=0` (the bias cancels).

For BeiDou the broadcast TGD is in two parts. Pass `field="TGD1"` or
`field="TGD2"` to `tgd_from_nav`.

```python
nav_bds = rp.load("BRDC00WRD_S_20231560000_01D_MN.rnx")
tgd1 = tgd_from_nav(nav_bds, epoch, field="TGD1")     # for B1I observations
tgd2 = tgd_from_nav(nav_bds, epoch, field="TGD2")     # for B2I observations
```

For Galileo the equivalent is the BGD (`BGDe5a` for E5a, `BGDe5b` for E5b).

The TGD is accurate to roughly 30 cm. For higher accuracy use the SINEX-BIAS
DCB / OSB.

## SINEX-BIAS

SINEX-BIAS is the modern format. The IGS, MGEX (Multi-GNSS), and several
analysis centres (CAS, DLR, CODE, JPL) publish daily SINEX-BIAS files.

### Reading

```python
from rinexpy.dcb import read_bsx

records = read_bsx("CAS0MGXRAP_20231560000_01D_01D_DCB.BSX")
print(records[0])
```

Each record is a dict:

```
{
    "bias_type": "OSB",          # or "DSB"
    "prn":       "G05",          # or empty for receiver biases
    "station":   "",              # 9-char station, or empty for satellite biases
    "obs1":      "C1W",          # RINEX 3 observation code
    "obs2":      "",              # only for DSB
    "start":     datetime(2023, 6, 5, 0, 0),
    "end":       datetime(2023, 6, 6, 0, 0),
    "unit":      "ns",            # always converted to metres in `value`
    "value":     -2.6,            # bias in metres
}
```

Two record types appear:

**OSB (observable-specific signal bias).** A per-(SV, code) absolute bias.
For example, the OSB for `G05 C1W` is the per-satellite bias on the C1W
pseudorange.

**DSB (differential signal bias).** A bias between two codes for the same
SV (e.g., `G05 C1W - C2W`). DSBs are what the legacy CODE monthly format
publishes.

### Application

The `get_bias` lookup retrieves one bias.

```python
from rinexpy.dcb import get_bias

bias_m = get_bias(
    records,
    prn="G05",
    obs1="C1W",
    obs2="",                     # empty for OSB lookups
    epoch=datetime(2023, 6, 5, 12),
)
# Returns the bias in metres, or None if no record fits.
```

The high-level helper applies it directly to a pseudorange:

```python
from rinexpy.dcb import correct_pseudorange

corrected = correct_pseudorange(
    pseudorange_m=24123456.789,
    prn="G05",
    obs_code="C1W",
    records=records,
    epoch=datetime(2023, 6, 5, 12),
    station="",                  # add a 9-char station for receiver bias
)
```

For an array of pseudoranges, loop or use `np.vectorize`:

```python
corrected = np.array([
    correct_pseudorange(pr, prn=sv, obs_code="C1W",
                        records=records, epoch=epoch)
    for pr, sv in zip(pr_array, sv_labels)
])
```

## Auto-download

For non-real-time work, the date-dispatched auto-loader picks the right
mirror.

```python
from datetime import datetime
from rinexpy.dcb_download import auto_load_dcb, download_dcb

records = auto_load_dcb(datetime(2024, 4, 15))   # post-2017: CAS MGEX
records = auto_load_dcb(datetime(2010, 6, 15))   # pre-2017: CODE P1-P2

# Or fetch only and parse later:
path = download_dcb(datetime(2024, 4, 15), product="CAS")
```

`auto_load_dcb` dispatches by date:

- Pre-2017: AIUB FTP, monthly CODE P1-P2 file.
- 2017 onwards: IGS BKG public mirror, daily CAS Rapid MGEX SINEX-BIAS.

Files are cached under `~/.cache/rinexpy/dcb/`. The CDDIS source is wired
but requires a NASA Earthdata Login in `~/.netrc`. The AIUB and BKG
mirrors are anonymous HTTP.

The product selector:

| `product=` | Format | Notes |
| --- | --- | --- |
| `"CAS"` (default) | SINEX-BIAS | CAS Rapid MGEX, daily, all systems |
| `"DLR"` | SINEX-BIAS | DLR MGEX, daily, all systems |
| `"CODE"` | SINEX-BIAS | CODE Final, daily, GPS only |
| `"JPL"` | SINEX-BIAS | JPL MGEX, daily, all systems |

### Legacy CODE format

```python
from rinexpy.dcb import read_code_dcb

records = read_code_dcb("P1P22406.DCB", year=2024, month=6)
```

The reader translates the CODE-style codes (P1, P2, C1, C2) into RINEX 3
codes (C1W, C2W, C1C, C2C) so the returned records share the schema with
`read_bsx`. The `year` and `month` arguments are needed for record dating
because the legacy file does not encode the date itself.

```python
from rinexpy.dcb_download import download_legacy_code_dcb, load_monthly_code_dcb

path = download_legacy_code_dcb(datetime(2010, 6, 15), product="P1P2")
records = load_monthly_code_dcb(datetime(2010, 6, 15), product="P1P2")
```

The combined call (`auto_load_dcb`) dispatches to the legacy reader for
pre-2017 dates automatically.

## Use in positioning

The SPP solver takes a DCB record set directly:

```python
import rinexpy as rp

sol = rp.spp_solve(
    sv_ecef, pseudoranges_m,
    sv_labels=["G05", "G10", ...],
    dcb_records=records,
    dcb_obs_code="C1W",
    dcb_station="STAT00BRA",      # optional, for receiver biases
    dcb_epoch=datetime(2023, 6, 5, 12),
)
```

The PPP driver also takes DCBs through `ppp_solve(dcb_records=...)`.

For the broadcast TGD path (no SINEX-BIAS available), pass `tgd_map=` to
`spp_solve` instead:

```python
tgd_map = tgd_from_nav(nav, epoch)
sol = rp.spp_solve(sv_ecef, pseudoranges_m,
                   sv_labels=svs, tgd_map=tgd_map, tgd_gamma=1.0)
```

## Constants

```python
from rinexpy.dcb import C_M_PER_S
print(C_M_PER_S)        # 299_792_458.0
```

The bias values stored in records are already converted to metres, so you
do not multiply by `c` yourself.

## Picking what to use

For PPP-class work, always use SINEX-BIAS OSBs. The accuracy is roughly
3-5 cm per satellite per code.

For single-frequency SPP, the broadcast TGD is enough; the accuracy is
roughly 30 cm.

For pre-2017 archive processing, the CODE monthly DCBs are what is
available; the accuracy is roughly 10-15 cm.

For real-time work, RTCM3 SSR code-bias messages (1059, 1065, 1242,
1248, 1254, 1260) replace the file-based path. The
[SSR corrections page](ssr.md) describes the integration.

## Related pages

- [Atmosphere products](../formats/atmosphere-products.md): the DCB readers.
- [SSR corrections](ssr.md): real-time alternative.
- [Single-point positioning](../positioning/spp.md): SPP with DCB / TGD.
- [Precise point positioning](../positioning/ppp.md): PPP with DCB.
