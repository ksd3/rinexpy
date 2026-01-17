"""RINEX 3 NAV writer — round-trip a fitted broadcast ephemeris back
to RINEX 3 NAV format. Companion to :mod:`rinexpy.nav3`.

Scope: GPS (G), Galileo (E), BeiDou (C), QZSS (J) Keplerian
ephemerides. GLONASS (R) and SBAS (S) use a different layout (ECEF
position / velocity / acceleration) and are emitted via a parallel
path below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _format_d_exp(value: float) -> str:
    """Format a float in RINEX D-style scientific notation, e.g.
    ``"-1.234567890123D+12"`` (19 chars wide)."""
    if not np.isfinite(value):
        return " 0.000000000000D+00"
    sign = "-" if value < 0 else " "
    a = abs(value)
    if a == 0.0:
        return f"{sign}0.000000000000D+00"
    exp = int(np.floor(np.log10(a)))
    mantissa = a / (10.0 ** exp)
    if mantissa >= 10.0:
        mantissa /= 10.0
        exp += 1
    elif mantissa < 1.0:
        mantissa *= 10.0
        exp -= 1
    return f"{sign}{mantissa:.12f}D{exp:+03d}"


def _write_keplerian_block(
    f, sv: str, sv_ds: xr.Dataset, time_idx: int
) -> None:
    """Write one 8-line Keplerian record for a single SV at one time."""
    t = sv_ds.time.values[time_idx]
    t_py = t.astype("datetime64[us]").tolist()
    # Line 1: PRN + epoch + clock polynomial.
    af0 = float(sv_ds["SVclockBias"].values[time_idx]) if "SVclockBias" in sv_ds else 0.0
    af1 = float(sv_ds["SVclockDrift"].values[time_idx]) if "SVclockDrift" in sv_ds else 0.0
    af2 = (
        float(sv_ds["SVclockDriftRate"].values[time_idx])
        if "SVclockDriftRate" in sv_ds else 0.0
    )
    f.write(
        f"{sv:3s} {t_py.year:4d} {t_py.month:2d} {t_py.day:2d} "
        f"{t_py.hour:2d} {t_py.minute:2d} {t_py.second:2d}"
        f"{_format_d_exp(af0)}{_format_d_exp(af1)}{_format_d_exp(af2)}\n"
    )
    # Helper: pull field, default zero.
    def _g(name):
        return (
            float(sv_ds[name].values[time_idx])
            if name in sv_ds else 0.0
        )
    # Each subsequent broadcast line is 4 fields × 19 chars per
    # line, right-justified. The 4-space indent corresponds to the
    # SV id column.
    pad = "    "
    # Line 2: IODE, Crs, DeltaN, M0.
    f.write(
        pad
        + _format_d_exp(_g("IODE"))
        + _format_d_exp(_g("Crs"))
        + _format_d_exp(_g("DeltaN"))
        + _format_d_exp(_g("M0"))
        + "\n"
    )
    # Line 3: Cuc, e, Cus, sqrtA.
    f.write(
        pad
        + _format_d_exp(_g("Cuc"))
        + _format_d_exp(_g("Eccentricity"))
        + _format_d_exp(_g("Cus"))
        + _format_d_exp(_g("sqrtA"))
        + "\n"
    )
    # Line 4: Toe, Cic, Omega0, Cis.
    f.write(
        pad
        + _format_d_exp(_g("Toe"))
        + _format_d_exp(_g("Cic"))
        + _format_d_exp(_g("Omega0"))
        + _format_d_exp(_g("Cis"))
        + "\n"
    )
    # Line 5: i0, Crc, omega, OmegaDot.
    f.write(
        pad
        + _format_d_exp(_g("Io"))
        + _format_d_exp(_g("Crc"))
        + _format_d_exp(_g("omega"))
        + _format_d_exp(_g("OmegaDot"))
        + "\n"
    )
    # Line 6: IDOT, codes, GPS Week, L2P data flag.
    f.write(
        pad
        + _format_d_exp(_g("IDOT"))
        + _format_d_exp(_g("CodesL2"))
        + _format_d_exp(_g("GPSWeek"))
        + _format_d_exp(_g("L2Pflag"))
        + "\n"
    )
    # Line 7: SV accuracy, SV health, TGD, IODC.
    f.write(
        pad
        + _format_d_exp(_g("SVacc"))
        + _format_d_exp(_g("health"))
        + _format_d_exp(_g("TGD"))
        + _format_d_exp(_g("IODC"))
        + "\n"
    )
    # Line 8: transmission time, fit interval, spare, spare.
    f.write(
        pad
        + _format_d_exp(_g("TransTime"))
        + _format_d_exp(_g("FitIntvl"))
        + _format_d_exp(0.0)
        + _format_d_exp(0.0)
        + "\n"
    )


def write_nav3(ds: xr.Dataset, outpath) -> Path:
    """Write a RINEX 3 NAV dataset to disk.

    Companion to :func:`rinexpy.nav3.rinexnav3`. The dataset shape
    expected matches what the reader emits: a ``time`` × ``sv``
    structure with each broadcast field as its own data variable.

    Scope today: Keplerian ephemerides (GPS / Galileo / BeiDou / QZSS).
    GLONASS and SBAS round-tripping is a TODO; their records use the
    ECEF position+velocity+acceleration form and need a separate
    write path.
    """
    out = Path(outpath).expanduser()
    sv_letter = (ds.attrs.get("svtype") or "M")
    with out.open("w") as f:
        # Minimal header.
        f.write(
            "     3.04           NAVIGATION DATA     M (MIXED)           "
            "RINEX VERSION / TYPE\n"
        )
        f.write(
            "rinexpy.nav_writer                                          "
            "PGM / RUN BY / DATE\n"
        )
        f.write(
            "                                                            "
            "END OF HEADER\n"
        )
        for sv in ds.sv.values:
            sv_ds = ds.sel(sv=sv)
            n_t = sv_ds.time.size
            for ti in range(n_t):
                # Skip rows where the clock-bias field is NaN -- those
                # are sparse-by-sv slots from the reader's fill pass.
                if "SVclockBias" in sv_ds and np.isnan(sv_ds["SVclockBias"].values[ti]):
                    continue
                _write_keplerian_block(f, str(sv), sv_ds, ti)
    return out


__all__ = ["write_nav3"]
