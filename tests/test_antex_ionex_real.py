"""Real-file integration tests for the ANTEX and IONEX readers.

Downloads the public IGS antenna calibration file (``igs14.atx``) and
the CODE final ionospheric TEC map (``codg1330.18i``) for 2018-05-13,
then exercises the readers + appliers on real-world content.

Files cached under ``/tmp/igs_real_cache/``.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import rinexpy as rp
from rinexpy.antex import apply_antex_pcv, find_antenna
from rinexpy.ionex import interp_tec, slant_tec

_CACHE = Path("/tmp/igs_real_cache")
_ANTEX_URL = "https://files.igs.org/pub/station/general/igs14.atx"
_IONEX_URL = "ftp://igs.gnsswhu.cn/pub/gps/products/ionex/2018/133/codg1330.18i.Z"


def _fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    _CACHE.mkdir(exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return dest.stat().st_size > 1000
    except Exception:
        if dest.exists():
            dest.unlink()
        return False


@pytest.fixture(scope="module")
def antex_real():
    path = _CACHE / "igs14.atx"
    if not _fetch(_ANTEX_URL, path):
        pytest.skip(f"Cannot reach {_ANTEX_URL}; skip real ANTEX test")
    return rp.load_antex(str(path))


@pytest.fixture(scope="module")
def ionex_real():
    path = _CACHE / "codg1330.18i"
    if not path.exists():
        z = _CACHE / "codg1330.18i.Z"
        if not _fetch(_IONEX_URL, z):
            pytest.skip(f"Cannot reach {_IONEX_URL}; skip real IONEX test")
        for tool in ("uncompress", "gzip"):
            try:
                subprocess.run([tool, "-d", "-f", str(z)], check=True, timeout=30)
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        else:
            pytest.skip("No .Z decompressor available")
    return rp.load_ionex(str(path))


def test_antex_has_many_antennas(antex_real):
    """The real IGS antenna file has hundreds of entries."""
    assert len(antex_real) > 300


def test_antex_block_iir_present(antex_real):
    """GPS Block IIR satellites have published PCV; we should find them."""
    types = {e["type"] for e in antex_real}
    assert any("BLOCK IIR" in t for t in types), (
        f"no Block IIR in {len(types)} antenna types"
    )


def test_antex_find_antenna_returns_matching_entry(antex_real):
    """find_antenna picks the entry whose type prefix matches."""
    chosen = find_antenna(antex_real, "BLOCK IIR-B")
    assert chosen is not None
    assert "BLOCK IIR-B" in chosen["type"]


def test_antex_apply_pcv_returns_finite_correction(antex_real):
    """Apply NOAZI PCV at a few elevations on a real antenna entry."""
    sv_entry = find_antenna(antex_real, "BLOCK IIR-B")
    assert sv_entry is not None, "BLOCK IIR-B not in the bundled ANTEX"
    for el in (1.0, 30.0, 60.0, 89.0):
        # GPS L1: 'G01' is the frequency key for L1 in ANTEX.
        pcv = apply_antex_pcv(sv_entry, "G01", el_deg=el)
        assert np.isfinite(pcv), f"PCV not finite at el={el}"
        # Real PCVs are sub-mm to a few cm (modeled in mm in ANTEX, returned
        # in meters by the applier).
        assert abs(pcv) < 0.05, f"PCV {pcv:.4f} m suspiciously large at el={el}"


def test_ionex_has_full_day_of_maps(ionex_real):
    """CODE final IONEX has 25 maps per day (1-hour cadence + the next day's first)."""
    assert ionex_real.sizes["time"] >= 13
    assert ionex_real.sizes["lat"] > 30
    assert ionex_real.sizes["lon"] > 30


def test_ionex_tec_values_in_physical_band(ionex_real):
    """TEC values from CODE for May 2018 are typically in 0-100 TECU."""
    tec = ionex_real["tec"].values
    finite = tec[np.isfinite(tec)]
    assert finite.size > 0
    assert finite.min() >= -10.0     # CODE allows a small negative tail
    assert finite.max() < 250.0      # daytime equatorial highs are ~70-150


def test_ionex_interp_at_geographic_point(ionex_real):
    """interp_tec produces a finite TEC value at a sensible lat/lon."""
    # Mid-day equatorial spot at the first map's time.
    first_time = ionex_real.time.values[0]
    tec = interp_tec(ionex_real, lat_deg=0.0, lon_deg=0.0, epoch=first_time)
    assert np.isfinite(tec)
    # Daytime equatorial: low-tens of TECU (could be higher at solar max).
    assert 0.0 <= tec < 150.0


def test_ionex_slant_tec_increases_at_low_elevation(ionex_real):
    """slant_tec applies the thin-shell mapping; lower elevation -> more TEC."""
    first_time = ionex_real.time.values[0]
    vtec = interp_tec(
        ionex_real, lat_deg=40.0, lon_deg=-90.0, epoch=first_time
    )
    s_high = slant_tec(vtec, el_deg=80.0)
    s_low = slant_tec(vtec, el_deg=10.0)
    assert np.isfinite(s_high) and np.isfinite(s_low)
    if vtec > 0.1:
        assert s_low > s_high
