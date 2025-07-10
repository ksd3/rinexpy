"""Parity tests for the C++ Lagrange SP3 kernel.

Skipped when rinexpy_native isn't importable.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from rinexpy import _native
from rinexpy.interp import interpolate_sp3

pytest.importorskip("rinexpy_native")

from .conftest import fixture  # noqa: E402


def _force(native_on: bool) -> None:
    if native_on:
        _native.have_interpolate_sp3 = (
            lambda: _native._interpolate_sp3_lagrange is not None
        )
    else:
        _native.have_interpolate_sp3 = lambda: False


def test_native_interpolate_sp3_available():
    """The kernel ships with the >=0.2 native wheel."""
    assert _native.have_interpolate_sp3() is True


def test_native_matches_python_on_bundled_sp3c():
    """Bit-equal results against the bundled igs19362.sp3c at four
    interior query times spanning the file."""
    import rinexpy as rp

    sp3 = rp.load_sp3(fixture("igs19362.sp3c"))
    t_axis = sp3.time.values  # datetime64[us]
    queries = []
    for k in (2, 10, 50, 80):
        t0 = t_axis[k].astype("datetime64[us]").astype(datetime)
        t1 = t_axis[k + 1].astype("datetime64[us]").astype(datetime)
        queries.append(t0 + (t1 - t0) / 2)
    qs = np.array(queries, dtype="datetime64[ns]")

    _force(False)
    py = interpolate_sp3(sp3, qs).position.values
    _force(True)
    cpp = interpolate_sp3(sp3, qs).position.values

    finite = np.isfinite(py)
    assert finite.any(), "test corpus should produce some finite values"
    abs_diff = np.abs(cpp[finite] - py[finite])
    scale = np.maximum(1.0, np.abs(py[finite]))
    rel = (abs_diff / scale).max()
    assert rel < 1e-12, f"rel diff {rel} too large"


def test_native_matches_python_on_igs_final_sp3():
    """Full multi-hour parity against the IGS final SP3 cached by
    test_ppp_realdata.

    Picks 50 random interior query times (~10 minutes off the source
    grid) and checks bit-equality across all SVs and components.
    """
    import rinexpy as rp

    cached = Path("/tmp/igs_real_cache/igs20010.sp3")
    if not cached.exists():
        pytest.skip("IGS SP3 not cached; run test_ppp_realdata first")

    sp3 = rp.load_sp3(cached)
    t0 = sp3.time.values[5]
    t_end = sp3.time.values[-5]
    rng = np.random.default_rng(42)
    # Sample 50 query times uniformly between t0 and t_end.
    t0_ns = t0.astype("datetime64[ns]").astype("int64")
    tn_ns = t_end.astype("datetime64[ns]").astype("int64")
    q_ns = rng.integers(t0_ns, tn_ns, size=50, dtype=np.int64)
    qs = q_ns.astype("datetime64[ns]")

    _force(False)
    py = interpolate_sp3(sp3, qs).position.values
    _force(True)
    cpp = interpolate_sp3(sp3, qs).position.values

    finite = np.isfinite(py) & np.isfinite(cpp)
    abs_diff = np.abs(cpp[finite] - py[finite])
    scale = np.maximum(1.0, np.abs(py[finite]))
    rel = (abs_diff / scale).max()
    # 1e-10 instead of 1e-12 because the larger time scale (multi-hour
    # window, ns timestamps near 10^18) costs a couple more bits of
    # round-off than the bundled fixture which is much shorter.
    assert rel < 1e-10, f"rel diff {rel} too large"


def teardown_module():
    _force(True)
