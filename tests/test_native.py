"""Tests for the optional rinexpy-native C++ extension.

These tests are skipped if rinexpy-native is not installed (the
package is shipped as a separate wheel and intentionally optional).
"""

from __future__ import annotations

import os

import pytest

from rinexpy import _native
from rinexpy.obs3 import rinexobs3

from .conftest import fixture

pytest.importorskip("rinexpy_native")


def test_native_is_available():
    assert _native.is_available() is True


def test_native_env_var_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RINEXPY_USE_NATIVE", raising=False)
    assert _native.is_enabled() is False


def test_native_env_var_enables(monkeypatch):
    monkeypatch.setenv("RINEXPY_USE_NATIVE", "1")
    assert _native.is_enabled() is True


def test_native_output_matches_python_small():
    """Native and pure-Python decoders agree on a small mixed-system file."""
    truth = rinexobs3(fixture("obs3.01gage.10o"), use_native=False)
    cpp = rinexobs3(fixture("obs3.01gage.10o"), use_native=True)
    assert truth.equals(cpp)


def test_native_output_matches_python_zip():
    """Same agreement on a zip-compressed multi-system file."""
    truth = rinexobs3(fixture("ABMF00GLP_R_20181330000_01D_30S_MO.zip"), use_native=False)
    cpp = rinexobs3(fixture("ABMF00GLP_R_20181330000_01D_30S_MO.zip"), use_native=True)
    assert truth.equals(cpp)


def test_native_with_indicators_matches():
    """Indicator columns match between native and pure-Python paths."""
    truth = rinexobs3(fixture("obs3.01gage.10o"), use_native=False, useindicators=True)
    cpp = rinexobs3(fixture("obs3.01gage.10o"), use_native=True, useindicators=True)
    assert truth.equals(cpp)


def test_native_explicit_kwarg_overrides_env(monkeypatch):
    """``use_native=False`` wins over ``RINEXPY_USE_NATIVE=1``."""
    monkeypatch.setenv("RINEXPY_USE_NATIVE", "1")
    obs = rinexobs3(fixture("obs3.01gage.10o"), use_native=False)
    assert obs.attrs["rinextype"] == "obs"


def test_native_takes_precedence_over_jit():
    """When both are enabled, native wins (it's the faster path)."""
    a = rinexobs3(fixture("obs3.01gage.10o"), use_native=True, use_jit=True)
    b = rinexobs3(fixture("obs3.01gage.10o"), use_native=True, use_jit=False)
    assert a.equals(b)


def test_native_call_with_too_small_buffer_raises():
    """The C++ binding rejects buffers that are smaller than n_lines * n_obs * 16."""
    import numpy as np

    flat = np.zeros(10, dtype=np.uint8)
    # nanobind translates the C++ std::invalid_argument into ValueError.
    with pytest.raises(ValueError, match="too small"):
        _native.decode_obs_batch(flat, n_lines=2, n_obs=8)


def test_native_kernel_round_trip():
    """The C++ kernel and the pure-Python decoder produce equal output."""
    import numpy as np

    from rinexpy.obs3 import _decode_sv_line

    line = "  22227666.760 6  25342359.370 0 ".ljust(2 * 16)
    py = _decode_sv_line(line, n_obs=2)
    flat = np.frombuffer(line.encode("ascii"), dtype=np.uint8)
    cpp = _native.decode_obs_batch(flat, 1, 2).reshape(2, 3)
    np.testing.assert_allclose(py, cpp, equal_nan=True, rtol=0, atol=0)


# A safety net: the import-skip above gates everything else, so it's
# fine to leave RINEXPY_USE_NATIVE unset for the rest of the suite.
def teardown_module():
    os.environ.pop("RINEXPY_USE_NATIVE", None)
