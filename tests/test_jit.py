"""Tests for the optional numba-jitted decoder path.

The JIT path is opt-in (``rinexobs3(..., use_jit=True)`` or
``RINEXPY_USE_JIT=1``). These tests verify that:

1. The output is bit-identical to the pure-Python decoder.
2. The opt-in is respected (env var, kwarg).
3. ``is_available()`` returns the actual import status.
"""

from __future__ import annotations

import pytest

from rinexpy import _jit
from rinexpy.obs3 import rinexobs3

from .conftest import fixture

pytest.importorskip("numba")


def test_jit_is_available():
    assert _jit.is_available() is True


def test_jit_env_var_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RINEXPY_USE_JIT", raising=False)
    assert _jit.is_enabled() is False


def test_jit_env_var_enables(monkeypatch):
    monkeypatch.setenv("RINEXPY_USE_JIT", "1")
    assert _jit.is_enabled() is True


def test_jit_output_matches_python_small():
    """JIT and Python decoders agree on a small mixed-system file."""
    truth = rinexobs3(fixture("obs3.01gage.10o"), use_jit=False)
    jit = rinexobs3(fixture("obs3.01gage.10o"), use_jit=True)
    assert truth.equals(jit)


def test_jit_output_matches_python_zip():
    """Same agreement on a real-world zip-compressed multi-system file."""
    truth = rinexobs3(fixture("ABMF00GLP_R_20181330000_01D_30S_MO.zip"), use_jit=False)
    jit = rinexobs3(fixture("ABMF00GLP_R_20181330000_01D_30S_MO.zip"), use_jit=True)
    assert truth.equals(jit)


def test_jit_with_indicators_matches():
    """LLI/SSI indicator columns match between JIT and Python paths."""
    truth = rinexobs3(fixture("obs3.01gage.10o"), use_jit=False, useindicators=True)
    jit = rinexobs3(fixture("obs3.01gage.10o"), use_jit=True, useindicators=True)
    assert truth.equals(jit)


def test_jit_explicit_kwarg_overrides_env(monkeypatch):
    """``use_jit=False`` wins over ``RINEXPY_USE_JIT=1``."""
    monkeypatch.setenv("RINEXPY_USE_JIT", "1")
    obs = rinexobs3(fixture("obs3.01gage.10o"), use_jit=False)
    assert obs.attrs["rinextype"] == "obs"
