"""Parity test for the C++ Joseph-form scalar EKF update kernel.

Skipped when rinexpy_native isn't importable.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rinexpy import _native
from rinexpy.kalman import StaticPPPFilter

pytest.importorskip("rinexpy_native")


def _force(native_on: bool) -> None:
    if native_on:
        _native.have_kalman_scalar_update = (
            lambda: _native._kalman_scalar_update_sparse is not None
        )
    else:
        _native.have_kalman_scalar_update = lambda: False


def test_native_kalman_available():
    assert _native.have_kalman_scalar_update() is True


def test_scalar_update_matches_python_code_obs():
    """Code update: hand-built sparse-H spec matches the dense numpy
    implementation to within float64 round-off."""
    n_sv = 8
    f_base = StaticPPPFilter(n_sv=n_sv, initial_position=(4e6, 2e6, 4e6))
    rng = np.random.default_rng(0)
    f_base.x = rng.normal(scale=10.0, size=4 + n_sv) + np.array(
        [4e6, 2e6, 4e6, 0.0] + [0.0] * n_sv
    )
    f_base.P = rng.normal(size=(4 + n_sv, 4 + n_sv))
    f_base.P = f_base.P @ f_base.P.T + np.eye(4 + n_sv) * 1e-3

    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    obs = 2e7 + rng.normal()
    rho = 2e7 + 5.0

    _force(False)
    f_py = copy.deepcopy(f_base)
    f_py._scalar_update(u, code=True, sv_index=2, obs=obs, rho=rho)

    _force(True)
    f_cpp = copy.deepcopy(f_base)
    f_cpp._scalar_update(u, code=True, sv_index=2, obs=obs, rho=rho)

    np.testing.assert_allclose(f_cpp.x, f_py.x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(f_cpp.P, f_py.P, rtol=0, atol=1e-9)


def test_scalar_update_matches_python_phase_obs():
    """Phase update with ambiguity slot."""
    n_sv = 8
    f_base = StaticPPPFilter(n_sv=n_sv, initial_position=(4e6, 2e6, 4e6))
    rng = np.random.default_rng(7)
    f_base.x = rng.normal(scale=10.0, size=4 + n_sv) + np.array(
        [4e6, 2e6, 4e6, 0.0] + [0.0] * n_sv
    )
    f_base.P = rng.normal(size=(4 + n_sv, 4 + n_sv))
    f_base.P = f_base.P @ f_base.P.T + np.eye(4 + n_sv) * 1e-3

    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    obs = 2e7 + rng.normal()
    rho = 2e7 + 5.0

    _force(False)
    f_py = copy.deepcopy(f_base)
    f_py._scalar_update(u, code=False, sv_index=3, obs=obs, rho=rho)

    _force(True)
    f_cpp = copy.deepcopy(f_base)
    f_cpp._scalar_update(u, code=False, sv_index=3, obs=obs, rho=rho)

    np.testing.assert_allclose(f_cpp.x, f_py.x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(f_cpp.P, f_py.P, rtol=0, atol=1e-9)


def teardown_module():
    _force(True)
