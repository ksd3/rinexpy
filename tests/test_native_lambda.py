"""Parity tests for the C++ LAMBDA ILS kernel.

Skipped when rinexpy_native isn't importable.
"""

from __future__ import annotations

import numpy as np
import pytest

from rinexpy import _native
from rinexpy.lambda_ar import (
    ILSAborted,
    integer_least_squares,
    lambda_resolve,
)

pytest.importorskip("rinexpy_native")


def test_native_lambda_ils_available():
    assert _native.have_lambda_ils() is True


def _force(native_on: bool) -> None:
    if native_on:
        _native.have_lambda_ils = lambda: _native._lambda_ils is not None
    else:
        _native.have_lambda_ils = lambda: False


@pytest.mark.parametrize("n", [3, 5, 8, 10, 15, 20])
def test_native_matches_python_on_well_conditioned(n):
    """Bit-identical best candidate and squared errors against the
    Python ILS on well-conditioned synthetic cases."""
    rng = np.random.default_rng(n)
    truth = rng.integers(-5, 6, size=n)
    Q = np.eye(n) * 0.05 + np.full((n, n), 0.01)
    a_float = truth + rng.normal(0, 0.05, size=n)

    _force(False)
    cands_py, sq_py, _ = integer_least_squares(a_float, Q, n_cands=2)
    _force(True)
    cands_cpp, sq_cpp, _ = integer_least_squares(a_float, Q, n_cands=2)

    np.testing.assert_array_equal(cands_cpp[0], cands_py[0])
    np.testing.assert_allclose(sq_cpp, sq_py, rtol=1e-12, atol=1e-12)


def test_native_recovers_truth():
    truth = np.array([5, -3, 2, 7, 1])
    Q = np.eye(5) * 0.01
    rng = np.random.default_rng(0)
    a_float = truth + rng.normal(0, 0.05, size=5)
    _force(True)
    cands, _, _ = integer_least_squares(a_float, Q, n_cands=2)
    np.testing.assert_array_equal(cands[0], truth)


def test_native_max_nodes_raises_ilsaborted():
    """A max_nodes=1 budget aborts and surfaces ILSAborted."""
    a_float = np.array([1.1, 2.2, 3.3, 4.4])
    Q = np.eye(4) * 0.01
    _force(True)
    with pytest.raises(ILSAborted) as exc:
        integer_least_squares(a_float, Q, max_nodes=1)
    # The bootstrap candidate is the best partial.
    assert exc.value.candidates.shape[1] == 4


def test_native_lambda_resolve_accepts_clean_case():
    """Clean noise -> high ratio -> accepted=True via the native path."""
    rng = np.random.default_rng(1)
    truth = np.array([3, -1, 4, 1, 5])
    Q = np.eye(5) * 0.01
    a_float = truth + rng.normal(0, 0.05, size=5)
    _force(True)
    sol = lambda_resolve(a_float, Q, ratio_threshold=3.0)
    assert sol["accepted"] is True
    np.testing.assert_array_equal(sol["a_int"], truth)


def teardown_module():
    """Re-arm the native dispatch so other tests see the production state."""
    _native.have_lambda_ils = lambda: _native._lambda_ils is not None
