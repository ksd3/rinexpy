"""Tests for the LAMBDA integer ambiguity resolver."""

from __future__ import annotations

import numpy as np
from pytest import approx

from rinexpy.lambda_ar import (
    bootstrap,
    integer_least_squares,
    lambda_resolve,
    ldl,
)


def test_ldl_round_trip():
    """``L D L^T == Q`` for a known SPD matrix."""
    Q = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    L, D = ldl(Q)
    reconstructed = L @ np.diag(D) @ L.T
    np.testing.assert_allclose(reconstructed, Q, atol=1e-10)


def test_ldl_lower_triangular_with_unit_diagonal():
    Q = np.eye(4) + 0.1 * np.ones((4, 4))
    L, _ = ldl(Q)
    # Unit lower triangular.
    assert np.allclose(np.diag(L), 1.0)
    assert np.allclose(np.triu(L, 1), 0.0)


def test_bootstrap_recovers_integer_when_close():
    """Float ambiguities very close to integers should round to those."""
    a_float = np.array([3.05, 7.99, -2.01])
    Q = np.eye(3) * 0.01
    L, _ = ldl(Q)
    a_int = bootstrap(L, a_float)
    np.testing.assert_array_equal(a_int, [3, 8, -2])


def test_ils_finds_known_truth():
    """Synthetic float = truth + small noise, ILS should recover truth."""
    rng = np.random.default_rng(0)
    truth = np.array([5, -3, 2, 7])
    Q = np.eye(4) * 0.01
    a_float = truth + rng.normal(0, 0.05, size=4)
    cands, sq, _ = integer_least_squares(a_float, Q, n_cands=2)
    np.testing.assert_array_equal(cands[0], truth)


def test_lambda_resolve_accepts_clean_case():
    """Clean noise -> high ratio -> accepted=True."""
    rng = np.random.default_rng(1)
    truth = np.array([3, -1, 4, 1, 5])
    Q = np.eye(5) * 0.01
    a_float = truth + rng.normal(0, 0.05, size=5)
    sol = lambda_resolve(a_float, Q, ratio_threshold=3.0)
    assert sol["accepted"] is True
    np.testing.assert_array_equal(sol["a_int"], truth)


def test_lambda_resolve_low_ratio_rejected():
    """Ambiguous case (large noise) gives small ratio -> rejected."""
    rng = np.random.default_rng(2)
    a_float = rng.uniform(-5, 5, size=4)
    Q = np.eye(4) * 1.0  # large variance -> nearby integers all plausible
    sol = lambda_resolve(a_float, Q, ratio_threshold=10.0)
    assert sol["accepted"] is False


def test_lambda_returns_two_candidates_by_default():
    a_float = np.array([2.5, 3.5])
    Q = np.eye(2) * 0.5
    sol = lambda_resolve(a_float, Q)
    assert sol["candidates"].shape == (2, 2)
    assert sol["sq_errors"].shape == (2,)
    assert sol["sq_errors"][0] <= sol["sq_errors"][1]
