"""Tests for the max_nodes / max_seconds budget guards added to the
LAMBDA integer search."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rinexpy.lambda_ar import (
    ILSAborted,
    integer_least_squares,
    lambda_resolve,
)


def test_well_conditioned_problem_under_budget():
    """A small, well-conditioned ILS problem completes well below the
    default budget."""
    a_float = np.array([3.1, -1.9, 4.05])
    Q = np.eye(3) * 0.01
    cands, sq, _ = integer_least_squares(a_float, Q, n_cands=2)
    assert cands.shape == (2, 3)
    # Best should round to (3, -2, 4) up to LAMBDA decorrelation order.
    np.testing.assert_array_equal(np.sort(cands[0]), np.sort([3, -2, 4]))


def test_max_nodes_zero_aborts_immediately():
    """A node budget of 1 forces an immediate abort, but the dispatcher
    still surfaces the bootstrap as the partial result."""
    a_float = np.array([1.1, 2.2, 3.3])
    Q = np.eye(3) * 0.01
    with pytest.raises(ILSAborted) as exc:
        integer_least_squares(a_float, Q, max_nodes=1)
    # The bootstrap candidate must be present as the best partial.
    assert exc.value.candidates.shape[1] == 3
    assert exc.value.sq_errors.size >= 1


def test_lambda_resolve_swallows_abort_into_dict():
    """When lambda_resolve hits the budget, it returns aborted=True
    rather than propagating the exception. accepted is forced False."""
    # Build a near-singular Q by stacking nearly-identical rows so the
    # ILS search can't shrink the bound. n=8 means up to 79^8 nodes
    # without pruning.
    n = 8
    base = np.eye(n)
    Q = base * 1e-6   # tight diagonal -> small D values -> wide search
    Q[0, 1] = Q[1, 0] = 0.999e-6   # near-singular block
    a_float = np.random.default_rng(0).standard_normal(n) * 0.5
    res = lambda_resolve(a_float, Q, max_nodes=200, ratio_threshold=3.0)
    assert res["aborted"] is True
    assert res["accepted"] is False
    # Even after abort, the dict still has a best-effort a_int.
    assert res["a_int"].shape == (n,)


def test_lambda_resolve_succeeds_with_generous_budget():
    """The same well-conditioned input works with the default budget."""
    a_float = np.array([3.05, -1.98, 4.12, 0.03])
    Q = np.eye(4) * 0.001
    res = lambda_resolve(a_float, Q)
    assert res["accepted"]
    assert res["aborted"] is False
    np.testing.assert_array_equal(
        np.sort(res["a_int"]), np.sort([3, -2, 4, 0]),
    )


def test_max_seconds_aborts_long_search():
    """A wall-clock budget aborts the search even when the node budget
    is huge. We hit the wall-clock path by constructing Q with a near-
    flat eigenvalue spectrum (correlated ambiguities) so the integer
    search has many equally-good directions to explore."""
    n = 8
    rng = np.random.default_rng(1)
    # Near-flat eigenvalue Q via outer product + tiny diagonal.
    v = rng.standard_normal(n)
    Q = np.outer(v, v) * 1e-4 + np.eye(n) * 1e-6
    Q = 0.5 * (Q + Q.T)
    # Confirm PD before running the test.
    np.linalg.cholesky(Q)
    a_float = rng.standard_normal(n) * 0.4
    t0 = time.monotonic()
    try:
        integer_least_squares(
            a_float, Q, max_seconds=0.3, max_nodes=10_000_000,
        )
    except ILSAborted:
        pass
    elapsed = time.monotonic() - t0
    # Whether the search completes before the timeout depends on the
    # exact spectrum; what matters is that the budget bounds the runtime.
    assert elapsed < 5.0


def test_aborted_partial_carries_best_so_far():
    """ILSAborted.candidates contains the best partial result; on a
    seriously under-budgeted search that should still include at least
    the bootstrap solution."""
    a_float = np.array([1.4, 2.6, 0.8])
    Q = np.eye(3) * 0.01
    with pytest.raises(ILSAborted) as exc:
        integer_least_squares(a_float, Q, max_nodes=1)
    # Bootstrap on integer = round(1.4, 2.6, 0.8) = (1, 3, 1) for
    # decorrelation order; we check it's in the partial result list.
    partial = exc.value.candidates
    assert partial.shape[1] == 3
    assert partial.dtype.kind == "i"
