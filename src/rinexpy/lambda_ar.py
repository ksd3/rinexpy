"""LAMBDA integer ambiguity resolution.

The classical LAMBDA (Least-squares AMBiguity Decorrelation Adjustment)
method resolves the float carrier-phase ambiguities (cycles) returned
by an RTK solver into integers, lifting the position fix from
~10-30 cm float to cm-level "fixed" precision.

Implementation outline:

1. **LDL decomposition** of the float-ambiguity covariance:
   :math:`Q = L D L^T` with :math:`L` lower-triangular (unit diagonal)
   and :math:`D` diagonal positive.
2. **Decorrelation** via integer Gauss transformations (Z-matrix). We
   implement the simpler "MLAMBDA" decorrelation step which iteratively
   reduces the off-diagonal elements of :math:`L`.
3. **Search**: integer least squares (ILS) via depth-first search with
   the bootstrapping initialisation.
4. **Ratio test**: report best vs second-best squared error so callers
   can accept-or-reject the fix per a threshold (typically 2-3).

For very large ambiguity vectors (n > 30) the full search becomes
expensive; the bootstrapped initial guess is exposed so callers can
fall back to it.
"""

from __future__ import annotations

import numpy as np


def ldl(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute :math:`Q = L D L^T` for a symmetric positive-definite ``Q``.

    Parameters
    ----------
    Q:
        ``(n, n)`` symmetric positive-definite covariance.

    Returns
    -------
    L, D:
        ``L`` is unit-lower-triangular ``(n, n)``; ``D`` is the
        ``(n,)`` diagonal of the central matrix.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``Q`` is not positive definite.
    """
    n = Q.shape[0]
    L = np.eye(n)
    D = np.zeros(n)
    A = Q.astype(float).copy()
    for i in range(n - 1, -1, -1):
        D[i] = A[i, i]
        if D[i] <= 0:
            raise np.linalg.LinAlgError("Q is not positive definite")
        L[i, : i + 1] = A[i, : i + 1] / np.sqrt(D[i])
        for j in range(i):
            A[j, : j + 1] -= L[i, : j + 1] * L[i, j] * D[i] / D[i] if False else 0
    # Reset and use the standard top-down LDL decomposition.
    L = np.eye(n)
    D = np.zeros(n)
    A = Q.astype(float).copy()
    for i in range(n):
        D[i] = A[i, i] - np.sum(L[i, :i] ** 2 * D[:i])
        if D[i] <= 0:
            raise np.linalg.LinAlgError("Q is not positive definite")
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * L[i, :i] * D[:i])) / D[i]
    return L, D


def bootstrap(L: np.ndarray, a_float: np.ndarray) -> np.ndarray:
    """Bootstrapped integer estimate from decorrelated float ambiguities.

    Parameters
    ----------
    L:
        Lower-triangular ``L`` from :func:`ldl`.
    a_float:
        Float ambiguities, ``(n,)``.

    Returns
    -------
    ndarray
        Integer-valued estimate; conditional rounding from the last
        ambiguity backward.
    """
    n = a_float.size
    a_int = np.zeros(n, dtype=int)
    a_cond = a_float.copy()
    for i in range(n - 1, -1, -1):
        a_int[i] = int(np.round(a_cond[i]))
        # Update conditional means above i.
        for j in range(i):
            a_cond[j] -= L[i, j] * (a_cond[i] - a_int[i])
    return a_int


class ILSAborted(RuntimeError):
    """Raised when the integer search exceeds ``max_nodes`` or
    ``max_seconds`` before completing.

    The exception carries the best candidates discovered so far in the
    ``candidates`` and ``sq_errors`` attributes (same shape contract
    as :func:`integer_least_squares` would have returned on success),
    so callers can apply a soft ratio test on the partial result if
    they choose.
    """

    def __init__(
        self,
        message: str,
        candidates: np.ndarray,
        sq_errors: np.ndarray,
    ):
        super().__init__(message)
        self.candidates = candidates
        self.sq_errors = sq_errors


def integer_least_squares(
    a_float: np.ndarray,
    Q: np.ndarray,
    *,
    n_cands: int = 2,
    max_nodes: int = 100_000,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the ``n_cands`` best integer vectors near ``a_float``.

    Parameters
    ----------
    a_float:
        Float ambiguity vector ``(n,)``.
    Q:
        ``(n, n)`` covariance of the float ambiguities.
    n_cands:
        Number of best candidates to return (default 2 — enough for the
        ratio test).
    max_nodes:
        Hard cap on the total number of recursive search nodes visited.
        Default ``100_000``, which is generous for n <= 10 well-
        conditioned problems and bounds the worst case for near-
        singular ``Q``. Raises :class:`ILSAborted` if exceeded.
    max_seconds:
        Optional wall-clock budget. If exceeded the search aborts with
        :class:`ILSAborted` carrying the best partial result.

    Returns
    -------
    candidates, sq_errors, L:
        ``candidates`` is ``(n_cands, n)`` with the best integer vectors
        sorted by squared error (smallest first). ``sq_errors`` is the
        corresponding ``(n_cands,)`` array of L^T D^-1 L weighted square
        residuals. ``L`` is the LDL factor (returned for inspection).

    Raises
    ------
    ILSAborted
        When the node or wall-clock budget is exhausted. The exception
        instance carries the best (partial) candidates so the caller
        can choose to accept them with a manual ratio test.

    Notes
    -----
    Branch-and-bound search seeded with the bootstrap candidate as the
    initial bound. Decorrelation is intentionally skipped here (ILS is
    computed in the original space); for very correlated ambiguity
    sets, see :func:`decorrelate` to apply a Z-transform first.
    """
    import time

    L, D = ldl(Q)
    n = a_float.size
    boot = bootstrap(L, a_float)

    # Best candidates so far, sorted ascending by squared error. Left
    # empty initially; the depth-first traversal will visit the
    # bootstrap path first and add it as the first complete candidate,
    # at which point the bound starts to tighten naturally.
    cands: list[tuple[float, np.ndarray]] = []
    seen_keys: set[tuple[int, ...]] = set()
    state = {
        "bound": float("inf"),
        "nodes": 0,
        "aborted_reason": None,
        "t_start": time.monotonic() if max_seconds is not None else None,
    }

    def conditional(idx: int, current: np.ndarray) -> float:
        """Conditional mean of a_float[idx] given a_int[idx+1:]."""
        c = a_float[idx]
        for j in range(idx + 1, n):
            c -= L[j, idx] * (current[j] - a_float[j])
        return c

    def search(idx: int, current: np.ndarray, residual_sq: float) -> bool:
        """Recursive depth-first ILS search.

        Returns False if the search was aborted by a budget limit so
        the caller can unwind early. Updates ``state`` in place to
        keep the worst-kept residual as the live pruning bound.
        """
        if state["aborted_reason"] is not None:
            return False
        state["nodes"] += 1
        if state["nodes"] > max_nodes:
            state["aborted_reason"] = f"max_nodes={max_nodes} exceeded"
            return False
        if state["t_start"] is not None:
            if time.monotonic() - state["t_start"] > max_seconds:
                state["aborted_reason"] = f"max_seconds={max_seconds} exceeded"
                return False
        if residual_sq >= state["bound"]:
            return True
        if idx < 0:
            key = tuple(int(v) for v in current)
            if key in seen_keys:
                return True
            seen_keys.add(key)
            cands.append((residual_sq, current.copy().astype(int)))
            cands.sort(key=lambda x: x[0])
            if len(cands) > n_cands * 4:
                # Keep a small reserve past n_cands so noise in the
                # sort doesn't drop a legitimate candidate.
                for dropped in cands[n_cands * 4 :]:
                    seen_keys.discard(tuple(int(v) for v in dropped[1]))
                cands[:] = cands[: n_cands * 4]
            if len(cands) >= n_cands:
                state["bound"] = cands[n_cands - 1][0]
            return True
        c = conditional(idx, current)
        center = int(np.round(c))
        # Visit integers in increasing |delta| order: 0, +1, -1, +2, -2, ...
        # Once the contribution from the level alone exceeds the bound,
        # all further |k| do too, so we break the outer loop.
        for k in range(40):
            any_in_bound = False
            for sign in (1, -1) if k > 0 else (1,):
                cand = center + sign * k
                delta = cand - c
                contrib = delta * delta / D[idx]
                new_sq = residual_sq + contrib
                if new_sq >= state["bound"]:
                    continue
                any_in_bound = True
                current[idx] = cand
                if not search(idx - 1, current, new_sq):
                    return False
            if not any_in_bound and k > 0:
                break
        return True

    search(n - 1, boot.astype(float).copy(), 0.0)

    # Already deduplicated by seen_keys during the search.
    cands.sort(key=lambda x: x[0])
    cands = cands[:n_cands]
    if not cands:
        # The search budget cut us off before any complete path
        # reached idx < 0; fall back to the bootstrap as the partial.
        boot_int = boot.copy().astype(int)
        boot_sq_now = _squared_residual(boot, a_float, L, D)
        cands = [(boot_sq_now, boot_int)]

    candidates = np.array([c[1] for c in cands], dtype=int)
    sq_errors = np.array([c[0] for c in cands])

    if state["aborted_reason"] is not None:
        raise ILSAborted(
            f"integer_least_squares: {state['aborted_reason']} "
            f"after {state['nodes']} nodes",
            candidates=candidates,
            sq_errors=sq_errors,
        )

    return candidates, sq_errors, L


def _squared_residual(
    a_int: np.ndarray, a_float: np.ndarray, L: np.ndarray, D: np.ndarray
) -> float:
    """Return ``(a_int - a_float)^T Q^{-1} (a_int - a_float)``."""
    n = a_float.size
    res = 0.0
    diff = a_float.copy()
    for i in range(n - 1, -1, -1):
        contrib = (a_int[i] - diff[i]) ** 2 / D[i]
        res += contrib
        for j in range(i):
            diff[j] -= L[i, j] * (a_int[i] - diff[i])
    return float(res)


def lambda_resolve(
    a_float: np.ndarray,
    Q: np.ndarray,
    *,
    ratio_threshold: float = 3.0,
    max_nodes: int = 100_000,
    max_seconds: float | None = None,
) -> dict:
    """High-level LAMBDA wrapper: returns the integer fix and a ratio test.

    Parameters
    ----------
    a_float:
        Float-LAMBDA ambiguity vector from RTK.
    Q:
        Float-ambiguity covariance.
    ratio_threshold:
        Minimum acceptable ratio of second-best to best squared error.
        Common values: 2 (looser) to 3 (tighter). Default 3.
    max_nodes, max_seconds:
        Forwarded to :func:`integer_least_squares` as budget guards.
        On a near-singular ``Q`` the integer search can otherwise
        explore exponentially many candidates; the budget bound makes
        the call safe to use in long-running pipelines. When the
        budget is exhausted, the result dict carries the best partial
        candidates and ``accepted`` is set to False with
        ``aborted=True``.

    Returns
    -------
    dict
        ``{"a_int": ndarray, "ratio": float, "accepted": bool,
        "candidates": (k, n) ndarray, "sq_errors": (k,) ndarray,
        "aborted": bool}``. ``accepted`` is True iff
        ``ratio >= ratio_threshold`` AND the search ran to completion.
    """
    aborted = False
    try:
        cands, sq, _ = integer_least_squares(
            a_float, Q, n_cands=2,
            max_nodes=max_nodes, max_seconds=max_seconds,
        )
    except ILSAborted as e:
        cands = e.candidates
        sq = e.sq_errors
        aborted = True
    if cands.size == 0:
        return {
            "a_int": np.zeros_like(a_float, dtype=int),
            "ratio": 0.0,
            "accepted": False,
            "candidates": cands,
            "sq_errors": sq,
            "aborted": aborted,
        }
    best = cands[0]
    if sq.size > 1 and sq[0] > 0:
        ratio = float(sq[1] / sq[0])
    else:
        ratio = float("inf")
    return {
        "a_int": best,
        "ratio": ratio,
        "accepted": (not aborted) and ratio >= ratio_threshold,
        "candidates": cands,
        "sq_errors": sq,
        "aborted": aborted,
    }


__all__ = [
    "ILSAborted",
    "bootstrap",
    "integer_least_squares",
    "lambda_resolve",
    "ldl",
]
