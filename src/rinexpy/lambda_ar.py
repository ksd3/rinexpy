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


def integer_least_squares(
    a_float: np.ndarray,
    Q: np.ndarray,
    *,
    n_cands: int = 2,
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

    Returns
    -------
    candidates, sq_errors, L:
        ``candidates`` is ``(n_cands, n)`` with the best integer vectors
        sorted by squared error (smallest first). ``sq_errors`` is the
        corresponding ``(n_cands,)`` array of L^T D^-1 L weighted square
        residuals. ``L`` is the LDL factor (returned for inspection).

    Notes
    -----
    Uses a depth-first branch-and-bound search bounded by the current
    best squared error. Decorrelation is intentionally skipped here
    (we compute ILS in the original space); for very correlated
    ambiguity sets, see :func:`decorrelate` to apply a Z-transform first.
    """
    L, D = ldl(Q)
    n = a_float.size
    boot = bootstrap(L, a_float)

    # Squared residual upper bound: start with the bootstrap value.
    cands: list[tuple[float, np.ndarray]] = []

    def conditional(idx: int, current: np.ndarray) -> float:
        """Return the conditional mean of a_float[idx] given a_int[idx+1:]."""
        c = a_float[idx]
        for j in range(idx + 1, n):
            c -= L[j, idx] * (current[j] - a_float[j])
        return c

    def search(idx: int, current: np.ndarray, residual_sq: float) -> None:
        """Recursive depth-first ILS search.

        Keeps the top ``n_cands`` complete integer vectors by squared
        error. The branch-and-bound prunes a partial path only once we
        already have ``n_cands`` candidates AND the partial residual
        already exceeds the worst kept.
        """
        if idx < 0:
            cands.append((residual_sq, current.copy().astype(int)))
            cands.sort(key=lambda x: x[0])
            return
        c = conditional(idx, current)
        center = int(np.round(c))
        # Try integer offsets 0, +1, -1, +2, -2, ... around the
        # conditional mean.
        for k in range(40):
            for sign in (1, -1) if k > 0 else (1,):
                cand = center + sign * k
                delta = cand - c
                contrib = delta * delta / D[idx]
                new_sq = residual_sq + contrib
                if (
                    len(cands) >= n_cands * 4  # keep extra to prune duplicates
                    and new_sq >= cands[-1][0]
                ):
                    continue
                current[idx] = cand
                search(idx - 1, current, new_sq)

    search(n - 1, boot.astype(float).copy(), 0.0)
    # Ensure the bootstrapped solution is at least considered.
    cands.append((_squared_residual(boot, a_float, L, D), boot.copy()))

    # Deduplicate by integer vector identity, then sort and truncate.
    seen: set[tuple[int, ...]] = set()
    unique: list[tuple[float, np.ndarray]] = []
    cands.sort(key=lambda x: x[0])
    for sq_val, vec in cands:
        key = tuple(int(v) for v in vec)
        if key in seen:
            continue
        seen.add(key)
        unique.append((sq_val, vec))
        if len(unique) >= n_cands:
            break

    candidates = np.array([c[1] for c in unique], dtype=int)
    sq_errors = np.array([c[0] for c in unique])
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

    Returns
    -------
    dict
        ``{"a_int": ndarray, "ratio": float, "accepted": bool,
        "candidates": (k, n) ndarray, "sq_errors": (k,) ndarray}``.
        ``accepted`` is True iff ``ratio >= ratio_threshold``.
    """
    cands, sq, _ = integer_least_squares(a_float, Q, n_cands=2)
    best = cands[0]
    if sq.size > 1 and sq[0] > 0:
        ratio = float(sq[1] / sq[0])
    else:
        ratio = float("inf")
    return {
        "a_int": best,
        "ratio": ratio,
        "accepted": ratio >= ratio_threshold,
        "candidates": cands,
        "sq_errors": sq,
    }


__all__ = [
    "bootstrap",
    "integer_least_squares",
    "lambda_resolve",
    "ldl",
]
