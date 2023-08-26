"""Optional numba-jitted hot-path replacements.

This module is wholly optional. If ``numba`` is not installed, importing
it is still safe — the public functions fall back to ``None`` (callers
check ``is_available()`` before dispatching).

The current jitted kernel is :func:`decode_obs_block`, which parses the
fixed-width per-epoch data section of a RINEX-3 OBS file. The pure-
Python version in :mod:`rinexpy.obs3` is ~7x slower per cell on
microbenchmarks; in end-to-end profiles the OBS3 read becomes ~1.4-1.8x
faster overall (the rest is I/O and xarray construction).

The fast path is **off by default**. Opt in either by passing
``rinexobs3(..., use_jit=True)`` or by setting the environment variable
``RINEXPY_USE_JIT=1``.
"""

from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False
    njit = None  # type: ignore[assignment]


def is_available() -> bool:
    """Return ``True`` if numba is importable in this Python."""
    return _HAVE_NUMBA


def is_enabled() -> bool:
    """Return ``True`` if the JIT path should be used.

    Looks at the ``RINEXPY_USE_JIT`` env var: ``"1"``, ``"true"``, ``"yes"``
    enable, anything else disables. Defaults to disabled.
    """
    if not _HAVE_NUMBA:
        return False
    return os.environ.get("RINEXPY_USE_JIT", "").lower() in {"1", "true", "yes"}


if _HAVE_NUMBA:

    @njit(cache=True)
    def _decode_one_cell(buf: np.ndarray, start: int, out: np.ndarray, row: int, col: int) -> None:
        """Parse a single 16-byte (value, LLI, SSI) cell into ``out[row, col*3:col*3+3]``.

        Fixed-width %14.3f for the value, then one digit each for LLI and
        SSI. Blank fields stay NaN. RINEX 3 OBS values do not use
        scientific notation (fixed format), so this parser is intentionally
        narrow: a minus sign, a sequence of digits, an optional decimal
        point, more digits.
        """
        v = 0.0
        sign = 1.0
        seen = False
        point = -1
        for i in range(14):
            c = buf[start + i]
            if c == 32:
                continue
            if c == 45:
                sign = -1.0
            elif c == 43:
                sign = 1.0
            elif c == 46:
                point = 0
            elif 48 <= c <= 57:
                v = v * 10.0 + (c - 48)
                if point >= 0:
                    point += 1
                seen = True
        if seen:
            if point > 0:
                v /= 10.0**point
            out[row, col * 3] = sign * v

        lli = buf[start + 14]
        if 48 <= lli <= 57:
            out[row, col * 3 + 1] = lli - 48

        ssi = buf[start + 15]
        if 48 <= ssi <= 57:
            out[row, col * 3 + 2] = ssi - 48

    @njit(cache=True)
    def decode_obs_block(bufs: list[np.ndarray], n_obs: int, n_sv: int) -> np.ndarray:
        """Decode an SV's worth of fixed-width cells into a (n_sv, n_obs*3) array.

        Parameters
        ----------
        bufs:
            One ``np.uint8`` buffer per SV, each padded to ``n_obs * 16``
            bytes long.
        n_obs:
            Number of measurement types per SV (the OBS3 ``Fmax``).
        n_sv:
            Number of SVs in this epoch (== ``len(bufs)``).

        Returns
        -------
        np.ndarray
            ``(n_sv, n_obs * 3)`` float64 array; NaN for blank cells. The
            three columns per measurement are (value, LLI, SSI).
        """
        out = np.full((n_sv, n_obs * 3), np.nan, dtype=np.float64)
        for s in range(n_sv):
            buf = bufs[s]
            for k in range(n_obs):
                _decode_one_cell(buf, k * 16, out, s, k)
        return out

    @njit(cache=True)
    def decode_obs_batch(flat: np.ndarray, n_lines: int, n_obs: int) -> np.ndarray:
        """Batch-decode N concatenated SV lines from one contiguous bytes buffer.

        Parameters
        ----------
        flat:
            ``np.uint8`` buffer of shape ``(n_lines * n_obs * 16,)`` —
            every SV line padded to the fixed width and concatenated.
        n_lines:
            Number of SV lines packed in ``flat``.
        n_obs:
            Number of measurement types (== ``Fmax`` from the header).

        Returns
        -------
        np.ndarray
            ``(n_lines, n_obs * 3)`` float64 array. The three columns per
            measurement are (value, LLI, SSI).

        Notes
        -----
        Batching the bytes-conversion to **once per file** rather than
        once per SV is the key to making the JIT path actually faster
        than the pure-Python loop. Per-call numba dispatch overhead is
        ~50-100 µs; calling it once instead of once-per-SV brings that
        cost down to negligible.
        """
        out = np.full((n_lines, n_obs * 3), np.nan, dtype=np.float64)
        cell = 16
        line_bytes = n_obs * cell
        for s in range(n_lines):
            base = s * line_bytes
            for k in range(n_obs):
                _decode_one_cell(flat, base + k * cell, out, s, k)
        return out
else:  # pragma: no cover

    def decode_obs_block(*args, **kwargs):
        raise ImportError("numba is not installed; install with `uv add 'rinexpy[jit]'`")

    def decode_obs_batch(*args, **kwargs):
        raise ImportError("numba is not installed; install with `uv add 'rinexpy[jit]'`")


__all__ = ["decode_obs_batch", "decode_obs_block", "is_available", "is_enabled"]
