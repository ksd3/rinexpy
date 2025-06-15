"""Optional ``rinexpy_native`` C++ extension dispatch.

This module is wholly optional. If ``rinexpy_native`` is not installed
(it ships as a separate compiled package), importing this module is
still safe — the public functions fall back to ``False`` /
``ImportError`` so callers can feature-test before dispatching.

The C++ extension provides exactly one kernel — ``decode_obs_batch``
— with the same signature and numerical contract as the numba kernel
in :mod:`rinexpy._jit`. It is the fastest of the three OBS3 decoder
back-ends (~3x over the JIT path, ~18x over pure Python on the inner
loop).

The fast path is **off by default** (the `rinexpy` install is
intentionally pure-Python). Opt in by either:

- installing the `rinexpy[native]` extra (which depends on
  ``rinexpy-native``) and calling ``rinexobs3(use_native=True)``;
- or setting ``RINEXPY_USE_NATIVE=1`` in the environment.
"""

from __future__ import annotations

import os

try:
    from rinexpy_native import decode_obs_batch as _decode_obs_batch

    _HAVE_NATIVE = True
except ImportError:  # pragma: no cover
    _HAVE_NATIVE = False
    _decode_obs_batch = None  # type: ignore[assignment]

# crc24q was added in rinexpy-native >= 0.2.0; older wheels that only
# ship decode_obs_batch are still supported.
try:
    from rinexpy_native import crc24q as _crc24q  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _crc24q = None  # type: ignore[assignment]

# read_bits was added alongside crc24q.
try:
    from rinexpy_native import read_bits as _read_bits  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _read_bits = None  # type: ignore[assignment]

# lambda_ils was added alongside crc24q / read_bits.
try:
    from rinexpy_native import lambda_ils as _lambda_ils  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _lambda_ils = None  # type: ignore[assignment]

# decode_msm landed in the same round as lambda_ils.
try:
    from rinexpy_native import decode_msm as _decode_msm  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _decode_msm = None  # type: ignore[assignment]


def is_available() -> bool:
    """Return ``True`` if ``rinexpy_native`` is importable in this Python."""
    return _HAVE_NATIVE


def is_enabled() -> bool:
    """Return ``True`` if the native path should be used by default.

    Looks at the ``RINEXPY_USE_NATIVE`` env var: ``"1"``, ``"true"``,
    ``"yes"`` enable, anything else disables. Defaults to disabled.
    """
    if not _HAVE_NATIVE:
        return False
    return os.environ.get("RINEXPY_USE_NATIVE", "").lower() in {"1", "true", "yes"}


def decode_obs_batch(flat, n_lines: int, n_obs: int):
    """Wrapper that raises a friendly ImportError when the extension is absent."""
    if _decode_obs_batch is None:
        raise ImportError(
            "rinexpy_native is not installed; "
            "from the rinexpy checkout run `uv sync --extra native`."
        )
    return _decode_obs_batch(flat, n_lines, n_obs)


def have_crc24q() -> bool:
    """Return ``True`` if the C++ ``crc24q`` kernel is available."""
    return _crc24q is not None


def crc24q(data: bytes) -> int:
    """RTCM3 CRC-24Q via the C++ kernel; raises if the extension is missing."""
    if _crc24q is None:
        raise ImportError(
            "rinexpy_native.crc24q is not installed; "
            "rebuild rinexpy-native >= 0.2.0 via `uv sync --extra native`."
        )
    return _crc24q(data)


def have_read_bits() -> bool:
    """Return ``True`` if the C++ ``read_bits`` kernel is available."""
    return _read_bits is not None


def read_bits(data: bytes, start_bit: int, n_bits: int,
              is_signed: bool = False) -> int:
    """MSB-first bit extraction via the C++ kernel."""
    if _read_bits is None:
        raise ImportError(
            "rinexpy_native.read_bits is not installed; "
            "rebuild rinexpy-native >= 0.2.0 via `uv sync --extra native`."
        )
    return _read_bits(data, start_bit, n_bits, is_signed)


def have_lambda_ils() -> bool:
    """Return ``True`` if the C++ LAMBDA ILS kernel is available."""
    return _lambda_ils is not None


def have_decode_msm() -> bool:
    """Return ``True`` if the C++ MSM decoder kernel is available."""
    return _decode_msm is not None


def decode_msm(body: bytes, msm_kind: int):
    """Full MSM4 / MSM7 frame decoder via the C++ kernel."""
    if _decode_msm is None:
        raise ImportError(
            "rinexpy_native.decode_msm is not installed; "
            "rebuild rinexpy-native >= 0.2.0 via `uv sync --extra native`."
        )
    return _decode_msm(body, msm_kind)


def lambda_ils(a_float, Q, n_cands: int, max_nodes: int,
               max_seconds: float):
    """LAMBDA branch-and-bound ILS via the C++ kernel.

    Returns ``(candidates, sq_errors, nodes_visited, aborted_reason)``
    where ``aborted_reason`` is ``0`` on completion, ``1`` for a
    ``max_nodes`` abort, ``2`` for ``max_seconds``. The Python wrapper
    in :mod:`rinexpy.lambda_ar` translates that into ``ILSAborted``.
    """
    if _lambda_ils is None:
        raise ImportError(
            "rinexpy_native.lambda_ils is not installed; "
            "rebuild rinexpy-native >= 0.2.0 via `uv sync --extra native`."
        )
    return _lambda_ils(a_float, Q, n_cands, max_nodes,
                       -1.0 if max_seconds is None else float(max_seconds))


__all__ = [
    "crc24q",
    "decode_msm",
    "decode_obs_batch",
    "have_crc24q",
    "have_decode_msm",
    "have_lambda_ils",
    "have_read_bits",
    "is_available",
    "is_enabled",
    "lambda_ils",
    "read_bits",
]
