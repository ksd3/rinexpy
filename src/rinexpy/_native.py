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


__all__ = [
    "crc24q",
    "decode_obs_batch",
    "have_crc24q",
    "have_read_bits",
    "is_available",
    "is_enabled",
    "read_bits",
]
