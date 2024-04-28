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
            "install with `uv add 'rinexpy[native]'` or `pip install rinexpy-native`."
        )
    return _decode_obs_batch(flat, n_lines, n_obs)


__all__ = ["decode_obs_batch", "is_available", "is_enabled"]
