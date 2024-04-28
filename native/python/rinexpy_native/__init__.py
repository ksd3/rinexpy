"""rinexpy_native — optional C++ acceleration for rinexpy.

This package contains a single compiled extension that accelerates
the OBS3 fixed-width decoder. It is imported automatically by
``rinexpy.obs3`` when present, so end users typically never call into
this package directly.

Install:

    uv add rinexpy-native

The pure-Python ``rinexpy`` package is unchanged whether this is
installed or not — installing ``rinexpy-native`` only adds a faster
back-end for one inner loop.
"""

from __future__ import annotations

from ._ext import decode_obs_batch

__version__ = "0.1.0"

__all__ = ["__version__", "decode_obs_batch"]
