"""rinexpy_native — optional C++ acceleration for rinexpy.

The compiled extension currently exposes:

- :func:`decode_obs_batch` — the OBS3 fixed-width decoder.
- :func:`crc24q` — the RTCM3 CRC-24Q checksum.
- :func:`read_bits` — MSB-first bit-cursor extraction (RTCM3 / SSR / MSM).
- :func:`lambda_ils` — LAMBDA branch-and-bound integer least squares.
- :func:`decode_msm` — full MSM4 / MSM7 frame decoder.
- :func:`interpolate_sp3_lagrange` — batched order-N SP3 Lagrange interp.

These are wired up by ``rinexpy.obs3`` / ``rinexpy.rtcm3`` when the
package is importable, so end users typically never call into here
directly.

Install from the parent rinexpy checkout:

    uv sync --extra native

The pure-Python ``rinexpy`` package is unchanged whether this is
installed or not — installing ``rinexpy-native`` only adds faster
back-ends for a few inner loops.
"""

from __future__ import annotations

from ._ext import (
    crc24q,
    decode_beidou_d1_sf1,
    decode_beidou_d2_page1,
    decode_lnav_subframe,
    decode_msm,
    decode_obs_batch,
    interpolate_sp3_lagrange,
    lambda_ils,
    read_bits,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "crc24q",
    "decode_beidou_d1_sf1",
    "decode_beidou_d2_page1",
    "decode_lnav_subframe",
    "decode_msm",
    "decode_obs_batch",
    "interpolate_sp3_lagrange",
    "lambda_ils",
    "read_bits",
]
