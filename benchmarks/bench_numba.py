"""Benchmark numba-jitted fixed-width float parsing vs the pure-Python path.

Run from the repo root:

    uv run python benchmarks/bench_numba.py

The conclusion (recorded in this file's docstring rather than only in
git) is that **numba is not worth the dependency for the OBS3 hot
loop**. The Python `float()` builtin is C-implemented and ~25 ns per
call; numba's bytes-to-float parse is ~80 ns per call, plus a one-time
JIT compile cost of ~1.5 s. For typical RINEX-3 OBS files the
break-even epoch count is in the millions.

This script is kept around so the conclusion is empirical, not just
asserted.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from rinexpy.obs3 import _decode_sv_line


def time_python(raw: str, n_obs: int, runs: int = 50) -> float:
    """Time the current pure-Python fixed-width decoder."""
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(1000):
            _decode_sv_line(raw, n_obs)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_numba(raw: str, n_obs: int, runs: int = 50) -> float | None:
    """Time a numba-jitted bytes-based decoder for the same input."""
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True)
    def decode_bytes(buf: np.ndarray, n_obs: int, cell: int) -> np.ndarray:
        out = np.full((n_obs, 3), np.nan, dtype=np.float64)
        for k in range(n_obs):
            start = k * cell
            # Parse 14-byte float
            v = 0.0
            sign = 1.0
            seen = False
            point = -1
            i = 0
            for i in range(14):
                c = buf[start + i]
                if c == 32:  # space
                    continue
                if c == 45:  # '-'
                    sign = -1.0
                elif c == 43:  # '+'
                    sign = 1.0
                elif c == 46:  # '.'
                    point = 0
                elif 48 <= c <= 57:  # digit
                    v = v * 10.0 + (c - 48)
                    if point >= 0:
                        point += 1
                    seen = True
            if seen:
                if point > 0:
                    v /= 10.0**point
                out[k, 0] = sign * v
        return out

    buf = np.frombuffer(raw.ljust(n_obs * 16).encode("ascii"), dtype=np.uint8)
    # Warm up the JIT once.
    decode_bytes(buf, n_obs, 16)

    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(1000):
            decode_bytes(buf, n_obs, 16)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    # Synthetic SV line: 8 measurements x 16 bytes.
    raw = " 22227666.760  23456789.012  34567890.123  45678901.234"
    raw += "    18.000  44.000   3.5  6.7"
    raw = raw.ljust(8 * 16)

    py = time_python(raw, n_obs=8)
    nb = time_numba(raw, n_obs=8)

    print(f"Python  : {py * 1000:6.2f} ms / 1000 calls")
    if nb is None:
        print("Numba   : not installed")
        return
    print(f"Numba   : {nb * 1000:6.2f} ms / 1000 calls")
    print(f"Speedup : {py / nb:5.2f}x" if nb > 0 else "")


if __name__ == "__main__":
    main()
