"""Benchmark the C++ kernels added to rinexpy-native v0.2: the RTCM3
CRC + bit cursor (used by rinexpy.rtcm3) and the LAMBDA ILS search
(used by rinexpy.lambda_ar).

Run from the repo root:

    uv run python benchmarks/bench_native_extra.py

If you also want the RTCM3 numbers, prefetch the RTKLIB GMSD7 capture::

    mkdir -p /tmp/igs_real_cache
    curl -fsSL -o /tmp/igs_real_cache/GMSD7_20121014.rtcm3 \\
      https://raw.githubusercontent.com/tomojitakasu/RTKLIB/\\
rtklib_2.4.3/test/data/rcvraw/GMSD7_20121014.rtcm3

The script falls back to a synthetic byte stream if the capture is
missing so the numbers it prints are still meaningful, just not
"real".
"""

from __future__ import annotations

import os
import random
import statistics
import time
from pathlib import Path

import numpy as np

import rinexpy.lambda_ar as la
import rinexpy.rtcm3 as r3
from rinexpy import _native


def _median_of(callable_, n: int = 5) -> float:
    """Run `callable_()` n times, return the median wall-clock seconds.

    A warm-up call runs first so JIT / page-fault costs don't poison
    the measurement.
    """
    callable_()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _force_native(flag: bool) -> None:
    """Toggle the dispatch helpers in rinexpy._native at runtime.

    All four functions short-circuit based on `have_xxx()` returning
    True/False, so we can swap implementations without re-importing
    anything.
    """
    if flag:
        _native.have_crc24q = lambda: _native._crc24q is not None
        _native.have_read_bits = lambda: _native._read_bits is not None
        _native.have_lambda_ils = lambda: _native._lambda_ils is not None
        _native.have_decode_msm = lambda: _native._decode_msm is not None
    else:
        _native.have_crc24q = lambda: False
        _native.have_read_bits = lambda: False
        _native.have_lambda_ils = lambda: False
        _native.have_decode_msm = lambda: False


# ----------------------------------------------------------------------
# RTCM3 corpus.
# ----------------------------------------------------------------------

_RTCM3_CAPTURE = Path("/tmp/igs_real_cache/GMSD7_20121014.rtcm3")


def _load_rtcm3_bytes() -> tuple[bytes, str]:
    """Return (data, label) for the RTCM3 benchmark."""
    if _RTCM3_CAPTURE.exists() and _RTCM3_CAPTURE.stat().st_size > 1000:
        return _RTCM3_CAPTURE.read_bytes(), f"GMSD7 ({_RTCM3_CAPTURE.stat().st_size // 1024} KB)"
    rng = random.Random(0)
    data = bytes(rng.randint(0, 255) for _ in range(256 * 1024))
    return data, "synthetic 256 KB random"


def bench_rtcm3() -> None:
    data, label = _load_rtcm3_bytes()
    print(f"\n== RTCM3 inner loops ==")
    print(f"  corpus: {label}")

    # 1. crc24q over the entire buffer.
    _force_native(False)
    t_py = _median_of(lambda: r3.crc24q(data))
    _force_native(True)
    t_cpp = _median_of(lambda: r3.crc24q(data))
    print(f"  crc24q(buffer)       py {t_py*1e3:7.2f} ms    cpp {t_cpp*1e3:7.2f} ms    "
          f"{t_py/t_cpp:6.1f}x")

    # 2. iter_messages end-to-end with CRC check on, walking the
    # dispatch stack one layer at a time so the per-kernel
    # contribution is visible. Only meaningful with the real capture.
    if _RTCM3_CAPTURE.exists():
        from io import BytesIO

        def run():
            return list(r3.iter_messages(BytesIO(data), check_crc=True))

        def configure(crc, bits, msm):
            _native.have_crc24q = (lambda: True) if crc else (lambda: False)
            _native.have_read_bits = (lambda: True) if bits else (lambda: False)
            _native.have_decode_msm = (lambda: True) if msm else (lambda: False)

        configure(False, False, False)
        t_py = _median_of(run, n=3)
        configure(True, False, False)
        t_crc = _median_of(run, n=3)
        configure(True, True, False)
        t_bits = _median_of(run, n=3)
        configure(True, True, True)
        t_all = _median_of(run, n=3)
        n_msgs = len(list(r3.iter_messages(BytesIO(data))))
        print(f"  iter_messages (CRC + decode), {n_msgs} msgs")
        print(f"    pure-Python              {t_py*1e3:7.2f} ms   1.00x")
        print(f"    + native crc24q          {t_crc*1e3:7.2f} ms   {t_py/t_crc:5.2f}x")
        print(f"    + native _bits           {t_bits*1e3:7.2f} ms   {t_py/t_bits:5.2f}x")
        print(f"    + native decode_msm      {t_all*1e3:7.2f} ms   {t_py/t_all:5.2f}x")

    _force_native(True)


# ----------------------------------------------------------------------
# LAMBDA ILS.
# ----------------------------------------------------------------------


def _lambda_case(n: int, seed: int, scale: float = 0.04, corr: float = 0.01,
                 noise_cycles: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a (a_float, Q) pair sized for a multi-GNSS DD vector.

    n is the number of independent ambiguities. A real dual-frequency
    multi-GNSS RTK epoch can produce 30-40 ambiguities, so we test up
    to that range. Q is built as `scale*I + corr*1` so the matrix is
    SPD but not diagonal — closer to what the joint LSQ in
    `rtk_fix` actually produces.
    """
    rng = np.random.default_rng(seed)
    truth = rng.integers(-5, 6, size=n)
    Q = scale * np.eye(n) + corr * np.ones((n, n))
    a_float = truth + rng.normal(0.0, noise_cycles, size=n)
    return np.ascontiguousarray(a_float, dtype=float), np.ascontiguousarray(Q, dtype=float)


def bench_lambda() -> None:
    print("\n== LAMBDA ILS (well-conditioned: clean RTK geometry) ==")
    cases = [
        ("n=5  L1-only GPS",   5),
        ("n=10 multi-GNSS L1", 10),
        ("n=20 dual-freq",     20),
        ("n=30 multi-GNSS",    30),
        ("n=40 multi-GNSS",    40),
    ]
    for label, n in cases:
        a, Q = _lambda_case(n, seed=n)
        _force_native(False)
        t_py = _median_of(
            lambda a=a, Q=Q: la.integer_least_squares(a, Q, n_cands=2),
            n=3 if n < 30 else 1,
        )
        _force_native(True)
        t_cpp = _median_of(
            lambda a=a, Q=Q: la.integer_least_squares(a, Q, n_cands=2),
            n=5,
        )
        print(f"  {label:22s}  py {t_py*1e3:8.2f} ms    cpp {t_cpp*1e3:8.2f} ms    "
              f"{t_py/t_cpp:6.1f}x")

    # Ill-conditioned cases: noisier float ambiguities + a more
    # correlated Q. These are what weak-geometry RTK (low elevation
    # cutoff, urban canyon, fresh L1 phase) produces, and they force
    # the branch-and-bound search to actually explore.
    print("\n== LAMBDA ILS (ill-conditioned: weak-geometry RTK) ==")
    hard_cases = [
        ("n=8  noisy",   8,  0.5),
        ("n=12 noisy",  12,  0.5),
        ("n=15 noisy",  15,  0.6),
        ("n=20 noisy",  20,  0.5),
    ]
    for label, n, noise in hard_cases:
        a, Q = _lambda_case(n, seed=n + 100, scale=0.2, corr=0.1,
                            noise_cycles=noise)
        _force_native(False)
        try:
            t_py = _median_of(
                lambda a=a, Q=Q: la.integer_least_squares(
                    a, Q, n_cands=2, max_nodes=10_000_000),
                n=1,
            )
        except la.ILSAborted:
            t_py = float("nan")
        _force_native(True)
        t_cpp = _median_of(
            lambda a=a, Q=Q: la.integer_least_squares(
                a, Q, n_cands=2, max_nodes=10_000_000),
            n=3,
        )
        if t_py != t_py:  # NaN
            print(f"  {label:22s}  py        ABORT    cpp {t_cpp*1e3:8.2f} ms")
        else:
            print(f"  {label:22s}  py {t_py*1e3:8.2f} ms    cpp {t_cpp*1e3:8.2f} ms    "
                  f"{t_py/t_cpp:6.1f}x")

    # End-to-end rtk_fix: this is what real callers see, because each
    # epoch of an RTK replay calls integer_least_squares once.
    from rinexpy.lambda_ar import lambda_resolve
    a, Q = _lambda_case(15, seed=7)
    _force_native(False)
    t_py = _median_of(lambda a=a, Q=Q: lambda_resolve(a, Q, ratio_threshold=3.0))
    _force_native(True)
    t_cpp = _median_of(lambda a=a, Q=Q: lambda_resolve(a, Q, ratio_threshold=3.0))
    print(f"  lambda_resolve(n=15)    py {t_py*1e3:8.2f} ms    cpp {t_cpp*1e3:8.2f} ms    "
          f"{t_py/t_cpp:6.1f}x")
    _force_native(True)


def main() -> None:
    if not _native.have_crc24q():
        print("rinexpy_native is missing. From the repo root:")
        print("  uv pip install -e ./native")
        return
    print(f"rinexpy_native available: crc24q={_native.have_crc24q()}, "
          f"read_bits={_native.have_read_bits()}, "
          f"lambda_ils={_native.have_lambda_ils()}, "
          f"decode_msm={_native.have_decode_msm()}")
    bench_rtcm3()
    bench_lambda()


if __name__ == "__main__":
    main()
