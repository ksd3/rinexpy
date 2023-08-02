"""Compare rinexpy.load vs georinex.load on the bundled corpus.

Run from the repo root:

    uv run python benchmarks/bench_obs3.py
"""

from __future__ import annotations

import statistics
import time
import warnings
from pathlib import Path

import georinex
import rinexpy

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"


def time_it(
    fn, *args, n: int = 3, max_seconds: float = 30.0, **kwargs
) -> float | None:
    """Run ``fn(*args, **kwargs)`` up to ``n`` times; return median seconds.

    Returns None if the first call raised. Stops early once total elapsed
    time exceeds ``max_seconds`` and reports the median of what we got.
    """
    times: list[float] = []
    total = 0.0
    for _ in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            try:
                fn(*args, **kwargs)
            except Exception:
                return None
            t1 = time.perf_counter()
        times.append(t1 - t0)
        total += t1 - t0
        if total > max_seconds:
            break
    return statistics.median(times) if times else None


def benchmark(fixture: str, *, n: int = 3, max_seconds: float = 30.0) -> None:
    """Time ``rinexpy.load`` and ``georinex.load`` on a single fixture."""
    fn = DATA / fixture
    if not fn.is_file():
        print(f"  [skip] {fixture} not present")
        return
    size_kb = fn.stat().st_size / 1024
    t_gx = time_it(georinex.load, fn, n=n, max_seconds=max_seconds)
    t_rp = time_it(rinexpy.load, fn, n=n, max_seconds=max_seconds)
    if t_rp is None:
        print(f"  {fixture:55s} {size_kb:7.0f} KB   rinexpy ERROR")
        return
    if t_gx is None:
        print(
            f"  {fixture:55s} {size_kb:7.0f} KB   "
            f"georinex CRASHED   rinexpy {t_rp*1000:7.1f} ms"
        )
        return
    speedup = t_gx / t_rp if t_rp > 0 else float("inf")
    print(
        f"  {fixture:55s} {size_kb:7.0f} KB   "
        f"georinex {t_gx*1000:8.1f} ms   "
        f"rinexpy {t_rp*1000:8.1f} ms   {speedup:5.2f}x"
    )


def main() -> None:
    print(f"# rinexpy {rinexpy.__version__} vs georinex {georinex.__version__}")
    print("# medians of up to 3 runs each (capped at 30s wall-clock per file)")
    print(f"# {'fixture':55s} {'size':8s} {'georinex':14s} {'rinexpy':14s} speedup\n")

    print("== RINEX 2 OBS ==")
    for f in [
        "minimal2.10o",
        "demo.10o",
        "rinex2onesat.10o",
        "ab430140.18o.zip",
        "ac660270.18o.Z",
    ]:
        benchmark(f)

    print("\n== RINEX 3 OBS ==")
    for f in [
        "obs3.01gage.10o",
        "ABMF00GLP_R_20181330000_01D_30S_MO.zip",
    ]:
        benchmark(f)

    print("\n== RINEX 2 NAV ==")
    for f in ["demo.10n", "ab422100.18n", "ceda2100.18e", "brdc2800.15n"]:
        benchmark(f)

    print("\n== RINEX 3 NAV ==")
    for f in [
        "demo_nav3.17n",
        "ELKO00USA_R_20182100000_01D_MN.rnx.gz",
    ]:
        benchmark(f, max_seconds=60.0)

    print("\n== SP3 ==")
    for f in ["example1.sp3a", "igs19362.sp3c", "minimal.sp3d"]:
        benchmark(f)


if __name__ == "__main__":
    main()
