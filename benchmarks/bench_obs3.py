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


def time_it(fn, *args, n: int = 5, **kwargs) -> float | None:
    """Run ``fn(*args, **kwargs)`` ``n`` times; return median seconds, or
    None if any call raised."""
    times: list[float] = []
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
    return statistics.median(times)


def benchmark(fixture: str, *, n: int = 5) -> None:
    """Time ``rinexpy.load`` and ``georinex.load`` on a single fixture."""
    fn = DATA / fixture
    if not fn.is_file():
        print(f"  [skip] {fixture} not present")
        return
    t_gx = time_it(georinex.load, fn, n=n)
    t_rp = time_it(rinexpy.load, fn, n=n)
    if t_rp is None:
        print(f"  {fixture:55s}  rinexpy ERROR")
        return
    if t_gx is None:
        print(
            f"  {fixture:55s}  georinex CRASHED   "
            f"rinexpy {t_rp*1000:7.1f} ms"
        )
        return
    speedup = t_gx / t_rp if t_rp > 0 else float("inf")
    print(
        f"  {fixture:55s}  georinex {t_gx*1000:7.1f} ms   "
        f"rinexpy {t_rp*1000:7.1f} ms   {speedup:5.2f}x"
    )


def main() -> None:
    print(f"# rinexpy {rinexpy.__version__} vs georinex {georinex.__version__}")
    print(f"# medians of {5} runs each, file size shown for context\n")

    print("== RINEX 2 OBS ==")
    for f in [
        "demo.10o",
        "minimal2.10o",
        "ab430140.18o.zip",
        "ac660270.18o.Z",
    ]:
        benchmark(f)

    print("\n== RINEX 3 OBS ==")
    for f in [
        "obs3.01gage.10o",
        "obs3.05gage.19o",
        "ABMF00GLP_R_20181330000_01D_30S_MO.zip",
        "CEDA00USA_R_20182100000_23H_15S_MO.rnx.gz",
        "CEBR00ESP_R_20182000000_01D_30S_MO.crx.gz",
    ]:
        benchmark(f)

    print("\n== RINEX 2 NAV ==")
    for f in ["demo.10n", "ab422100.18n", "ceda2100.18e", "brdc2800.15n"]:
        benchmark(f)

    print("\n== RINEX 3 NAV ==")
    for f in [
        "demo_nav3.17n",
        "VILL00ESP_R_20181700000_01D_MN.rnx.gz",
        "ELKO00USA_R_20182100000_01D_MN.rnx.gz",
    ]:
        benchmark(f, n=3)

    print("\n== SP3 ==")
    for f in ["example1.sp3a", "igs19362.sp3c", "minimal.sp3d"]:
        benchmark(f)


if __name__ == "__main__":
    main()
