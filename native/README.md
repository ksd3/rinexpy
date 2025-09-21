# rinexpy-native

Optional C++ acceleration for [`rinexpy`](https://github.com/ksd3/rinexpy).

A single compiled extension (`rinexpy_native._ext`) that ships a
handful of hot-loop kernels in C++17. The rinexpy modules pick them up
transparently — the public Python API is unchanged whether this is
installed or not.

Kernels currently shipped:

| kernel                | replaces                                     | typical speedup |
|-----------------------|----------------------------------------------|-----------------|
| `decode_obs_batch`    | the OBS3 fixed-width inner decoder           | ~40x vs georinex on a 23h file |
| `crc24q`              | `rinexpy.rtcm3.crc24q`                       | ~150x (510 MB/s) |
| `read_bits`           | `rinexpy.rtcm3._bits` (MSB-first bit cursor) | ~9x; with `_bits` alone, `iter_messages` ~5-6x |
| `decode_msm`          | `rinexpy.rtcm3._decode_msm_header` (full MSM frame body) | end-to-end `iter_messages` ~9x |
| `lambda_ils`          | `rinexpy.lambda_ar.integer_least_squares`    | 30-220x depending on n |
| `interpolate_sp3_lagrange` | `rinexpy.interp.interpolate_sp3` (order-N Lagrange) | ~43x on 1000 queries x 32 SVs |
| `decode_lnav_subframe` | `rinexpy.gps_lnav.decode_lnav_subframe{1,2,3}` | ~3.6x per subframe (end-to-end via public API) |
| `decode_beidou_d1_sf1` | `rinexpy.beidou.decode_d1_subframe1` | ~3.2x per subframe |
| `decode_beidou_d2_page1` | `rinexpy.beidou.decode_d2_page1` | ~3.2x per subframe |
| `kalman_scalar_update_sparse` | `StaticPPPFilter._scalar_update` (all 3 variants) | ~5.4x on 1000 updates (n=24) |
| `apply_ssr_orbit_correction` | `RealtimeOrbitClock.apply_orbit_correction` | ~10.5x on 3200 corrections |
| `apply_ssr_clock_correction` | `RealtimeOrbitClock.apply_clock_correction` | similar |

## Separate package

rinexpy stays pure-Python so a clone plus `uv sync` works on any
Python 3.11+ on any OS without a compiler. Some users want extra
speed at the cost of a compiled wheel; that's what this package
is for.

From the parent repo:

```sh
uv sync --extra native       # resolves this package from ./native/
```

Or build it standalone (see [Build](#build) below).

## Performance

OBS3 reader:

| OBS3 path                 | Time (23h 15s OBS3 file) | vs georinex |
|---------------------------|--------------------------|-------------|
| `georinex` baseline       | ~1100 ms                 | 1.0x        |
| `rinexpy` (pure Python)   | ~83 ms                   | 13x         |
| `rinexpy[jit]`            | ~44 ms                   | 25x         |
| `rinexpy[native]`         | ~25-30 ms                | ~40x        |

RTCM3 (RTKLIB GMSD7 multi-GNSS capture, 1143 messages, 1028 of which
are MSM4 / MSM7 frames):

|                                 | pure-Python | with native | speedup |
|---------------------------------|-------------|-------------|---------|
| `crc24q(256 KB)`                | 78 ms       | 0.49 ms     | 160x    |
| `iter_messages(check_crc=True)` | 217 ms      | 24 ms       | 9.0x    |

Walking the dispatch stack one kernel at a time on the same capture:

| layer                  | wall   | speedup |
|------------------------|--------|---------|
| pure-Python            | 217 ms | 1.00x   |
| + native `crc24q`      | 138 ms | 1.58x   |
| + native `_bits`       | 39 ms  | 5.63x   |
| + native `decode_msm`  | 24 ms  | 9.0x    |

LAMBDA integer search (`integer_least_squares`):

| case                       | pure-Python | with native | speedup |
|----------------------------|-------------|-------------|---------|
| n=5  L1-only GPS           | 0.08 ms     | 0.00 ms     | 31x     |
| n=20 dual-freq             | 0.70 ms     | 0.01 ms     | 127x    |
| n=40 multi-GNSS            | 2.50 ms     | 0.01 ms     | 233x    |
| n=20 noisy / weak-geometry | 2.17 ms     | 0.02 ms     | 97x     |

See `benchmarks/bench_native_extra.py` and
`benchmarks/native_extra_last_run.txt` for the verbatim run.

## Build

```sh
cd native/
uv pip install scikit-build-core nanobind cmake
uv pip install -e .
```

Compiles the `_ext` extension and installs the package in editable
mode.

## License

MIT, like the parent project.
