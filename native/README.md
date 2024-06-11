# rinexpy-native

Optional C++ acceleration for [`rinexpy`](https://github.com/ksd3/rinexpy).

A single compiled extension (`rinexpy_native._ext`) that replaces
the OBS3 fixed-width decoder with a C++17 implementation. Once
installed, `rinexpy.obs3` picks it up and uses it transparently.
The rinexpy API doesn't change.

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

| OBS3 path                 | Time (23h 15s OBS3 file) | vs georinex |
|---------------------------|--------------------------|-------------|
| `georinex` baseline       | ~1100 ms                 | 1.0x        |
| `rinexpy` (pure Python)   | ~83 ms                   | 13x         |
| `rinexpy[jit]`            | ~44 ms                   | 25x         |
| `rinexpy[native]`         | ~25-30 ms                | ~40x        |

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
