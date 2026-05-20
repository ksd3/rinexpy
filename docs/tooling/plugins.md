# Plugin system

For file formats that rinexpy does not natively support, the `plugins`
module lets external packages register their own readers under a Python
entry-point group. The dispatcher tries the built-in `load` first and
falls back to plugins on failure.

The plugin system is intentionally minimal: a single entry-point group,
a single function signature, no per-format configuration.

## Registering a plugin

A third-party package declares an entry point under the
`rinexpy.readers` group in its `pyproject.toml`:

```toml
[project.entry-points."rinexpy.readers"]
ublox_log = "my_package.readers:read_ublox_log"
```

The function on the right (here `my_package.readers:read_ublox_log`) is
the reader. Its signature must be:

```python
def read_ublox_log(path: Path | str) -> xr.Dataset:
    """Read a u-blox log file and return an xarray.Dataset."""
    ...
```

The function should raise `ValueError` if the file is not in its
recognised format.

## Discovering plugins

```python
from rinexpy.plugins import discover_plugins

readers = discover_plugins()
print(readers)
# {'ublox_log': <_PluginReader object>}
```

The function uses `importlib.metadata.entry_points` to enumerate the
group. The keys are the entry-point names; the values are wrapper
objects that hold the loaded callable.

## Loading with fallback

```python
from rinexpy.plugins import load_with_plugins

ds = load_with_plugins("file.unknown")
```

The function:

1. Tries `rinexpy.load("file.unknown")` first.
2. If `load` raises (typically `ValueError` for an unrecognised file), it
   iterates the discovered plugins and calls each one until one returns a
   Dataset.
3. If every plugin also raises, the original `load` error is re-raised.

This pattern lets you point existing code at unrecognised file types
without changing your loaders:

```python
# Before:
ds = rp.load(path)

# With plugin fallback:
from rinexpy.plugins import load_with_plugins
ds = load_with_plugins(path)
```

## Plugin authoring guide

A minimal plugin package looks like:

```
my_rinexpy_plugin/
├── pyproject.toml
├── my_rinexpy_plugin/
│   ├── __init__.py
│   └── reader.py
```

`pyproject.toml`:

```toml
[project]
name = "my-rinexpy-plugin"
version = "0.1.0"
dependencies = ["xarray>=2024.1"]

[project.entry-points."rinexpy.readers"]
my_format = "my_rinexpy_plugin.reader:read_my_format"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

`my_rinexpy_plugin/reader.py`:

```python
from pathlib import Path
import numpy as np
import xarray as xr

def read_my_format(path) -> xr.Dataset:
    path = Path(path)
    if path.suffix != ".myfmt":
        raise ValueError(f"not a .myfmt file: {path}")
    # ... parse the file ...
    return xr.Dataset({"data": ("time", values)}, coords={"time": times})
```

After installing the package alongside rinexpy (`uv pip install
my-rinexpy-plugin`), the entry point appears in `discover_plugins`
automatically.

## When to write a plugin

The plugin path is intended for genuinely external file formats: vendor
binary protocols that are not in scope for the rinexpy core, internal
formats specific to your organisation, or experimental readers that you
want to keep separate.

For standard GNSS formats (RINEX, SP3, RTCM, NMEA, UBX, SBF, NovAtel,
BINEX, GW10), the built-in readers are the right answer. If you need a
fix to one of them, file an issue rather than writing a plugin around it.

## Related pages

- [RINEX observation files](../formats/rinex-obs.md): the built-in dispatch.
- [Receiver binary formats](../formats/receiver-binary.md): the in-tree binary readers.
