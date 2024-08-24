"""Plugin discovery for third-party readers.

External packages can register their own file-format readers under the
``rinexpy.readers`` entry-point group:

.. code-block:: toml

    # In a downstream package's pyproject.toml
    [project.entry-points."rinexpy.readers"]
    my_format = "my_package.readers:load_my_format"

Each registered entry must be a callable ``f(path) -> xarray.Dataset``
that either returns the parsed dataset or raises ``ValueError`` /
``NotImplementedError`` to signal "this format isn't mine, try the next
reader". :func:`load_with_plugins` calls the built-in
:func:`rinexpy.load` first and falls back to the registered plugins on
failure, returning the first dataset any plugin can produce.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

_PluginReader = Callable[..., Any]
_ENTRY_POINT_GROUP = "rinexpy.readers"


def discover_plugins() -> dict[str, _PluginReader]:
    """Return all readers registered under the ``rinexpy.readers`` group.

    Each value is the loaded entry-point callable, ready to invoke. The
    discovery happens at call time, so newly-installed plugins are picked
    up without restarting the process.

    Returns
    -------
    dict
        ``{name: callable}``. Empty if no plugins are installed.
    """
    out: dict[str, _PluginReader] = {}
    for ep in entry_points(group=_ENTRY_POINT_GROUP):
        try:
            out[ep.name] = ep.load()
        except Exception:
            # A broken plugin shouldn't take the whole stack down.
            continue
    return out


def load_with_plugins(
    path,
    *,
    plugin_readers: dict[str, _PluginReader] | None = None,
):
    """Load a file via the built-in reader, falling back to plugins.

    The built-in :func:`rinexpy.load` is tried first. If it raises (for an
    unrecognised format, typically ``ValueError`` or
    ``NotImplementedError``), every registered plugin is tried in turn
    until one returns a dataset. If all plugins also fail, the original
    error is re-raised.

    Parameters
    ----------
    path:
        File path or path-like to load.
    plugin_readers:
        Optional preset mapping of plugin name to callable. Pass an
        explicit dict to skip the entry-point discovery (handy for tests
        and for environments where entry-point loading is expensive).
        Defaults to :func:`discover_plugins`.

    Returns
    -------
    xarray.Dataset
        Whatever the winning reader returned.

    Raises
    ------
    Exception
        The original exception from the built-in :func:`rinexpy.load` if
        no plugin can handle the file either.
    """
    from . import load as _builtin_load

    try:
        return _builtin_load(path)
    except (ValueError, NotImplementedError, OSError) as builtin_err:
        readers = plugin_readers if plugin_readers is not None else discover_plugins()
        for name, reader in readers.items():
            try:
                return reader(path)
            except (ValueError, NotImplementedError, OSError):
                continue
            except Exception:
                # Any other plugin error: skip and try the next.
                continue
        raise builtin_err


__all__ = [
    "discover_plugins",
    "load_with_plugins",
]
