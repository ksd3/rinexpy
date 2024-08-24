"""Tests for the plugin reader discovery and load_with_plugins fallback."""

from __future__ import annotations

import pytest
import xarray as xr

from rinexpy.plugins import discover_plugins, load_with_plugins


def test_discover_plugins_returns_dict():
    """With no third-party packages installed, discovery returns an empty (or
    near-empty) dict but never raises."""
    out = discover_plugins()
    assert isinstance(out, dict)


def test_load_with_plugins_uses_builtin_for_known_format():
    """A real OBS file is recognised by rinexpy.load; no plugin needed."""
    ds = load_with_plugins("tests/data/demo.10o")
    assert isinstance(ds, xr.Dataset)
    assert "L1" in ds.data_vars


def test_load_with_plugins_falls_back_to_plugin_on_unknown_format(tmp_path):
    """An unrecognised file makes load_with_plugins try the registered plugins."""
    # Create a fake file the built-in loader won't understand.
    bogus = tmp_path / "thing.weird"
    bogus.write_bytes(b"not a real format\n")
    calls = []

    def fake_reader(path):
        calls.append(str(path))
        return xr.Dataset({"answer": ("x", [42])})

    ds = load_with_plugins(bogus, plugin_readers={"weird": fake_reader})
    assert ds["answer"].values.tolist() == [42]
    assert calls == [str(bogus)]


def test_load_with_plugins_re_raises_when_no_plugin_handles_file(tmp_path):
    """If neither the built-in nor any plugin can read the file, the original
    built-in error propagates."""
    bogus = tmp_path / "thing.weird"
    bogus.write_bytes(b"junk\n")

    def fake_reader(path):
        raise ValueError("not my format either")

    with pytest.raises((ValueError, NotImplementedError, OSError)):
        load_with_plugins(bogus, plugin_readers={"weird": fake_reader})


def test_load_with_plugins_tries_plugins_in_order(tmp_path):
    """First plugin to return a dataset wins; later ones aren't called."""
    bogus = tmp_path / "thing.weird"
    bogus.write_bytes(b"junk\n")

    second_called = []

    def first(path):
        return xr.Dataset({"who": ("x", ["first"])})

    def second(path):
        second_called.append(path)
        return xr.Dataset({"who": ("x", ["second"])})

    ds = load_with_plugins(
        bogus, plugin_readers={"first": first, "second": second}
    )
    assert ds["who"].values.tolist() == ["first"]
    assert second_called == []


def test_load_with_plugins_skips_a_plugin_that_raises(tmp_path):
    """A plugin that raises is skipped; the next one is tried."""
    bogus = tmp_path / "thing.weird"
    bogus.write_bytes(b"junk\n")

    def broken(path):
        raise ValueError("broken plugin")

    def good(path):
        return xr.Dataset({"who": ("x", ["good"])})

    ds = load_with_plugins(
        bogus, plugin_readers={"broken": broken, "good": good}
    )
    assert ds["who"].values.tolist() == ["good"]
