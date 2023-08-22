"""Smoke tests for the matplotlib plotting helpers.

Plots are not visually verified; the assertions just check that the
helper runs end-to-end and that the right number of figures were
created. The tests are skipped if matplotlib is not installed.
"""

from __future__ import annotations

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # non-interactive backend
plt = pytest.importorskip("matplotlib.pyplot")

import rinexpy as rp  # noqa: E402
from rinexpy.plots import (  # noqa: E402
    navtimeseries,
    obstimeseries,
    receiver_locations,
    timeseries,
)

from .conftest import fixture  # noqa: E402


@pytest.fixture(autouse=True)
def _close_all():
    yield
    plt.close("all")


def test_obstimeseries_runs():
    obs = rp.load(fixture("demo.10o"))
    obstimeseries(obs)
    assert len(plt.get_fignums()) >= 1


def test_obstimeseries_silent_on_wrong_type():
    obstimeseries("not a dataset")  # type: ignore[arg-type]
    assert plt.get_fignums() == []


def test_navtimeseries_runs():
    pytest.importorskip("pymap3d")
    nav = rp.load(fixture("demo.10n"))
    navtimeseries(nav)
    # GPS records present -> at least one figure.
    assert len(plt.get_fignums()) >= 1


def test_timeseries_dispatch_obs():
    obs = rp.load(fixture("demo.10o"))
    timeseries(obs)
    assert len(plt.get_fignums()) >= 1


def test_timeseries_dispatch_nav():
    pytest.importorskip("pymap3d")
    nav = rp.load(fixture("demo.10n"))
    timeseries(nav)
    assert len(plt.get_fignums()) >= 1


def test_receiver_locations_smoke():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {"lat": [40.0, 50.0], "lon": [-3.0, 7.0], "interval": [30.0, 5.0]},
        index=["site_a", "site_b"],
    )
    receiver_locations(df)
    assert len(plt.get_fignums()) == 1


def test_receiver_locations_silent_on_wrong_type():
    receiver_locations({"not": "a frame"})  # type: ignore[arg-type]
    assert plt.get_fignums() == []
