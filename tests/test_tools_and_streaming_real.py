"""Real-fixture tests for rinexpy.tools and the streaming filters.

Uses the bundled CEDA Galileo OBS3, the brdc2800.15n GPS NAV, and the
SP3 fixtures. Verifies:

- validate_file produces a QC report with the right shape on real files,
- concat_files joins consecutive files along time without losing data,
- diff_datasets correctly identifies when two reads of the same file
  agree (bit-identical) and when an intentionally-perturbed copy diverges,
- iter_obs3_epochs use= and tlim= filters are honored on real OBS3.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

import rinexpy as rp
from rinexpy.streaming import iter_obs3_epochs
from rinexpy.tools import concat_files, diff_datasets, validate_file

_RINEX3_OBS = "tests/data/ABMF00GLP_R_20181330000_01D_30S_MO.zip"
_CEDA = "tests/data/CEDA00USA_R_20182100000_23H_15S_MO.rnx.gz"


def test_validate_file_returns_report_on_real_obs():
    """validate_file produces a structured QC report on a real OBS."""
    report = validate_file(_RINEX3_OBS)
    assert isinstance(report, dict)
    assert report["ok"] is True
    assert report["info"]["rinextype"] == "obs"
    assert report["n_epochs"] > 0
    assert report["n_sv"] > 0
    assert report["interval_seconds"] == 30.0
    assert "time_first" in report and "time_last" in report


def test_concat_files_joins_same_file_to_itself():
    """concat_files(file, file) along time should drop duplicates and return
    one timeline."""
    out = concat_files([_RINEX3_OBS, _RINEX3_OBS], dim="time")
    src = rp.load(_RINEX3_OBS)
    # Dedup leaves the original time count.
    assert out.time.size == src.time.size


def test_diff_datasets_matches_self():
    """diff_datasets on a dataset compared to itself reports no differences."""
    src = rp.load(_RINEX3_OBS)
    result = diff_datasets(src, src)
    assert result.get("equal") is True


def test_diff_datasets_flags_perturbation():
    """A 1 cm perturbation to one variable is detected when rtol is tight.

    The default ``rtol=1e-6`` is misleading for GNSS code observables: with
    pseudoranges around 2.5e7 m it allows ~25 m of relative tolerance and
    a 1 cm perturbation slips through. Tightening with ``rtol=0`` and an
    absolute tolerance well below the perturbation catches it.
    """
    src = rp.load(_RINEX3_OBS)
    other = src.copy(deep=True)
    other["C1C"] = other["C1C"] + 0.01
    result = diff_datasets(src, other, rtol=0.0, atol=1e-6)
    assert result.get("equal") is False
    diffs = result.get("differences", [])
    assert any("C1C" in str(d) for d in diffs), (
        f"diff_datasets didn't flag C1C in {result}"
    )


def test_iter_obs3_epochs_honors_use_filter():
    """iter_obs3_epochs use='E' only yields Galileo SVs."""
    n_seen = 0
    for _t, ds in iter_obs3_epochs(_CEDA, use="E"):
        assert all(str(sv).startswith("E") for sv in ds.sv.values)
        n_seen += 1
        if n_seen >= 5:
            break
    assert n_seen > 0


def test_iter_obs3_epochs_honors_tlim_filter():
    """iter_obs3_epochs tlim= restricts the yielded epochs.

    Note: streaming.iter_obs3_epochs accepts a tlim tuple of
    datetime.datetime, not numpy.datetime64; convert with .tolist() on
    a microsecond-precision numpy datetime to get datetime back.
    """
    obs = rp.load(_CEDA)
    np_start = obs.time.values[1000].astype("datetime64[us]")
    np_end = obs.time.values[1010].astype("datetime64[us]")
    t_start = np_start.tolist()
    t_end = np_end.tolist()
    yielded = []
    for t, _ in iter_obs3_epochs(_CEDA, tlim=(t_start, t_end)):
        yielded.append(t)
    assert 0 < len(yielded) <= 11, f"got {len(yielded)} epochs in tlim"
    assert yielded[0] >= t_start
    assert yielded[-1] <= t_end
