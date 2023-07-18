"""Tests for shared helpers in :mod:`rinexpy._common`."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from rinexpy._common import (
    check_unique_times,
    determine_time_system,
    fortran_float,
    globber,
)


def test_fortran_float_with_d():
    assert fortran_float("1.23D+04") == 12300.0
    assert fortran_float("-2.5D-3") == -0.0025


def test_fortran_float_without_d():
    assert fortran_float("1.5e2") == 150.0
    assert fortran_float("3.14") == 3.14


def test_fortran_float_invalid():
    with pytest.raises(ValueError):
        fortran_float("not a number")


def test_check_unique_times_ok():
    assert check_unique_times(np.array([1, 2, 3])) is True


def test_check_unique_times_dup(caplog):
    caplog.set_level(logging.ERROR)
    assert check_unique_times(np.array([1, 1, 2])) is False
    assert "unique" in caplog.text.lower()


def test_determine_time_system_gps():
    assert determine_time_system({"systems": "G"}) == "GPS"


def test_determine_time_system_galileo():
    assert determine_time_system({"systems": "E"}) == "GAL"


def test_determine_time_system_unknown():
    with pytest.raises(ValueError):
        determine_time_system({"systems": "Z"})


def test_determine_time_system_mixed_via_header():
    hdr = {
        "systems": "M",
        "TIME OF FIRST OBS": " " * 48 + "GPS" + " " * 9,
    }
    assert determine_time_system(hdr) == "GPS"


def test_globber_single_file(tmp_path):
    f = tmp_path / "x.10o"
    f.write_text("hi")
    assert globber(f, "*o") == [f]


def test_globber_directory(tmp_path):
    a = tmp_path / "a.10o"
    a.write_text("a")
    b = tmp_path / "b.10n"
    b.write_text("b")
    matches = sorted(globber(tmp_path, "*o"))
    assert matches == [a]


def test_globber_multiple_patterns(tmp_path):
    a = tmp_path / "a.10o"
    a.write_text("a")
    b = tmp_path / "b.10n"
    b.write_text("b")
    matches = sorted(globber(tmp_path, ["*o", "*n"]))
    assert matches == [a, b]
