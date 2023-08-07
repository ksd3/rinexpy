"""Tests for the version / filetype / system detectors."""

from __future__ import annotations

import io

import pytest

from rinexpy._version import (
    detect_filetype,
    detect_systems,
    first_nonblank_line,
    rinex_version,
)


def _padded(s: str) -> str:
    """Pad ``s`` to 80 chars on the right with spaces (RINEX header convention)."""
    return s + " " * (80 - len(s))


def test_rinex_version_obs2():
    line = "     2.11           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE"
    v, is_crinex = rinex_version(line)
    assert v == 2.11
    assert is_crinex is False


def test_rinex_version_nav3():
    line = "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE"
    v, is_crinex = rinex_version(line)
    assert v == 3.04
    assert is_crinex is False


def test_rinex_version_crinex():
    line = "1.0                 COMPACT RINEX FORMAT                    CRINEX VERS   / TYPE"
    v, is_crinex = rinex_version(line)
    assert v == 1.0
    assert is_crinex is True


def test_rinex_version_sp3a():
    assert rinex_version("#aP2019  1  1  0  0  0.00000000   ") == ("sp3a", False)


def test_rinex_version_sp3d():
    assert rinex_version("#dP2019  1  1  0  0  0.00000000   ") == ("sp3d", False)


def test_rinex_version_bad_sp3():
    with pytest.raises(ValueError, match="SP3 versions"):
        rinex_version("#xP2019  1  1")


def test_rinex_version_too_short():
    with pytest.raises(ValueError):
        rinex_version("a")


def test_rinex_version_wrong_type():
    with pytest.raises(TypeError):
        rinex_version(b"not a string")  # type: ignore[arg-type]


def test_rinex_version_corrupt_marker():
    # Pad to 80 chars but with a bogus marker in cols 60-80.
    line = "     2.11           OBSERVATION DATA    M (MIXED)           BOGUSBOGUSBOGUSBOGUS"
    assert len(line) == 80
    with pytest.raises(ValueError, match="corrupted"):
        rinex_version(line)


def test_detect_filetype_obs():
    line = _padded("     2.11           OBSERVATION DATA    M (MIXED)")
    assert detect_filetype(line, 2.11) == "obs"


def test_detect_filetype_nav():
    line = _padded("     2.11           N: GPS NAV DATA")
    assert detect_filetype(line, 2.11) == "nav"


def test_detect_filetype_sp3():
    assert detect_filetype("#aP", "sp3a") == "sp3"


def test_detect_systems_rinex2_nav():
    line = _padded("     2.11           N: GPS NAV DATA")
    assert detect_systems(line, 2.11) == "G"


def test_detect_systems_rinex2_glonass():
    line = _padded("     2.11           G: GLONASS NAV DATA")
    assert detect_systems(line, 2.11) == "R"


def test_detect_systems_rinex3():
    line = _padded("     3.04           N: GNSS NAV DATA    M: MIXED")
    assert detect_systems(line, 3.04) == "M"


def test_first_nonblank_line():
    s = io.StringIO("\n\n   \nhello world\nignored\n")
    assert first_nonblank_line(s).strip() == "hello world"


def test_first_nonblank_line_max_lines_too_low():
    with pytest.raises(ValueError):
        first_nonblank_line(io.StringIO("x\n"), max_lines=0)


def test_first_nonblank_line_only_blanks():
    with pytest.raises(ValueError):
        first_nonblank_line(io.StringIO("\n\n\n\n"))
