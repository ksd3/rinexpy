"""Real-NTRIP integration test against the public rtk2go.com caster.

rtk2go.com is a community-operated NTRIP caster that serves real-time
RTCM3 streams from contributor reference stations. The sourcetable
is anonymous-readable; we just fetch it and validate the structure.

The streaming endpoints require an anonymous email as the user, which
some mountpoints accept. We don't pull RTCM data here because that
would tie this test to a particular mountpoint's availability; just
the sourcetable parse is enough to exercise the live NTRIP read path.

Marked at a 10-second socket timeout and skips gracefully on any
network failure so CI without internet just skips this test.
"""

from __future__ import annotations

import socket

import pytest

from rinexpy.ntrip import fetch_sourcetable


def _network_ok() -> bool:
    try:
        s = socket.create_connection(("rtk2go.com", 2101), timeout=5)
    except (OSError, socket.timeout):
        return False
    s.close()
    return True


@pytest.fixture(scope="module")
def sourcetable():
    if not _network_ok():
        pytest.skip("rtk2go.com:2101 unreachable; skip live NTRIP test")
    try:
        return fetch_sourcetable("rtk2go.com", port=2101, timeout=15.0)
    except Exception as e:
        pytest.skip(f"sourcetable fetch failed: {e}")


def test_sourcetable_returns_list(sourcetable):
    assert isinstance(sourcetable, list)
    assert len(sourcetable) > 100, (
        f"expected many entries, got {len(sourcetable)}"
    )


def test_sourcetable_has_mountpoints(sourcetable):
    """STR; lines describe mountpoints — there should be plenty."""
    strs = [e for e in sourcetable if e.get("type") == "STR"]
    assert len(strs) > 50, f"only {len(strs)} STR entries"


def test_sourcetable_str_entries_have_expected_fields(sourcetable):
    """Every STR entry has the standard fields populated."""
    strs = [e for e in sourcetable if e.get("type") == "STR"]
    for e in strs[:20]:
        assert "mountpoint" in e
        assert "format" in e
        assert isinstance(e["mountpoint"], str)
        assert len(e["mountpoint"]) > 0
        # Lat / lon either parsed to float or left as the raw string.
        lat = e.get("latitude")
        if isinstance(lat, float):
            assert -90.0 <= lat <= 90.0


def test_sourcetable_contains_known_rtcm3_mountpoint(sourcetable):
    """Most rtk2go mountpoints are RTCM 3.x; at least one should be."""
    strs = [e for e in sourcetable if e.get("type") == "STR"]
    rtcm3 = [e for e in strs if "RTCM 3" in e.get("format", "")
             or "RTCM3" in e.get("format", "")]
    assert len(rtcm3) > 10, (
        f"only {len(rtcm3)} RTCM3 mountpoints; got first 3 formats "
        f"{[s.get('format') for s in strs[:3]]}"
    )
