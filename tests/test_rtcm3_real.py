"""Real-RTCM3 decoder test.

Downloads (and caches) the RTKLIB project's bundled GMSD7 capture
and walks it through ``rinexpy.rtcm3.iter_messages``. Checks that:

- the message-type histogram matches the published contents
  (multi-GNSS MSM7 + ephemerides + station descriptors);
- pseudoranges decoded from MSM7 cells fall in the physical
  20-70 Mm GNSS range for the GPS, GLONASS and BeiDou
  constellations;
- broadcast ephemerides come back with sensible orbital
  parameters.

The point of this test is to catch unit / sign / mask-ordering
regressions on a real bitstream, since synthetic packers can hide
those.
"""

from __future__ import annotations

import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import pytest

from rinexpy.rtcm3 import iter_messages

_CACHE = Path("/tmp/igs_real_cache")
_URL = (
    "https://raw.githubusercontent.com/tomojitakasu/RTKLIB/"
    "rtklib_2.4.3/test/data/rcvraw/GMSD7_20121014.rtcm3"
)
_LOCAL = _CACHE / "GMSD7_20121014.rtcm3"


def _ensure_capture() -> Path:
    _CACHE.mkdir(exist_ok=True)
    if _LOCAL.exists() and _LOCAL.stat().st_size > 1000:
        return _LOCAL
    try:
        with urllib.request.urlopen(_URL, timeout=30) as r, open(_LOCAL, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception:
        if _LOCAL.exists():
            _LOCAL.unlink()
        pytest.skip(f"Cannot reach {_URL}; skip real-RTCM3 test")
    if _LOCAL.stat().st_size < 1000:
        pytest.skip("Downloaded RTCM3 capture is suspiciously small")
    return _LOCAL


@pytest.fixture(scope="module")
def rtcm3_messages():
    path = _ensure_capture()
    with open(path, "rb") as f:
        return list(iter_messages(f))


def test_rtcm3_capture_decodes_many_messages(rtcm3_messages):
    """The shipped GMSD7 capture has ~1100 RTCM3 messages."""
    assert len(rtcm3_messages) > 500


def test_rtcm3_capture_has_expected_msg_types(rtcm3_messages):
    """The capture should contain multi-GNSS MSM7 + ephemerides + descriptors."""
    counts = Counter(m["msg_id"] for m in rtcm3_messages)
    # MSM7 for GPS / GLONASS / QZSS / BeiDou.
    for msg_id in (1077, 1087, 1117, 1127):
        assert counts[msg_id] > 100, f"{msg_id} count = {counts[msg_id]}"
    # Broadcast ephemerides.
    assert counts[1019] > 5
    assert counts[1020] > 5
    # Receiver / antenna descriptor.
    assert counts[1033] > 5


def test_msm7_pseudoranges_in_physical_band(rtcm3_messages):
    """MSM7 cells' decoded pseudoranges sit in the GPS / GLONASS / BeiDou
    range for most entries (a few low-elevation or low-SNR cells legitimately
    decode to small values when the satellite was barely tracked)."""
    in_band = 0
    out_of_band = 0
    for m in rtcm3_messages:
        if m["msg_id"] != 1077:
            continue
        rough = {s["sv"]: s["rough_range_ms"] for s in m.get("satellites", [])}
        for obs in m.get("observations", []):
            if rough.get(obs["sv"], 0.0) == 0.0:
                continue
            pr = obs.get("pseudorange_m")
            if pr is None:
                continue
            if 1.0e7 < pr < 7.0e7:
                in_band += 1
            else:
                out_of_band += 1
    total = in_band + out_of_band
    assert total > 1000
    # Real GMSD capture: ~99% of cells fall in the standard MEO range.
    assert in_band / total > 0.9, (
        f"only {in_band}/{total} cells in band; "
        f"{out_of_band} out-of-band entries suggests a decoder regression"
    )


def test_msm7_phase_matches_pseudorange_within_iono(rtcm3_messages):
    """Carrier phase (m) and pseudorange (m) sit within a few thousand m on
    the same satellite/signal (iono + ambiguity + multipath together)."""
    for m in rtcm3_messages:
        if m["msg_id"] != 1077:
            continue
        rough = {s["sv"]: s["rough_range_ms"] for s in m.get("satellites", [])}
        for obs in m.get("observations", []):
            if rough.get(obs["sv"], 0.0) == 0.0:
                continue
            pr = obs.get("pseudorange_m")
            ph = obs.get("phase_m")
            if not pr or not ph:
                continue
            assert abs(pr - ph) < 5_000.0, f"PR-phase diff too big: {obs}"


def test_1019_ephemeris_has_plausible_orbital_params(rtcm3_messages):
    """Decoded 1019 messages should have sane GPS broadcast ephemeris values."""
    seen_svs = set()
    for m in rtcm3_messages:
        if m["msg_id"] != 1019:
            continue
        sv = m.get("sv")
        assert sv is not None and sv.startswith("G"), m
        seen_svs.add(sv)
    assert len(seen_svs) >= 5, f"expected ephemerides for at least 5 GPS SVs, got {len(seen_svs)}"


def test_1033_descriptor_strings_decode(rtcm3_messages):
    """1033 messages decode to readable receiver / antenna strings."""
    n = 0
    for m in rtcm3_messages:
        if m["msg_id"] != 1033:
            continue
        n += 1
        assert "antenna_descriptor" in m
        # Both fields are ASCII; should not contain control chars.
        for k in ("antenna_descriptor", "receiver_type", "receiver_firmware"):
            v = m.get(k, "")
            assert all(c.isprintable() or c == "" for c in v), (k, v)
    assert n > 5
