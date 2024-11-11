"""Real-file integration tests for the Furuno GW-10 framer + SBAS extractor.

Uses the RTKLIB sample ``gw10_20110121.sbas`` capture (256 KB of GW-10
binary output from a Furuno GW-10 III receiver). Downloaded by the
receiver-fixtures test; reused here.
"""

from __future__ import annotations

import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import pytest

from rinexpy.gw10 import (
    SYNC,
    decode_sbas,
    iter_frames,
    iter_sbas_messages,
)

_CACHE = Path("/tmp/igs_real_cache")
_URL = (
    "https://raw.githubusercontent.com/tomojitakasu/RTKLIB/"
    "rtklib_2.4.3/test/data/rcvraw/gw10_20110121.sbas"
)


def _capture() -> Path:
    path = _CACHE / "gw10_20110121.sbas"
    if path.exists() and path.stat().st_size > 1000:
        return path
    _CACHE.mkdir(exist_ok=True)
    try:
        with urllib.request.urlopen(_URL, timeout=60) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception:
        if path.exists():
            path.unlink()
        pytest.skip(f"Cannot reach {_URL}; skip GW-10 real test")
    return path


@pytest.fixture(scope="module")
def gw10_frames():
    path = _capture()
    with open(path, "rb") as f:
        return list(iter_frames(f))


@pytest.fixture(scope="module")
def sbas_messages():
    path = _capture()
    with open(path, "rb") as f:
        return list(iter_sbas_messages(f))


def test_gw10_decodes_thousands_of_frames(gw10_frames):
    """The 256 KB capture has thousands of GW-10 frames."""
    assert len(gw10_frames) > 1000


def test_gw10_frames_have_sane_ids(gw10_frames):
    """Every frame's ID is one we know about (we only emit known IDs)."""
    ids = Counter(f["id"] for f in gw10_frames)
    # The known IDs from gw10.py's _MSG_LENGTHS table.
    known = {0x02, 0x03, 0x06, 0x07, 0x08, 0x20, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27}
    assert set(ids) <= known, f"unexpected IDs: {set(ids) - known}"


def test_gw10_sbas_messages_dominate_the_capture(gw10_frames):
    """gw10_20110121.sbas is named for its SBAS content; SBAS frames should be common."""
    n_sbas = sum(1 for f in gw10_frames if f["id"] == 0x03)
    assert n_sbas > 100, f"only {n_sbas} SBAS frames in capture"


def test_gw10_checksums_pass(gw10_frames):
    """Every frame iter_frames yields must already have passed its checksum."""
    assert all(f["checksum_ok"] for f in gw10_frames)


def test_sbas_messages_have_real_preambles(sbas_messages):
    """SBAS L1 messages use a rotating 8-bit preamble: 0x53, 0x9A, 0xC6."""
    preambles = Counter(m["preamble"] for m in sbas_messages)
    canonical = {0x53, 0x9A, 0xC6}
    # The canonical preambles should dominate (>90% of messages).
    canonical_count = sum(preambles[p] for p in canonical)
    total = sum(preambles.values())
    assert total > 100
    assert canonical_count / total > 0.9, (
        f"only {canonical_count}/{total} messages start with a canonical "
        f"SBAS preamble; top values: {preambles.most_common(5)}"
    )


def test_sbas_prns_in_geo_band(sbas_messages):
    """SBAS GEO satellites have PRN 120-158."""
    prns = set(m["prn"] for m in sbas_messages)
    out_of_band = [p for p in prns if not (120 <= p <= 158)]
    assert not out_of_band, f"unexpected PRNs: {out_of_band}"


def test_sbas_message_types_in_valid_range(sbas_messages):
    """SBAS L1 message types are 0-63 (6-bit field)."""
    for m in sbas_messages[:100]:
        assert 0 <= m["message_type"] < 64


def test_decode_sbas_truncated_payload():
    """A short payload reports truncated rather than crashing."""
    out = decode_sbas(b"\x00" * 10)
    assert out.get("truncated") is True


def test_sync_constant():
    assert SYNC == 0x8B
