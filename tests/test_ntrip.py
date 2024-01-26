"""Tests for the NTRIP sourcetable parser.

The actual TCP/TLS network paths aren't covered (would require a live
caster); we test the parser against a captured sourcetable fragment and
the auth-header builder.
"""

from __future__ import annotations

from rinexpy.ntrip import _basic_auth, _parse_sourcetable

_SOURCETABLE = """\
SOURCETABLE 200 OK
Server: NTRIP test
Content-Type: text/plain

CAS;example.org;2101;CASTER;Operator;0;USA;40.0;-74.0;0.0.0.0;0;none
NET;EXAMPLE_NET;EXAMPLE;B;none;http://example.org;none;contact@example.org;none
STR;MOUNT01;MOUNT01;RTCM 3.3;1004(1),1019(0.1);2;GPS;EXAMPLE_NET;USA;40.0;-74.0;0;0;Generic;none;B;N;9600
STR;MOUNT02;MOUNT02;RTCM 3.2;1077(1),1087(1);2;GPS+GLO;EXAMPLE_NET;USA;41.0;-75.0;0;0;Other;none;N;N;19200
ENDSOURCETABLE
"""


def test_basic_auth_encoding():
    assert _basic_auth("user", "pass") == "dXNlcjpwYXNz"


def test_parse_sourcetable_extracts_mountpoints():
    entries = _parse_sourcetable(_SOURCETABLE)
    str_entries = [e for e in entries if e["type"] == "STR"]
    assert len(str_entries) == 2
    assert str_entries[0]["mountpoint"] == "MOUNT01"
    assert str_entries[0]["format"] == "RTCM 3.3"
    assert str_entries[0]["latitude"] == 40.0
    assert str_entries[0]["longitude"] == -74.0


def test_parse_sourcetable_keeps_cas_and_net():
    entries = _parse_sourcetable(_SOURCETABLE)
    types = [e["type"] for e in entries]
    assert "CAS" in types
    assert "NET" in types


def test_parse_sourcetable_stops_at_endsourcetable():
    body = (
        _SOURCETABLE + "STR;EXTRA;...;...;...;...;...;...;...;...;...;...;...;...;...;...;...;...\n"
    )
    # The parser should ignore the trailing extra line.
    entries = _parse_sourcetable(body)
    str_entries = [e for e in entries if e["type"] == "STR"]
    assert len(str_entries) == 2
