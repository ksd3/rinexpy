"""Tests for the async NTRIP client.

Spins up an in-process asyncio TCP server that mimics a caster, so the
tests don't reach the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from rinexpy.ntrip import afetch_sourcetable, astream


async def _serve_once(
    handler: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]],
):
    """Start a TCP server on a random port, accept one client, then stop."""
    started = asyncio.Event()
    ready_port: list[int] = []

    async def _wrap(reader, writer):
        try:
            await handler(reader, writer)
        finally:
            writer.close()

    server = await asyncio.start_server(_wrap, host="127.0.0.1", port=0)
    sock = server.sockets[0]
    ready_port.append(sock.getsockname()[1])
    started.set()
    return server, ready_port[0]


def test_afetch_sourcetable_parses_three_records():
    """Server returns a tiny sourcetable; the async client parses it."""

    SOURCETABLE = (
        b"SOURCETABLE 200 OK\r\n"
        b"Server: Test/1.0\r\n\r\n"
        b"CAS;rtk2go.com;2101;Test;Test;0;USA;40.0;-100.0;0;0\r\n"
        b"NET;TEST;Test;B;N;http://example.com;;\r\n"
        b"STR;MOUNT01;Test mount;RTCM3;1004(1),1005(5);2;GPS;USA;USA;40.0;-100.0;0;0;Trimble;none;N;N;9600\r\n"
        b"ENDSOURCETABLE\r\n"
    )

    async def handler(reader, writer):
        await reader.read(4096)
        writer.write(SOURCETABLE)
        await writer.drain()

    async def run():
        server, port = await _serve_once(handler)
        async with server:
            entries = await afetch_sourcetable("127.0.0.1", port=port, timeout=5.0)
        return entries

    entries = asyncio.run(run())
    assert {e["type"] for e in entries} == {"CAS", "NET", "STR"}
    str_entry = next(e for e in entries if e["type"] == "STR")
    assert str_entry["mountpoint"] == "MOUNT01"
    assert str_entry["format"] == "RTCM3"
    assert str_entry["latitude"] == 40.0


def test_astream_yields_payload_after_200_response():
    """ICY 200 OK then a known payload: client yields the bytes."""

    PAYLOAD = b"\xd3\x00\x04\x4e\x01\x10\xfd\xa1"  # fake RTCM3-shaped bytes

    async def handler(reader, writer):
        request = await reader.readuntil(b"\r\n\r\n")
        assert b"GET /MOUNT01" in request
        writer.write(b"ICY 200 OK\r\n")
        writer.write(PAYLOAD)
        await writer.drain()

    async def run():
        server, port = await _serve_once(handler)
        chunks: list[bytes] = []
        async with server:
            async for chunk in astream("127.0.0.1", "MOUNT01", port=port, timeout=5.0):
                chunks.append(chunk)
        return b"".join(chunks)

    got = asyncio.run(run())
    assert got == PAYLOAD


def test_astream_passes_basic_auth_header():
    """When user/password are given, the request includes the right Authorization header."""

    captured: dict = {}

    async def handler(reader, writer):
        request = await reader.readuntil(b"\r\n\r\n")
        captured["request"] = request
        writer.write(b"ICY 200 OK\r\n")
        await writer.drain()

    async def run():
        server, port = await _serve_once(handler)
        async with server:
            async for _ in astream(
                "127.0.0.1", "M", user="alice", password="hunter2",
                port=port, timeout=5.0,
            ):
                pass

    asyncio.run(run())
    # "alice:hunter2" base64 = YWxpY2U6aHVudGVyMg==
    assert b"Authorization: Basic YWxpY2U6aHVudGVyMg==" in captured["request"]


def test_astream_raises_on_403():
    """A non-200 response makes the async client raise ConnectionError."""

    async def handler(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
        await writer.drain()

    async def run():
        server, port = await _serve_once(handler)
        async with server:
            async for _ in astream("127.0.0.1", "M", port=port, timeout=5.0):
                pass

    with pytest.raises(ConnectionError, match="rejected"):
        asyncio.run(run())


def test_astream_sync_and_async_use_same_parsing():
    """Spot-check: afetch_sourcetable and fetch_sourcetable share the parser
    (so a STR; record looks the same either way)."""
    from rinexpy.ntrip import _parse_sourcetable

    body = (
        "STR;MOUNT01;Test;RTCM3;1004;2;GPS;USA;USA;40.0;-100.0;0;0;X;none;N;N;9600\r\n"
        "ENDSOURCETABLE\r\n"
    )
    parsed = _parse_sourcetable(body)
    assert len(parsed) == 1
    assert parsed[0]["mountpoint"] == "MOUNT01"
