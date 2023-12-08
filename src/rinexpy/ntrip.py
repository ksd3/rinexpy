"""NTRIP v1 / v2 client.

NTRIP (Networked Transport of RTCM via Internet Protocol) is essentially
HTTP/1.1 with a custom GET that opens an indefinite RTCM3 byte stream.

Two entry points:

- :func:`fetch_sourcetable` — return the caster's sourcetable (mountpoint
  catalog) as a list of dicts.
- :func:`stream` — open a mountpoint and yield the raw bytes (suitable
  to feed straight into :func:`rinexpy.rtcm3.iter_messages`).

Auth is HTTP Basic. The TLS variant (NTRIP-over-HTTPS) is requested by
passing ``port=443`` and a normal hostname.
"""

from __future__ import annotations

import base64
import socket
import ssl
from collections.abc import Iterator

_USER_AGENT = "NTRIP rinexpy/0.1"


def _open_connection(host: str, port: int, *, timeout: float = 30.0) -> socket.socket:
    """TCP-connect to ``(host, port)``, with TLS for port 443."""
    sock = socket.create_connection((host, port), timeout=timeout)
    if port == 443:
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(sock, server_hostname=host)
    return sock


def _basic_auth(user: str, password: str) -> str:
    """Encode ``user:password`` for the HTTP Basic Authorization header."""
    return base64.b64encode(f"{user}:{password}".encode()).decode("ascii")


def fetch_sourcetable(
    host: str,
    *,
    port: int = 2101,
    timeout: float = 30.0,
) -> list[dict]:
    """Fetch and parse the caster's sourcetable.

    Parameters
    ----------
    host:
        Caster hostname (e.g. ``"rtk2go.com"``).
    port:
        Caster TCP port. Default 2101 (NTRIP v1); use 443 for TLS-NTRIP.
    timeout:
        Socket timeout in seconds.

    Returns
    -------
    list[dict]
        One dict per ``STR;`` (mountpoint) line. Keys: ``mountpoint``,
        ``identifier``, ``format``, ``format_details``, ``carrier``,
        ``nav_system``, ``network``, ``country``, ``latitude``,
        ``longitude``, ``nmea``, ``solution``, ``generator``,
        ``compr_encrp``, ``authentication``, ``fee``, ``bitrate``.
        ``CAS;`` (caster) and ``NET;`` (network) lines come back with
        ``type`` set to ``"CAS"`` / ``"NET"`` and the raw fields under
        ``raw``.
    """
    sock = _open_connection(host, port, timeout=timeout)
    request = (
        f"GET / HTTP/1.0\r\n"
        f"Host: {host}\r\n"
        f"User-Agent: {_USER_AGENT}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("ascii")
    try:
        sock.sendall(request)
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        sock.close()

    body = b"".join(chunks).decode("ascii", errors="ignore")
    return _parse_sourcetable(body)


def _parse_sourcetable(text: str) -> list[dict]:
    """Parse the body of a sourcetable response into structured records."""
    out: list[dict] = []
    str_fields = [
        "mountpoint", "identifier", "format", "format_details",
        "carrier", "nav_system", "network", "country",
        "latitude", "longitude", "nmea", "solution",
        "generator", "compr_encrp", "authentication",
        "fee", "bitrate",
    ]
    for line in text.splitlines():
        if line.startswith("ENDSOURCETABLE"):
            break
        if not line:
            continue
        if line.startswith("STR;"):
            parts = line.split(";")
            entry: dict = {"type": "STR"}
            for i, key in enumerate(str_fields, start=1):
                entry[key] = parts[i] if i < len(parts) else ""
            try:
                entry["latitude"] = float(entry["latitude"])
                entry["longitude"] = float(entry["longitude"])
            except (ValueError, KeyError):
                pass
            out.append(entry)
        elif line.startswith(("CAS;", "NET;")):
            out.append({"type": line[:3], "raw": line[4:].split(";")})
    return out


def stream(
    host: str,
    mountpoint: str,
    *,
    user: str = "",
    password: str = "",
    port: int = 2101,
    timeout: float = 30.0,
    chunk_size: int = 4096,
) -> Iterator[bytes]:
    """Open a mountpoint and yield raw RTCM3 bytes indefinitely.

    Parameters
    ----------
    host, port:
        Caster hostname and port.
    mountpoint:
        Mountpoint name (from the sourcetable's STR; lines).
    user, password:
        Optional HTTP Basic credentials. Empty strings request anonymous.
    timeout:
        Socket timeout in seconds.
    chunk_size:
        Bytes per ``recv()`` call. Default 4 KB.

    Yields
    ------
    bytes
        Raw bytes off the socket. Feed straight into
        :func:`rinexpy.rtcm3.iter_messages` after wrapping in a
        ``BytesIO`` (or a generator-to-stream adapter).

    Raises
    ------
    ConnectionError
        If the caster doesn't return ``ICY 200`` / ``HTTP/1.x 200``
        (the NTRIP success signatures).
    """
    sock = _open_connection(host, port, timeout=timeout)
    headers = [
        f"GET /{mountpoint} HTTP/1.0",
        f"Host: {host}",
        f"User-Agent: {_USER_AGENT}",
        "Ntrip-Version: Ntrip/2.0",
    ]
    if user or password:
        headers.append(f"Authorization: Basic {_basic_auth(user, password)}")
    headers.append("Connection: close")
    request = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii")

    try:
        sock.sendall(request)
        # Read the response status line — NTRIP1 sends "ICY 200 OK\r\n",
        # NTRIP2 sends "HTTP/1.1 200 OK\r\n\r\n".
        buf = b""
        while b"\r\n\r\n" not in buf and b"ICY 200 OK\r\n" not in buf:
            chunk = sock.recv(256)
            if not chunk:
                break
            buf += chunk
        if not (b"ICY 200" in buf or b"200 OK" in buf):
            raise ConnectionError(f"NTRIP caster rejected request: {buf[:200]!r}")
        # Anything after the headers is RTCM payload.
        if b"\r\n\r\n" in buf:
            _, _, leftover = buf.partition(b"\r\n\r\n")
        else:
            _, _, leftover = buf.partition(b"\r\n")
        if leftover:
            yield leftover
        while True:
            chunk = sock.recv(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        sock.close()


__all__ = ["fetch_sourcetable", "stream"]
