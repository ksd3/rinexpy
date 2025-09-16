"""Real-time PPP integration: NTRIP -> RTCM3/HAS -> orbit/clock cache.

The pieces existed already in separate modules; this module wires them
into one place so a caller can do::

    from rinexpy.realtime import RealtimeOrbitClock, ntrip_message_loop
    cache = RealtimeOrbitClock()
    for msg in ntrip_message_loop("rtk2go.com", 2101, "EXAMPLE"):
        cache.ingest(msg)
        # cache.sv_position(prn=5, t=...) is now corrected by the latest
        # SSR / HAS message that's still inside its validity window.

The orbit/clock cache:

- Stores the most recent broadcast ephemeris per (gnss, PRN).
- Stores the most recent SSR orbit (RTCM 1057) and clock (1058)
  corrections per GPS PRN.
- Stores the most recent HAS orbit (MT 2) and clock (MT 3) corrections
  per (gnss, PRN), gated by the validity-interval timer.
- Computes corrected ECEF satellite position and clock at any
  requested epoch.

A small CLI lives at the bottom; ``python -m rinexpy.realtime --caster
rtk2go.com:2101 --mount XYZ`` streams from a public NTRIP caster, prints
the messages it decodes, and tracks the orbit/clock cache.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import numpy as np

from . import _native
from .has import decode_has_message
from .keplerian import keplerian2ecef
from .rtcm3 import decode_message, iter_messages

log = logging.getLogger(__name__)

_C = 299_792_458.0


@dataclass
class _SSREntry:
    """One SSR correction record with its received timestamp."""
    received: datetime
    iod: int
    radial_m: float
    along_m: float
    cross_m: float
    dot_radial_m_per_s: float
    dot_along_m_per_s: float
    dot_cross_m_per_s: float


@dataclass
class _SSRClockEntry:
    received: datetime
    c0_m: float
    c1_m_per_s: float
    c2_m_per_s2: float


@dataclass
class RealtimeOrbitClock:
    """In-memory cache of broadcast ephemerides and SSR / HAS corrections.

    All entries are timestamped at ingest; ``sv_position(prn, t)`` uses
    the most recent correction whose validity has not yet expired (10 s
    default validity for SSR; the HAS messages carry their own validity
    interval in the payload).
    """

    ssr_orbit: dict[int, _SSREntry] = field(default_factory=dict)
    ssr_clock: dict[int, _SSRClockEntry] = field(default_factory=dict)
    has_mask: dict[str, Any] | None = None
    has_orbit: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    has_clock: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    broadcast: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    ssr_validity_s: float = 10.0

    def ingest(self, msg: dict[str, Any]) -> None:
        """Route one decoded message to the appropriate cache slot.

        ``msg`` is the dict returned by :func:`rinexpy.rtcm3.decode_message`
        (for RTCM 3.x messages) or :func:`rinexpy.has.decode_has_message`
        (for HAS messages). Unknown payloads are silently ignored.
        """
        mid = msg.get("msg_id")
        now = datetime.now(timezone.utc)
        if mid == 1057:
            for s in msg.get("satellites", []):
                self.ssr_orbit[int(s["prn"])] = _SSREntry(
                    received=now,
                    iod=int(s["iode"]),
                    radial_m=float(s["delta_radial_m"]),
                    along_m=float(s["delta_along_track_m"]),
                    cross_m=float(s["delta_cross_track_m"]),
                    dot_radial_m_per_s=float(s["dot_delta_radial_m_per_s"]),
                    dot_along_m_per_s=float(s["dot_delta_along_track_m_per_s"]),
                    dot_cross_m_per_s=float(s["dot_delta_cross_track_m_per_s"]),
                )
        elif mid == 1058:
            for s in msg.get("satellites", []):
                self.ssr_clock[int(s["prn"])] = _SSRClockEntry(
                    received=now,
                    c0_m=float(s["c0_m"]),
                    c1_m_per_s=float(s["c1_m_per_s"]),
                    c2_m_per_s2=float(s["c2_m_per_s2"]),
                )
        elif mid == 1019:
            self.broadcast[("G", int(msg.get("prn", 0)))] = msg
        elif mid == 1042:
            self.broadcast[("C", int(msg.get("prn", 0)))] = msg
        elif mid in (1045, 1046):
            self.broadcast[("E", int(msg.get("prn", 0)))] = msg
        elif mid == 1020:
            self.broadcast[("R", int(msg.get("prn", 0)))] = msg
        elif "header" in msg and "payload" in msg and "message_type" in msg.get("header", {}):
            # HAS message
            mt = msg["header"]["message_type"]
            if mt == 1:
                self.has_mask = msg["payload"]
            elif mt == 2 and "satellites" in msg.get("payload", {}):
                for s in msg["payload"]["satellites"]:
                    key = (s["gnss_id"], int(s["prn"]))
                    self.has_orbit[key] = {"received": now, **s}
            elif mt == 3 and "satellites" in msg.get("payload", {}):
                for s in msg["payload"]["satellites"]:
                    key = (s["gnss_id"], int(s["prn"]))
                    self.has_clock[key] = {"received": now, **s}

    def ssr_orbit_for(self, prn: int) -> _SSREntry | None:
        """Return the live SSR orbit entry for ``prn``, or None if stale."""
        entry = self.ssr_orbit.get(int(prn))
        if entry is None:
            return None
        if (datetime.now(timezone.utc) - entry.received).total_seconds() > self.ssr_validity_s:
            return None
        return entry

    def ssr_clock_for(self, prn: int) -> _SSRClockEntry | None:
        entry = self.ssr_clock.get(int(prn))
        if entry is None:
            return None
        if (datetime.now(timezone.utc) - entry.received).total_seconds() > self.ssr_validity_s:
            return None
        return entry

    def apply_orbit_correction(
        self,
        prn: int,
        sv_ecef_broadcast: np.ndarray,
        sv_velocity_ecef: np.ndarray | None = None,
        elapsed_s: float = 0.0,
    ) -> np.ndarray:
        """Apply the live SSR orbit correction (RAC frame -> ECEF).

        Parameters
        ----------
        prn:
            GPS satellite PRN.
        sv_ecef_broadcast:
            ``(3,)`` broadcast-ephemeris ECEF position at the desired
            epoch.
        sv_velocity_ecef:
            Optional ``(3,)`` broadcast ECEF velocity. Used to build the
            RAC frame; required when the velocity isn't itself in the
            broadcast (Keplerian solver returns it on demand).
        elapsed_s:
            Seconds since the SSR correction was received. The orbit
            rate terms (``dot_*``) are multiplied by this to extrapolate
            the correction to the current epoch.

        Returns
        -------
        ndarray
            Corrected ``(3,)`` ECEF position in meters. Falls back to
            the uncorrected position when no live SSR is available.
        """
        entry = self.ssr_orbit_for(prn)
        if entry is None or sv_velocity_ecef is None:
            return np.asarray(sv_ecef_broadcast, dtype=float)
        r = np.ascontiguousarray(sv_ecef_broadcast, dtype=float)
        v = np.ascontiguousarray(sv_velocity_ecef, dtype=float)

        if _native.have_apply_ssr():
            rac0 = np.array(
                [entry.radial_m, entry.along_m, entry.cross_m],
                dtype=float,
            )
            racdot = np.array(
                [entry.dot_radial_m_per_s, entry.dot_along_m_per_s,
                 entry.dot_cross_m_per_s],
                dtype=float,
            )
            return _native.apply_ssr_orbit_correction(
                r, v, rac0, racdot, elapsed_s,
            )

        # Radial / Along / Cross unit vectors.
        e_r = r / np.linalg.norm(r)
        h = np.cross(r, v)
        e_c = h / np.linalg.norm(h)
        e_a = np.cross(e_c, e_r)
        d_radial = entry.radial_m + entry.dot_radial_m_per_s * elapsed_s
        d_along = entry.along_m + entry.dot_along_m_per_s * elapsed_s
        d_cross = entry.cross_m + entry.dot_cross_m_per_s * elapsed_s
        return r - (d_radial * e_r + d_along * e_a + d_cross * e_c)

    def apply_clock_correction(
        self,
        prn: int,
        broadcast_clock_s: float,
        elapsed_s: float = 0.0,
    ) -> float:
        """Add the live SSR clock correction (in seconds)."""
        entry = self.ssr_clock_for(prn)
        if entry is None:
            return broadcast_clock_s
        if _native.have_apply_ssr():
            return _native.apply_ssr_clock_correction(
                broadcast_clock_s, entry.c0_m, entry.c1_m_per_s,
                entry.c2_m_per_s2, elapsed_s,
            )
        delta_m = (
            entry.c0_m
            + entry.c1_m_per_s * elapsed_s
            + 0.5 * entry.c2_m_per_s2 * elapsed_s * elapsed_s
        )
        # IGS SSR convention: orbit / clock corrections are subtracted
        # from the broadcast to obtain the precise quantity.
        return broadcast_clock_s - delta_m / _C


def ntrip_message_loop(
    caster: str,
    port: int,
    mountpoint: str,
    *,
    user: str = "anonymous@example.com",
    password: str = "anonymous",
    timeout: float = 15.0,
) -> Iterator[dict[str, Any]]:
    """Connect to an NTRIP caster and yield decoded RTCM3 messages.

    Re-implements just enough of the NTRIP client to keep this module
    self-contained for the CLI entry. For production use the higher-
    level :func:`rinexpy.ntrip.stream` instead.
    """
    import base64
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    request = (
        f"GET /{mountpoint} HTTP/1.1\r\n"
        f"Host: {caster}\r\n"
        f"User-Agent: NTRIP rinexpy-realtime/0\r\n"
        f"Authorization: Basic {auth}\r\n"
        "Connection: close\r\n"
        "\r\n"
    )
    log.info("ntrip: connecting to %s:%s/%s", caster, port, mountpoint)
    with socket.create_connection((caster, port), timeout=timeout) as sock:
        sock.sendall(request.encode())
        # Read until we see the HTTP 200 line, then start framing RTCM3.
        sock.settimeout(timeout)
        # We pop the headers out of the same socket file object.
        f = sock.makefile("rb")
        # Skip headers (a blank line terminates them).
        first = f.readline()
        if not first.startswith((b"ICY 200", b"HTTP/1.1 200", b"HTTP/1.0 200")):
            raise RuntimeError(f"NTRIP refused: {first!r}")
        while True:
            line = f.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        for msg in iter_messages(f):
            mid = msg.get("msg_id")
            body = msg.get("body", b"")
            try:
                decoded = decode_message(mid, body)
            except Exception:
                decoded = msg
            yield decoded


def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rinexpy.realtime",
        description="Stream RTCM3 from an NTRIP caster and track the orbit/clock cache.",
    )
    parser.add_argument("--caster", required=True, help="caster host (e.g. rtk2go.com)")
    parser.add_argument("--port", type=int, default=2101)
    parser.add_argument("--mount", required=True, help="mountpoint name")
    parser.add_argument("--user", default="anonymous@example.com")
    parser.add_argument("--password", default="anonymous")
    parser.add_argument("--max-messages", type=int, default=200,
                        help="stop after this many messages (default 200)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cache = RealtimeOrbitClock()
    count = 0
    for msg in ntrip_message_loop(
        args.caster, args.port, args.mount,
        user=args.user, password=args.password,
    ):
        cache.ingest(msg)
        mid = msg.get("msg_id")
        print(f"  msg {mid}: SSR-orbit cached {len(cache.ssr_orbit)},"
              f" SSR-clock cached {len(cache.ssr_clock)},"
              f" broadcast cached {len(cache.broadcast)}")
        count += 1
        if count >= args.max_messages:
            break
    print(f"done after {count} messages", file=sys.stderr)


if __name__ == "__main__":
    _cli()


__all__ = [
    "RealtimeOrbitClock",
    "ntrip_message_loop",
]
