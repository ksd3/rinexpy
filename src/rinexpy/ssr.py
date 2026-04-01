"""SSR (State-Space Representation) correction composer.

RTCM 3.x State-Space Representation messages (1057-1068 GPS, 1240-1263
Galileo, plus IGS-SSR MT 4076 and the QZSS / SBAS / BeiDou / GLONASS
extensions) carry real-time orbit, clock, and code-bias corrections to
the broadcast ephemeris. This module wraps the decoded messages into a
single :class:`SSRCorrections` object that:

- absorbs heterogeneous SSR messages by ``msg_id`` (1057 orbit, 1058
  clock, 1059 code-bias, plus the constellation-specific equivalents
  1063/1240/...);
- exposes per-(sv, epoch) orbit + clock corrections so callers can
  shift broadcast / SP3 positions and clocks to precise values;
- exposes per-(sv, obs_code) code biases so callers can correct
  pseudoranges.

The intended consumer is :func:`rinexpy.ppp.ppp_solve`, which accepts an
optional ``ssr=SSRCorrections(...)`` argument and applies the
corrections in its per-epoch loop. This is the PPP-side half of the
roadmap's SSR acceptance criterion: ``rtcm3.iter_messages`` returns
decoded dicts; PPP consumes them in place of CLK.

Sign conventions follow RTCM 10403.3 §3.5.9 / §3.5.10:

- Orbit corrections (1057-style): the radial / along-track / cross-
  track delta is defined as ``X_broadcast - X_precise``, so the
  precise position is ``X_broadcast - delta_orbit``.
- Clock corrections (1058-style): the c0/c1/c2 polynomial gives the
  delta as ``t_broadcast - t_precise`` in meters of range, so the
  precise satellite-clock value (s) is
  ``t_broadcast_s - (c0 + c1*dt + c2*dt^2) / c``.
- Code biases (1059-style): the bias is added to the raw pseudorange
  to obtain the bias-corrected pseudorange.

References
----------
- RTCM Standard 10403.3 §3.5.9-3.5.12 (SSR family).
- IGS-SSR Format Specification v1.0 (RTCM proprietary MT 4076).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import numpy as np

_C_M_PER_S = 299_792_458.0


#: Default per-system signal-ID -> RINEX-3 observation-code mapping for
#: SSR code-bias messages (1059 / 1242 / ...). Sourced from RTCM 10403.3
#: Tables 3.5-91 through 3.5-95.
_DEFAULT_SIGNAL_MAP: dict[str, dict[int, str]] = {
    "G": {
        0: "C1C", 1: "C1P", 2: "C1W",
        5: "C2C", 6: "C2D", 7: "C2S", 8: "C2L", 9: "C2X",
        10: "C2P", 11: "C2W",
        14: "C5I", 15: "C5Q", 16: "C5X",
    },
    "E": {
        0: "C1A", 1: "C1B", 2: "C1C", 3: "C1X", 4: "C1Z",
        5: "C5I", 6: "C5Q", 7: "C5X",
        8: "C7I", 9: "C7Q", 10: "C7X",
        11: "C8I", 12: "C8Q", 13: "C8X",
        14: "C6A", 15: "C6B", 16: "C6C", 17: "C6X", 18: "C6Z",
    },
    "R": {
        0: "C1C", 1: "C1P",
        2: "C2C", 3: "C2P",
    },
    "J": {
        0: "C1C", 1: "C1S", 2: "C1L", 3: "C1X",
        4: "C2S", 5: "C2L", 6: "C2X",
        7: "C5I", 8: "C5Q", 9: "C5X",
    },
    "C": {
        0: "C2I", 1: "C2Q", 2: "C2X",
        3: "C6I", 4: "C6Q", 5: "C6X",
        6: "C7I", 7: "C7Q", 8: "C7X",
    },
    "S": {
        0: "C1C", 1: "C5I", 2: "C5Q", 3: "C5X",
    },
}


def _system_letter(msg_id: int) -> str:
    """Return the constellation letter for a given SSR msg_id."""
    if 1057 <= msg_id <= 1062:
        return "G"
    if 1063 <= msg_id <= 1068:
        return "R"
    if 1240 <= msg_id <= 1245:
        return "E"
    if 1246 <= msg_id <= 1251:
        return "J"
    if 1252 <= msg_id <= 1257:
        return "S"
    if 1258 <= msg_id <= 1263:
        return "C"
    return ""


def _kind_of(msg_id: int) -> str:
    """Return ``"orbit"``, ``"clock"``, ``"combined"``, ``"code_bias"``,
    ``"ura"``, ``"high_rate_clock"``, or ``""`` based on RTCM 10403.3
    layout: each 6-message block of one system is in this fixed order."""
    if msg_id in (1057, 1063, 1240, 1246, 1252, 1258):
        return "orbit"
    if msg_id in (1058, 1064, 1241, 1247, 1253, 1259):
        return "clock"
    if msg_id in (1059, 1065, 1242, 1248, 1254, 1260):
        return "code_bias"
    if msg_id in (1060, 1066, 1243, 1249, 1255, 1261):
        return "combined"
    if msg_id in (1061, 1067, 1244, 1250, 1256, 1262):
        return "ura"
    if msg_id in (1062, 1068, 1245, 1251, 1257, 1263):
        return "high_rate_clock"
    return ""


class SSRCorrections:
    """Pool of decoded SSR messages indexed by (system, sv, kind).

    Construction:

    ::

        from rinexpy.rtcm3 import iter_messages
        from rinexpy.ssr import SSRCorrections
        ssr = SSRCorrections(iter_messages(stream))

    Then pass to PPP: ``ppp_solve(obs, sp3, clk=None, ssr=ssr)``.

    The class is intentionally minimal: each SV's *latest* orbit /
    clock / code-bias message is kept. A future revision can stack a
    time-series for cross-checking against the IOD SSR continuity, but
    a single latest-message store covers the dominant PPP use case
    (the SSR stream tops up every 5-60 seconds).
    """

    def __init__(self, messages: Iterable[dict[str, Any]] | None = None) -> None:
        # Maps sv -> dict (most recent payload for that kind).
        self._orbit: dict[str, dict[str, Any]] = {}
        self._clock: dict[str, dict[str, Any]] = {}
        self._combined: dict[str, dict[str, Any]] = {}
        # Maps (sv, obs_code) -> bias_m.
        self._code_bias: dict[tuple[str, str], float] = {}
        if messages:
            for m in messages:
                self.add_message(m)

    def add_message(self, msg: dict[str, Any]) -> None:
        """Absorb one decoded SSR message into the internal pool."""
        msg_id = msg.get("msg_id")
        if msg_id is None:
            return
        kind = _kind_of(int(msg_id))
        if not kind:
            return
        system = msg.get("system") or _system_letter(int(msg_id))
        sats = msg.get("satellites", [])
        header = msg.get("header", {})
        epoch_s = header.get("epoch_time_s")
        for sat in sats:
            sv = sat.get("sv") or self._label(system, sat["prn"])
            if kind == "orbit":
                self._orbit[sv] = {**sat, "_epoch_s": epoch_s}
            elif kind == "clock":
                self._clock[sv] = {**sat, "_epoch_s": epoch_s}
            elif kind == "combined":
                self._combined[sv] = {**sat, "_epoch_s": epoch_s}
                # Combined messages also carry the orbit + clock half.
                self._orbit[sv] = {**sat, "_epoch_s": epoch_s}
                self._clock[sv] = {**sat, "_epoch_s": epoch_s}
            elif kind == "code_bias":
                sig_map = _DEFAULT_SIGNAL_MAP.get(system, {})
                for sig in sat.get("signals", []):
                    code = sig_map.get(sig["signal_id"])
                    if code is None:
                        continue
                    self._code_bias[(sv, code)] = float(sig["bias_m"])

    @staticmethod
    def _label(system: str, prn: int) -> str:
        if system == "S":
            return f"S{prn + 120:03d}" if prn > 0 else f"S{prn:02d}"
        return f"{system}{prn:02d}"

    # -- correction lookups ------------------------------------------------

    def orbit_correction_ecef(
        self,
        sv: str,
        sat_pos_ecef: np.ndarray,
        sat_vel_ecef: np.ndarray,
        epoch_seconds_of_week: float,
    ) -> np.ndarray:
        """Return the orbit correction ``delta_X_ecef`` for one SV at one
        epoch, in meters. The convention is ``X_precise = X_broadcast -
        delta_X_ecef``.

        ``sat_pos_ecef`` and ``sat_vel_ecef`` are the broadcast / SP3
        position and velocity at the same epoch (needed to build the
        radial-along-cross frame).
        """
        rec = self._orbit.get(sv)
        if rec is None:
            return np.zeros(3)
        epoch_s = rec.get("_epoch_s")
        if epoch_s is None:
            dt = 0.0
        else:
            dt = float(epoch_seconds_of_week) - float(epoch_s)
        d_r = rec.get("delta_radial_m", 0.0) + rec.get("dot_delta_radial_m_per_s", 0.0) * dt
        d_a = rec.get("delta_along_track_m", 0.0) + rec.get("dot_delta_along_track_m_per_s", 0.0) * dt
        d_c = rec.get("delta_cross_track_m", 0.0) + rec.get("dot_delta_cross_track_m_per_s", 0.0) * dt

        pos = np.asarray(sat_pos_ecef, dtype=float)
        vel = np.asarray(sat_vel_ecef, dtype=float)
        norm_pos = np.linalg.norm(pos)
        norm_vel = np.linalg.norm(vel)
        if norm_pos == 0.0 or norm_vel == 0.0:
            return np.zeros(3)
        radial = pos / norm_pos
        cross = np.cross(pos, vel)
        norm_cross = np.linalg.norm(cross)
        if norm_cross == 0.0:
            return np.zeros(3)
        cross_unit = cross / norm_cross
        along = np.cross(cross_unit, radial)
        return d_r * radial + d_a * along + d_c * cross_unit

    def clock_correction_s(
        self, sv: str, epoch_seconds_of_week: float
    ) -> float:
        """Return the clock correction in **seconds** for one SV at one
        epoch. The convention is ``t_precise_s = t_broadcast_s -
        clock_correction_s``.
        """
        rec = self._clock.get(sv)
        if rec is None:
            return 0.0
        epoch_s = rec.get("_epoch_s")
        dt = 0.0 if epoch_s is None else float(epoch_seconds_of_week) - float(epoch_s)
        c0 = rec.get("c0_m", 0.0)
        c1 = rec.get("c1_m_per_s", 0.0)
        c2 = rec.get("c2_m_per_s2", 0.0)
        delta_m = c0 + c1 * dt + c2 * dt * dt
        return delta_m / _C_M_PER_S

    def code_bias_m(self, sv: str, obs_code: str) -> float:
        """Per-SV per-obs-code bias correction in meters. Returns 0 if
        no SSR bias has been seen for the (sv, obs_code) pair."""
        return self._code_bias.get((sv, obs_code), 0.0)

    # -- introspection -----------------------------------------------------

    def known_satellites(self) -> list[str]:
        """Set of SVs for which any correction has been absorbed."""
        return sorted(
            set(self._orbit) | set(self._clock) | {sv for sv, _ in self._code_bias}
        )

    def has_clock(self, sv: str) -> bool:
        return sv in self._clock

    def has_orbit(self, sv: str) -> bool:
        return sv in self._orbit


__all__ = ["SSRCorrections"]
