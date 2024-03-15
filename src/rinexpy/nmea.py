"""NMEA-0183 sentence decoder.

NMEA-0183 sentences are ASCII lines of the form:

    $TALKER,field1,field2,...,fieldN*HH

where ``TALKER`` is a 5-letter mnemonic (``GPGGA``, ``GNRMC``, ``GLGSV``,
etc.) and ``HH`` is a two-digit hex XOR checksum over everything between
``$`` and ``*``.

We decode the most common positioning sentences:

- **GGA** — Global Positioning System Fix Data (lat/lon/alt/quality/n_sat)
- **RMC** — Recommended Minimum (lat/lon/speed/course/date)
- **GSA** — Active SVs + DOP
- **GSV** — Satellites in View (az/el/SNR per SV)
- **VTG** — Course over ground

Unknown talker prefixes are accepted; unknown sentence types come back
with the raw ``fields`` list and ``type`` set to the trailing 3 letters.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date, datetime, time
from typing import Any


def checksum(sentence: str) -> int:
    """Compute the NMEA-0183 XOR checksum over the body of ``sentence``.

    Parameters
    ----------
    sentence:
        Either the full sentence (``"$..*HH"``) or just the body
        between the ``$`` and the ``*``.

    Returns
    -------
    int
        XOR of every byte between ``$`` (exclusive) and ``*``
        (exclusive). Compare against ``int(suffix, 16)``.
    """
    body = sentence
    if body.startswith("$"):
        body = body[1:]
    if "*" in body:
        body = body.split("*", 1)[0]
    cs = 0
    for ch in body.encode("ascii", errors="ignore"):
        cs ^= ch
    return cs


def parse_sentence(line: str, *, check_crc: bool = True) -> dict[str, Any] | None:
    """Parse one NMEA-0183 line into a structured dict.

    Parameters
    ----------
    line:
        A single ASCII NMEA sentence, with or without the trailing
        ``\\r\\n``.
    check_crc:
        Validate the trailing ``*HH`` checksum. If the checksum is
        wrong (or missing) and ``check_crc=True``, returns ``None``
        instead of raising.

    Returns
    -------
    dict | None
        Always contains ``talker`` (the 2-letter prefix, e.g. ``"GP"``,
        ``"GN"``, ``"GL"``), ``type`` (the 3-letter sentence type),
        ``raw`` (the input line), ``fields`` (the body split by ``,``).
        Decoded sentence types add structured fields. Returns ``None``
        when the line doesn't look like NMEA at all or the CRC fails.
    """
    line = line.strip()
    if not line.startswith("$") or len(line) < 7:
        return None
    if "*" in line:
        body, suffix = line[1:].rsplit("*", 1)
    else:
        body, suffix = line[1:], ""
    if check_crc:
        if not suffix:
            # No checksum present; we treat missing-checksum as invalid
            # when CRC checking is on. Pass check_crc=False to accept
            # checksumless sentences (e.g. some test/replay corpora).
            return None
        try:
            if checksum(line) != int(suffix[:2], 16):
                return None
        except ValueError:
            return None

    fields = body.split(",")
    head = fields[0]
    if len(head) < 5:
        return None
    talker = head[:2]
    sentence_type = head[2:]

    out: dict[str, Any] = {
        "talker": talker,
        "type": sentence_type,
        "raw": line,
        "fields": fields[1:],
    }
    decoder = _DECODERS.get(sentence_type)
    if decoder is not None:
        out.update(decoder(fields[1:]))
    return out


def iter_lines(stream: Iterable[str], *, check_crc: bool = True) -> Iterator[dict[str, Any]]:
    """Yield decoded sentence dicts from an iterable of NMEA lines.

    Parameters
    ----------
    stream:
        Anything iterating ASCII text lines (a file, ``stdin``, the
        output of ``socket.makefile("r")``).
    check_crc:
        Forwarded to :func:`parse_sentence`.

    Yields
    ------
    dict
        One per recognisable sentence; lines that fail validation are
        silently skipped.
    """
    for line in stream:
        msg = parse_sentence(line, check_crc=check_crc)
        if msg is not None:
            yield msg


# ---------------------------------------------------------------------------
# Per-sentence decoders. Each takes the comma-split fields (no talker
# prefix) and returns a dict of structured keys to merge into the result.
# ---------------------------------------------------------------------------


def _parse_lat_lon(value: str, hemi: str) -> float | None:
    """Decode an NMEA ddmm.mmmm + hemisphere field into signed degrees."""
    if not value or not hemi:
        return None
    try:
        dot = value.index(".")
    except ValueError:
        return None
    if dot < 3:
        return None
    deg = int(value[: dot - 2])
    minutes = float(value[dot - 2 :])
    decimal = deg + minutes / 60.0
    if hemi.upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def _parse_time_hms(value: str) -> time | None:
    """Decode an NMEA HHMMSS.ss timestamp into ``datetime.time``."""
    if not value or len(value) < 6:
        return None
    try:
        h = int(value[:2])
        m = int(value[2:4])
        s = float(value[4:])
    except ValueError:
        return None
    micro = int(round((s - int(s)) * 1_000_000))
    return time(h, m, int(s), micro)


def _parse_date_dmy(value: str) -> date | None:
    """Decode an NMEA DDMMYY date into ``datetime.date`` (1980-2079 pivot)."""
    if not value or len(value) != 6:
        return None
    try:
        d = int(value[:2])
        m = int(value[2:4])
        y = int(value[4:6])
    except ValueError:
        return None
    y += 2000 if y < 80 else 1900
    return date(y, m, d)


def _decode_gga(f: list[str]) -> dict[str, Any]:
    """GGA: lat/lon/altitude + fix quality + n_sat + HDOP."""
    out: dict[str, Any] = {}
    if len(f) >= 14:
        out["time"] = _parse_time_hms(f[0])
        out["lat"] = _parse_lat_lon(f[1], f[2])
        out["lon"] = _parse_lat_lon(f[3], f[4])
        out["fix_quality"] = int(f[5]) if f[5].isdigit() else None
        out["n_sat"] = int(f[6]) if f[6].isdigit() else None
        out["hdop"] = float(f[7]) if f[7] else None
        out["altitude_m"] = float(f[8]) if f[8] else None
        out["geoid_sep_m"] = float(f[10]) if f[10] else None
    return out


def _decode_rmc(f: list[str]) -> dict[str, Any]:
    """RMC: time + lat/lon + speed/course + date."""
    out: dict[str, Any] = {}
    if len(f) >= 9:
        out["time"] = _parse_time_hms(f[0])
        out["status"] = f[1]
        out["lat"] = _parse_lat_lon(f[2], f[3])
        out["lon"] = _parse_lat_lon(f[4], f[5])
        out["speed_kn"] = float(f[6]) if f[6] else None
        out["course_deg"] = float(f[7]) if f[7] else None
        out["date"] = _parse_date_dmy(f[8])
        if out["time"] is not None and out["date"] is not None:
            out["datetime"] = datetime.combine(out["date"], out["time"])
    return out


def _decode_gsa(f: list[str]) -> dict[str, Any]:
    """GSA: mode + 12-SV active list + PDOP/HDOP/VDOP."""
    out: dict[str, Any] = {}
    if len(f) >= 17:
        out["mode_auto"] = f[0]
        out["fix_type"] = int(f[1]) if f[1].isdigit() else None
        out["sv_ids"] = [int(s) for s in f[2:14] if s.isdigit()]
        out["pdop"] = float(f[14]) if f[14] else None
        out["hdop"] = float(f[15]) if f[15] else None
        out["vdop"] = float(f[16].split("*")[0]) if f[16] else None
    return out


def _decode_gsv(f: list[str]) -> dict[str, Any]:
    """GSV: total messages, message N of M, satellites in view (4 per sentence)."""
    out: dict[str, Any] = {}
    if len(f) < 3:
        return out
    out["n_messages"] = int(f[0]) if f[0].isdigit() else None
    out["message_num"] = int(f[1]) if f[1].isdigit() else None
    out["sv_in_view"] = int(f[2]) if f[2].isdigit() else None
    sats: list[dict[str, Any]] = []
    for i in range(3, len(f), 4):
        chunk = f[i : i + 4]
        if len(chunk) < 4:
            break
        sat = {
            "prn": int(chunk[0]) if chunk[0].isdigit() else None,
            "elevation_deg": int(chunk[1]) if chunk[1].isdigit() else None,
            "azimuth_deg": int(chunk[2]) if chunk[2].isdigit() else None,
            "snr_dbhz": int(chunk[3].split("*")[0]) if chunk[3].split("*")[0].isdigit() else None,
        }
        sats.append(sat)
    out["satellites"] = sats
    return out


def _decode_vtg(f: list[str]) -> dict[str, Any]:
    """VTG: course (true + magnetic) and speed (knots + km/h)."""
    out: dict[str, Any] = {}
    if len(f) >= 8:
        out["course_true_deg"] = float(f[0]) if f[0] else None
        out["course_mag_deg"] = float(f[2]) if f[2] else None
        out["speed_kn"] = float(f[4]) if f[4] else None
        out["speed_kmh"] = float(f[6]) if f[6] else None
    return out


_DECODERS = {
    "GGA": _decode_gga,
    "RMC": _decode_rmc,
    "GSA": _decode_gsa,
    "GSV": _decode_gsv,
    "VTG": _decode_vtg,
}


__all__ = ["checksum", "iter_lines", "parse_sentence"]
