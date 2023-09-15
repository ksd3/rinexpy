"""GPS time conversions and the IGS leap-second table.

The IGS publishes leap-second corrections as TAI-UTC; GPS time runs at
TAI-19s, so GPS-UTC = TAI-UTC - 19. This module hard-codes the leap
seconds from the IERS Bulletin C up to 2017 (the most recent leap as of
publication); newer leaps can be appended to ``LEAP_SECONDS``.

All datetimes are naive and treated as UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta

#: GPS epoch (00:00:00 UTC, 1980-01-06).
GPS_EPOCH = datetime(1980, 1, 6)

#: Seconds in one GPS week.
SECONDS_PER_WEEK = 604_800

#: (date, leap_seconds) pairs from IERS Bulletin C. Each entry is the
#: UTC-TAI offset in effect *after* that date. GPS-UTC = -(offset + 19).
LEAP_SECONDS: list[tuple[datetime, int]] = [
    (datetime(1981, 7, 1), 20),
    (datetime(1982, 7, 1), 21),
    (datetime(1983, 7, 1), 22),
    (datetime(1985, 7, 1), 23),
    (datetime(1988, 1, 1), 24),
    (datetime(1990, 1, 1), 25),
    (datetime(1991, 1, 1), 26),
    (datetime(1992, 7, 1), 27),
    (datetime(1993, 7, 1), 28),
    (datetime(1994, 7, 1), 29),
    (datetime(1996, 1, 1), 30),
    (datetime(1997, 7, 1), 31),
    (datetime(1999, 1, 1), 32),
    (datetime(2006, 1, 1), 33),
    (datetime(2009, 1, 1), 34),
    (datetime(2012, 7, 1), 35),
    (datetime(2015, 7, 1), 36),
    (datetime(2017, 1, 1), 37),
]


def leap_seconds_at(t: datetime) -> int:
    """Return TAI-UTC in seconds for UTC datetime ``t``.

    Returns 19 (= GPS offset) for any time before the GPS epoch.
    """
    if t < GPS_EPOCH:
        return 19
    last = 19
    for date, n in LEAP_SECONDS:
        if t >= date:
            last = n
        else:
            break
    return last


def datetime_to_gps(t: datetime) -> tuple[int, float]:
    """Convert UTC datetime to ``(gps_week, seconds_into_week)``.

    Parameters
    ----------
    t:
        UTC datetime (naive, assumed UTC).

    Returns
    -------
    week:
        GPS week number (no roll-over correction).
    sow:
        Seconds into the GPS week, in [0, 604800).
    """
    delta = t - GPS_EPOCH + timedelta(seconds=leap_seconds_at(t) - 19)
    total = delta.total_seconds()
    week = int(total // SECONDS_PER_WEEK)
    sow = total - week * SECONDS_PER_WEEK
    return week, sow


def gps_to_datetime(week: int, sow: float) -> datetime:
    """Convert ``(gps_week, seconds_of_week)`` to a UTC ``datetime``.

    Inverts :func:`datetime_to_gps` to within microsecond precision.
    """
    gps_t = GPS_EPOCH + timedelta(seconds=week * SECONDS_PER_WEEK + sow)
    # Subtract leap-second offset to get UTC.
    return gps_t - timedelta(seconds=leap_seconds_at(gps_t) - 19)


def datetime_to_gps_seconds(t: datetime) -> float:
    """Return continuous GPS-seconds since the GPS epoch (no week wrap)."""
    delta = t - GPS_EPOCH + timedelta(seconds=leap_seconds_at(t) - 19)
    return delta.total_seconds()


def gps_seconds_to_datetime(s: float) -> datetime:
    """Inverse of :func:`datetime_to_gps_seconds`."""
    gps_t = GPS_EPOCH + timedelta(seconds=s)
    return gps_t - timedelta(seconds=leap_seconds_at(gps_t) - 19)


def gps_week_rollover(week_mod_1024: int, reference: datetime) -> int:
    """Resolve a rollover-ambiguous 10-bit GPS week against a reference date.

    GPS broadcasts the week number modulo 1024; older receivers and some
    file formats only store the low 10 bits. This helper returns the most
    plausible full week number close to ``reference``.
    """
    ref_week, _ = datetime_to_gps(reference)
    ref_era = ref_week // 1024
    candidates = (
        (ref_era - 1) * 1024 + week_mod_1024,
        ref_era * 1024 + week_mod_1024,
        (ref_era + 1) * 1024 + week_mod_1024,
    )
    return min(candidates, key=lambda w: abs(w - ref_week))


__all__ = [
    "GPS_EPOCH",
    "LEAP_SECONDS",
    "SECONDS_PER_WEEK",
    "datetime_to_gps",
    "datetime_to_gps_seconds",
    "gps_seconds_to_datetime",
    "gps_to_datetime",
    "gps_week_rollover",
    "leap_seconds_at",
]
