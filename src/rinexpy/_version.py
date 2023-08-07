"""RINEX / CRINEX / SP3 version and file-type detection.

These helpers all operate on the very first non-blank line of a file. The
first 80 characters of a RINEX file are a fixed-width header record whose
columns 0-9 contain the version (right-justified ``%9.2f``), columns 20-39
contain the file type description, columns 40-59 contain the satellite system
code, and columns 60-80 contain the literal ``RINEX VERSION / TYPE`` (or
``CRINEX VERS   / TYPE`` for Hatanaka-compressed files).

SP3 files start with ``#a``, ``#c``, or ``#d`` indicating the SP3 version.
"""

from __future__ import annotations

from typing import IO

# Sentinels that appear in column 60-80 of the first RINEX header line.
_RINEX_TAG = "RINEX VERSION / TYPE"
_CRINEX_TAG = "CRINEX VERS   / TYPE"

# Sentinel that appears in column 20-40 of CRINEX (Hatanaka) files.
_CRINEX_FORMAT_TAG = "COMPACT RINEX FORMAT"

#: SP3 version letters we know how to read.
_SP3_VERSIONS = frozenset({"a", "c", "d"})


def rinex_version(line: str) -> tuple[float | str, bool]:
    """Decode the version and Hatanaka flag from the first line of a file.

    Parameters
    ----------
    line:
        The first non-blank line of a RINEX, CRINEX or SP3 file. Must be at
        least two characters long.

    Returns
    -------
    version:
        - For RINEX/CRINEX files, the numeric version as a float (e.g. ``3.04``).
        - For SP3 files, the string ``"sp3"`` followed by the version letter
          (one of ``"sp3a"``, ``"sp3c"``, ``"sp3d"``).
    is_crinex:
        ``True`` if the file is a Hatanaka-compressed RINEX (CRINEX) file,
        ``False`` otherwise (always ``False`` for SP3 files).

    Raises
    ------
    TypeError
        If ``line`` is not a ``str``.
    ValueError
        If the line is too short, the SP3 version letter is unsupported, or
        the RINEX header marker is missing/invalid.

    Examples
    --------
    >>> rinex_version("     3.03           OBSERVATION DATA    M (MIXED)" + " " * 11 + "RINEX VERSION / TYPE")
    (3.03, False)
    >>> rinex_version("#dP2019  1  1  0  0  0.00000000     192    ORBIT IGS14 HLM  IGS")
    ('sp3d', False)
    """
    if not isinstance(line, str):
        raise TypeError("need first line of RINEX/SP3 file as str")
    if len(line) < 2:
        raise ValueError(f"cannot decode RINEX/SP3 version from line:\n{line!r}")

    # SP3 files begin with '#' followed by the version letter.
    if line[0] == "#":
        if line[1] not in _SP3_VERSIONS:
            raise ValueError(f"SP3 versions handled: {sorted(_SP3_VERSIONS)}, got {line[1]!r}")
        return f"sp3{line[1]}", False

    # RINEX files have a standard 80-byte first line.
    if len(line) >= 80 and line[60:80] not in (_RINEX_TAG, _CRINEX_TAG):
        raise ValueError("the first line of the RINEX file header is corrupted.")

    try:
        version = float(line[:9])
    except ValueError as err:
        raise ValueError(f"could not parse RINEX version from {line[:9]!r}: {err}") from err

    return version, line[20:40] == _CRINEX_FORMAT_TAG


def detect_filetype(line: str, version: float | str) -> str:
    """Decode the RINEX file type letter from the first header line.

    Parameters
    ----------
    line:
        The first non-blank line of a RINEX file (≥ 41 chars).
    version:
        Numeric RINEX version, as returned by :func:`rinex_version`. Used
        only to distinguish RINEX 2 from RINEX 3 file-type semantics.

    Returns
    -------
    str
        One of ``"obs"``, ``"nav"``, ``"sp3"`` or — if the column-20 letter is
        not recognised — the raw character from column 20 of the header.

    Notes
    -----
    The mapping is asymmetric across versions: RINEX 2 uses ``N``/``G``/``E``
    in column 20 to distinguish GPS / GLONASS / Galileo navigation files,
    whereas RINEX 3 always uses ``N`` for navigation regardless of system and
    encodes the system in column 40.
    """
    if isinstance(version, str) and version.startswith("sp3"):
        return "sp3"

    flag = line[20]
    if flag in ("O", "C"):
        return "obs"
    if flag == "N" or "NAV" in line[20:40]:
        return "nav"
    return flag


def detect_systems(line: str, version: float | str) -> str:
    """Decode the satellite-system letter from the first header line.

    Parameters
    ----------
    line:
        The first non-blank line of a RINEX file (≥ 41 chars).
    version:
        Numeric RINEX version, as returned by :func:`rinex_version`.

    Returns
    -------
    str
        Single-character system code: one of ``G``, ``R``, ``E``, ``C``,
        ``J``, ``S``, ``I``, or ``M`` for mixed.
    """
    if int(version) == 2:
        match line[20]:
            case "N":
                return "G"
            case "G":
                return "R"
            case "E":
                return "E"
            case _:
                return line[40]
    return line[40]


def first_nonblank_line(stream: IO[str], max_lines: int = 10) -> str:
    """Return the first non-blank, ≤ 81-character line from ``stream``.

    Parameters
    ----------
    stream:
        Any text stream (file, ``StringIO``, etc.) that supports ``readline``.
    max_lines:
        Maximum number of lines to scan before giving up. Defaults to 10.

    Returns
    -------
    str
        The first line (with its trailing newline retained) whose stripped
        content is non-empty.

    Raises
    ------
    ValueError
        If ``max_lines`` is below 1, or no non-blank line was found within
        ``max_lines`` reads.
    """
    if max_lines < 1:
        raise ValueError("must read at least one line")

    line = ""
    for _ in range(max_lines):
        line = stream.readline(81)
        if line.strip():
            return line

    raise ValueError("could not find a valid first header line")


__all__ = [
    "detect_filetype",
    "detect_systems",
    "first_nonblank_line",
    "rinex_version",
]
