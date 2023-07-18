"""RINEX 2/3 OBS/NAV header parsing and the public :func:`rinexinfo` helper.

The public surface of this module is small:

- :func:`rinexinfo`: peek at a file (or text stream) and return basic metadata
  (version, file type, system code, file kind).
- :func:`obsheader2`, :func:`obsheader3`: parse a complete OBS header dict.
- :func:`navheader2`, :func:`navheader3`: parse a complete NAV header dict.
- :func:`rinexheader`: dispatch to the right header parser based on
  :func:`rinexinfo`.

All parsers return ``dict``s rather than dataclasses to remain a drop-in
replacement for georinex; downstream code routinely indexes these dicts by
the literal RINEX header label (e.g. ``"APPROX POSITION XYZ"``).
"""

from __future__ import annotations

import io
import logging
from collections.abc import Hashable
from math import ceil
from pathlib import Path
from typing import IO, Any

import numpy as np
import xarray as xr

from ._common import fortran_float
from ._io import opener
from ._time import parse_header_epoch
from ._types import FileLike, MeasSelection
from ._version import (
    detect_filetype,
    detect_systems,
    first_nonblank_line,
    rinex_version,
)

log = logging.getLogger(__name__)

try:
    from pymap3d import ecef2geodetic as _ecef2geodetic
except ImportError:
    _ecef2geodetic = None  # type: ignore[assignment]


def rinexinfo(fn: FileLike) -> dict[Hashable, Any]:
    """Return a small dict describing a RINEX, CRINEX, SP3, or NetCDF file.

    Parameters
    ----------
    fn:
        Path to a file, or an already-open text stream.

    Returns
    -------
    dict
        Always contains ``rinextype`` (one of ``"obs"``, ``"nav"``, ``"sp3"``,
        or — for ``.nc`` inputs — a list of types found). For RINEX/CRINEX/SP3
        also contains ``version``, ``filetype``, and ``systems``.

    Raises
    ------
    ValueError
        If the file is unrecognisable as a RINEX-family file.
    """
    if isinstance(fn, (str, Path)):
        path = Path(fn).expanduser()
        if path.suffix == ".nc":
            rinex_types: list[str] = []
            attrs: dict[Hashable, Any] = {}
            for grp in ("OBS", "NAV"):
                try:
                    ds = xr.open_dataset(path, group=grp)
                except OSError:
                    continue
                rinex_types.append(grp.lower())
                # Merge group attrs but never let the group's own
                # 'rinextype' string clobber our accumulating list.
                for k, v in ds.attrs.items():
                    if k != "rinextype":
                        attrs[k] = v
            attrs["rinextype"] = rinex_types
            return attrs

        with opener(path, header=True) as stream:
            return rinexinfo(stream)

    # Stream branch.
    fn.seek(0)
    try:
        line = first_nonblank_line(fn)
    except (TypeError, AttributeError, ValueError) as e:
        raise ValueError(f"not a known/valid RINEX file: {e}") from e

    # SP3 short-circuit.
    if line.startswith(("#a", "#c", "#d")):
        return {"version": line[1], "rinextype": "sp3"}

    try:
        version, _ = rinex_version(line)
    except ValueError as e:
        raise ValueError(f"not a known/valid RINEX file: {e}") from e

    return {
        "version": version,
        "filetype": line[20],
        "rinextype": detect_filetype(line, version),
        "systems": detect_systems(line, version),
    }


def _iter_header_lines(stream: IO[str]):
    """Yield header lines until ``END OF HEADER`` is seen (consuming it)."""
    for line in stream:
        if "END OF HEADER" in line:
            return
        yield line


def _maybe_compute_geodetic(hdr: dict[Hashable, Any]) -> None:
    """Populate ``hdr['position_geodetic']`` from ECEF if pymap3d is available."""
    if "position" not in hdr or _ecef2geodetic is None or len(hdr["position"]) != 3:
        return
    hdr["position_geodetic"] = _ecef2geodetic(*hdr["position"])


def obsheader2(
    f: FileLike,
    *,
    useindicators: bool = False,
    meas: MeasSelection = None,
) -> dict[Hashable, Any]:
    """Parse a RINEX-2 OBS header.

    Parameters
    ----------
    f:
        Path or open text stream positioned at the start of the file.
    useindicators:
        If True, multiplies the ``Nobsused`` count by 3 so the caller knows
        how many output columns to allocate (one for value + LLI + SSI).
    meas:
        Optional list of measurement-label prefixes to *keep*. Other
        measurements stay in ``hdr["fields"]`` but are filtered out via
        ``hdr["fields_ind"]``.

    Returns
    -------
    dict
        Header dict with the original RINEX label keys plus the derived
        keys ``systems``, ``Nobs``, ``Nl_sv``, ``Nobsused``, ``fields``,
        ``fields_ind``, optionally ``t0``, ``t1``, ``interval``, ``rxmodel``,
        ``position``, ``position_geodetic``.
    """
    if isinstance(f, (str, Path)):
        with opener(f, header=True) as h:
            return obsheader2(h, useindicators=useindicators, meas=meas)

    f.seek(0)

    if isinstance(meas, str):
        meas = [meas]
    if not meas or not meas[0].strip():
        meas = None

    hdr = rinexinfo(f)
    n_obs = 0

    for line in _iter_header_lines(f):
        label = line[60:80].strip()
        content = line[:60]

        if label == "# / TYPES OF OBSERV":
            if n_obs == 0:
                n_obs = int(content[:6])
                hdr[label] = content[6:].split()
            else:
                hdr[label] += content[6:].split()
        elif label not in hdr:
            hdr[label] = content
        else:
            hdr[label] += " " + content

    try:
        hdr["systems"] = hdr["RINEX VERSION / TYPE"][40]
    except (KeyError, IndexError):
        pass

    hdr["Nobs"] = n_obs
    hdr["Nl_sv"] = ceil(n_obs / 5)

    try:
        hdr["position"] = [float(x) for x in hdr["APPROX POSITION XYZ"].split()]
        _maybe_compute_geodetic(hdr)
    except (KeyError, ValueError):
        pass

    try:
        fields = hdr["# / TYPES OF OBSERV"]
        if hdr["Nobs"] != len(fields):
            log.error(
                "%s: declared Nobs (%d) != number of fields (%d)",
                getattr(f, "name", "<stream>"),
                hdr["Nobs"],
                len(fields),
            )
            hdr["Nobs"] = len(fields)

        if isinstance(meas, (tuple, list, np.ndarray)):
            mask = np.zeros(len(fields), dtype=bool)
            for prefix in meas:
                for i, field in enumerate(fields):
                    if field.startswith(prefix):
                        mask[i] = True
            hdr["fields_ind"] = np.nonzero(mask)[0]
            hdr["fields"] = [fields[i] for i in hdr["fields_ind"]]
        else:
            hdr["fields_ind"] = np.arange(hdr["Nobs"])
            hdr["fields"] = list(fields)
    except KeyError:
        pass

    hdr["Nobsused"] = hdr["Nobs"]
    if useindicators:
        hdr["Nobsused"] *= 3

    try:
        hdr["# OF SATELLITES"] = int(hdr["# OF SATELLITES"][:6])
    except (KeyError, ValueError):
        pass

    for key in ("TIME OF FIRST OBS", "TIME OF LAST OBS"):
        if key in hdr:
            try:
                hdr["t0" if key == "TIME OF FIRST OBS" else "t1"] = parse_header_epoch(hdr[key])
            except ValueError:
                pass

    if "INTERVAL" in hdr:
        try:
            hdr["interval"] = float(hdr["INTERVAL"][:10])
        except ValueError:
            pass

    if "REC # / TYPE / VERS" in hdr:
        try:
            hdr["rxmodel"] = " ".join(hdr["REC # / TYPE / VERS"].split()[1:-1])
        except ValueError:
            pass

    return hdr


def obsheader3(
    f: FileLike,
    *,
    use: set[str] | None = None,
    meas: MeasSelection = None,
) -> dict[Hashable, Any]:
    """Parse a RINEX-3 OBS header.

    Parameters
    ----------
    f:
        Path or open text stream positioned at the start of the file.
    use:
        Optional set of single-letter system codes to keep (e.g. ``{"G", "E"}``).
    meas:
        Optional list of measurement-label prefixes to keep.

    Returns
    -------
    dict
        Header dict. Notable derived keys: ``fields`` (per-system list of
        observation labels), ``fields_ind`` (per-system NumPy index/slice
        into the per-epoch raw array), ``Fmax`` (max number of observation
        types across all systems, used to size the per-epoch buffer),
        optionally ``t0``, ``interval``, ``position``, ``position_geodetic``.
    """
    if isinstance(f, (str, Path)):
        with opener(f, header=True) as h:
            return obsheader3(h, use=use, meas=meas)

    if isinstance(meas, str):
        meas = [meas]
    if not meas or not meas[0].strip():
        meas = None

    fields: dict[str, list[str]] = {}
    f_max = 0

    hdr = rinexinfo(f)

    for line in _iter_header_lines(f):
        label = line[60:80]
        content = line[:60]

        if "SYS / # / OBS TYPES" in label:
            sys_letter = content[0]
            n_types = int(content[3:6])
            fields[sys_letter] = content[6:60].split()

            f_max = max(n_types, f_max)
            n_per_line = 13
            for _ in range(n_types // n_per_line):
                cont = f.readline()
                if "SYS / # / OBS TYPES" not in cont[60:]:
                    log.warning("OBS-type continuation line missing label: %r", cont[60:])
                fields[sys_letter] += cont[6:60].split()

            if sys_letter == "S":
                # SBAS may declare additional ranging codes
                if len(fields[sys_letter]) not in {n_types, n_types * 2}:
                    log.warning("SBAS field count mismatch for header line %r", line)
            elif len(fields[sys_letter]) != n_types:
                log.warning("field count mismatch for system %s", sys_letter)
            continue

        key = label.strip()
        if key not in hdr:
            hdr[key] = content
        else:
            hdr[key] += " " + content

    try:
        hdr["position"] = [float(x) for x in hdr["APPROX POSITION XYZ"].split()][:3]
        _maybe_compute_geodetic(hdr)
    except (KeyError, ValueError):
        pass

    if "TIME OF FIRST OBS" in hdr:
        try:
            hdr["t0"] = parse_header_epoch(hdr["TIME OF FIRST OBS"])
        except ValueError:
            pass

    if "INTERVAL" in hdr:
        try:
            hdr["interval"] = float(hdr["INTERVAL"][:10])
        except ValueError:
            pass

    if use:
        if not set(fields) & set(use):
            raise KeyError(f"system type {use} not found in RINEX file")
        fields = {k: fields[k] for k in use if k in fields}

    sys_ind: dict[str, Any] = {}
    if isinstance(meas, (tuple, list, np.ndarray)):
        for sk in fields:
            mask = np.zeros(len(fields[sk]), dtype=bool)
            for prefix in meas:
                for i, field in enumerate(fields[sk]):
                    if field.startswith(prefix):
                        mask[i] = True
            fields[sk] = [f for f, keep in zip(fields[sk], mask) if keep]
            # Per-system masks: each obs type uses 3 cells (value, LLI, SSI).
            row = np.empty(f_max * 3, dtype=bool)
            for j, keep in enumerate(mask):
                row[j * 3 : j * 3 + 3] = keep
            sys_ind[sk] = row
    else:
        sys_ind = dict.fromkeys(fields, np.s_[:])

    hdr["fields"] = fields
    hdr["fields_ind"] = sys_ind
    hdr["Fmax"] = f_max
    return hdr


def navheader2(f: FileLike) -> dict[Hashable, Any]:
    """Parse a RINEX-2 NAV header into a dict."""
    if isinstance(f, (str, Path)):
        with opener(f, header=True) as h:
            return navheader2(h)

    hdr = rinexinfo(f)
    for line in _iter_header_lines(f):
        label = line[60:].strip()
        hdr[label] = line[:60]
    return hdr


def navheader3(f: FileLike) -> dict[Hashable, Any]:
    """Parse a RINEX-3 NAV header into a dict, decoding correction tables."""
    if isinstance(f, (str, Path)):
        with opener(f, header=True) as h:
            return navheader3(h)

    hdr = rinexinfo(f)

    for line in _iter_header_lines(f):
        label = line[60:].strip()
        content = line[:60]

        if label == "IONOSPHERIC CORR":
            bucket = hdr.setdefault(label, {})
            kind = content[:4].strip()
            n = 3 if kind == "GAL" else 4
            bucket[kind] = [
                fortran_float(content[5 + i * 12 : 5 + (i + 1) * 12]) for i in range(n)
            ]
        elif label == "TIME SYSTEM CORR":
            bucket = hdr.setdefault(label, {})
            kind = content[:4].strip()
            bucket[kind] = [
                fortran_float(content[5:22]),
                fortran_float(content[22:38]),
                int(content[38:45]),
                int(content[45:50]),
            ]
        else:
            hdr[label] = content
    return hdr


def rinexheader(fn: FileLike) -> dict[Hashable, Any]:
    """Return a parsed header dict for any RINEX-family input.

    Dispatch is purely on the result of :func:`rinexinfo` (version + file
    kind); the caller does not need to know which sub-parser to invoke.

    Parameters
    ----------
    fn:
        Path or open text stream.

    Returns
    -------
    dict
        The RINEX header. For ``.nc`` inputs this is just the metadata as
        returned by :func:`rinexinfo`.

    Raises
    ------
    TypeError
        If ``fn`` is not a Path, str, ``StringIO`` or ``TextIOWrapper``.
    ValueError
        If ``fn`` is some other RINEX flavor that we don't have a header
        parser for (e.g. SP3).
    """
    if isinstance(fn, (str, Path)):
        path = Path(fn).expanduser()
        if path.suffix == ".nc":
            return rinexinfo(path)
        with opener(path, header=True) as h:
            return rinexheader(h)

    if isinstance(fn, io.StringIO):
        fn.seek(0)
    elif isinstance(fn, io.TextIOWrapper):
        pass
    elif hasattr(fn, "read"):
        pass
    else:
        raise TypeError(f"unknown RINEX file type {type(fn).__name__}")

    info = rinexinfo(fn)
    version = int(info["version"])
    rinex_type = info["rinextype"]

    if version in {1, 2}:
        if rinex_type == "obs":
            return obsheader2(fn)
        if rinex_type == "nav":
            return navheader2(fn)
    elif version == 3:
        if rinex_type == "obs":
            return obsheader3(fn)
        if rinex_type == "nav":
            return navheader3(fn)

    raise ValueError(f"no header parser for {info}")


__all__ = [
    "navheader2",
    "navheader3",
    "obsheader2",
    "obsheader3",
    "rinexheader",
    "rinexinfo",
]
