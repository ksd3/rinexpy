"""CRINEX (Hatanaka) decompressor: in-tree CRINEX 1 / 3 -> RINEX 2 / 3 OBS.

Reference: Hatanaka, Y. (2008), "A Compression Format and Tools for
GNSS Observation Data". The format applies two complementary
compression schemes:

- **TextDiff** (character-wise positional delta) for text fields:
  the epoch line, the per-SV LLI/SSI 2*n_obs-char string.
- **NumDiff** (k-th order integer differencing, default k=3) for
  numeric observation values.

This module drives the C++ kernels in :mod:`rinexpy_native` to
reconstruct an absolute RINEX OBS byte stream from a compressed
CRINEX stream. Drop-in replacement for the upstream ``hatanaka``
Python package's :func:`crx2rnx` for both CRINEX 1 (wrapping
RINEX 2 OBS) and CRINEX 3 (wrapping RINEX 3 OBS).
"""

from __future__ import annotations

from typing import Any

from . import _native


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def _parse_obs_header_v3(rinex_header_lines: list[str]) -> dict[str, list[str]]:
    """Walk the RINEX 3 header and return the per-system observation list.

    Output maps a single-letter system code (G, R, E, C, S, J, I) to
    the ordered list of observation codes for that system.
    """
    systems: dict[str, list[str]] = {}
    current_sys: str | None = None
    remaining = 0
    for line in rinex_header_lines:
        label = line[60:80].rstrip()
        if label == "SYS / # / OBS TYPES":
            head = line[:60]
            sys_char = head[0]
            if sys_char.strip():
                current_sys = sys_char
                remaining = int(head[3:6])
                systems[current_sys] = []
            obs_part = head[7:]
            codes = obs_part.split()
            for c in codes:
                if remaining > 0 and current_sys is not None:
                    systems[current_sys].append(c)
                    remaining -= 1
    return systems


def _parse_obs_header_v2(rinex_header_lines: list[str]) -> list[str]:
    """Walk a RINEX 2 header and return the global observation list.

    RINEX 2 uses one `# / TYPES OF OBSERV` block (no per-system
    breakdown). The first line carries `Nobs` in cols 0-6 plus up
    to 9 codes; if Nobs > 9, the remaining codes spill onto
    continuation lines (each carrying up to 9 more).
    """
    codes: list[str] = []
    remaining = 0
    for line in rinex_header_lines:
        label = line[60:80].rstrip()
        if label == "# / TYPES OF OBSERV":
            head = line[:60]
            if remaining == 0:
                try:
                    remaining = int(head[:6])
                except ValueError:
                    continue
            obs_part = head[6:]
            # Each code lives in a 6-char field but is typically 2-3 chars
            # right-justified. Use whitespace split.
            for c in obs_part.split():
                if remaining > 0:
                    codes.append(c)
                    remaining -= 1
            if remaining == 0:
                continue
    return codes


# ---------------------------------------------------------------------------
# CRINEX 3 -> RINEX 3 (existing path, kept intact)
# ---------------------------------------------------------------------------


def _decode_epoch_v3(
    crx_lines: list[str],
    idx: int,
    epoch_state: dict,
    sv_state: dict,
    systems: dict[str, list[str]],
) -> tuple[int, list[str]]:
    """Decode one CRINEX 3 epoch into RINEX 3 OBS lines."""
    epoch_in = crx_lines[idx]
    idx += 1
    if epoch_in and epoch_in[0] == "&":
        epoch_state["text"].reset()
    epoch_text = epoch_state["text"].step(epoch_in)
    while idx < len(crx_lines) and crx_lines[idx].strip() == "":
        idx += 1
    n_sv = int(epoch_text[32:35])

    sv_field = epoch_text[35 + 6:]
    sv_list = [sv_field[i:i + 3] for i in range(0, 3 * n_sv, 3)]

    out_lines = [epoch_text[: 32 + 3].rstrip()]

    for sv in sv_list:
        if idx >= len(crx_lines):
            break
        data_line = crx_lines[idx]
        idx += 1
        sys_char = sv[0]
        obs_codes = systems.get(sys_char, [])
        n_obs = len(obs_codes)
        if n_obs == 0:
            continue
        dec = sv_state.get(sv)
        if dec is None:
            dec = _native.CrinexSVDecoder(sv, n_obs)
            sv_state[sv] = dec
        obs_str = dec.decode_line(data_line)
        out_lines.append((sv + obs_str).rstrip())

    return idx, out_lines


# ---------------------------------------------------------------------------
# CRINEX 1 -> RINEX 2 OBS
# ---------------------------------------------------------------------------


def _wrap_obs_rinex2(obs_str: str, n_obs: int) -> list[str]:
    """Wrap a contiguous n_obs * 16-char obs string into RINEX 2 lines.

    RINEX 2 OBS data is laid out in lines of up to 5 obs each (80
    chars max). For each line the reference output:

    - finds the rightmost non-empty 16-char obs slot,
    - truncates the line at the end of that slot,
    - if NO slot is non-empty, keeps a single 16-char slot of spaces
      (a placeholder so the line isn't silently dropped).
    """
    OBS_PER_LINE = 5
    lines: list[str] = []
    for start in range(0, n_obs, OBS_PER_LINE):
        end = min(start + OBS_PER_LINE, n_obs)
        chunk = obs_str[start * 16 : end * 16]
        last_nonempty = -1
        for slot in range(end - start):
            base = slot * 16
            if chunk[base : base + 16] != " " * 16:
                last_nonempty = slot
        if last_nonempty < 0:
            lines.append("")
        else:
            keep = (last_nonempty + 1) * 16
            # rstrip the final kept slot's trailing LLI/SSI spaces to
            # match hatanaka's output convention (a slot ending with
            # blank flags drops them; non-blank flags are preserved).
            lines.append(chunk[:keep].rstrip())
    return lines


def _read_epoch_line_v2(
    crx_lines: list[str], idx: int, epoch_state: dict
) -> tuple[int, str]:
    """Read one CRINEX 1 epoch header line through TextDiff.

    Handles the leading-``&`` reinitialisation marker: when present,
    the TextDiff state is discarded before stepping, so the rest of
    the line becomes the absolute reconstruction (with the ``&``
    itself decoded to a leading space).
    """
    epoch_in = crx_lines[idx]
    idx += 1
    if epoch_in and epoch_in[0] == "&":
        epoch_state["text"].reset()
    epoch_text = epoch_state["text"].step(epoch_in)
    return idx, epoch_text


def _decode_epoch_v2(
    crx_lines: list[str],
    idx: int,
    epoch_state: dict,
    sv_state: dict,
    obs_codes: list[str],
) -> tuple[int, list[str]]:
    """Decode one CRINEX 1 epoch into RINEX 2 OBS lines.

    Handles event-flag epochs (flag in 2..6: pass through event-data
    lines verbatim) and normal data epochs (flag 0/1: parse SV list,
    decode each SV with the fused CrinexSVDecoder, wrap output at 5
    obs per line).
    """
    idx, epoch_text = _read_epoch_line_v2(crx_lines, idx, epoch_state)
    # Skip the blank line some encoders emit between the epoch
    # header and the first SV data row.
    while idx < len(crx_lines) and crx_lines[idx].strip() == "":
        idx += 1
    # RINEX 2 epoch line layout:
    #   col 0      : ' '   (leading space)
    #   cols 1- 25 : ' YY MM DD HH MM SS.SSSSSSS' (date+time)
    #   col 26     : ' '
    #   col 28     : event flag (single digit at col 28? typically col 28)
    #   cols 29-32 : n_sv (3-char right-justified)
    #   cols 32+   : SV PRN list (3 chars per SV, up to 12 on this line)
    # When the epoch has a clock-offset reading, it appears in cols 68-80
    # as F12.9. CRINEX 1 transmits it as a NumDiff in the trailing part
    # of the data section (one line per epoch after the SV rows).
    try:
        epoch_flag = int(epoch_text[28:29])
    except (ValueError, IndexError):
        epoch_flag = 0
    try:
        n_sv = int(epoch_text[29:32])
    except (ValueError, IndexError):
        n_sv = 0

    if epoch_flag in (2, 3, 4, 5, 6):
        # Event-flag epoch: pass through n_sv lines of free text.
        out: list[str] = [epoch_text.rstrip()]
        for _ in range(n_sv):
            if idx >= len(crx_lines):
                break
            out.append(crx_lines[idx])
            idx += 1
        return idx, out

    # Normal data epoch: parse the SV list (all on one line in CRINEX 1,
    # regardless of n_sv).
    sv_field = epoch_text[32:]
    sv_list = [sv_field[i:i + 3] for i in range(0, 3 * n_sv, 3)]

    # Build the RINEX 2 output epoch line(s): up to 12 SVs on the first
    # line, then continuation lines (32 leading spaces + up to 12 SVs).
    head = epoch_text[:32]
    first_chunk = sv_list[:12]
    out_lines = [(head + "".join(first_chunk)).rstrip()]
    for start in range(12, len(sv_list), 12):
        chunk = sv_list[start:start + 12]
        out_lines.append((" " * 32 + "".join(chunk)).rstrip())

    # Per-SV data rows.
    for sv in sv_list:
        if idx >= len(crx_lines):
            break
        data_line = crx_lines[idx]
        idx += 1
        n_obs = len(obs_codes)
        if n_obs == 0:
            continue
        dec = sv_state.get(sv)
        if dec is None:
            # CRINEX 1 transmits LLI/SSI text as absolute per epoch
            # (no TextDiff against prev); flag this in the decoder.
            dec = _native.CrinexSVDecoder(sv, n_obs, True)
            sv_state[sv] = dec
        obs_str = dec.decode_line(data_line)
        out_lines.extend(_wrap_obs_rinex2(obs_str, n_obs))

    return idx, out_lines


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def crx2rnx(crx_text: str) -> str:
    """Decompress a CRINEX text stream into a RINEX OBS text stream.

    Handles both CRINEX 1 (wrapping RINEX 2 OBS) and CRINEX 3
    (wrapping RINEX 3 OBS) byte-for-byte against the upstream
    ``hatanaka`` reference. Returns the reconstructed RINEX text.
    """
    lines = crx_text.split("\n")
    if not lines or "CRINEX VERS" not in (lines[0][60:80] if len(lines[0]) >= 60 else ""):
        raise ValueError("not a CRINEX file (missing CRINEX VERS header)")
    crinex_version = lines[0][:20].strip()

    # Find end of RINEX header.
    body_start = None
    for i, line in enumerate(lines[2:], start=2):
        if line[60:80].rstrip() == "END OF HEADER":
            body_start = i + 1
            break
    if body_start is None:
        raise ValueError("CRINEX file: no END OF HEADER")

    out_header = lines[2:body_start]

    if crinex_version.startswith("3"):
        return _decode_v3_body(out_header, lines[body_start:])
    if crinex_version.startswith("1"):
        return _decode_v1_body(out_header, lines[body_start:])
    raise NotImplementedError(
        f"unsupported CRINEX version {crinex_version!r}"
    )


def _decode_v3_body(out_header: list[str], body_lines: list[str]) -> str:
    systems = _parse_obs_header_v3(out_header)
    epoch_state = {"text": _native.TextDiffState()}
    sv_state: dict[str, Any] = {}
    out_data: list[str] = []
    idx = 0
    n_lines = len(body_lines)
    while idx < n_lines:
        line = body_lines[idx]
        if line.strip() == "":
            idx += 1
            continue
        idx, ep_out = _decode_epoch_v3(body_lines, idx, epoch_state, sv_state, systems)
        out_data.extend(ep_out)
    return "\n".join(out_header + out_data) + "\n"


def _decode_v1_body(out_header: list[str], body_lines: list[str]) -> str:
    obs_codes = _parse_obs_header_v2(out_header)
    epoch_state = {"text": _native.TextDiffState()}
    sv_state: dict[str, Any] = {}
    out_data: list[str] = []
    idx = 0
    n_lines = len(body_lines)
    while idx < n_lines:
        line = body_lines[idx]
        if line.strip() == "":
            idx += 1
            continue
        idx, ep_out = _decode_epoch_v2(body_lines, idx, epoch_state, sv_state, obs_codes)
        out_data.extend(ep_out)
    return "\n".join(out_header + out_data) + "\n"


__all__ = ["crx2rnx"]
