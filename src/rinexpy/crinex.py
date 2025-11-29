"""CRINEX (Hatanaka) decompressor: in-tree CRINEX 3 -> RINEX 3 obs decoder.

Reference: Hatanaka, Y. (2008), "A Compression Format and Tools for
GNSS Observation Data". The format applies two complementary
compression schemes:

- **TextDiff** (character-wise positional delta) for text fields:
  the epoch line, the per-SV LLI/SSI 2*n_obs-char string.
- **NumDiff** (k-th order integer differencing, default k=3) for
  numeric observation values.

This module drives the C++ kernels in ``rinexpy_native`` to
reconstruct an absolute RINEX 3 OBS byte stream from a compressed
CRINEX 3 stream. It is a drop-in replacement for the optional
``hatanaka`` Python package's :func:`crx2rnx`; when ``rinexpy_native``
is importable, :func:`crx2rnx` here calls into it directly so the
``[hatanaka]`` extra is no longer required for CRINEX 3 input.

Scope: RINEX 3 / CRINEX 3 only. Older CRINEX 1 / RINEX 2 input falls
back to the ``hatanaka`` package when present.
"""

from __future__ import annotations

import re
from typing import Any

from . import _native


_INIT_TOKEN_RE = re.compile(r"^(\d+)&(-?\d+)$")
# Default differencing order if a channel does not carry an explicit
# "k&" marker on its first epoch (per the Hatanaka spec, k=3).
_DEFAULT_ORDER = 3


def _parse_obs_header(rinex_header_lines: list[str]) -> dict[str, list[str]]:
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
            # First line for a system: column 0 = letter, then n_obs,
            # then up to 13 obs codes.
            head = line[:60]
            sys_char = head[0]
            if sys_char.strip():
                current_sys = sys_char
                remaining = int(head[3:6])
                systems[current_sys] = []
            # Each obs code is in 4-char fields starting at column 7,
            # one space then 3 chars per code.
            obs_part = head[7:]
            codes = obs_part.split()
            # Take min(remaining, len(codes)) and decrement.
            for c in codes:
                if remaining > 0 and current_sys is not None:
                    systems[current_sys].append(c)
                    remaining -= 1
    return systems


def _split_numeric_tokens(line: str, n_obs: int) -> tuple[list[str], str]:
    """Split a CRINEX data line into ``n_obs`` numeric tokens and a
    trailing LLI/SSI string.

    Each numeric obs is space-terminated; an empty token between two
    adjacent spaces means "this observation is missing at this epoch".
    The remainder of the line (after the n_obs-th separator) is the
    LLI/SSI delta string, possibly trimmed of trailing spaces.
    """
    tokens: list[str] = []
    pos = 0
    n = len(line)
    for _ in range(n_obs):
        # Find next space starting from pos.
        sp = line.find(" ", pos)
        if sp < 0:
            # No more spaces. This token consumes the rest, and the
            # remainder is empty.
            tokens.append(line[pos:])
            pos = n
            break
        tokens.append(line[pos:sp])
        pos = sp + 1
    # Pad with empty tokens if we ran out of input.
    while len(tokens) < n_obs:
        tokens.append("")
    flags = line[pos:] if pos <= n else ""
    return tokens, flags


def _format_obs_value(int_value: int) -> str:
    """Format a NumDiff-reconstructed integer (= value_m * 1000) as
    the standard RINEX 3 F14.3 14-character field.

    Matches the hatanaka reference encoder's leading-zero convention:
    for values with ``|x| < 1`` the leading "0" before the decimal
    point is omitted (so 0.192 -> ".192", -0.192 -> "-.192").
    """
    if int_value < 0:
        sign = "-"
        absv = -int_value
    else:
        sign = ""
        absv = int_value
    whole, frac = divmod(absv, 1000)
    if whole == 0:
        body = f"{sign}.{frac:03d}"
    else:
        body = f"{sign}{whole}.{frac:03d}"
    return body.rjust(14)


def _decode_epoch(
    crx_lines: list[str],
    idx: int,
    epoch_state: dict,
    sv_state: dict,
    systems: dict[str, list[str]],
) -> tuple[int, list[str]]:
    """Decode one CRINEX epoch starting at line index ``idx``.

    Mutates ``epoch_state`` (TextDiff for the epoch line) and
    ``sv_state`` (per-SV TextDiff + per-channel NumDiff) in place.

    Returns ``(next_idx, output_lines)``: the index of the line right
    after the consumed epoch, and the RINEX 3 obs lines for this
    epoch (epoch header + per-SV data lines).
    """
    # Epoch line.
    epoch_in = crx_lines[idx]
    idx += 1
    epoch_text = epoch_state["text"].step(epoch_in)
    # Some CRINEX 3 encoders emit a blank line between the epoch
    # header and the first SV data row (legacy carryover from CRINEX
    # 1's separate SV-list line). Skip it so we don't mis-consume it
    # as a data line.
    while idx < len(crx_lines) and crx_lines[idx].strip() == "":
        idx += 1
    # The epoch line is "> YYYY MM DD HH MM SS.SSSSSSS  flag n_sv [sv list]".
    # n_sv lives in columns 32-35.
    n_sv = int(epoch_text[32:35])

    sv_field = epoch_text[35 + 6:]  # after time + flag + n_sv + 6 reserved spaces
    sv_list = [sv_field[i:i + 3] for i in range(0, 3 * n_sv, 3)]

    # Output the RINEX 3 epoch header (no SV list).
    out_lines = [epoch_text[: 32 + 3].rstrip()]
    # Strip trailing whitespace and keep "0 NN" right-padded if needed.
    # Actually hatanaka emits the epoch line including the n_sv field
    # padded — examine the test sample to verify.

    # Walk each SV's data line.
    for sv in sv_list:
        if idx >= len(crx_lines):
            break
        data_line = crx_lines[idx]
        idx += 1
        sys_char = sv[0]
        obs_codes = systems.get(sys_char, [])
        n_obs = len(obs_codes)
        if n_obs == 0:
            # Header didn't declare this constellation — skip.
            continue
        tokens, flag_delta = _split_numeric_tokens(data_line, n_obs)

        # Get / create per-SV state.
        st = sv_state.get(sv)
        if st is None:
            st = {
                "channels": [_native.CrinexChannel() for _ in range(n_obs)],
                "filled": [False] * n_obs,
                "flags": _native.TextDiffState(),
            }
            sv_state[sv] = st
        # Each numeric token decodes via NumDiff.
        values: list[str] = []  # 14-char obs fields
        for i, tok in enumerate(tokens):
            tok = tok.strip()
            if tok == "":
                # Missing obs at this epoch. Reset its channel state so
                # the next non-empty token must carry an init marker.
                st["channels"][i] = _native.CrinexChannel()
                st["filled"][i] = False
                values.append(" " * 14)
                continue
            m = _INIT_TOKEN_RE.match(tok)
            if m:
                order = int(m.group(1))
                value = int(m.group(2))
                ch = _native.CrinexChannel()
                ch.reset(order)
                reconstructed = ch.step(value)
                st["channels"][i] = ch
                st["filled"][i] = True
                values.append(_format_obs_value(reconstructed))
            else:
                if not st["filled"][i]:
                    # No init pending and the token isn't an init —
                    # use the default order with this value as init.
                    ch = _native.CrinexChannel()
                    ch.reset(_DEFAULT_ORDER)
                    reconstructed = ch.step(int(tok))
                    st["channels"][i] = ch
                    st["filled"][i] = True
                else:
                    reconstructed = st["channels"][i].step(int(tok))
                values.append(_format_obs_value(reconstructed))

        # Pad the LLI/SSI delta to 2*n_obs chars (trimmed trailing spaces).
        flag_padded = flag_delta.ljust(2 * n_obs)[: 2 * n_obs]
        flag_text = st["flags"].step(flag_padded)
        flag_text = flag_text.ljust(2 * n_obs)[: 2 * n_obs]

        # Compose the RINEX 3 obs line: "PRN" + per-obs (value + LLI + SSI).
        pieces = [sv]
        for k in range(n_obs):
            pieces.append(values[k])
            pieces.append(flag_text[2 * k : 2 * k + 2])
        out_lines.append("".join(pieces).rstrip())

    return idx, out_lines


def crx2rnx(crx_text: str) -> str:
    """Decompress a CRINEX 3 text stream into a RINEX 3 OBS text stream.

    Pure-Python orchestration driving the C++ TextDiff + NumDiff
    kernels in :mod:`rinexpy_native`. Validates that the input
    is CRINEX 3.0; falls back to the ``hatanaka`` package for older
    streams when that package is importable, and raises otherwise.
    """
    lines = crx_text.split("\n")
    # Parse the 2-line CRINEX header.
    if not lines or "CRINEX VERS" not in (lines[0][60:80] if len(lines[0]) >= 60 else ""):
        raise ValueError("not a CRINEX file (missing CRINEX VERS header)")
    crinex_version = lines[0][:20].strip()
    if not crinex_version.startswith("3"):
        try:
            import hatanaka  # type: ignore
        except ImportError:
            raise NotImplementedError(
                f"in-tree CRINEX decoder supports CRINEX 3 only "
                f"(got version {crinex_version!r}); install the "
                "`hatanaka` extra for older streams"
            ) from None
        return hatanaka.crx2rnx(crx_text)

    # Find end of RINEX header.
    body_start = None
    for i, line in enumerate(lines[2:], start=2):
        if line[60:80].rstrip() == "END OF HEADER":
            body_start = i + 1
            break
    if body_start is None:
        raise ValueError("CRINEX file: no END OF HEADER")

    # Build the output: drop CRINEX header (lines 0..1), keep the
    # RINEX header (lines 2..body_start-1) unchanged.
    out_header = lines[2:body_start]
    systems = _parse_obs_header(out_header)

    # Decode the data section.
    epoch_state = {"text": _native.TextDiffState()}
    sv_state: dict[str, Any] = {}
    out_data: list[str] = []
    idx = body_start
    n_lines = len(lines)
    while idx < n_lines:
        # Skip blank lines (CRINEX uses one blank line after the epoch
        # header line on the FIRST epoch only? Actually let's check.)
        line = lines[idx]
        if line.strip() == "":
            idx += 1
            continue
        if not line.startswith(">") and not line.startswith(" ") \
           and not line.startswith("&") and not (line[0].isdigit()
                                                  or line[0] == "-"):
            # Unrecognised; skip defensively.
            idx += 1
            continue
        idx, ep_out = _decode_epoch(lines, idx, epoch_state, sv_state, systems)
        out_data.extend(ep_out)

    return "\n".join(out_header + out_data) + "\n"


__all__ = ["crx2rnx"]
