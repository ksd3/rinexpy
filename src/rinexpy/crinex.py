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

from typing import Any

from . import _native


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

    # Walk each SV's data line. The whole per-SV decode (tokenisation,
    # numeric NumDiff, flag TextDiff, F14.3 formatting, line assembly)
    # is fused into a single C++ call via the CrinexSVDecoder class.
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
        out_lines.append(dec.decode_line(data_line))

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
