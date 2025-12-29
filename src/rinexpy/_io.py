"""Transparent file opener for RINEX-family files.

A single public entry point, :func:`opener`, returns a context manager that
yields a text stream for any supported input:

- plain ASCII RINEX (any extension);
- gzip (``.gz``) — including gzip-compressed CRINEX;
- bzip2 (``.bz2``) — including bzip2-compressed CRINEX;
- zip (``.zip``) — yields the first member as a text stream;
- LZW (``.Z``) — requires the optional ``ncompress`` extra;
- Hatanaka CRINEX (``.crx`` and any compressed variant) — requires the
  optional ``hatanaka`` extra;
- in-memory ``StringIO`` / generic text streams (passed through).

Magic-number detection is preferred over extension matching, because the
upstream world is unfortunately full of mis-extended files. Extensions are
used only as a tie-breaker.
"""

from __future__ import annotations

import bz2
import gzip
import io
import logging
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from ._types import FileLike
from ._version import first_nonblank_line, rinex_version

log = logging.getLogger(__name__)

# Optional extras — kept lazy so the package imports cleanly without them.
try:
    from hatanaka import crx2rnx as _crx2rnx
except ImportError:
    _crx2rnx = None  # type: ignore[assignment]

try:
    from ncompress import decompress as _unlzw
except ImportError:
    _unlzw = None  # type: ignore[assignment]

# File-signature byte sequences. See https://en.wikipedia.org/wiki/List_of_file_signatures
_MAGIC_GZIP = b"\x1f\x8b"
_MAGIC_BZIP2 = b"\x42\x5a\x68"
_MAGIC_ZIP = b"\x50\x4b"
_MAGIC_LZW = b"\x1f\x9d"

# Files larger than this trigger an INFO log message at open time so the user
# notices when they're about to wait for a multi-hundred-MB read.
_LARGE_FILE_THRESHOLD_BYTES = 100_000_000

# Plain-text files this big or bigger get mmap'd instead of fully read.
# Below this we just read() because the syscall overhead of mmap dominates.
_MMAP_THRESHOLD_BYTES = 50_000_000


def _mmap_text(path: Path) -> str:
    """Return ``path`` decoded as ASCII, via ``mmap``.

    Saves a copy through the heap on multi-GB plain-text files. Decode
    is still required because Python text streams want ``str``, not
    ``bytes``; the savings come from skipping the explicit ``read()``.
    """
    import mmap

    with path.open("rb") as fp, mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return mm[:].decode("ascii", errors="ignore")


def _is_crinex_stream(stream: IO[str]) -> bool:
    """Peek at ``stream`` and return whether it begins with a CRINEX header.

    The stream is rewound to position 0 on return, so this is non-destructive.
    """
    stream.seek(0)
    try:
        _, is_crinex = rinex_version(first_nonblank_line(stream))
    except ValueError:
        is_crinex = False
    stream.seek(0)
    return is_crinex


def _decode_crinex(stream: IO[str]) -> io.StringIO:
    """Decode a CRINEX text stream into a RINEX text stream.

    Prefers the in-tree decoder (:mod:`rinexpy.crinex`, backed by the
    ``rinexpy_native`` C++ kernels) which handles both CRINEX 1 and
    CRINEX 3 byte-for-byte against the upstream reference. Falls back
    to the legacy ``hatanaka`` Python package only when the in-tree
    path is unavailable.
    """
    raw = stream.read()
    # In-tree decoder first when the C++ kernels are available.
    try:
        from . import _native
        if _native.TextDiffState is not None and _native.CrinexChannel is not None:
            from .crinex import crx2rnx as _intree_crx2rnx
            return io.StringIO(_intree_crx2rnx(raw))
    except (ImportError, ValueError, NotImplementedError):
        pass
    if _crx2rnx is None:
        raise ImportError(
            "CRINEX (.crx) input needs the C++ extension: install with "
            "`uv sync --extra native`. (As a Python-only fallback, the "
            "legacy `[hatanaka]` extra still works.)"
        )
    return io.StringIO(_crx2rnx(raw))


@contextmanager
def opener(fn: FileLike, *, header: bool = False) -> Iterator[IO[str]]:
    """Yield a text stream for ``fn``, transparently decompressing as needed.

    Parameters
    ----------
    fn:
        Path or open text stream to read. Strings are expanded via
        :meth:`pathlib.Path.expanduser`.
    header:
        If True, the caller only intends to read the header. CRINEX files are
        therefore *not* run through ``hatanaka.crx2rnx``, since the header is
        unchanged by Hatanaka compression and the conversion is expensive.
        Default ``False``.

    Yields
    ------
    IO[str]
        A text stream positioned at the start of the (decompressed) data.

    Raises
    ------
    FileNotFoundError
        If ``fn`` is a path that does not exist.
    ImportError
        If the file uses LZW or CRINEX compression and the corresponding
        optional extra is not installed.
    OSError
        If ``fn`` is of a type rinexpy does not know how to open.
    """
    if isinstance(fn, str):
        fn = Path(fn).expanduser()

    if isinstance(fn, io.StringIO):
        fn.seek(0)
        yield fn
        return

    if not isinstance(fn, Path):
        # Generic text stream (e.g. TextIOWrapper). Pass through as-is.
        yield fn
        return

    if not fn.is_file():
        raise FileNotFoundError(fn)

    finfo = fn.stat()
    if finfo.st_size > _LARGE_FILE_THRESHOLD_BYTES:
        log.info("opening %.1f MB %s", finfo.st_size / 1e6, fn.name)

    # Read the magic bytes for content-based dispatch.
    with fn.open("rb") as raw:
        magic = raw.read(4)

    suffix = fn.suffix.lower()

    if suffix == ".gz" or magic.startswith(_MAGIC_GZIP):
        with gzip.open(fn, "rt", encoding="ascii", errors="ignore") as f:
            if not header and _is_crinex_stream(f):
                # gzip's transparent decoding doesn't materialize content
                # until the first read, so we have to fully consume it before
                # the Hatanaka decoder can see it.
                yield _decode_crinex(f)
            else:
                yield f
        return

    if suffix == ".bz2" or magic.startswith(_MAGIC_BZIP2):
        with bz2.open(fn, "rt", encoding="ascii", errors="ignore") as f:
            if not header and _is_crinex_stream(f):
                yield _decode_crinex(f)
            else:
                yield f
        return

    if suffix == ".zip" or magic.startswith(_MAGIC_ZIP):
        with zipfile.ZipFile(fn, "r") as z:
            for member in z.namelist():
                with z.open(member, "r") as raw:
                    text = io.TextIOWrapper(raw, encoding="ascii", errors="ignore").read()
                    yield io.StringIO(text)
        return

    if suffix == ".z" or magic.startswith(_MAGIC_LZW):
        if _unlzw is None:
            raise ImportError(
                "ncompress extra is required for .Z (LZW) input: "
                "`uv add 'rinexpy[lzw]'` or `pip install ncompress`."
            )
        with fn.open("rb") as raw:
            text = _unlzw(raw.read()).decode("ascii", errors="ignore")
        stream = io.StringIO(text)
        if not header and _is_crinex_stream(stream):
            yield _decode_crinex(stream)
        else:
            yield stream
        return

    # Plain text fallback — could still be Hatanaka CRINEX though.
    if not header and finfo.st_size > _MMAP_THRESHOLD_BYTES:
        # For really large local plain-text files, mmap saves the read()
        # of the whole file into the heap — we get a virtual view that
        # the OS pages in on demand.
        text = _mmap_text(fn)
        with io.StringIO(text) as f:
            if _is_crinex_stream(f):
                yield _decode_crinex(f)
            else:
                yield f
        return
    with fn.open("r", encoding="ascii", errors="ignore") as f:
        if not header and _is_crinex_stream(f):
            yield _decode_crinex(f)
        else:
            yield f


__all__ = ["opener"]
