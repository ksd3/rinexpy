"""Error-message helpers that attach source line numbers.

Wrap a stream with :class:`LineCountingStream` to surface the current
line on every readline; format errors via :func:`format_parse_error`.
"""

from __future__ import annotations

from typing import IO


class LineCountingStream:
    """Wrap a text stream and track the line counter on every ``readline``.

    Used by the streaming readers to surface the current source line in
    parse errors, so users can grep right at the offending byte.
    """

    def __init__(self, inner: IO[str], name: str | None = None) -> None:
        self.inner = inner
        self.line_no = 0
        self.name = name or getattr(inner, "name", "<stream>")

    def readline(self, *args, **kwargs) -> str:
        line = self.inner.readline(*args, **kwargs)
        if line:
            self.line_no += 1
        return line

    def read(self, *args, **kwargs) -> str:
        return self.inner.read(*args, **kwargs)

    def __iter__(self):
        for line in self.inner:
            self.line_no += 1
            yield line

    def seek(self, *args, **kwargs):
        self.line_no = 0
        return self.inner.seek(*args, **kwargs)

    def tell(self):
        return self.inner.tell()


def format_parse_error(name: str, line_no: int, line: str, message: str) -> str:
    """Build a parse-error string with file:line context.

    ``name`` is typically the filename or ``<stream>``; ``line`` is the
    text of the offending line (truncated to 80 chars).
    """
    snippet = line[:80].rstrip()
    return f"{name}:{line_no}: {message} | line: {snippet!r}"


__all__ = ["LineCountingStream", "format_parse_error"]
