"""Connect to an NTRIP caster and decode the live RTCM3 stream.

Two modes:

- live: opens an actual TCP connection (requires network + a public
  caster + valid mountpoint).
- offline: replays a pre-captured byte stream from disk so the example
  is runnable without network access.

Run from the repo root:

    uv run python examples/08_ntrip_to_rtcm.py --offline
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from rinexpy.rtcm3 import iter_messages


def _replay(path: Path):
    """Yield byte chunks from a captured RTCM3 dump for offline replay."""
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(4096)
            if not chunk:
                break
            yield chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="replay a captured stream instead of touching the net")
    parser.add_argument("--host", default="rtk2go.com")
    parser.add_argument("--port", type=int, default=2101)
    parser.add_argument("--mount", default="MOUNT01")
    parser.add_argument("--user", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--limit", type=int, default=20,
                        help="stop after N decoded messages")
    parser.add_argument("--capture", default=str(
        Path(__file__).resolve().parent / "_rtcm3_sample.bin"
    ), help="path of the captured byte stream for --offline mode")
    ns = parser.parse_args()

    if ns.offline:
        cap = Path(ns.capture)
        if not cap.is_file():
            # Synthesize a minimal valid frame on the fly so the example
            # runs without any external file.
            from rinexpy.rtcm3 import PREAMBLE, crc24q
            body = bytes([0x3E, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # msg 1000-ish
            head = bytes([PREAMBLE, 0, len(body)])
            crc = crc24q(head + body)
            frame = head + body + bytes([crc >> 16, (crc >> 8) & 0xFF, crc & 0xFF])
            cap.write_bytes(frame * 5)
            print(f"Wrote synthetic 5-frame capture to {cap}")
        chunks = _replay(cap)
    else:
        from rinexpy.ntrip import stream
        chunks = stream(
            ns.host, ns.mount,
            user=ns.user, password=ns.password,
            port=ns.port,
        )

    # Glue the chunks into a single byte stream for the RTCM3 framer.
    buf = io.BytesIO()
    for chunk in chunks:
        buf.write(chunk)
    buf.seek(0)

    counts: dict[int, int] = {}
    for i, msg in enumerate(iter_messages(buf), start=1):
        msg_id = msg["msg_id"]
        counts[msg_id] = counts.get(msg_id, 0) + 1
        print(f"#{i:3d}  msg {msg_id:4d}  "
              f"{', '.join(f'{k}={v!r}' for k, v in msg.items() if k != 'payload_bytes' and not isinstance(v, list))[:80]}")
        if i >= ns.limit:
            break

    print()
    print("Message-type counts:")
    for mid, n in sorted(counts.items()):
        print(f"  {mid}: {n}")


if __name__ == "__main__":
    main()
