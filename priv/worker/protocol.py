"""Framed wire protocol for the nx_tinygrad worker.

Two layers:

1. Port packet framing: the Erlang Port is opened with ``packet: 4``, so every
   message on stdin/stdout is prefixed with a 4-byte big-endian length. We read
   and write those prefixes manually (:func:`read_packet` / :func:`write_packet`).

2. The XTG1 frame carried inside each packet::

       4  bytes  magic "XTG1"
       8  bytes  request id (u64 BE)
       4  bytes  JSON metadata length (u32 BE)
       2  bytes  blob count (u16 BE)
       2  bytes  reserved
       8*N       blob lengths (u64 BE)
       M         UTF-8 JSON metadata
       ...       concatenated blob bytes

stdout is reserved for protocol frames only; all logging goes to stderr.
"""
from __future__ import annotations

import json
import struct

from errors import ProtocolError

MAGIC = b"XTG1"
_HEADER = struct.Struct(">4sQIH2x")  # magic, req_id, json_len, blob_count, 2 reserved
MAX_FRAME_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB hard bound


def read_exact(stream, n: int) -> bytes | None:
    """Read exactly ``n`` bytes, or return None on clean EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def read_packet(stream) -> bytes | None:
    """Read one ``packet: 4`` framed message payload, or None on EOF."""
    header = read_exact(stream, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length > MAX_FRAME_BYTES:
        raise ProtocolError(f"incoming frame too large: {length} bytes")
    payload = read_exact(stream, length)
    if payload is None:
        raise ProtocolError("truncated frame: EOF inside packet body")
    return payload


def write_packet(stream, payload: bytes) -> None:
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def decode_frame(payload: bytes) -> tuple[int, dict, list[bytes]]:
    """Decode an XTG1 frame into ``(request_id, metadata, blobs)``."""
    if len(payload) < _HEADER.size:
        raise ProtocolError("frame shorter than header")
    magic, req_id, json_len, blob_count = _HEADER.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: {magic!r}")

    offset = _HEADER.size
    lengths_size = blob_count * 8
    if offset + lengths_size > len(payload):
        raise ProtocolError("truncated blob length table")

    blob_lengths = []
    for _ in range(blob_count):
        (blen,) = struct.unpack_from(">Q", payload, offset)
        offset += 8
        blob_lengths.append(blen)

    meta_bytes = payload[offset : offset + json_len]
    offset += json_len
    if len(meta_bytes) != json_len:
        raise ProtocolError("truncated metadata")

    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"invalid JSON metadata: {exc}") from exc
    if not isinstance(meta, dict):
        raise ProtocolError("metadata must be a JSON object")

    blobs = []
    for blen in blob_lengths:
        blob = payload[offset : offset + blen]
        offset += blen
        if len(blob) != blen:
            raise ProtocolError("truncated blob")
        blobs.append(blob)

    if offset != len(payload):
        raise ProtocolError("trailing bytes after frame")

    return req_id, meta, blobs


def encode_frame(req_id: int, meta: dict, blobs: list[bytes] | None = None) -> bytes:
    blobs = blobs or []
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    parts = [
        _HEADER.pack(MAGIC, req_id, len(meta_bytes), len(blobs)),
        *[struct.pack(">Q", len(b)) for b in blobs],
        meta_bytes,
        *blobs,
    ]
    return b"".join(parts)
