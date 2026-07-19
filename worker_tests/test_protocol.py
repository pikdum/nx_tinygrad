import io

import pytest

import protocol
from errors import ProtocolError


def test_frame_roundtrip_no_blobs():
    frame = protocol.encode_frame(42, {"command": "hello", "args": {"protocol_version": 1}}, [])
    req_id, meta, blobs = protocol.decode_frame(frame)
    assert req_id == 42
    assert meta == {"command": "hello", "args": {"protocol_version": 1}}
    assert blobs == []


def test_frame_roundtrip_multiple_blobs():
    payload = [b"\x01\x02\x03", b"", b"\xab" * 1000]
    frame = protocol.encode_frame(7, {"command": "execute"}, payload)
    req_id, meta, blobs = protocol.decode_frame(frame)
    assert req_id == 7
    assert blobs == payload


def test_large_request_id():
    big = 0xFFFFFFFFFFFFFFFF
    frame = protocol.encode_frame(big, {"command": "x"}, [])
    req_id, _, _ = protocol.decode_frame(frame)
    assert req_id == big


def test_bad_magic_rejected():
    with pytest.raises(ProtocolError):
        protocol.decode_frame(b"NOPE" + b"\x00" * 16)


def test_packet_framing_roundtrip():
    frame = protocol.encode_frame(1, {"command": "stats"}, [b"data"])
    out = io.BytesIO()
    protocol.write_packet(out, frame)
    inp = io.BytesIO(out.getvalue())
    got = protocol.read_packet(inp)
    assert got == frame
    # A second read hits EOF cleanly.
    assert protocol.read_packet(inp) is None


def test_truncated_packet_raises():
    inp = io.BytesIO(b"\x00\x00\x00\x10short")  # claims 16 bytes, has 5
    with pytest.raises(ProtocolError):
        protocol.read_packet(inp)
