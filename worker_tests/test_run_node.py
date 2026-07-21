"""The run_node command: eager single-op application to worker buffers."""
import numpy as np

import device
from main import Handler


def _handler():
    return Handler(device.parse_device("CPU"))


def _ok(meta):
    assert meta["ok"] is True, meta
    return meta["result"]


def _upload(handler, arr, dtype="f32"):
    meta, _ = handler.handle(
        {"command": "upload", "args": {"shape": list(arr.shape), "dtype": dtype}},
        [arr.tobytes()],
    )
    return _ok(meta)["id"]


def _run_node(handler, op, inputs, attrs, shape, dtype="f32", blobs=()):
    meta, _ = handler.handle(
        {
            "command": "run_node",
            "args": {"op": op, "attrs": attrs, "shape": shape, "dtype": dtype, "inputs": inputs},
        },
        list(blobs),
    )
    return meta


def _download(handler, buffer_id):
    meta, blobs = handler.handle({"command": "download", "args": {"id": buffer_id}}, [])
    result = _ok(meta)
    return np.frombuffer(blobs[0], dtype=np.float32).reshape(result["shape"])


def test_run_node_applies_op_to_handles():
    handler = _handler()
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    tid = _upload(handler, arr)

    meta = _run_node(
        handler, "transpose", [{"kind": "handle", "id": tid}], {"axes": [1, 0]}, [3, 2]
    )
    out = _ok(meta)
    assert out["shape"] == [3, 2]
    assert np.array_equal(_download(handler, out["id"]), arr.T)


def test_run_node_accepts_blob_inputs():
    handler = _handler()
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tid = _upload(handler, arr)

    meta = _run_node(
        handler,
        "multiply",
        [
            {"kind": "handle", "id": tid},
            {"kind": "blob", "blob_index": 0, "shape": [], "dtype": "f32"},
        ],
        {},
        [3],
        blobs=[np.float32(2.5).tobytes()],
    )
    out = _ok(meta)
    assert np.allclose(_download(handler, out["id"]), arr * 2.5)


def test_run_node_rejects_traced_function_ops():
    handler = _handler()
    meta = _run_node(handler, "while", [], {}, [1])
    assert meta["ok"] is False
    assert meta["error"]["class"] == "UnsupportedOperation"


def test_run_node_results_feed_captured_executables():
    # Movement-op results (permute views) must be materialized to plain
    # buffers: TinyJit captures with plain-buffer dummies, and replay rejects
    # inputs whose view structure differs (the SD "args mismatch in JIT"
    # regression).
    handler = _handler()
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    tid = _upload(handler, arr)

    meta = _run_node(
        handler, "transpose", [{"kind": "handle", "id": tid}], {"axes": [1, 0]}, [3, 2]
    )
    transposed_id = _ok(meta)["id"]

    graph = {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [3, 2], "dtype": "f32"}],
        "constants": [],
        "nodes": [{"id": 1, "op": "add", "inputs": [0, 0], "attrs": {}, "shape": [3, 2], "dtype": "f32"}],
        "outputs": [{"node": 1, "shape": [3, 2], "dtype": "f32"}],
    }
    compile_meta, _ = handler.handle(
        {"command": "compile", "args": {"graph": graph, "validate_capture": True}}, []
    )
    exec_id = _ok(compile_meta)["executable_id"]

    execute_meta, _ = handler.handle(
        {
            "command": "execute",
            "args": {
                "executable_id": exec_id,
                "output": "device",
                "inputs": [{"kind": "handle", "id": transposed_id, "dtype": "f32", "shape": [3, 2]}],
            },
        },
        [],
    )
    out = _ok(execute_meta)["outputs"][0]
    assert np.array_equal(_download(handler, out["id"]), arr.T * 2)
