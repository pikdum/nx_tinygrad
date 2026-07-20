import numpy as np
import pytest
from tinygrad import Tensor

import executable as executable_mod
from compiler import compile_graph
from errors import CompileError, GraphValidationError


def _T(data):
    return Tensor(np.array(data, dtype=np.float32))


def test_compile_and_run_add():
    graph = {
        "version": 1,
        "inputs": [
            {"id": 0, "index": 0, "shape": [2], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [2], "dtype": "f32"},
        ],
        "constants": [],
        "nodes": [{"id": 2, "op": "add", "inputs": [0, 1], "attrs": {}, "shape": [2], "dtype": "f32"}],
        "outputs": [{"node": 2, "shape": [2], "dtype": "f32"}],
    }
    ex = compile_graph(1, graph, [], "CPU")
    [out] = ex.run([_T([1, 2]), _T([3, 4])])
    assert np.allclose(out.numpy(), [4, 6])


def test_scalar_constant_materialized():
    graph = {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [3], "dtype": "f32"}],
        "constants": [{"id": 1, "value": 10.0, "shape": [], "dtype": "f32"}],
        "nodes": [{"id": 2, "op": "add", "inputs": [0, 1], "attrs": {}, "shape": [3], "dtype": "f32"}],
        "outputs": [{"node": 2, "shape": [3], "dtype": "f32"}],
    }
    ex = compile_graph(2, graph, [], "CPU")
    [out] = ex.run([_T([1, 2, 3])])
    assert np.allclose(out.numpy(), [11, 12, 13])


def test_constant_from_blob():
    blob = np.array([100.0, 200.0], dtype=np.float32).tobytes()
    graph = {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [2], "dtype": "f32"}],
        "constants": [{"id": 1, "data_index": 0, "shape": [2], "dtype": "f32"}],
        "nodes": [{"id": 2, "op": "add", "inputs": [0, 1], "attrs": {}, "shape": [2], "dtype": "f32"}],
        "outputs": [{"node": 2, "shape": [2], "dtype": "f32"}],
    }
    ex = compile_graph(3, graph, [blob], "CPU")
    [out] = ex.run([_T([1, 2])])
    assert np.allclose(out.numpy(), [101, 202])


def test_validation_rejects_wrong_constant_blob_size():
    graph = {
        "version": 1,
        "inputs": [],
        "constants": [{"id": 0, "data_index": 0, "shape": [2], "dtype": "f32"}],
        "nodes": [],
        "outputs": [{"node": 0, "shape": [2], "dtype": "f32"}],
    }

    with pytest.raises(GraphValidationError, match="byte size"):
        compile_graph(13, graph, [b"too short"], "CPU")


def test_validation_rejects_output_spec_mismatch():
    graph = {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [2], "dtype": "f32"}],
        "constants": [],
        "nodes": [],
        "outputs": [{"node": 0, "shape": [1, 2], "dtype": "f32"}],
    }

    with pytest.raises(GraphValidationError, match="output shape"):
        compile_graph(14, graph, [], "CPU")


def test_validation_rejects_unknown_op():
    graph = {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [2], "dtype": "f32"}],
        "constants": [],
        "nodes": [{"id": 1, "op": "fft", "inputs": [0], "attrs": {}, "shape": [2], "dtype": "f32"}],
        "outputs": [{"node": 1, "shape": [2], "dtype": "f32"}],
    }
    with pytest.raises(GraphValidationError):
        compile_graph(4, graph, [], "CPU")


def test_tinyjit_capture_and_replay():
    graph = {
        "version": 1,
        "inputs": [
            {"id": 0, "index": 0, "shape": [2], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [2], "dtype": "f32"},
        ],
        "constants": [],
        "nodes": [{"id": 2, "op": "multiply", "inputs": [0, 1], "attrs": {}, "shape": [2], "dtype": "f32"}],
        "outputs": [{"node": 2, "shape": [2], "dtype": "f32"}],
    }
    ex = compile_graph(10, graph, [], "CPU", validate_capture=True)
    assert ex.kernel_count >= 1
    [o1] = ex.run([_T([1, 2]), _T([3, 4])])
    assert np.allclose(o1.numpy(), [3, 8])
    [o2] = ex.run([_T([10, 20]), _T([2, 2])])
    assert np.allclose(o2.numpy(), [20, 40])


def test_capture_validation_compares_replayed_values(monkeypatch):
    real_tinyjit = executable_mod.TinyJit

    class CorruptingTinyJit:
        def __init__(self, fn):
            self.inner = real_tinyjit(fn)
            self.calls = 0

        def __call__(self, *inputs):
            self.calls += 1
            outputs = self.inner(*inputs)
            if self.calls == 3:
                outputs[0] = outputs[0] + 1
            return outputs

    monkeypatch.setattr(executable_mod, "TinyJit", CorruptingTinyJit)

    graph = {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [2], "dtype": "f32"}],
        "constants": [],
        "nodes": [{"id": 1, "op": "negate", "inputs": [0], "attrs": {}, "shape": [2], "dtype": "f32"}],
        "outputs": [{"node": 1, "shape": [2], "dtype": "f32"}],
    }

    with pytest.raises(CompileError, match="replay mismatch"):
        compile_graph(12, graph, [], "CPU", validate_capture=True)


def test_duplicate_input_cloning():
    graph = {
        "version": 1,
        "inputs": [
            {"id": 0, "index": 0, "shape": [3], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [3], "dtype": "f32"},
        ],
        "constants": [],
        "nodes": [{"id": 2, "op": "add", "inputs": [0, 1], "attrs": {}, "shape": [3], "dtype": "f32"}],
        "outputs": [{"node": 2, "shape": [3], "dtype": "f32"}],
    }
    ex = compile_graph(11, graph, [], "CPU")
    same = _T([1, 2, 3])
    [out] = ex.run([same, same])
    assert ex.duplicate_input_clones == 1
    assert np.allclose(out.numpy(), [2, 4, 6])


def test_executable_registry_release_is_idempotent():
    registry = executable_mod.ExecutableRegistry()

    class FakeExecutable:
        id = registry.allocate_id()

    registry.put(FakeExecutable())
    assert registry.count() == 1
    assert registry.release([FakeExecutable.id]) == 1
    assert registry.release([FakeExecutable.id]) == 0
    assert registry.count() == 0


def test_validation_rejects_dangling_reference():
    graph = {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [2], "dtype": "f32"}],
        "constants": [],
        "nodes": [{"id": 1, "op": "add", "inputs": [0, 99], "attrs": {}, "shape": [2], "dtype": "f32"}],
        "outputs": [{"node": 1, "shape": [2], "dtype": "f32"}],
    }
    with pytest.raises(GraphValidationError):
        compile_graph(5, graph, [], "CPU")
