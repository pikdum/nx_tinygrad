import numpy as np
import pytest
from tinygrad import Tensor

from compiler import compile_graph
from errors import GraphValidationError


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
