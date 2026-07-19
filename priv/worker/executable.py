"""Compiled executables: constant materialization plus graph evaluation.

For M2 an Executable evaluates the graph by walking its (topologically ordered)
nodes on every ``run``. tinygrad builds these lazily, so this is symbolic until
realized. M3 wraps the same evaluation in ``TinyJit`` for capture/replay.
"""
from __future__ import annotations

import math

import numpy as np
from tinygrad import Tensor

import operations
from dtype import numpy_dtype, tinygrad_dtype


def _decode_value(value):
    if isinstance(value, str):
        return {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}[value]
    return value


def _make_constant(const: dict, blobs: list[bytes], device: str) -> Tensor:
    shape = tuple(const["shape"])
    dtype = const["dtype"]
    if "data_index" in const:
        arr = np.frombuffer(blobs[const["data_index"]], dtype=numpy_dtype(dtype)).reshape(shape).copy()
        return Tensor(arr, device=device).realize()

    value = _decode_value(const["value"])
    tg_dtype = tinygrad_dtype(dtype)
    if shape == ():
        return Tensor(value, device=device, dtype=tg_dtype).realize()
    return Tensor.full(shape, value, device=device, dtype=tg_dtype).realize()


class Executable:
    def __init__(self, exec_id: int, graph: dict, blobs: list[bytes], device: str):
        self.id = exec_id
        self.graph = graph
        self.device = device
        self.input_specs = graph["inputs"]
        self.output_specs = graph["outputs"]
        self.constants = {c["id"]: _make_constant(c, blobs, device) for c in graph["constants"]}
        self.kernel_count = 0

    def run(self, input_tensors: list[Tensor]) -> list[Tensor]:
        env = dict(self.constants)
        for spec, tensor in zip(self.input_specs, input_tensors):
            env[spec["id"]] = tensor
        for node in self.graph["nodes"]:
            env[node["id"]] = operations.apply(node, env)
        return [env[out["node"]] for out in self.output_specs]


class ExecutableRegistry:
    def __init__(self):
        self._executables: dict[int, Executable] = {}
        self._next_id = 1

    def allocate_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def put(self, executable: Executable) -> None:
        self._executables[executable.id] = executable

    def get(self, exec_id: int) -> Executable:
        try:
            return self._executables[exec_id]
        except KeyError:
            from errors import StaleReference

            raise StaleReference(f"executable {exec_id} not found", details={"executable_id": exec_id}) from None

    def count(self) -> int:
        return len(self._executables)
