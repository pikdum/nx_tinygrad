"""Compiled executables backed by tinygrad's ``TinyJit``.

At compile time the graph function is captured: it runs once normally, once to
capture, and (optionally) a third time to validate replay. Subsequent ``run``
calls replay the captured kernels, which is where the single-RPC, fused
execution happens.

Captured outputs reuse the same physical buffers on every replay, so callers
must copy (host mode: ``.numpy()``) or clone (device mode) outputs before the
next execution — see the worker's execute handler.
"""
from __future__ import annotations

import math

import numpy as np
from tinygrad import Tensor, TinyJit

import operations
from dtype import numpy_dtype, tinygrad_dtype
from errors import CompileError


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
    def __init__(self, exec_id: int, graph: dict, blobs: list[bytes], device: str, validate_capture: bool = True):
        self.id = exec_id
        self.graph = graph
        self.device = device
        self.input_specs = graph["inputs"]
        self.output_specs = graph["outputs"]
        self.constants = {c["id"]: _make_constant(c, blobs, device) for c in graph["constants"]}
        self.duplicate_input_clones = 0
        self.kernel_count = 0
        self._jit = None
        self._capture(validate_capture)

    # -- graph evaluation ---------------------------------------------------

    def _graph_fn(self, *inputs):
        env = dict(self.constants)
        for spec, tensor in zip(self.input_specs, inputs):
            env[spec["id"]] = tensor
        for node in self.graph["nodes"]:
            env[node["id"]] = operations.apply(node, env)
        outs = [env[out["node"]] for out in self.output_specs]
        if outs:
            outs[0].realize(*outs[1:])
        return outs

    def _dummy(self, spec) -> Tensor:
        return (
            Tensor.zeros(tuple(spec["shape"]), dtype=tinygrad_dtype(spec["dtype"]), device=self.device)
            .contiguous()
            .realize()
        )

    def _capture(self, validate: bool) -> None:
        # A graph with no inputs is a pure constant computation; TinyJit has
        # nothing to replay, so evaluate it directly.
        if not self.input_specs:
            return

        self._jit = TinyJit(self._graph_fn)
        rounds = 3 if validate else 2
        outs = None
        for _ in range(rounds):
            outs = self._jit(*[self._dummy(spec) for spec in self.input_specs])

        if len(outs) != len(self.output_specs):
            raise CompileError(f"captured {len(outs)} outputs, expected {len(self.output_specs)}")
        for out, spec in zip(outs, self.output_specs):
            if list(out.shape) != list(spec["shape"]):
                raise CompileError(f"captured output shape {list(out.shape)} != {spec['shape']}")

        self.kernel_count = self._count_kernels()

    def _count_kernels(self) -> int:
        # tinygrad's captured JIT stores its kernel/copy calls in linear.src.
        captured = getattr(self._jit, "captured", None)
        try:
            return len(captured.linear.src) if captured is not None else 0
        except Exception:  # noqa: BLE001
            return 0

    def _dedup(self, inputs: list[Tensor]) -> list[Tensor]:
        # TinyJit rejects the same underlying buffer in multiple input slots.
        # Realize first so a clone gets a genuinely distinct buffer, then clone
        # repeated occurrences.
        seen: set[int] = set()
        result = []
        for tensor in inputs:
            tensor = tensor.realize()
            if id(tensor) in seen:
                result.append(tensor.clone().realize())
                self.duplicate_input_clones += 1
            else:
                seen.add(id(tensor))
                result.append(tensor)
        return result

    def run(self, inputs: list[Tensor]) -> list[Tensor]:
        if self._jit is None:
            return self._graph_fn(*inputs)
        return self._jit(*self._dedup(inputs))


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
