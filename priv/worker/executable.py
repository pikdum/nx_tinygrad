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
from tinygrad import Tensor, TinyJit, Device

import operations
from dtype import component_dtype, is_complex, numpy_dtype, tinygrad_dtype, wire_numpy, wire_tensor
from errors import CompileError
from operations import Cx


def _expected_shape(spec) -> list:
    # A complex tensor is stored as a real float tensor with a trailing [2] axis.
    return list(spec["shape"]) + [2] if is_complex(spec["dtype"]) else list(spec["shape"])


def immutable_copy(t: Tensor, stats=None) -> Tensor:
    """A fresh, independent realized copy of ``t``.

    Captured TinyJit outputs reuse the same buffer on every replay, so a returned
    handle must be copied to its own buffer to stay immutable. ``clone().realize()``
    does this but pays tinygrad's full (~600us) scheduling cost per output. When
    the device allocator supports a raw device-to-device transfer (HCQ/AMD SDMA),
    we copy at the buffer level instead — ~3x cheaper and size-independent. The
    HCQ transfer signals the device timeline, so later kernels reading the copy
    correctly wait for it. Falls back to ``clone().realize()`` otherwise (e.g. CPU).
    """
    try:
        t.realize()
        src = t.uop.base.realized
        # Only fast-path a standalone, whole-buffer output. If the output is a
        # view into a larger/shared buffer (its base holds more elements than
        # this tensor — as some gradient outputs do), a raw base-buffer copy
        # would read the wrong region, so fall through to the safe clone.
        if src is None or src.size != math.prod(t.shape):
            raise NotImplementedError("output is a view/shared buffer")
        dev = Device[src.device]
        allocator = dev.allocator
        if not hasattr(allocator, "_transfer"):
            raise NotImplementedError("allocator has no _transfer")
        dst = Tensor.empty(*t.shape, dtype=t.dtype, device=t.device)
        dst.uop.buffer.allocate()
        allocator._transfer(dst.uop.buffer._buf, src._buf, src.nbytes, dev, dev)
        if stats is not None:
            stats.immutable_copy_fast += 1
        return dst
    except Exception:  # noqa: BLE001 — any failure falls back to the safe path
        if stats is not None:
            stats.immutable_copy_fallback += 1
        return t.clone().realize()


def _requires_eager(node) -> bool:
    # Nodes that must read a runtime scalar (breaking static JIT capture).
    if node["op"] == "while":
        return True
    if node["op"] == "slice" and len(node["inputs"]) > 1:
        return True
    if node["op"] == "put_slice" and len(node["inputs"]) > 2:
        return True
    return False


def _decode_value(value):
    if isinstance(value, str):
        return {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}[value]
    return value


def _make_constant(const: dict, blobs: list[bytes], device: str):
    shape = tuple(const["shape"])
    dtype = const["dtype"]

    if is_complex(dtype):
        comp = tinygrad_dtype(component_dtype(dtype))
        if "data_index" in const:
            arr = np.frombuffer(blobs[const["data_index"]], dtype=numpy_dtype(dtype)).reshape(shape).copy()
            return Cx(wire_tensor(arr, dtype, device).realize())
        value = const["value"]  # {"re": ..., "im": ...}
        re = Tensor.full(shape or (1,), _decode_value(value["re"]), device=device, dtype=comp)
        im = Tensor.full(shape or (1,), _decode_value(value["im"]), device=device, dtype=comp)
        if not shape:
            re, im = re.reshape(()), im.reshape(())
        return Cx(operations._cxt(re, im).realize())

    if "data_index" in const:
        arr = np.frombuffer(blobs[const["data_index"]], dtype=numpy_dtype(dtype)).reshape(shape).copy()
        return wire_tensor(arr, dtype, device).realize()

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
        self.blobs = blobs
        self.input_specs = graph["inputs"]
        self.output_specs = graph["outputs"]
        self.constants = {c["id"]: _make_constant(c, blobs, device) for c in graph["constants"]}
        self.duplicate_input_clones = 0
        self.kernel_count = 0
        self._jit = None
        # Some nodes read runtime scalar values (a `while`'s condition, a dynamic
        # slice start), which can't be JIT-captured — such graphs run eagerly
        # node-by-node instead of via TinyJit replay.
        self._eager = any(_requires_eager(node) for node in graph["nodes"])
        self._capture(validate_capture)

    # -- graph evaluation ---------------------------------------------------

    def _eval_nodes(self, nodes, env):
        for node in nodes:
            if node["op"] == "while":
                for out, result in zip(node["outputs"], self._run_while(node, env)):
                    env[out["id"]] = result
            elif node["op"] == "reduce":
                env[node["id"]] = self._run_reduce(node, env)
            elif node["op"] == "window_reduce":
                env[node["id"]] = self._run_window_reduce(node, env)
            else:
                env[node["id"]] = operations.apply(node, env)

    def _run_reduce(self, node, env):
        import math

        t = env[node["inputs"][0]]
        acc = env[node["inputs"][1]]
        axes = node["attrs"]["axes"]
        keep = node["attrs"]["keep_axes"]
        body = node["attrs"]["fn"]
        rank = len(t.shape)
        non_axes = [a for a in range(rank) if a not in axes]

        # Move reduced axes to the end and flatten them, so the fold is 1-D.
        batch = [t.shape[a] for a in non_axes]
        count = math.prod([t.shape[a] for a in axes]) if axes else 1
        flat = t.permute(tuple(non_axes + list(axes))).reshape(tuple(batch) + (count,))

        if batch and acc.shape == ():
            acc = acc.reshape((1,) * len(batch)).expand(tuple(batch))
        for i in range(count):
            acc = self._interpret(body, [acc, flat[..., i]])[0]

        if keep:
            acc = acc.reshape(tuple(1 if a in axes else t.shape[a] for a in range(rank)))
        return acc

    def _run_window_reduce(self, node, env):
        import math

        t = env[node["inputs"][0]]
        acc = env[node["inputs"][1]]
        a = node["attrs"]
        rank = len(t.shape)
        if any(lo or hi for (lo, hi) in a["padding"]):
            raise CompileError("window_reduce with padding is not supported")

        pooled = t._pool(tuple(a["window"]), stride=tuple(a["strides"]), dilation=tuple(a["window_dilations"]))
        out_dims = list(pooled.shape[:rank])
        count = math.prod(pooled.shape[rank:])
        flat = pooled.reshape(tuple(out_dims) + (count,))

        if acc.shape == ():
            acc = acc.reshape((1,) * len(out_dims)).expand(tuple(out_dims))
        for i in range(count):
            acc = self._interpret(a["fn"], [acc, flat[..., i]])[0]
        return acc

    def _interpret(self, subgraph, params):
        # Evaluate a self-contained sub-graph with its loop-var parameters bound
        # to `params` (by input index). Constants may index the shared blob list.
        env = {c["id"]: _make_constant(c, self.blobs, self.device) for c in subgraph["constants"]}
        for inp in subgraph["inputs"]:
            env[inp["id"]] = params[inp["index"]]
        self._eval_nodes(subgraph["nodes"], env)
        return [env[o["node"]] for o in subgraph["outputs"]]

    def _run_while(self, node, env):
        state = [env[i].realize() for i in node["inputs"]]
        cond, body = node["attrs"]["cond"], node["attrs"]["body"]

        # The body is a static computation (only the trip count is dynamic), so
        # JIT-capture it for fast replay — unless it itself needs eager execution
        # (a nested while or dynamic slice), in which case interpret it directly.
        if any(_requires_eager(n) for n in body["nodes"]):
            step = lambda s: self._interpret(body, s)  # noqa: E731
        else:
            jit = TinyJit(lambda *s: self._interpret(body, list(s)))
            step = lambda s: jit(*[t.clone().realize() for t in s])  # noqa: E731

        # Read the scalar condition each iteration (realizes it), then step.
        while bool(self._interpret(cond, state)[0].item()):
            state = [t.realize() for t in step(state)]
        return state

    def _graph_fn(self, *inputs):
        env = dict(self.constants)
        for spec, tensor in zip(self.input_specs, inputs):
            env[spec["id"]] = Cx(tensor) if is_complex(spec["dtype"]) else tensor
        self._eval_nodes(self.graph["nodes"], env)
        # Force row-major contiguous outputs inside the captured graph (runs at
        # replay speed). This is a no-op for already-contiguous outputs and lets
        # immutable_copy do a raw buffer transfer safely even for outputs that
        # would otherwise be strided views (e.g. gradients from dot/transpose).
        outs = []
        for out in self.output_specs:
            tensor = env[out["node"]]
            if isinstance(tensor, Cx):
                tensor = tensor.t  # complex output -> real [..., 2] tensor for transport
            contiguous = tensor.contiguous()
            # contiguous() is a no-op for an already-contiguous input or view
            # such as reshape. TinyJit would then retain the capture-time input
            # buffer as an output instead of rebinding it on replay.
            if contiguous.uop is tensor.uop:
                contiguous = tensor.clone()
            outs.append(contiguous)
        if outs:
            outs[0].realize(*outs[1:])
        return outs

    def _dummy(self, spec, seed: int) -> Tensor:
        shape = tuple(spec["shape"])
        if is_complex(spec["dtype"]):
            comp = numpy_dtype(component_dtype(spec["dtype"]))
            arr = (np.arange(math.prod(shape) * 2, dtype=comp) + seed).reshape(shape + (2,))
            return Tensor(arr, device=self.device).clone().realize()
        size = math.prod(shape)
        arr = (np.arange(size, dtype=numpy_dtype(spec["dtype"])) + seed).reshape(shape)
        # clone() forces scalars and other constant-foldable inputs to own real
        # buffers, as required by TinyJit input replacement.
        return wire_tensor(arr, spec["dtype"], self.device).clone().realize()

    def _capture(self, validate: bool) -> None:
        # A graph with no inputs is a pure constant computation, and graphs that
        # read runtime scalars (while, dynamic slice) can't be captured; both run
        # eagerly via run().
        if not self.input_specs or self._eager:
            return

        self._jit = TinyJit(self._graph_fn)
        outs = self._jit(*[self._dummy(spec, 1) for spec in self.input_specs])
        outs = self._jit(*[self._dummy(spec, 2) for spec in self.input_specs])

        if validate:
            validation_inputs = [self._dummy(spec, 3) for spec in self.input_specs]
            expected = self._graph_fn(*validation_inputs)
            reference = [
                np.ascontiguousarray(wire_numpy(out, spec["dtype"])).copy()
                for out, spec in zip(expected, self.output_specs)
            ]
            outs = self._jit(*validation_inputs)

        if len(outs) != len(self.output_specs):
            raise CompileError(f"captured {len(outs)} outputs, expected {len(self.output_specs)}")
        for out, spec in zip(outs, self.output_specs):
            if list(out.shape) != _expected_shape(spec):
                raise CompileError(f"captured output shape {list(out.shape)} != {_expected_shape(spec)}")

        if validate:
            self._validate_replay(reference, outs)

        self.kernel_count = self._count_kernels()

    def _validate_replay(self, reference: list[np.ndarray], replayed: list[Tensor]) -> None:
        for index, (expected, tensor, spec) in enumerate(zip(reference, replayed, self.output_specs)):
            actual = np.ascontiguousarray(wire_numpy(tensor, spec["dtype"]))
            expected_dtype = numpy_dtype(spec["dtype"])
            if actual.dtype != expected_dtype:
                raise CompileError(
                    f"captured output {index} dtype {actual.dtype} != {expected_dtype}"
                )

            if np.issubdtype(actual.dtype, np.floating):
                equal = np.array_equal(actual, expected, equal_nan=True)
            else:
                equal = np.array_equal(actual, expected)

            if not equal:
                raise CompileError(f"TinyJit replay mismatch for output {index}")

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

    def release(self, ids) -> int:
        """Remove compiled executables. Unknown ids are ignored."""
        removed = 0
        for exec_id in ids:
            if self._executables.pop(exec_id, None) is not None:
                removed += 1
        return removed
