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
import sys

import numpy as np
from tinygrad import Tensor, TinyJit, Device, Variable

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


def _log(msg: str) -> None:
    print(f"[nx_tinygrad.worker] {msg}", file=sys.stderr, flush=True)


# Per-iteration while-step counters, exposed via the stats command so a lost
# fast path (e.g. tinygrad API drift breaking symbolic capture) shows up in
# telemetry instead of silently degrading to interpretation.
WHILE_STATS = {
    "while_steps_jit": 0,
    "while_steps_symbolic": 0,
    "while_steps_interpreted": 0,
    "while_jit_fallbacks": 0,
    # Top-level segments of eager graphs (the static regions between while /
    # dynamic-slice nodes, e.g. SD's text encoder and VAE decoder).
    "graph_segments_jit": 0,
    "graph_segments_interpreted": 0,
    "graph_segment_fallbacks": 0,
}


def _requires_eager(node) -> bool:
    # Nodes that must read a runtime scalar (breaking static JIT capture).
    if node["op"] == "while":
        return True
    if node["op"] == "slice" and len(node["inputs"]) > 1:
        return True
    if node["op"] == "put_slice" and len(node["inputs"]) > 2:
        return True
    return False


# First dynamic-start input position per op: slice inputs are [tensor, *starts],
# put_slice inputs are [target, slice, *starts].
_DYN_FIRST = {"slice": 1, "put_slice": 2}


def _dyn_start_count(node) -> int:
    first = _DYN_FIRST.get(node["op"])
    return 0 if first is None else max(0, len(node["inputs"]) - first)


def _sub_requires_eager(sub) -> bool:
    """True when a while body can't be JIT-captured even with symbolic starts:
    a nested while, or dynamic starts hidden inside a reduce/window_reduce fn
    (whose evaluation would read runtime scalars mid-trace)."""
    for node in sub["nodes"]:
        if node["op"] == "while":
            return True
        if node["op"] in ("reduce", "window_reduce"):
            fn = node["attrs"]["fn"]
            if _sub_requires_eager(fn) or any(_dyn_start_count(n) for n in fn["nodes"]):
                return True
    return False


def _node_jit_safe(node) -> bool:
    # Can this top-level node live inside a TinyJit trace?
    if _requires_eager(node):
        return False
    if node["op"] in ("reduce", "window_reduce"):
        fn = node["attrs"]["fn"]
        return not _sub_requires_eager(fn) and not any(_dyn_start_count(n) for n in fn["nodes"])
    return True


def _node_defined_ids(node) -> list:
    if node["op"] == "while":
        return [out["id"] for out in node["outputs"]]
    return [node["id"]]


def _symbolic_while_plan(body):
    """Rewrite a while body so dynamic slice/put_slice starts become symbolic
    tinygrad Variables, making the whole body TinyJit-capturable.

    Returns ``(sym_body, uses, anc_nodes, anc_consts)``: the rewritten body,
    one descriptor per Variable use ({"id": start node id, "hi": clamp upper
    bound, "name": variable name}), and the node/constant subset needed to
    eagerly evaluate each start's runtime value every iteration.
    """
    shapes = {}
    for inp in body["inputs"]:
        shapes[inp["id"]] = inp["shape"]
    for const in body["constants"]:
        shapes[const["id"]] = const["shape"]
    for node in body["nodes"]:
        shapes[node["id"]] = node["shape"]

    uses = []
    dyn_ids: set[int] = set()
    new_nodes = []
    for node in body["nodes"]:
        if not _dyn_start_count(node):
            new_nodes.append(node)
            continue
        first = _DYN_FIRST[node["op"]]
        dyn = node["inputs"][first:]
        dims = shapes[node["inputs"][0]]
        # The clamp bound per axis: slice windows span `lengths`; put_slice
        # windows span the slice operand's shape.
        spans = node["attrs"]["lengths"] if node["op"] == "slice" else shapes[node["inputs"][1]]

        new_starts = []
        for axis, spec in enumerate(node["attrs"]["starts"]):
            if "static" in spec:
                new_starts.append(spec)
                continue
            hi = dims[axis] - spans[axis]
            if hi <= 0:
                # Nx clamps the start into [0, hi]; hi == 0 pins it statically.
                new_starts.append({"static": 0})
                continue
            start_id = dyn[spec["input"]]
            uses.append({"id": start_id, "hi": hi, "name": f"nxw{len(uses)}"})
            dyn_ids.add(start_id)
            new_starts.append({"symbolic": len(uses) - 1})

        new_node = dict(node)
        new_node["attrs"] = dict(node["attrs"], starts=new_starts)
        new_nodes.append(new_node)

    sym_body = dict(body, nodes=new_nodes)

    # Ancestor closure of the start ids (node list is topologically ordered).
    needed = set(dyn_ids)
    anc_nodes = []
    for node in reversed(body["nodes"]):
        if node.get("id") in needed:
            anc_nodes.append(node)
            needed.update(node["inputs"])
    anc_nodes.reverse()
    anc_consts = [c for c in body["constants"] if c["id"] in needed]
    return sym_body, uses, anc_nodes, anc_consts


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
        # Per-while-node step functions, cached across execute calls so replays
        # skip re-capture (the cached TinyJit keeps its intermediate buffers
        # alive between calls — that is what makes replays fast — and is freed
        # with the executable).
        self._while_steps: dict[int, tuple] = {}
        # Top-level segments for eager graphs (see _plan_segments) and their
        # cached per-segment step functions.
        self._segments = None
        self._segment_steps: dict[int, tuple] = {}
        self._id_dtype: dict[int, str] = {}
        # Some nodes read runtime scalar values (a `while`'s condition, a dynamic
        # slice start), which can't be JIT-captured — such graphs run eagerly
        # node-by-node instead of via TinyJit replay.
        self._eager = any(_requires_eager(node) for node in graph["nodes"])
        if self._eager:
            self._segments = self._plan_segments()
        self._capture(validate_capture)

    # -- top-level segmentation ----------------------------------------------
    #
    # A graph containing a while (or a top-level dynamic slice) runs eagerly,
    # but everything BETWEEN those nodes is static — e.g. Stable Diffusion's
    # text encoder and VAE decoder around the denoise loop. Split the node
    # list into maximal jit-safe segments and eager singletons; each segment
    # is TinyJit-captured once and replayed on later executes.

    def _plan_segments(self):
        for inp in self.graph["inputs"]:
            self._id_dtype[inp["id"]] = inp["dtype"]
        for const in self.graph["constants"]:
            self._id_dtype[const["id"]] = const["dtype"]
        for node in self.graph["nodes"]:
            if node["op"] == "while":
                for out in node["outputs"]:
                    self._id_dtype[out["id"]] = out["dtype"]
            else:
                self._id_dtype[node["id"]] = node["dtype"]

        plan = []
        current = []

        def flush():
            if current:
                plan.append(("jit", {"nodes": list(current)}))
                current.clear()

        for node in self.graph["nodes"]:
            if _node_jit_safe(node):
                current.append(node)
            else:
                flush()
                plan.append(("eager", node))
        flush()

        if not any(kind == "jit" for kind, _ in plan):
            return None

        # ids needed strictly after each position (for segment outputs).
        after = [set() for _ in plan]
        needed = {o["node"] for o in self.graph["outputs"]}
        for pos in range(len(plan) - 1, -1, -1):
            after[pos] = set(needed)
            kind, item = plan[pos]
            for node in [item] if kind == "eager" else item["nodes"]:
                needed.update(node["inputs"])

        constants = set(self.constants)
        for pos, (kind, item) in enumerate(plan):
            if kind != "jit":
                continue
            defined = set()
            inputs = []
            for node in item["nodes"]:
                for ref in node["inputs"]:
                    if ref not in defined and ref not in constants and ref not in inputs:
                        inputs.append(ref)
                defined.update(_node_defined_ids(node))
            item["input_ids"] = inputs
            item["output_ids"] = [i for i in sorted(defined) if i in after[pos]]

        return plan

    def _eval_segments(self, env):
        for idx, (kind, item) in enumerate(self._segments):
            if kind == "eager":
                self._eval_nodes([item], env)
                continue
            if not item["output_ids"]:
                continue  # dead segment: nothing downstream reads it
            step, mode = self._segment_step(idx, item)
            if mode == "interpret":
                WHILE_STATS["graph_segments_interpreted"] += 1
                self._eval_nodes(item["nodes"], env)
                continue
            try:
                results = step([env[i] for i in item["input_ids"]])
            except Exception as exc:  # noqa: BLE001 — capture/replay failure
                _log(f"graph segment JIT failed ({type(exc).__name__}: {exc}); "
                     "falling back to interpretation")
                WHILE_STATS["graph_segment_fallbacks"] += 1
                self._segment_steps[idx] = (None, "interpret")
                self._eval_nodes(item["nodes"], env)
                continue
            WHILE_STATS["graph_segments_jit"] += 1
            for out_id, value in zip(item["output_ids"], results):
                env[out_id] = value

    def _segment_step(self, idx, item):
        cached = self._segment_steps.get(idx)
        if cached is not None:
            return cached

        in_dtypes = [self._id_dtype[i] for i in item["input_ids"]]
        out_dtypes = [self._id_dtype[i] for i in item["output_ids"]]

        def fn(*args):
            env = dict(self.constants)
            for ref, dtype, tensor in zip(item["input_ids"], in_dtypes, args):
                env[ref] = Cx(tensor) if is_complex(dtype) else tensor
            self._eval_nodes(item["nodes"], env)
            outs = []
            for ref in item["output_ids"]:
                value = env[ref]
                if isinstance(value, Cx):
                    value = value.t
                # Realized-in-trace outputs (input passthrough, constant view)
                # schedule no kernel and would replay stale; clone them.
                if value.uop.base.realized is not None:
                    value = value.clone()
                outs.append(value)
            return outs

        jit = TinyJit(fn)

        def step(values):
            seen: set[int] = set()
            args = []
            for value in values:
                tensor = value.t if isinstance(value, Cx) else value
                if tensor.uop.base.realized is None:
                    # constant-foldable value: force a real buffer (jit input)
                    tensor = tensor.clone().realize()
                else:
                    # A strided view input (e.g. from an interpreted dynamic
                    # slice) bakes its runtime offsets into the capture and
                    # would args-mismatch on the next execute; materialize it.
                    contiguous = tensor.contiguous()
                    if contiguous.uop is not tensor.uop:
                        tensor = contiguous.realize()
                base = tensor.uop.base
                if id(base) in seen:
                    tensor = tensor.clone().realize()  # TinyJit rejects dup buffers
                else:
                    seen.add(id(base))
                args.append(tensor)
            outs = jit(*args)
            return [
                Cx(out) if is_complex(dtype) else out for out, dtype in zip(outs, out_dtypes)
            ]

        cached = (step, "jit")
        self._segment_steps[idx] = cached
        return cached

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

    def _interpret(self, subgraph, params, sym=None):
        # Evaluate a self-contained sub-graph with its loop-var parameters bound
        # to `params` (by input index). Constants may index the shared blob list.
        # `sym` binds symbolic slice starts (see _symbolic_while_plan) under a
        # reserved key that cannot collide with the integer node ids.
        env = {c["id"]: _make_constant(c, self.blobs, self.device) for c in subgraph["constants"]}
        if sym is not None:
            env["__sym__"] = list(sym)
        for inp in subgraph["inputs"]:
            env[inp["id"]] = params[inp["index"]]
        self._eval_nodes(subgraph["nodes"], env)
        return [env[o["node"]] for o in subgraph["outputs"]]

    def _make_while_step(self, body, n_state):
        """Build the per-iteration step function for a while body.

        Bodies whose only runtime scalars are dynamic slice/put_slice starts
        are JIT-captured whole, with the starts as bound tinygrad Variables
        passed as jit call arguments (TinyJit collects bindings from call args
        into var_vals on every replay). Bodies with no runtime scalars are the
        degenerate case with zero Variables. Everything else (nested while,
        dynamic starts inside reduce fns) is interpreted node-by-node.

        Loop-invariant vars — body output j is exactly loop-var input j, which
        is how Nx carries closed-over constants like model weights through a
        while — bypass the jit entirely: they are passed as inputs without the
        per-iteration anti-aliasing clone (they are never jit outputs, so no
        buffer reuse can touch them) and returned unchanged Python-side.
        """
        if _sub_requires_eager(body):
            return (lambda s: self._interpret(body, s)), "interpret"

        sym_body, uses, anc_nodes, anc_consts = _symbolic_while_plan(body)

        input_id_by_index = {inp["index"]: inp["id"] for inp in body["inputs"]}
        invariant = {
            j for j, out in enumerate(body["outputs"]) if input_id_by_index.get(j) == out["node"]
        }

        def fn(*args):
            outs = self._interpret(sym_body, list(args[:n_state]), sym=args[n_state:])
            fixed = []
            for j, out in enumerate(outs):
                if j in invariant:
                    continue
                # An output that is already realized inside the trace (an
                # input passed through under a different index, a view of an
                # input, a constant) schedules no kernel, so TinyJit replay
                # would return the stale capture-time buffer. clone() forces a
                # captured copy kernel that reads the replay-time input.
                if out.uop.base.realized is not None:
                    out = out.clone()
                fixed.append(out)
            return fixed

        jit = TinyJit(fn)
        const_env = {c["id"]: _make_constant(c, self.blobs, self.device) for c in anc_consts}

        def step(state):
            # Evaluate just the start-index subgraph eagerly (cheap scalar
            # chains), clamp host-side, and bind one Variable per use.
            bound = []
            if uses:
                scratch = dict(const_env)
                for inp in body["inputs"]:
                    scratch[inp["id"]] = state[inp["index"]]
                self._eval_nodes(anc_nodes, scratch)
                for use in uses:
                    raw = int(scratch[use["id"]].item())
                    bound.append(Variable(use["name"], 0, use["hi"]).bind(max(0, min(raw, use["hi"]))))
            # TinyJit rejects two inputs backed by the same buffer. Invariant
            # vars can legitimately share one (params preallocated as views,
            # tied weights) — clone the repeats. Non-invariant clones are
            # fresh buffers already.
            seen: set[int] = set()
            inputs = []
            for j, t in enumerate(state):
                if j in invariant:
                    base = t.uop.base
                    if id(base) in seen:
                        t = t.clone().realize()
                    else:
                        seen.add(id(base))
                else:
                    t = t.clone().realize()
                inputs.append(t)
            outs = iter(jit(*inputs, *bound))
            return [state[j] if j in invariant else next(outs) for j in range(n_state)]

        return step, ("symbolic" if uses else "jit")

    def _run_while(self, node, env):
        # TinyJit input replacement needs inputs backed by real buffers, and
        # constant-foldable loop-var inits (scalars, iota) have none after
        # realize(); clone() forces one (the _dummy trick). Doing it once here
        # keeps invariant vars real-buffered for the whole loop.
        state = [
            t if t.uop.base.realized is not None else t.clone().realize()
            for t in (env[i].realize() for i in node["inputs"])
        ]
        cond, body = node["attrs"]["cond"], node["attrs"]["body"]

        key = id(node)
        cached = self._while_steps.get(key)
        if cached is None:
            cached = self._make_while_step(body, len(state))
            self._while_steps[key] = cached
        step, kind = cached

        # Read the scalar condition each iteration (realizes it), then step.
        while bool(self._interpret(cond, state)[0].item()):
            if kind == "interpret":
                new_state = step(state)
            else:
                try:
                    new_state = step(state)
                except Exception as exc:  # noqa: BLE001 — capture/replay failure
                    # tinygrad internals shift between versions; never wrong
                    # data — fall back to interpretation from the same state.
                    _log(f"while {kind} JIT step failed ({type(exc).__name__}: {exc}); "
                         "falling back to interpretation")
                    WHILE_STATS["while_jit_fallbacks"] += 1
                    step, kind = (lambda s: self._interpret(body, s)), "interpret"
                    self._while_steps[key] = (step, kind)
                    new_state = step(state)
            WHILE_STATS[f"while_steps_{'interpreted' if kind == 'interpret' else kind}"] += 1
            state = [t.realize() for t in new_state]
        return state

    def _graph_fn(self, *inputs):
        env = dict(self.constants)
        for spec, tensor in zip(self.input_specs, inputs):
            env[spec["id"]] = Cx(tensor) if is_complex(spec["dtype"]) else tensor
        if self._eager and self._segments is not None:
            self._eval_segments(env)
        else:
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
