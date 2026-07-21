"""JIT capture of `while` bodies, including symbolic dynamic-slice starts.

Every test compares the jitted execution against forced node-by-node
interpretation of the same graph — interpretation is the semantic reference.
"""
import numpy as np
from tinygrad import Tensor

import executable as executable_mod
from compiler import compile_graph


def _T(data, dtype=np.float32):
    return Tensor(np.array(data, dtype=dtype))


def _run(graph, make_inputs, interpret=False):
    """Compile and run `graph`, optionally forcing full interpretation
    (no while-step JIT, no top-level segment JIT) as the semantic reference."""
    if interpret:
        orig_step = executable_mod.Executable._make_while_step
        orig_plan = executable_mod.Executable._plan_segments

        def forced(self, body, n_state):
            return (lambda s: self._interpret(body, s)), "interpret"

        executable_mod.Executable._make_while_step = forced
        executable_mod.Executable._plan_segments = lambda self: None
        try:
            ex = compile_graph(1, graph, [], "CPU")
            return [np.array(o.numpy()) for o in ex.run(make_inputs())]
        finally:
            executable_mod.Executable._make_while_step = orig_step
            executable_mod.Executable._plan_segments = orig_plan
    ex = compile_graph(1, graph, [], "CPU")
    return [np.array(o.numpy()) for o in ex.run(make_inputs())]


def _assert_parity(graph, make_inputs):
    expected = _run(graph, make_inputs, interpret=True)
    actual = _run(graph, make_inputs)
    for exp, act in zip(expected, actual):
        assert np.array_equal(exp, act), f"jit result diverged:\n{act}\n!=\n{exp}"


def _stats():
    return dict(executable_mod.WHILE_STATS)


def _delta(before):
    return {k: v - before[k] for k, v in executable_mod.WHILE_STATS.items()}


# -- graph builders -----------------------------------------------------------


def _counter_cond(iters, input_index=1):
    return {
        "inputs": [{"id": 0, "index": input_index, "shape": [], "dtype": "s32"}],
        "constants": [{"id": 1, "value": iters, "shape": [], "dtype": "s32"}],
        "nodes": [{"id": 2, "op": "less", "inputs": [0, 1], "attrs": {}, "shape": [], "dtype": "u8"}],
        "outputs": [{"node": 2, "shape": [], "dtype": "u8"}],
    }


def _sd_shaped_graph(iters, n_coeffs, length=1):
    """acc' = acc*0.5 + coeffs[i]; i' = i+1; coeffs invariant.

    The scheduler-style `coeffs[i]` dynamic slice is what forces SD's denoise
    loop onto the eager path before symbolic capture.
    """
    body = {
        "inputs": [
            {"id": 0, "index": 0, "shape": [4], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [], "dtype": "s32"},
            {"id": 2, "index": 2, "shape": [n_coeffs], "dtype": "f32"},
        ],
        "constants": [
            {"id": 3, "value": 1, "shape": [], "dtype": "s32"},
            {"id": 4, "value": 0.5, "shape": [], "dtype": "f32"},
        ],
        "nodes": [
            {"id": 5, "op": "multiply", "inputs": [0, 4], "attrs": {}, "shape": [4], "dtype": "f32"},
            {
                "id": 6,
                "op": "slice",
                "inputs": [2, 1],
                "attrs": {"starts": [{"input": 0}], "lengths": [length], "strides": [1]},
                "shape": [length],
                "dtype": "f32",
            },
            {"id": 7, "op": "sum", "inputs": [6], "attrs": {"axes": [0], "keep_axes": False}, "shape": [], "dtype": "f32"},
            {"id": 8, "op": "add", "inputs": [5, 7], "attrs": {}, "shape": [4], "dtype": "f32"},
            {"id": 9, "op": "add", "inputs": [1, 3], "attrs": {}, "shape": [], "dtype": "s32"},
        ],
        "outputs": [
            {"node": 8, "shape": [4], "dtype": "f32"},
            {"node": 9, "shape": [], "dtype": "s32"},
            {"node": 2, "shape": [n_coeffs], "dtype": "f32"},
        ],
    }
    return {
        "version": 1,
        "inputs": [
            {"id": 0, "index": 0, "shape": [4], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [n_coeffs], "dtype": "f32"},
        ],
        "constants": [{"id": 2, "value": 0, "shape": [], "dtype": "s32"}],
        "nodes": [
            {
                "id": 10,
                "op": "while",
                "inputs": [0, 2, 1],
                "attrs": {"cond": _counter_cond(iters), "body": body},
                "outputs": [
                    {"id": 10, "shape": [4], "dtype": "f32"},
                    {"id": 11, "shape": [], "dtype": "s32"},
                    {"id": 12, "shape": [n_coeffs], "dtype": "f32"},
                ],
            }
        ],
        "outputs": [
            {"node": 10, "shape": [4], "dtype": "f32"},
            {"node": 12, "shape": [n_coeffs], "dtype": "f32"},
        ],
    }


def _sd_inputs(n_coeffs):
    return lambda: [_T([1.0, 2.0, 3.0, 4.0]), _T((np.arange(n_coeffs) + 1.0) / n_coeffs)]


def _linalg_shaped_graph(iters, n=6):
    """Carried matrix: row i is read (dynamic slice), doubled, and written back
    (dynamic put_slice) — the iterative-linalg shape (cholesky/qr/lu) that the
    old body-partition prototype silently broke."""
    body = {
        "inputs": [
            {"id": 0, "index": 0, "shape": [n, n], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [], "dtype": "s32"},
        ],
        "constants": [
            {"id": 2, "value": 1, "shape": [], "dtype": "s32"},
            {"id": 3, "value": 2.0, "shape": [], "dtype": "f32"},
        ],
        "nodes": [
            {
                "id": 4,
                "op": "slice",
                "inputs": [0, 1],
                "attrs": {
                    "starts": [{"input": 0}, {"static": 0}],
                    "lengths": [1, n],
                    "strides": [1, 1],
                },
                "shape": [1, n],
                "dtype": "f32",
            },
            {"id": 5, "op": "multiply", "inputs": [4, 3], "attrs": {}, "shape": [1, n], "dtype": "f32"},
            {
                "id": 6,
                "op": "put_slice",
                "inputs": [0, 5, 1],
                "attrs": {"starts": [{"input": 0}, {"static": 0}]},
                "shape": [n, n],
                "dtype": "f32",
            },
            {"id": 7, "op": "add", "inputs": [1, 2], "attrs": {}, "shape": [], "dtype": "s32"},
        ],
        "outputs": [
            {"node": 6, "shape": [n, n], "dtype": "f32"},
            {"node": 7, "shape": [], "dtype": "s32"},
        ],
    }
    return {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [n, n], "dtype": "f32"}],
        "constants": [{"id": 1, "value": 0, "shape": [], "dtype": "s32"}],
        "nodes": [
            {
                "id": 10,
                "op": "while",
                "inputs": [0, 1],
                "attrs": {"cond": _counter_cond(iters), "body": body},
                "outputs": [
                    {"id": 10, "shape": [n, n], "dtype": "f32"},
                    {"id": 11, "shape": [], "dtype": "s32"},
                ],
            }
        ],
        "outputs": [{"node": 10, "shape": [n, n], "dtype": "f32"}],
    }


def _linalg_inputs(n=6):
    return lambda: [_T(np.arange(n * n, dtype=np.float32).reshape(n, n) / 10.0)]


# -- tests --------------------------------------------------------------------


def test_symbolic_while_matches_interpretation():
    before = _stats()
    _assert_parity(_sd_shaped_graph(iters=8, n_coeffs=8), _sd_inputs(8))
    delta = _delta(before)
    assert delta["while_steps_symbolic"] == 8
    assert delta["while_steps_interpreted"] == 8  # the forced reference run
    assert delta["while_jit_fallbacks"] == 0


def test_symbolic_while_clamps_out_of_range_starts():
    # 8 iterations over a 6-entry table: i = 6, 7 clamp to 5 (Nx semantics).
    _assert_parity(_sd_shaped_graph(iters=8, n_coeffs=6), _sd_inputs(6))


def test_symbolic_while_computes_expected_values():
    n = 8
    [acc, coeffs_out] = _run(_sd_shaped_graph(iters=n, n_coeffs=n), _sd_inputs(n))
    expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    table = (np.arange(n) + 1.0).astype(np.float32) / n
    for i in range(n):
        expected = expected * np.float32(0.5) + table[i]
    assert np.array_equal(acc, expected)
    assert np.array_equal(coeffs_out, table)  # invariant loop var passes through


def test_symbolic_while_dynamic_put_slice():
    before = _stats()
    _assert_parity(_linalg_shaped_graph(iters=6), _linalg_inputs())
    assert _delta(before)["while_steps_symbolic"] == 6


def test_swap_style_passthrough_is_not_stale():
    # Body {a, b} -> {b, a + b}: output 0 is input b passed through under a
    # different index. Without the realized-output clone inside the traced fn,
    # TinyJit replay would return the stale capture-time buffer for it.
    body = {
        "inputs": [
            {"id": 0, "index": 0, "shape": [3], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [3], "dtype": "f32"},
            {"id": 2, "index": 2, "shape": [], "dtype": "s32"},
        ],
        "constants": [{"id": 3, "value": 1, "shape": [], "dtype": "s32"}],
        "nodes": [
            {"id": 4, "op": "add", "inputs": [0, 1], "attrs": {}, "shape": [3], "dtype": "f32"},
            {"id": 5, "op": "add", "inputs": [2, 3], "attrs": {}, "shape": [], "dtype": "s32"},
        ],
        "outputs": [
            {"node": 1, "shape": [3], "dtype": "f32"},
            {"node": 4, "shape": [3], "dtype": "f32"},
            {"node": 5, "shape": [], "dtype": "s32"},
        ],
    }
    graph = {
        "version": 1,
        "inputs": [
            {"id": 0, "index": 0, "shape": [3], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [3], "dtype": "f32"},
        ],
        "constants": [{"id": 2, "value": 0, "shape": [], "dtype": "s32"}],
        "nodes": [
            {
                "id": 10,
                "op": "while",
                "inputs": [0, 1, 2],
                "attrs": {"cond": _counter_cond(5, input_index=2), "body": body},
                "outputs": [
                    {"id": 10, "shape": [3], "dtype": "f32"},
                    {"id": 11, "shape": [3], "dtype": "f32"},
                    {"id": 12, "shape": [], "dtype": "s32"},
                ],
            }
        ],
        "outputs": [
            {"node": 10, "shape": [3], "dtype": "f32"},
            {"node": 11, "shape": [3], "dtype": "f32"},
        ],
    }
    make_inputs = lambda: [_T([1.0, 1.0, 1.0]), _T([2.0, 4.0, 8.0])]  # noqa: E731
    _assert_parity(graph, make_inputs)


def test_static_body_uses_plain_jit():
    # No dynamic starts at all: the zero-Variable degenerate case.
    body = {
        "inputs": [
            {"id": 0, "index": 0, "shape": [4], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [], "dtype": "s32"},
        ],
        "constants": [{"id": 2, "value": 1, "shape": [], "dtype": "s32"}],
        "nodes": [
            {"id": 3, "op": "multiply", "inputs": [0, 0], "attrs": {}, "shape": [4], "dtype": "f32"},
            {"id": 4, "op": "add", "inputs": [1, 2], "attrs": {}, "shape": [], "dtype": "s32"},
        ],
        "outputs": [
            {"node": 3, "shape": [4], "dtype": "f32"},
            {"node": 4, "shape": [], "dtype": "s32"},
        ],
    }
    graph = {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [4], "dtype": "f32"}],
        "constants": [{"id": 1, "value": 0, "shape": [], "dtype": "s32"}],
        "nodes": [
            {
                "id": 10,
                "op": "while",
                "inputs": [0, 1],
                "attrs": {"cond": _counter_cond(3), "body": body},
                "outputs": [
                    {"id": 10, "shape": [4], "dtype": "f32"},
                    {"id": 11, "shape": [], "dtype": "s32"},
                ],
            }
        ],
        "outputs": [{"node": 10, "shape": [4], "dtype": "f32"}],
    }
    make_inputs = lambda: [_T([1.1, 0.9, 1.0, 0.5])]  # noqa: E731
    before = _stats()
    _assert_parity(graph, make_inputs)
    delta = _delta(before)
    assert delta["while_steps_jit"] == 3
    assert delta["while_steps_symbolic"] == 0


def test_full_axis_dynamic_start_pins_to_static():
    # The slice covers the whole axis, so the Nx clamp pins the start to 0 no
    # matter how large the counter gets; the body becomes fully static.
    before = _stats()
    _assert_parity(_sd_shaped_graph(iters=6, n_coeffs=4, length=4), _sd_inputs(4))
    delta = _delta(before)
    assert delta["while_steps_jit"] == 6
    assert delta["while_steps_symbolic"] == 0


def test_while_step_jit_is_cached_across_executes():
    constructions = 0
    real_jit = executable_mod.TinyJit

    class CountingJit(real_jit):
        def __init__(self, fxn):
            nonlocal constructions
            constructions += 1
            super().__init__(fxn)

    executable_mod.TinyJit = CountingJit
    try:
        graph = _sd_shaped_graph(iters=6, n_coeffs=6)
        ex = compile_graph(1, graph, [], "CPU")
        first = [np.array(o.numpy()) for o in ex.run(_sd_inputs(6)())]
        second = [np.array(o.numpy()) for o in ex.run(_sd_inputs(6)())]
    finally:
        executable_mod.TinyJit = real_jit

    assert constructions == 1  # the second execute reuses the captured step
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_top_level_segments_around_while_are_jitted():
    # prefix (multiply) -> while -> suffix (add) : the static regions around
    # the while must run as cached TinyJit segments, with exact parity.
    n = 6
    body = {
        "inputs": [
            {"id": 0, "index": 0, "shape": [4], "dtype": "f32"},
            {"id": 1, "index": 1, "shape": [], "dtype": "s32"},
        ],
        "constants": [{"id": 2, "value": 1, "shape": [], "dtype": "s32"}],
        "nodes": [
            {"id": 3, "op": "multiply", "inputs": [0, 0], "attrs": {}, "shape": [4], "dtype": "f32"},
            {"id": 4, "op": "add", "inputs": [1, 2], "attrs": {}, "shape": [], "dtype": "s32"},
        ],
        "outputs": [
            {"node": 3, "shape": [4], "dtype": "f32"},
            {"node": 4, "shape": [], "dtype": "s32"},
        ],
    }
    graph = {
        "version": 1,
        "inputs": [{"id": 0, "index": 0, "shape": [4], "dtype": "f32"}],
        "constants": [
            {"id": 1, "value": 0, "shape": [], "dtype": "s32"},
            {"id": 2, "value": 0.5, "shape": [], "dtype": "f32"},
        ],
        "nodes": [
            # prefix segment
            {"id": 10, "op": "multiply", "inputs": [0, 2], "attrs": {}, "shape": [4], "dtype": "f32"},
            {"id": 11, "op": "tanh", "inputs": [10], "attrs": {}, "shape": [4], "dtype": "f32"},
            # eager while
            {
                "id": 20,
                "op": "while",
                "inputs": [11, 1],
                "attrs": {"cond": _counter_cond(3), "body": body},
                "outputs": [
                    {"id": 20, "shape": [4], "dtype": "f32"},
                    {"id": 21, "shape": [], "dtype": "s32"},
                ],
            },
            # suffix segment
            {"id": 30, "op": "add", "inputs": [20, 2], "attrs": {}, "shape": [4], "dtype": "f32"},
        ],
        "outputs": [{"node": 30, "shape": [4], "dtype": "f32"}],
    }
    make_inputs = lambda: [_T([0.5, 1.0, 1.5, 2.0])]  # noqa: E731

    before = _stats()
    _assert_parity(graph, make_inputs)
    delta = _delta(before)
    assert delta["graph_segments_jit"] == 2  # prefix + suffix, one non-forced run
    assert delta["graph_segment_fallbacks"] == 0

    # segments and while steps are cached across executes of one executable
    ex = compile_graph(2, graph, [], "CPU")
    r1 = [np.array(o.numpy()) for o in ex.run(make_inputs())]
    r2 = [np.array(o.numpy()) for o in ex.run(make_inputs())]
    assert len(ex._segment_steps) == 2
    for a, b in zip(r1, r2):
        assert np.array_equal(a, b)


def test_falls_back_to_interpretation_when_jit_breaks():
    class BrokenJit:
        def __init__(self, fxn):
            self.fxn = fxn

        def __call__(self, *args, **kwargs):
            raise RuntimeError("simulated tinygrad drift")

    orig = executable_mod.TinyJit
    executable_mod.TinyJit = BrokenJit
    try:
        before = _stats()
        actual = _run(_sd_shaped_graph(iters=8, n_coeffs=8), _sd_inputs(8))
        delta = _delta(before)
    finally:
        executable_mod.TinyJit = orig

    expected = _run(_sd_shaped_graph(iters=8, n_coeffs=8), _sd_inputs(8), interpret=True)
    for exp, act in zip(expected, actual):
        assert np.array_equal(exp, act)
    assert delta["while_jit_fallbacks"] == 1
    assert delta["while_steps_interpreted"] == 8
