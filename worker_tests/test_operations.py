import numpy as np
from tinygrad import Tensor

import operations


def run(op, inputs, dtype="f32", attrs=None, shape=None):
    env = {i: t for i, t in enumerate(inputs)}
    node = {
        "op": op,
        "inputs": list(range(len(inputs))),
        "attrs": attrs or {},
        "shape": shape if shape is not None else list(inputs[0].shape),
        "dtype": dtype,
    }
    return operations.apply(node, env).numpy()


def T(data, dtype=np.float32):
    return Tensor(np.array(data, dtype=dtype))


def test_add_subtract_multiply_divide():
    a, b = T([1, 2, 4]), T([2, 2, 2])
    assert np.allclose(run("add", [a, b]), [3, 4, 6])
    assert np.allclose(run("subtract", [a, b]), [-1, 0, 2])
    assert np.allclose(run("multiply", [a, b]), [2, 4, 8])
    assert np.allclose(run("divide", [a, b]), [0.5, 1, 2])


def test_expm1_and_log1p_are_stable_near_zero():
    values = np.array([1e-8, -1e-8, 1e-6, -1e-6, 1e-3], dtype=np.float32)
    tensor = Tensor(values)

    assert np.allclose(run("expm1", [tensor]), np.expm1(values), rtol=1e-6, atol=1e-12)
    assert np.allclose(run("log1p", [tensor]), np.log1p(values), rtol=1e-6, atol=1e-12)


def test_abs_canonicalizes_negative_zero():
    out = run("abs", [T([-0.0, 0.0, -1.0])])
    assert out.tolist() == [0.0, 0.0, 1.0]
    assert not np.signbit(out[0])


def test_broadcasting_scalar_plus_vector():
    out = run("add", [T(1.0), T([1, 2, 3])], shape=[3])
    assert np.allclose(out, [2, 3, 4])


def test_comparison_yields_u8():
    out = run("greater", [T([1, 5, 3]), T([2, 2, 2])], dtype="u8")
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 1, 1]


def test_select():
    pred = Tensor(np.array([1, 0, 1], dtype=np.uint8))
    out = run("select", [pred, T([10, 20, 30]), T([1, 2, 3])], shape=[3])
    assert np.allclose(out, [10, 2, 30])


def test_sum_and_reduce_max_with_axes():
    x = T([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(run("sum", [x], attrs={"axes": [1], "keep_axes": False}, shape=[2]), [6, 15])
    assert np.allclose(run("reduce_max", [x], attrs={"axes": [0], "keep_axes": False}, shape=[3]), [4, 5, 6])


def test_matmul_via_dot():
    a = T([[1, 2, 3], [4, 5, 6]])  # (2,3)
    b = T([[1, 0], [0, 1], [1, 1]])  # (3,2)
    attrs = {"contract_left": [1], "contract_right": [0], "batch_left": [], "batch_right": []}
    out = run("dot", [a, b], attrs=attrs, shape=[2, 2])
    assert np.allclose(out, np.array([[1, 2, 3], [4, 5, 6]]) @ np.array([[1, 0], [0, 1], [1, 1]]))


def test_transpose_and_reshape():
    x = T([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(run("transpose", [x], attrs={"axes": [1, 0]}, shape=[3, 2]), [[1, 4], [2, 5], [3, 6]])
    assert np.allclose(run("reshape", [x], attrs={"shape": [6]}, shape=[6]), [1, 2, 3, 4, 5, 6])
