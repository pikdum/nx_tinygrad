"""Translation of graph IR nodes into tinygrad Tensor operations.

Each operation takes the node dict and an ``env`` mapping graph-id -> tinygrad
Tensor, and returns the resulting Tensor. Results are cast to the node's declared
dtype so the output matches Nx exactly (notably: comparisons produce u8, and Nx's
output-type promotion is authoritative).

tinygrad broadcasting follows numpy rules, which matches Nx for the binary ops we
support, so operands are not explicitly broadcast here.
"""
from __future__ import annotations

import math

from tinygrad import Tensor, dtypes

from dtype import component_dtype, is_complex, tinygrad_dtype
from errors import UnsupportedOperation


def _cast(t: Tensor, node) -> Tensor:
    return t.cast(tinygrad_dtype(node["dtype"]))


# -- complex numbers ----------------------------------------------------------
#
# tinygrad has no complex dtype, so a complex tensor of logical shape S is held
# as a real float tensor of shape S + [2] (last axis = [real, imag]), wrapped in
# Cx so it is self-identifying as it flows through the executor env.


class Cx:
    __slots__ = ("t",)

    def __init__(self, t: Tensor):
        self.t = t  # float tensor of shape S + [2]


def _re(t: Tensor) -> Tensor:
    return t[..., 0]


def _im(t: Tensor) -> Tensor:
    return t[..., 1]


def _cxt(re: Tensor, im: Tensor) -> Tensor:
    return re.reshape(tuple(re.shape) + (1,)).cat(im.reshape(tuple(im.shape) + (1,)), dim=-1)


def _promote(x) -> Tensor:
    # A real operand in a complex expression becomes (x, 0).
    if isinstance(x, Cx):
        return x.t
    return _cxt(x, x * 0)


def _dft(t: Tensor, n: int, axis: int, inverse: bool) -> Tensor:
    # Discrete Fourier transform along `axis` via a real cos/sin matrix (O(n^2),
    # always correct). t is a complex S+[2] tensor; returns a complex S+[2].
    k = Tensor.arange(n).reshape((n, 1))
    m = Tensor.arange(n).reshape((1, n))
    theta = (2.0 * math.pi / n) * (k * m).cast(dtypes.float32)
    wr = theta.cos()
    wi = (theta if inverse else -theta).sin()  # sign flips for inverse

    xr, xi = _re(t), _im(t)
    # Move the fft axis to the last logical position, contract with W, move back.
    rank = len(xr.shape)
    perm = [a for a in range(rank) if a != axis] + [axis]
    inv = [0] * rank
    for pos, a in enumerate(perm):
        inv[a] = pos
    xr, xi = xr.permute(tuple(perm)), xi.permute(tuple(perm))

    # X[..., k] = sum_m x[..., m] * W[k, m]  ->  x @ W^T
    wr_t, wi_t = wr.transpose(0, 1), wi.transpose(0, 1)
    yr = xr.matmul(wr_t) - xi.matmul(wi_t)
    yi = xr.matmul(wi_t) + xi.matmul(wr_t)
    if inverse:
        yr, yi = yr * (1.0 / n), yi * (1.0 / n)

    yr, yi = yr.permute(tuple(inv)), yi.permute(tuple(inv))
    return _cxt(yr, yi)


def _apply_complex(node, ins):
    op = node["op"]
    shape = node["shape"]
    rank = len(shape)

    if op == "add":
        return Cx(_promote(ins[0]) + _promote(ins[1]))
    if op == "subtract":
        return Cx(_promote(ins[0]) - _promote(ins[1]))
    if op == "negate":
        return Cx(-_promote(ins[0]))
    if op == "multiply":
        a, b = _promote(ins[0]), _promote(ins[1])
        return Cx(_cxt(_re(a) * _re(b) - _im(a) * _im(b), _re(a) * _im(b) + _im(a) * _re(b)))
    if op == "divide":
        a, b = _promote(ins[0]), _promote(ins[1])
        d = _re(b) * _re(b) + _im(b) * _im(b)
        return Cx(_cxt((_re(a) * _re(b) + _im(a) * _im(b)) / d, (_im(a) * _re(b) - _re(a) * _im(b)) / d))
    if op == "conjugate":
        a = _promote(ins[0])
        return Cx(_cxt(_re(a), -_im(a)))
    if op == "exp":
        a = _promote(ins[0])
        e = _re(a).exp()
        return Cx(_cxt(e * _im(a).cos(), e * _im(a).sin()))
    if op == "real":
        return _re(_promote(ins[0]))
    if op == "imag":
        return _im(_promote(ins[0]))
    if op == "abs":
        a = _promote(ins[0])
        return (_re(a) * _re(a) + _im(a) * _im(a)).sqrt()
    if op == "as_type":
        return Cx(_promote(ins[0]))
    if op in ("reshape", "squeeze"):
        return Cx(_promote(ins[0]).reshape(tuple(shape) + (2,)))
    if op == "transpose":
        axes = node["attrs"]["axes"]
        return Cx(_promote(ins[0]).permute(tuple(axes) + (rank,)))
    if op == "reverse":
        return Cx(_promote(ins[0]).flip(tuple(node["attrs"]["axes"])))
    if op == "broadcast":
        out = node["attrs"]["shape"]
        axes = node["attrs"]["axes"]
        a = _promote(ins[0])
        inter = [1] * len(out) + [2]
        for src_dim, out_dim in enumerate(axes):
            inter[out_dim] = a.shape[src_dim]
        return Cx(a.reshape(tuple(inter)).expand(tuple(out) + (2,)))
    if op == "slice":
        a = _promote(ins[0])
        starts = [s["static"] for s in node["attrs"]["starts"]]
        lengths, strides = node["attrs"]["lengths"], node["attrs"]["strides"]
        idx = tuple(slice(s, s + l, st) for s, l, st in zip(starts, lengths, strides)) + (slice(None),)
        return Cx(a[idx])
    if op == "concatenate":
        parts = [_promote(x) for x in ins]
        return Cx(parts[0].cat(*parts[1:], dim=node["attrs"]["axis"]))
    if op in ("sum", "reduce_max", "reduce_min"):
        a = _promote(ins[0])
        axes = tuple(node["attrs"]["axes"])
        keep = node["attrs"]["keep_axes"]
        red = _re(a).sum(axis=axes, keepdim=keep), _im(a).sum(axis=axes, keepdim=keep)
        return Cx(_cxt(*red))
    if op == "dot":
        a, b = _promote(ins[0]), _promote(ins[1])
        attrs = node["attrs"]
        ca, cb, ba, bb = attrs["contract_left"], attrs["contract_right"], attrs["batch_left"], attrs["batch_right"]
        ar, ai, br, bi = _re(a), _im(a), _re(b), _im(b)
        re = _einsum_dot(ar, br, ca, cb, ba, bb) - _einsum_dot(ai, bi, ca, cb, ba, bb)
        im = _einsum_dot(ar, bi, ca, cb, ba, bb) + _einsum_dot(ai, br, ca, cb, ba, bb)
        return Cx(_cxt(re, im))
    if op in ("fft", "ifft"):
        return Cx(_dft(_promote(ins[0]), node["attrs"]["length"], node["attrs"]["axis"], op == "ifft"))
    if op == "iota":
        return Cx(_promote(_iota(node)))
    if op == "select":
        cond = ins[0].cast(dtypes.bool)
        a, b = _promote(ins[1]), _promote(ins[2])
        return Cx(cond.reshape(tuple(cond.shape) + (1,)).where(a, b))

    raise UnsupportedOperation(f"complex op not supported: {op}", details={"op": op})


def _in(node, env, i=0):
    return env[node["inputs"][i]]


# -- elementwise unary --------------------------------------------------------

def _expm1(t: Tensor) -> Tensor:
    # tinygrad has no expm1 primitive. exp(x)-1 catastrophically cancels near
    # zero, so use a short Taylor expansion there and the regular expression
    # elsewhere. The threshold leaves the omitted term far below f32 precision.
    x2 = t * t
    small = t + x2 * (1 / 2) + x2 * t * (1 / 6) + x2 * x2 * (1 / 24) + x2 * x2 * t * (1 / 120)
    return (t.abs() < 1e-2).where(small, t.exp() - 1)


def _log1p(t: Tensor) -> Tensor:
    # As above, avoid losing small x when forming 1+x in the input dtype.
    x2 = t * t
    small = t - x2 * (1 / 2) + x2 * t * (1 / 3) - x2 * x2 * (1 / 4) + x2 * x2 * t * (1 / 5)
    return (t.abs() < 1e-2).where(small, (t + 1).log())


def _abs(t: Tensor) -> Tensor:
    # tinygrad preserves the sign bit for -0.0 here; Nx.abs returns +0.0.
    return (t == 0).where(0.0, t.abs())


_BIT_WIDTH = {"u8": 8, "s8": 8, "u16": 16, "s16": 16, "u32": 32, "s32": 32, "u64": 64, "s64": 64}


def _erf_inv(t: Tensor) -> Tensor:
    # Inverse error function via Mike Giles' single-precision rational
    # approximation (accurate to ~1e-6, matching Nx's own implementation).
    w = -((1.0 - t) * (1.0 + t)).log()
    lt = w < 5.0

    w1 = w - 2.5
    p1 = 2.81022636e-08
    for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
              -0.00125372503, -0.00417768164, 0.246640727, 1.50140941):
        p1 = c + p1 * w1

    w2 = w.sqrt() - 3.0
    p2 = -0.000200214257
    for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
              -0.0076224613, 0.00943887047, 1.00167406, 2.83297682):
        p2 = c + p2 * w2

    return lt.where(p1, p2) * t


def _bit_reduce(node, env):
    # count_leading_zeros / population_count over the operand's fixed bit width.
    # Uses arithmetic shifts, which give the correct result for both signed
    # (negative -> all-ones after smearing) and unsigned patterns.
    t = _in(node, env)
    width = _BIT_WIDTH[node["dtype"]]

    if node["op"] == "count_leading_zeros":
        smeared = t
        shift = 1
        while shift < width:
            smeared = smeared | (smeared >> shift)
            shift *= 2
        highest = smeared
    else:
        highest = t

    total = highest & 1
    for i in range(1, width):
        total = total + ((highest >> i) & 1)

    return width - total if node["op"] == "count_leading_zeros" else total


def _cbrt(t: Tensor) -> Tensor:
    # tinygrad has no cbrt, and pow of a negative base is nan. Take the root of
    # the magnitude and restore the sign so cbrt(-8) == -2 (matches Nx).
    r = t.abs() ** (1.0 / 3.0)
    return (t < 0).where(-r, r)


_UNARY = {
    "negate": lambda t: -t,
    "abs": _abs,
    "exp": lambda t: t.exp(),
    "expm1": _expm1,
    "log": lambda t: t.log(),
    "log1p": _log1p,
    "sqrt": lambda t: t.sqrt(),
    "rsqrt": lambda t: t.rsqrt(),
    "tanh": lambda t: t.tanh(),
    "sigmoid": lambda t: t.sigmoid(),
    "sin": lambda t: t.sin(),
    "cos": lambda t: t.cos(),
    "tan": lambda t: t.tan(),
    "asin": lambda t: t.asin(),
    "acos": lambda t: t.acos(),
    "atan": lambda t: t.atan(),
    "sinh": lambda t: t.sinh(),
    "cosh": lambda t: t.cosh(),
    "asinh": lambda t: t.asinh(),
    "acosh": lambda t: t.acosh(),
    "atanh": lambda t: t.atanh(),
    "erf": lambda t: t.erf(),
    "erfc": lambda t: 1 - t.erf(),
    "erf_inv": _erf_inv,
    # conjugate of a real tensor is the identity (complex is not supported).
    "conjugate": lambda t: t,
    "cbrt": _cbrt,
    "sign": lambda t: t.sign(),
    # Nx rounds half away from zero; tinygrad's round is half-to-even. Compose.
    "round": lambda t: t.sign() * (t.abs() + 0.5).floor(),
    "is_nan": lambda t: t.isnan(),
    "is_infinity": lambda t: t.isinf(),
    "bitwise_not": lambda t: t.bitwise_not(),
    "floor": lambda t: t.floor(),
    "ceil": lambda t: t.ceil(),
}

# -- elementwise binary -------------------------------------------------------

def _remainder(a: Tensor, b: Tensor) -> Tensor:
    # Nx.remainder takes the sign of the dividend (truncated division), while
    # tinygrad's % follows Python floor semantics. Compose to match Nx.
    return a - b * (a / b).trunc()


def _quotient(a: Tensor, b: Tensor) -> Tensor:
    # Integer division truncating toward zero (Nx semantics; the final cast
    # restores the integer output dtype).
    return (a / b).trunc()


def _atan2(y: Tensor, x: Tensor) -> Tensor:
    # tinygrad has no atan2; reconstruct it from atan with quadrant correction.
    base = (y / x).atan()
    adj = (x < 0).where((y >= 0).where(math.pi, -math.pi), 0.0)
    r = base + adj
    on_axis = (y > 0).where(math.pi / 2, (y < 0).where(-math.pi / 2, 0.0))
    return (x == 0).where(on_axis, r)


_BINARY = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
    "max": lambda a, b: a.maximum(b),
    "min": lambda a, b: a.minimum(b),
    "remainder": _remainder,
    "quotient": _quotient,
    "atan2": _atan2,
    "bitwise_and": lambda a, b: a & b,
    "bitwise_or": lambda a, b: a | b,
    "bitwise_xor": lambda a, b: a ^ b,
    "left_shift": lambda a, b: a << b,
    "right_shift": lambda a, b: a >> b,
    # Nx logical ops treat any nonzero as true and yield u8 0/1 (final cast).
    "logical_and": lambda a, b: (a != 0) & (b != 0),
    "logical_or": lambda a, b: (a != 0) | (b != 0),
    "logical_xor": lambda a, b: (a != 0) ^ (b != 0),
}

# -- comparisons (result cast to node dtype, i.e. u8) -------------------------

_COMPARISON = {
    "equal": lambda a, b: a == b,
    "not_equal": lambda a, b: a != b,
    "less": lambda a, b: a < b,
    "less_equal": lambda a, b: a <= b,
    "greater": lambda a, b: a > b,
    "greater_equal": lambda a, b: a >= b,
}


def _reduce(node, env):
    t = _in(node, env)
    axes = node["attrs"]["axes"]
    keep = node["attrs"]["keep_axes"]
    op = node["op"]
    if not axes:
        # No-op reduction (e.g. scalar input); Nx keeps the value.
        return t
    axis = tuple(axes)
    if op == "sum":
        return t.sum(axis=axis, keepdim=keep)
    if op == "product":
        return t.prod(axis=axis, keepdim=keep)
    if op == "reduce_max":
        return t.max(axis=axis, keepdim=keep)
    if op == "reduce_min":
        return t.min(axis=axis, keepdim=keep)
    if op in ("all", "any"):
        mask = (t != 0).cast(dtypes.int32)
        reduced = mask.min(axis=axis, keepdim=keep) if op == "all" else mask.max(axis=axis, keepdim=keep)
        return reduced
    raise UnsupportedOperation(f"unsupported reduction: {op}")


def _argreduce(node, env):
    t = _in(node, env)
    op = node["op"]
    axis = node["attrs"]["axis"]  # int, or None for flattened argreduce
    keep = node["attrs"]["keep_axis"]
    tie = node["attrs"]["tie_break"]  # "low" (first index) or "high" (last)

    def reduce(x, ax=None, keepdim=False):
        return x.argmax(ax, keepdim) if op == "argmax" else x.argmin(ax, keepdim)

    # tinygrad breaks ties toward the first index, matching Nx's default :low.
    if tie == "low":
        return reduce(t) if axis is None else reduce(t, axis, keep)

    # :high wants the last winning index: flip the reduction axis, reduce, remap.
    if axis is None:
        flat = t.reshape((t.numel(),))
        return flat.shape[0] - 1 - reduce(flat.flip(0))
    return t.shape[axis] - 1 - reduce(t.flip(axis), axis, keep)


def _broadcast(node, env):
    t = _in(node, env)
    out_shape = node["attrs"]["shape"]
    axes = node["attrs"]["axes"]
    # Reshape source dims into their output positions (1 elsewhere), then expand.
    inter = [1] * len(out_shape)
    for src_dim, out_dim in enumerate(axes):
        inter[out_dim] = t.shape[src_dim]
    return t.reshape(tuple(inter)).expand(tuple(out_shape))


_SPECIAL_NUMBERS = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}


def _num(v):
    return _SPECIAL_NUMBERS[v] if isinstance(v, str) else v


def _pad(node, env):
    t = _in(node, env)
    config = node["attrs"]["config"]  # [[low, high, interior], ...]
    if any(interior != 0 for (_lo, _hi, interior) in config):
        raise UnsupportedOperation("interior padding is not supported", details={"config": config})
    if any(lo < 0 or hi < 0 for (lo, hi, _i) in config):
        raise UnsupportedOperation("negative (cropping) padding is not supported", details={"config": config})
    padding = tuple((lo, hi) for (lo, hi, _i) in config)
    return t.pad(padding, value=float(_num(node["attrs"]["value"])))


def _sort(node, env):
    t = _in(node, env)
    return t.sort(node["attrs"]["axis"], node["attrs"]["descending"])[0]


def _argsort(node, env):
    t = _in(node, env)
    return t.argsort(node["attrs"]["axis"], node["attrs"]["descending"])


def _slice(node, env):
    t = _in(node, env, 0)
    dyn = [env[i] for i in node["inputs"][1:]]
    sym = env.get("__sym__") or []
    lengths = node["attrs"]["lengths"]
    strides = node["attrs"]["strides"]

    starts = []
    symbolic = False
    for axis, spec in enumerate(node["attrs"]["starts"]):
        if "static" in spec:
            starts.append(spec["static"])
        elif "symbolic" in spec:
            # A bound tinygrad Variable (clamped at bind time) so the slice
            # stays JIT-capturable inside a while body.
            starts.append(sym[spec["symbolic"]])
            symbolic = True
        else:
            # Dynamic start (a scalar tensor); Nx clamps it so the slice fits.
            raw = int(dyn[spec["input"]].item())
            starts.append(max(0, min(raw, t.shape[axis] - lengths[axis])))

    if symbolic:
        out = t.shrink(tuple((s, s + l) for s, l in zip(starts, lengths)))
        if any(st != 1 for st in strides):
            out = out[tuple(slice(None, None, st) for st in strides)]
        return out

    idx = tuple(slice(s, s + l, st) for s, l, st in zip(starts, lengths, strides))
    return t[idx]


def _clip(node, env):
    return _in(node, env, 0).maximum(_in(node, env, 1)).minimum(_in(node, env, 2))


def _stack(node, env):
    # Insert the new axis into each operand, then concatenate along it.
    axis = node["attrs"]["axis"]
    parts = []
    for i in node["inputs"]:
        x = env[i]
        shp = list(x.shape)
        shp.insert(axis, 1)
        parts.append(x.reshape(tuple(shp)))
    return parts[0].cat(*parts[1:], dim=axis) if len(parts) > 1 else parts[0]


def _eye(node):
    shape = node["shape"]
    r, c = shape[-2], shape[-1]
    diag = Tensor.arange(r).reshape((r, 1)) == Tensor.arange(c).reshape((1, c))
    return diag.reshape((1,) * (len(shape) - 2) + (r, c)).expand(tuple(shape))


_WINDOW_IDENTITY = {
    "window_sum": 0.0,
    "window_max": -math.inf,
    "window_min": math.inf,
    "window_product": 1.0,
}


def _window(node, env):
    # Nx windowed reduction: pool over every axis (size-1 windows on non-pooled
    # axes), then reduce the trailing window axes _pool exposes. Padding uses the
    # reduction's identity so it never changes the result.
    t = _in(node, env)
    op = node["op"]
    rank = len(t.shape)
    padding = node["attrs"]["padding"]
    if any(lo or hi for (lo, hi) in padding):
        t = t.pad(tuple((lo, hi) for (lo, hi) in padding), value=float(_WINDOW_IDENTITY[op]))

    pooled = t._pool(
        tuple(node["attrs"]["window"]),
        stride=tuple(node["attrs"]["strides"]),
        dilation=tuple(node["attrs"]["window_dilations"]),
    )
    axes = tuple(range(rank, 2 * rank))
    if op == "window_sum":
        return pooled.sum(axis=axes)
    if op == "window_max":
        return pooled.max(axis=axes)
    if op == "window_min":
        return pooled.min(axis=axes)
    return pooled.prod(axis=axes)


def _window_scatter(node, env):
    # Select-and-scatter (max-pool / min-pool backward): for each window of the
    # operand, scatter the corresponding `source` value onto the window's extreme
    # element, accumulating across overlapping windows. Nx breaks ties toward the
    # last (highest-index) extreme within a window.
    op = node["op"]
    t = _in(node, env, 0)
    source = _in(node, env, 1)
    init = float(_num(node["attrs"]["init"]))
    window = node["attrs"]["window"]
    strides = node["attrs"]["strides"]
    padding = node["attrs"]["padding"]
    rank = len(t.shape)
    identity = -math.inf if op == "window_scatter_max" else math.inf

    padded = t
    if any(lo or hi for (lo, hi) in padding):
        padded = t.pad(tuple((lo, hi) for (lo, hi) in padding), value=identity)
    pshape = padded.shape

    windows = padded._pool(tuple(window), stride=tuple(strides), dilation=1)
    out_dims = list(windows.shape[:rank])
    win_dims = list(windows.shape[rank:])
    n_win = math.prod(win_dims) if win_dims else 1

    flat_win = windows.reshape(tuple(out_dims) + (n_win,))
    best = flat_win.max(axis=rank, keepdim=True) if op == "window_scatter_max" else flat_win.min(axis=rank, keepdim=True)
    widx = Tensor.arange(n_win).reshape((1,) * len(out_dims) + (n_win,))
    last = (flat_win == best).where(widx, -1).max(axis=rank, keepdim=True)
    selected = (widx == last)  # one-hot at the last extreme per window
    contrib = (selected * source.reshape(tuple(out_dims) + (1,))).reshape(tuple(out_dims) + tuple(win_dims))

    # Linear index into the flat padded operand for each (out.., win..) element.
    pstrides = [0] * rank
    acc = 1
    for a in range(rank - 1, -1, -1):
        pstrides[a] = acc
        acc *= pshape[a]

    total_rank = 2 * rank
    lin = None
    for a in range(rank):
        out_shape = [1] * total_rank
        out_shape[a] = out_dims[a]
        oc = Tensor.arange(out_dims[a]).reshape(tuple(out_shape))
        win_shape = [1] * total_rank
        win_shape[rank + a] = win_dims[a]
        wc = Tensor.arange(win_dims[a]).reshape(tuple(win_shape))
        coord = oc * strides[a] + wc
        term = coord * pstrides[a]
        lin = term if lin is None else lin + term

    lin = lin.expand(tuple(out_dims) + tuple(win_dims)).reshape((-1,)).cast(dtypes.int32)
    vals = contrib.reshape((-1,))
    scattered = Tensor.zeros(math.prod(pshape), dtype=vals.dtype).scatter_reduce(0, lin, vals, reduce="sum")
    out = scattered.reshape(pshape) + init

    if any(lo or hi for (lo, hi) in padding):
        out = out[tuple(slice(lo, lo + t.shape[a]) for a, (lo, hi) in enumerate(padding))]
    return out


def _iota(node):
    # Index counter along `axis` (or over the flattened tensor when axis is None).
    shape = node["shape"]
    axis = node["attrs"]["axis"]
    if axis is None:
        return Tensor.arange(math.prod(shape)).reshape(tuple(shape))
    view = [1] * len(shape)
    view[axis] = shape[axis]
    return Tensor.arange(shape[axis]).reshape(tuple(view)).expand(tuple(shape))


def _gather(node, env):
    # Nx coordinate gather: idx[..., j] indexes t along axes[j]. Output shape is
    # idx.shape[:-1] ++ (t's non-indexed axes). We move the indexed axes to the
    # front, collapse them, turn each coordinate into one linear index, then
    # index_select along that combined axis via tinygrad fancy indexing.
    t = _in(node, env, 0)
    idx = _in(node, env, 1)
    axes = list(node["attrs"]["axes"])
    rank = len(t.shape)
    non_axes = [d for d in range(rank) if d not in axes]

    a_sizes = [t.shape[a] for a in axes]
    b_sizes = [t.shape[d] for d in non_axes]
    k = idx.shape[-1]
    batch = tuple(idx.shape[:-1])
    rows = math.prod(batch) if batch else 1

    combined = math.prod(a_sizes) if a_sizes else 1
    t3 = t.permute(tuple(axes) + tuple(non_axes)).reshape((combined,) + tuple(b_sizes))

    # row-major strides over the indexed axes
    strides = [0] * k
    acc = 1
    for j in range(k - 1, -1, -1):
        strides[j] = acc
        acc *= a_sizes[j]

    flat_idx = idx.reshape((rows, k)).cast(dtypes.int32)
    lin = flat_idx[:, 0] * strides[0]
    for j in range(1, k):
        lin = lin + flat_idx[:, j] * strides[j]

    return t3[lin].reshape(batch + tuple(b_sizes))


def _put_slice(node, env):
    # Overwrite target[starts : starts+slice.shape] with the slice, composed from
    # pad + select. Nx clamps starts so the slice fits fully inside the target.
    target = _in(node, env, 0)
    sl = _in(node, env, 1)
    dyn = [env[i] for i in node["inputs"][2:]]
    sym = env.get("__sym__") or []
    pad_config = []
    for dim, spec in enumerate(node["attrs"]["starts"]):
        if "symbolic" in spec:
            # A bound tinygrad Variable, already clamped at bind time; pad
            # accepts symbolic amounts, keeping the body JIT-capturable.
            start = sym[spec["symbolic"]]
        else:
            raw = spec["static"] if "static" in spec else int(dyn[spec["input"]].item())
            start = max(0, min(int(raw), target.shape[dim] - sl.shape[dim]))
        pad_config.append((start, target.shape[dim] - sl.shape[dim] - start))
    padded = sl.pad(tuple(pad_config))
    mask = (sl * 0 + 1).pad(tuple(pad_config))
    return (mask != 0).where(padded, target)


def _indexed(node, env):
    # Nx indexed_add/indexed_put: idx is {K, len(axes)} coordinates, updates is
    # {K} ++ (non-indexed axes). Collapse the indexed axes into one, scatter along
    # it, then restore the original axis order.
    t = _in(node, env, 0)
    idx = _in(node, env, 1).cast(dtypes.int32)
    upd = _in(node, env, 2)
    axes = list(node["attrs"]["axes"])
    rank = len(t.shape)
    non_axes = [d for d in range(rank) if d not in axes]

    a_sizes = [t.shape[a] for a in axes]
    b_sizes = [t.shape[d] for d in non_axes]
    combined = math.prod(a_sizes) if a_sizes else 1
    rows = idx.shape[0]

    perm = tuple(axes) + tuple(non_axes)
    t2 = t.permute(perm).reshape((combined,) + tuple(b_sizes))

    strides = [0] * len(a_sizes)
    acc = 1
    for j in range(len(a_sizes) - 1, -1, -1):
        strides[j] = acc
        acc *= a_sizes[j]
    lin = idx[:, 0] * strides[0]
    for j in range(1, len(a_sizes)):
        lin = lin + idx[:, j] * strides[j]

    # tinygrad scatter requires src and self to share a dtype.
    src = upd.reshape((rows,) + tuple(b_sizes)).cast(t2.dtype)
    index = lin.reshape((rows,) + (1,) * len(b_sizes)).expand((rows,) + tuple(b_sizes))

    if node["op"] == "indexed_add":
        out = t2.scatter_reduce(0, index, src, reduce="sum", include_self=True)
    else:
        out = t2.scatter(0, index, src)

    out = out.reshape(tuple(a_sizes) + tuple(b_sizes))
    inverse = [0] * rank
    for new_pos, orig_axis in enumerate(perm):
        inverse[orig_axis] = new_pos
    return out.permute(tuple(inverse))


def _concatenate(node, env):
    tensors = [env[i] for i in node["inputs"]]
    axis = node["attrs"]["axis"]
    return tensors[0].cat(*tensors[1:], dim=axis) if len(tensors) > 1 else tensors[0]


def _dilate(x, dilation):
    # Insert (d-1) zeros between elements along each spatial axis (axes 2..) —
    # i.e. input dilation, as used by conv's gradient w.r.t. its input.
    for i, d in enumerate(dilation):
        if d <= 1:
            continue
        axis = 2 + i
        shape = list(x.shape)
        n = shape[axis]
        x = x.reshape(tuple(shape[: axis + 1] + [1] + shape[axis + 1 :]))
        pad = [(0, 0)] * len(x.shape)
        pad[axis + 1] = (0, d - 1)
        x = x.pad(tuple(pad))
        x = x.reshape(tuple(shape[:axis] + [n * d] + shape[axis + 1 :]))
        idx = tuple(slice(0, (n - 1) * d + 1) if j == axis else slice(None) for j in range(len(x.shape)))
        x = x[idx]
    return x


def _conv(node, env):
    # Nx.conv onto tinygrad conv2d (general over spatial rank). We transpose the
    # operands into canonical [batch, channels, *spatial] / [out, in, *spatial]
    # layout, honor Nx's asymmetric padding and input dilation ourselves, then
    # transpose the output back out of canonical layout.
    x = _in(node, env, 0)
    w = _in(node, env, 1)
    a = node["attrs"]

    if a["batch_group_size"] != 1:
        raise UnsupportedOperation("conv batch_group_size != 1 is not supported")

    x = x.permute(tuple(a["input_permutation"]))
    w = w.permute(tuple(a["kernel_permutation"]))

    if any(d != 1 for d in a["input_dilation"]):
        x = _dilate(x, a["input_dilation"])

    pad_config = [(0, 0), (0, 0)] + [(lo, hi) for (lo, hi) in a["padding"]]
    if any(lo or hi for (lo, hi) in pad_config):
        x = x.pad(tuple(pad_config))

    out = x.conv2d(
        w,
        groups=a["feature_group_size"],
        stride=tuple(a["strides"]),
        dilation=tuple(a["kernel_dilation"]),
        padding=0,
    )

    # out is canonical [batch, out_channels, *spatial]; undo output_permutation.
    op = a["output_permutation"]
    inverse = [0] * len(op)
    for pos, axis in enumerate(op):
        inverse[axis] = pos
    return out.permute(tuple(inverse))


def _triangular_solve(node, env):
    # Solve a @ x = b for triangular `a` by statically unrolled forward/back
    # substitution (the matrix dimension is known, so no dynamic loop).
    a = _in(node, env, 0)
    b = _in(node, env, 1)
    attrs = node["attrs"]

    if not attrs["left_side"]:
        raise UnsupportedOperation("triangular_solve with left_side=false is not supported")

    lower = attrs["lower"]
    transform = attrs["transform_a"]
    if transform == "transpose":
        a = a.permute(1, 0)
        lower = not lower
    elif transform != "none":
        raise UnsupportedOperation(f"triangular_solve transform_a={transform!r} is not supported")

    n = a.shape[0]
    vector = len(b.shape) == 1
    rhs = b.reshape((n, 1)) if vector else b

    solved = {}
    for i in range(n) if lower else range(n - 1, -1, -1):
        acc = rhs[i]
        for j, xj in solved.items():
            acc = acc - a[i, j] * xj
        solved[i] = acc / a[i, i]

    stacked = None
    for i in range(n):
        row = solved[i].reshape((1,) + solved[i].shape)
        stacked = row if stacked is None else stacked.cat(row, dim=0)

    return stacked.reshape((n,)) if vector else stacked


def _dot(node, env):
    a, b = _in(node, env, 0), _in(node, env, 1)
    attrs = node["attrs"]
    return _einsum_dot(a, b, attrs["contract_left"], attrs["contract_right"], attrs["batch_left"], attrs["batch_right"])


_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _einsum_dot(a, b, ca, cb, ba, bb):
    """General Nx.dot via einsum: batch axes and contraction axes are shared
    between operands; free axes are unique. Output = batch ++ free_a ++ free_b."""
    ra, rb = len(a.shape), len(b.shape)
    sub_a = [None] * ra
    sub_b = [None] * rb
    nxt = 0

    def letter():
        nonlocal nxt
        c = _LETTERS[nxt]
        nxt += 1
        return c

    batch_letters = []
    for la, lb in zip(ba, bb):
        c = letter()
        sub_a[la] = c
        sub_b[lb] = c
        batch_letters.append(c)

    for la, lb in zip(ca, cb):
        c = letter()
        sub_a[la] = c
        sub_b[lb] = c

    free_a = []
    for i in range(ra):
        if sub_a[i] is None:
            sub_a[i] = letter()
            free_a.append(sub_a[i])
    free_b = []
    for j in range(rb):
        if sub_b[j] is None:
            sub_b[j] = letter()
            free_b.append(sub_b[j])

    out = batch_letters + free_a + free_b
    formula = f"{''.join(sub_a)},{''.join(sub_b)}->{''.join(out)}"
    return Tensor.einsum(formula, a, b)


def apply(node, env) -> Tensor:
    op = node["op"]

    # Complex path: a complex output, or any complex (Cx) operand.
    if is_complex(node["dtype"]) or any(isinstance(env[i], Cx) for i in node["inputs"]):
        result = _apply_complex(node, [env[i] for i in node["inputs"]])
        if isinstance(result, Cx):
            return Cx(result.t.cast(tinygrad_dtype(component_dtype(node["dtype"]))))
        return _cast(result, node)

    if op in _UNARY:
        result = _UNARY[op](_in(node, env))
    elif op in _BINARY:
        result = _BINARY[op](_in(node, env, 0), _in(node, env, 1))
    elif op in _COMPARISON:
        result = _COMPARISON[op](_in(node, env, 0), _in(node, env, 1))
    elif op == "select":
        cond = _in(node, env, 0).cast(dtypes.bool)
        # tinygrad's where requires both branches to share a dtype; cast to the
        # declared output dtype (Nx may hand us mismatched branch dtypes).
        out_dt = tinygrad_dtype(node["dtype"])
        result = cond.where(_in(node, env, 1).cast(out_dt), _in(node, env, 2).cast(out_dt))
    elif op in ("sum", "product", "reduce_max", "reduce_min", "all", "any"):
        result = _reduce(node, env)
    elif op in ("argmax", "argmin"):
        result = _argreduce(node, env)
    elif op in ("window_sum", "window_max", "window_min", "window_product"):
        result = _window(node, env)
    elif op in ("window_scatter_max", "window_scatter_min"):
        result = _window_scatter(node, env)
    elif op in ("count_leading_zeros", "population_count"):
        result = _bit_reduce(node, env)
    elif op == "reshape":
        result = _in(node, env).reshape(tuple(node["attrs"]["shape"]))
    elif op == "squeeze":
        result = _in(node, env).reshape(tuple(node["shape"]))
    elif op == "broadcast":
        result = _broadcast(node, env)
    elif op == "transpose":
        result = _in(node, env).permute(tuple(node["attrs"]["axes"]))
    elif op == "reverse":
        result = _in(node, env).flip(tuple(node["attrs"]["axes"]))
    elif op == "concatenate":
        result = _concatenate(node, env)
    elif op == "slice":
        result = _slice(node, env)
    elif op == "pad":
        result = _pad(node, env)
    elif op == "sort":
        result = _sort(node, env)
    elif op == "argsort":
        result = _argsort(node, env)
    elif op == "gather":
        result = _gather(node, env)
    elif op == "iota":
        result = _iota(node)
    elif op == "put_slice":
        result = _put_slice(node, env)
    elif op in ("indexed_add", "indexed_put"):
        result = _indexed(node, env)
    elif op == "clip":
        result = _clip(node, env)
    elif op == "stack":
        result = _stack(node, env)
    elif op == "eye":
        result = _eye(node)
    elif op == "as_type":
        result = _in(node, env)
    elif op == "bitcast":
        # Reinterpret the bits as the target dtype (no value conversion).
        # A bitcast is a buffer VIEW; a view of a TinyJit *input* buffer is not
        # rebound by jit input replacement on replay (observed on tinygrad
        # master: replays return capture-time bytes). Copy realized sources
        # first — the copy reads the input directly (rebindable) and the
        # bitcast then views jit-internal data. Unrealized sources are
        # jit-internal already; views of them are rewritten every replay.
        src = _in(node, env)
        if src.uop.base.realized is not None:
            src = src.clone()
        result = src.bitcast(tinygrad_dtype(node["dtype"]))
    elif op == "dot":
        result = _dot(node, env)
    elif op == "conv":
        result = _conv(node, env)
    elif op == "triangular_solve":
        result = _triangular_solve(node, env)
    else:
        raise UnsupportedOperation(f"unsupported Nx operation: {op}", details={"op": op})

    return _cast(result, node)


SUPPORTED_OPS = (
    set(_UNARY)
    | set(_BINARY)
    | set(_COMPARISON)
    | {"select", "sum", "product", "reduce_max", "reduce_min", "all", "any", "argmax", "argmin"}
    | {"window_sum", "window_max", "window_min", "window_product"}
    | {"window_scatter_max", "window_scatter_min"}
    | {"count_leading_zeros", "population_count"}
    | {"reshape", "squeeze", "broadcast", "transpose", "reverse", "concatenate", "slice", "as_type", "bitcast", "dot", "conv"}
    | {"triangular_solve"}
    | {"pad", "sort", "argsort", "gather", "iota", "clip", "stack", "eye"}
    | {"put_slice", "indexed_add", "indexed_put"}
    | {"while"}
    | {"fft", "ifft", "real", "imag"}
    | {"reduce", "window_reduce"}
)
