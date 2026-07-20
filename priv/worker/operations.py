"""Translation of graph IR nodes into tinygrad Tensor operations.

Each operation takes the node dict and an ``env`` mapping graph-id -> tinygrad
Tensor, and returns the resulting Tensor. Results are cast to the node's declared
dtype so the output matches Nx exactly (notably: comparisons produce u8, and Nx's
output-type promotion is authoritative).

tinygrad broadcasting follows numpy rules, which matches Nx for the binary ops we
support, so operands are not explicitly broadcast here.
"""
from __future__ import annotations

from tinygrad import Tensor, dtypes

from dtype import tinygrad_dtype
from errors import UnsupportedOperation


def _cast(t: Tensor, node) -> Tensor:
    return t.cast(tinygrad_dtype(node["dtype"]))


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


_UNARY = {
    "negate": lambda t: -t,
    "abs": lambda t: t.abs(),
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
    "floor": lambda t: t.floor(),
    "ceil": lambda t: t.ceil(),
}

# -- elementwise binary -------------------------------------------------------

_BINARY = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
    "max": lambda a, b: a.maximum(b),
    "min": lambda a, b: a.minimum(b),
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
    if op == "reduce_max":
        return t.max(axis=axis, keepdim=keep)
    if op == "reduce_min":
        return t.min(axis=axis, keepdim=keep)
    if op in ("all", "any"):
        mask = (t != 0).cast(dtypes.int32)
        reduced = mask.min(axis=axis, keepdim=keep) if op == "all" else mask.max(axis=axis, keepdim=keep)
        return reduced
    raise UnsupportedOperation(f"unsupported reduction: {op}")


def _broadcast(node, env):
    t = _in(node, env)
    out_shape = node["attrs"]["shape"]
    axes = node["attrs"]["axes"]
    # Reshape source dims into their output positions (1 elsewhere), then expand.
    inter = [1] * len(out_shape)
    for src_dim, out_dim in enumerate(axes):
        inter[out_dim] = t.shape[src_dim]
    return t.reshape(tuple(inter)).expand(tuple(out_shape))


def _slice(node, env):
    t = _in(node, env)
    starts = node["attrs"]["starts"]
    lengths = node["attrs"]["lengths"]
    strides = node["attrs"]["strides"]
    idx = tuple(slice(s, s + l, st) for s, l, st in zip(starts, lengths, strides))
    return t[idx]


def _concatenate(node, env):
    tensors = [env[i] for i in node["inputs"]]
    axis = node["attrs"]["axis"]
    return tensors[0].cat(*tensors[1:], dim=axis) if len(tensors) > 1 else tensors[0]


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

    if op in _UNARY:
        result = _UNARY[op](_in(node, env))
    elif op in _BINARY:
        result = _BINARY[op](_in(node, env, 0), _in(node, env, 1))
    elif op in _COMPARISON:
        result = _COMPARISON[op](_in(node, env, 0), _in(node, env, 1))
    elif op == "select":
        cond = _in(node, env, 0).cast(dtypes.bool)
        result = cond.where(_in(node, env, 1), _in(node, env, 2))
    elif op in ("sum", "reduce_max", "reduce_min", "all", "any"):
        result = _reduce(node, env)
    elif op == "reshape":
        result = _in(node, env).reshape(tuple(node["attrs"]["shape"]))
    elif op == "squeeze":
        result = _in(node, env).reshape(tuple(node["shape"]))
    elif op == "broadcast":
        result = _broadcast(node, env)
    elif op == "transpose":
        result = _in(node, env).permute(tuple(node["attrs"]["axes"]))
    elif op == "concatenate":
        result = _concatenate(node, env)
    elif op == "slice":
        result = _slice(node, env)
    elif op == "as_type":
        result = _in(node, env)
    elif op == "dot":
        result = _dot(node, env)
    else:
        raise UnsupportedOperation(f"unsupported Nx operation: {op}", details={"op": op})

    return _cast(result, node)


SUPPORTED_OPS = (
    set(_UNARY)
    | set(_BINARY)
    | set(_COMPARISON)
    | {"select", "sum", "reduce_max", "reduce_min", "all", "any"}
    | {"reshape", "squeeze", "broadcast", "transpose", "concatenate", "slice", "as_type", "dot"}
)
