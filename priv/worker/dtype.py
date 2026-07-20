"""Dtype mapping between nx_tinygrad's stable names, numpy, and tinygrad.

nx_tinygrad owns a small set of stable dtype names that both Elixir and Python
agree on. This module is the single source of truth on the Python side.

v0.1 required: f32, s32, u8. We also map a handful of others that tinygrad and
numpy support natively; unsupported names raise UnsupportedDtype.
"""
from __future__ import annotations

import numpy as np

from errors import UnsupportedDtype

# stable name -> numpy dtype. bf16 has no numpy type, so it uses uint16 as a
# 2-byte transport carrier (the raw bit pattern moves unchanged); the worker
# bitcasts to/from tinygrad's bfloat16 at the boundary.
_NUMPY = {
    "f32": np.float32,
    "f16": np.float16,
    "f64": np.float64,
    "bf16": np.uint16,
    "c64": np.complex64,
    "c128": np.complex128,
    "s8": np.int8,
    "s16": np.int16,
    "s32": np.int32,
    "s64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
}

# stable names that are the v0.1 "required" set.
REQUIRED = ("f32", "s32", "u8")


def numpy_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(_NUMPY[name])
    except KeyError:
        raise UnsupportedDtype(f"unsupported dtype: {name!r}") from None


def tinygrad_dtype(name: str):
    """Return the tinygrad DType for a stable name (imported lazily)."""
    from tinygrad import dtypes

    table = {
        "f32": dtypes.float32,
        "f16": dtypes.float16,
        "f64": dtypes.float64,
        "bf16": dtypes.bfloat16,
        "s8": dtypes.int8,
        "s16": dtypes.int16,
        "s32": dtypes.int32,
        "s64": dtypes.int64,
        "u8": dtypes.uint8,
        "u16": dtypes.uint16,
        "u32": dtypes.uint32,
        "u64": dtypes.uint64,
    }
    try:
        return table[name]
    except KeyError:
        raise UnsupportedDtype(f"unsupported dtype: {name!r}") from None


_COMPLEX_COMPONENT = {"c64": "f32", "c128": "f64"}


def is_complex(name: str) -> bool:
    return name in _COMPLEX_COMPONENT


def component_dtype(name: str) -> str:
    """The real float dtype backing a complex wire name (c64 -> f32)."""
    return _COMPLEX_COMPONENT[name]


def wire_tensor(arr, name: str, device):
    """Build a tinygrad tensor of wire dtype `name` from a numpy array whose
    dtype is the transport carrier. bf16 rides uint16 (then bitcast); complex
    types are stored as a real float tensor with a trailing [real, imag] axis."""
    import numpy as np
    from tinygrad import Tensor

    if is_complex(name):
        comp = numpy_dtype(component_dtype(name))
        stacked = np.stack([arr.real.astype(comp), arr.imag.astype(comp)], axis=-1)
        return Tensor(np.ascontiguousarray(stacked), device=device)

    t = Tensor(arr, device=device)
    if name == "bf16":
        t = t.bitcast(tinygrad_dtype("bf16"))
    return t


def wire_numpy(tensor, name: str):
    """Convert a tinygrad tensor to a numpy array in the transport carrier.
    bf16 is bitcast to uint16; a complex tensor's trailing [real, imag] axis is
    folded back into a numpy complex array."""
    if name == "bf16":
        from tinygrad import dtypes

        tensor = tensor.bitcast(dtypes.uint16)
        return tensor.numpy()

    if is_complex(name):
        import numpy as np

        arr = tensor.numpy()
        out = arr[..., 0].astype(numpy_dtype(name)) + 1j * arr[..., 1]
        return np.ascontiguousarray(out.astype(numpy_dtype(name)))

    return tensor.numpy()


def itemsize(name: str) -> int:
    return numpy_dtype(name).itemsize


def is_supported(name: str) -> bool:
    return name in _NUMPY
