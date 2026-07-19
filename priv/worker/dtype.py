"""Dtype mapping between ex_tinygrad's stable names, numpy, and tinygrad.

ex_tinygrad owns a small set of stable dtype names that both Elixir and Python
agree on. This module is the single source of truth on the Python side.

v0.1 required: f32, s32, u8. We also map a handful of others that tinygrad and
numpy support natively; unsupported names raise UnsupportedDtype.
"""
from __future__ import annotations

import numpy as np

from errors import UnsupportedDtype

# stable name -> numpy dtype
_NUMPY = {
    "f32": np.float32,
    "f16": np.float16,
    "f64": np.float64,
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


def itemsize(name: str) -> int:
    return numpy_dtype(name).itemsize


def is_supported(name: str) -> bool:
    return name in _NUMPY
