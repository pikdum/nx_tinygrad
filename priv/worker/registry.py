"""Worker-side registry of resident device buffers.

Each entry holds a realized tinygrad Tensor plus enough metadata to report
statistics and validate `execute` inputs. Buffer ids are monotonic and never
reused within a worker generation.
"""
from __future__ import annotations

from dataclasses import dataclass

from errors import StaleReference


@dataclass
class Buffer:
    id: int
    tensor: object  # tinygrad Tensor
    shape: tuple
    dtype: str
    nbytes: int


class TensorRegistry:
    def __init__(self):
        self._buffers: dict[int, Buffer] = {}
        self._next_id = 1

    def allocate_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def put(self, tensor, shape, dtype: str, nbytes: int, buffer_id: int | None = None) -> int:
        buffer_id = self.allocate_id() if buffer_id is None else buffer_id
        self._buffers[buffer_id] = Buffer(buffer_id, tensor, tuple(shape), dtype, nbytes)
        return buffer_id

    def get(self, buffer_id: int) -> Buffer:
        try:
            return self._buffers[buffer_id]
        except KeyError:
            raise StaleReference(
                f"buffer {buffer_id} is not resident in this worker generation",
                details={"buffer_id": buffer_id},
            ) from None

    def release(self, ids) -> int:
        """Remove buffers. Idempotent: unknown ids are ignored."""
        removed = 0
        for buffer_id in ids:
            if self._buffers.pop(buffer_id, None) is not None:
                removed += 1
        return removed

    def count(self) -> int:
        return len(self._buffers)

    def total_bytes(self) -> int:
        return sum(buf.nbytes for buf in self._buffers.values())

    def clear(self) -> None:
        self._buffers.clear()
