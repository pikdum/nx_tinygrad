import pytest

from registry import TensorRegistry
from errors import StaleReference


def test_put_get_roundtrip():
    reg = TensorRegistry()
    sentinel = object()
    bid = reg.put(sentinel, (2, 3), "f32", 24)
    buf = reg.get(bid)
    assert buf.tensor is sentinel
    assert buf.shape == (2, 3)
    assert buf.dtype == "f32"
    assert buf.nbytes == 24


def test_ids_are_monotonic_and_unique():
    reg = TensorRegistry()
    ids = [reg.put(object(), (1,), "u8", 1) for _ in range(5)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 5


def test_get_missing_raises_stale():
    reg = TensorRegistry()
    with pytest.raises(StaleReference):
        reg.get(123)


def test_release_is_idempotent():
    reg = TensorRegistry()
    bid = reg.put(object(), (1,), "u8", 1)
    assert reg.release([bid]) == 1
    assert reg.release([bid]) == 0
    assert reg.count() == 0


def test_stats_track_bytes_and_count():
    reg = TensorRegistry()
    reg.put(object(), (4,), "f32", 16)
    reg.put(object(), (2,), "f32", 8)
    assert reg.count() == 2
    assert reg.total_bytes() == 24
