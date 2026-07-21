defmodule NxTinygrad.EagerBackendTest do
  @moduledoc """
  Differential coverage of the eager `NxTinygrad.Backend` op surface against
  `Nx.BinaryBackend` — the same operations the compiled path supports, but
  dispatched one node per call via the worker's `run_node` command (the model
  weight-load path).
  """
  use ExUnit.Case, async: false

  alias NxTinygrad.Backend

  defp to_device(tensor), do: Nx.backend_transfer(tensor, {Backend, worker: :default})

  # Run `fun` on device (eagerly) and on BinaryBackend; results must agree.
  defp assert_same(fun, tensors, opts \\ []) do
    expected = apply(fun, tensors)
    actual = apply(fun, Enum.map(tensors, &to_device/1)) |> Nx.backend_transfer(Nx.BinaryBackend)

    assert Nx.type(actual) == Nx.type(expected)
    assert Nx.shape(actual) == Nx.shape(expected)

    close = Nx.all_close(actual, expected, atol: Keyword.get(opts, :atol, 1.0e-6))
    assert Nx.to_number(close) == 1, "eager result diverged:\n#{inspect(actual)}\n!=\n#{inspect(expected)}"
  end

  test "movement ops: transpose, reshape, squeeze, broadcast, reverse, pad, slice, concatenate, stack" do
    t = Nx.iota({2, 3, 4}, type: :f32)

    assert_same(&Nx.transpose(&1, axes: [2, 0, 1]), [t])
    assert_same(&Nx.reshape(&1, {4, 6}), [t])
    assert_same(&Nx.squeeze(Nx.reshape(&1, {2, 1, 3, 4, 1})), [t])
    assert_same(&Nx.broadcast(&1, {2, 2, 3, 4}, axes: [1, 2, 3]), [Nx.iota({2, 3, 4}, type: :f32)])
    assert_same(&Nx.reverse(&1, axes: [1]), [t])
    assert_same(&Nx.pad(&1, 0.5, [{1, 0, 0}, {0, 2, 0}, {0, 0, 0}]), [t])
    assert_same(&Nx.pad(&1, 0.0, [{-1, 0, 0}, {0, -2, 0}, {1, 1, 0}]), [t])
    assert_same(&Nx.slice(&1, [0, 1, 2], [2, 2, 2]), [t])
    assert_same(&Nx.slice(&1, [0, 0, 0], [2, 3, 4], strides: [1, 2, 3]), [t])
    assert_same(&Nx.concatenate([&1, &2], axis: 1), [t, t])
    assert_same(&Nx.stack([&1, &2], axis: 0), [t, t])
  end

  test "dynamic slice / put_slice starts given as tensors" do
    t = Nx.iota({4, 4}, type: :f32)
    i = Nx.tensor(2)

    assert_same(&Nx.slice(&1, [Nx.tensor(1), Nx.tensor(9)], [2, 2]), [t])
    assert_same(&Nx.put_slice(&1, [i, Nx.tensor(0)], Nx.broadcast(9.0, {1, 4})), [t])
    assert_same(&Nx.put_slice(&1, [1, 1], Nx.broadcast(7.0, {2, 2})), [t])
  end

  test "type ops: as_type upcast/downcast and bitcast" do
    t16 = Nx.tensor([[1.0, -2.5], [0.25, 3.0]], type: :f16)
    assert_same(&Nx.as_type(&1, :f32), [t16])
    assert_same(&Nx.as_type(&1, :s32), [Nx.tensor([1.9, -1.9, 0.0])])
    assert_same(&Nx.bitcast(&1, :u32), [Nx.tensor([1, -1, 7], type: :s32)])
  end

  test "elementwise binary, comparison, and unary ops with scalar operands" do
    a = Nx.tensor([[1.0, -2.0], [3.5, 0.0]])
    b = Nx.tensor([[2.0, 2.0], [-1.0, 4.0]])

    assert_same(&Nx.add/2, [a, b])
    assert_same(&Nx.multiply(&1, 2.5), [a])
    assert_same(&Nx.divide(2.0, &1), [b])
    assert_same(&Nx.pow(&1, 2), [a])
    assert_same(&Nx.min/2, [a, b])
    assert_same(&Nx.greater/2, [a, b])
    assert_same(&Nx.equal(&1, 0.0), [a])
    assert_same(&Nx.logical_and/2, [Nx.tensor([1, 0, 1]), Nx.tensor([1, 1, 0])])
    assert_same(&Nx.exp/1, [a], atol: 1.0e-5)
    assert_same(&Nx.abs/1, [a])
    assert_same(&Nx.negate/1, [a])
    assert_same(&Nx.sqrt/1, [Nx.tensor([0.0, 1.0, 4.0, 9.0])])
    assert_same(&Nx.quotient/2, [Nx.tensor([7, -7, 9]), Nx.tensor([2, 2, 3])])
    assert_same(&Nx.remainder/2, [Nx.tensor([7.0, -7.0]), Nx.tensor([3.0, 3.0])])
  end

  test "creation ops: iota, eye, and implicit scalar constants" do
    expected = Nx.iota({3, 4}, axis: 1, type: :s32)
    actual = Nx.iota({3, 4}, axis: 1, type: :s32, backend: Backend) |> Nx.backend_transfer()
    assert Nx.to_flat_list(actual) == Nx.to_flat_list(expected)

    eye = Nx.eye(4, backend: Backend) |> Nx.backend_transfer()
    assert Nx.to_flat_list(eye) == Nx.to_flat_list(Nx.eye(4))
  end

  test "reductions: sum, product, max/min, all/any, argmax/argmin with options" do
    t = Nx.tensor([[1.0, 5.0, 2.0], [8.0, 5.0, 0.0]])

    assert_same(&Nx.sum(&1, axes: [1]), [t])
    assert_same(&Nx.sum(&1, axes: [0], keep_axes: true), [t])
    assert_same(&Nx.product(&1, axes: [-1]), [t])
    assert_same(&Nx.reduce_max(&1), [t])
    assert_same(&Nx.reduce_min(&1, axes: [0]), [t])
    assert_same(&Nx.all(&1, axes: [1]), [t])
    assert_same(&Nx.any(&1), [Nx.tensor([0, 0, 0])])
    assert_same(&Nx.argmax(&1, axis: 1), [t])
    assert_same(&Nx.argmax(&1, axis: 0, tie_break: :high), [t])
    assert_same(&Nx.argmin(&1, keep_axis: false), [t])
  end

  test "contractions and selection: dot, select, clip" do
    a = Nx.iota({3, 4}, type: :f32)
    b = Nx.iota({4, 2}, type: :f32)

    assert_same(&Nx.dot/2, [a, b])
    assert_same(&Nx.dot(&1, [0], &2, [0]), [a, Nx.iota({3, 5}, type: :f32)])
    assert_same(&Nx.select(Nx.greater(&1, 5.0), &1, Nx.negate(&1)), [a])
    assert_same(&Nx.clip(&1, 2.0, 8.0), [a])
  end

  test "indexing: gather, sort, argsort, indexed_add/indexed_put" do
    t = Nx.tensor([[10.0, 20.0], [30.0, 40.0]])
    idx = Nx.tensor([[1, 0], [0, 1]])

    assert_same(&Nx.gather(&1, idx), [t])
    assert_same(&Nx.sort(&1, axis: 1, direction: :desc), [t])
    assert_same(&Nx.argsort(&1, axis: 0), [t])
    assert_same(&Nx.indexed_add(&1, Nx.tensor([[0, 0], [1, 1]]), Nx.tensor([5.0, 5.0])), [t])
    assert_same(&Nx.indexed_put(&1, Nx.tensor([[0, 1]]), Nx.tensor([99.0])), [t])
  end

  test "take and take_along_axis compose from eager primitives" do
    t = Nx.tensor([[10.0, 20.0], [30.0, 40.0]])
    assert_same(&Nx.take(&1, Nx.tensor([1, 0, 1])), [t])
    assert_same(&Nx.take_along_axis(&1, Nx.tensor([[1, 0], [0, 0]]), axis: 1), [t])
  end

  test "windowed ops" do
    t = Nx.iota({1, 6}, type: :f32)
    assert_same(&Nx.window_sum(&1, {1, 3}, strides: [1, 1], padding: :valid), [t])
    assert_same(&Nx.window_max(&1, {1, 2}, strides: [1, 2], padding: :valid), [t])
  end

  test "conv runs eagerly" do
    img = Nx.iota({1, 1, 5, 5}, type: :f32)
    kernel = Nx.broadcast(1.0, {1, 1, 3, 3})
    assert_same(&Nx.conv(&1, &2, padding: :same), [img, kernel], atol: 1.0e-5)
  end

  test "chained eager pipeline stays on device (Bumblebee load shape)" do
    # f16 checkpoint param -> upcast -> transpose -> reshape, all device-side.
    param = Nx.iota({8, 4}, type: :f16) |> to_device()

    remapped =
      param
      |> Nx.as_type(:f32)
      |> Nx.transpose(axes: [1, 0])
      |> Nx.reshape({2, 2, 8})

    assert %Backend{} = remapped.data

    expected =
      Nx.iota({8, 4}, type: :f16)
      |> Nx.as_type(:f32)
      |> Nx.transpose(axes: [1, 0])
      |> Nx.reshape({2, 2, 8})

    assert Nx.to_flat_list(Nx.backend_transfer(remapped)) == Nx.to_flat_list(expected)
  end

  test "eager ops report worker run_node stats" do
    before = NxTinygrad.worker_stats()["run_node_count"]
    _ = Nx.tensor([1.0, 2.0]) |> to_device() |> Nx.multiply(3.0)
    assert NxTinygrad.worker_stats()["run_node_count"] > before
  end

  test "backend_copy within the same worker stays device-side" do
    t = Nx.tensor([1.0, 2.0, 3.0]) |> to_device()

    upload_before = NxTinygrad.worker_stats()["upload_bytes"]
    copy = Nx.backend_copy(t, {Backend, worker: :default})

    assert %Backend{} = copy.data
    assert Backend.handle(copy.data) != Backend.handle(t.data)
    # no bytes crossed the transport: the copy was minted device-side
    assert NxTinygrad.worker_stats()["upload_bytes"] == upload_before

    # independent lifetimes: releasing the original leaves the copy readable
    assert Nx.backend_deallocate(t) == :ok
    assert Nx.to_flat_list(Nx.backend_transfer(copy)) == [1.0, 2.0, 3.0]
  end
end
