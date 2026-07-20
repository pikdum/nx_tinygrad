defmodule NxTinygrad.GPU.AmdOpsTest do
  @moduledoc "Coverage-march ops verified on the real AMD device (KFD+AMD:LLVM)."
  use ExUnit.Case, async: false
  @moduletag :gpu

  import Nx.Defn
  import NxTinygrad.TestGraphs, only: [assert_close: 3]

  setup_all do
    NxTinygrad.GPUHelpers.ensure_amd_worker()
    :ok
  end

  defp amd(fun), do: NxTinygrad.jit(fun, worker: :amd)

  test "extended elementwise + reductions parity on GPU" do
    x = Nx.tensor([[-2.0, -0.5, 0.5, 2.0], [1.0, 3.0, 0.25, 4.0]], type: :f32)

    fun = fn t ->
      {Nx.sinh(t), Nx.atan(t), Nx.erf(t), Nx.cbrt(t), Nx.round(t), Nx.product(t, axes: [1]),
       Nx.cumulative_sum(t, axis: 1), Nx.argmax(t, axis: 1)}
    end

    assert_close(amd(fun).(x), fun.(x), atol: 1.0e-4, rtol: 1.0e-4)
  end

  test "indexing, scatter, sort, and pad parity on GPU" do
    t = Nx.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]], type: :f32)
    ids = Nx.tensor([[0, 2], [1, 1]], type: :s64)
    coords = Nx.tensor([[0, 0], [2, 1]], type: :s64)
    upd = Nx.tensor([100.0, 200.0], type: :f32)

    fun = fn t, ids, coords, upd ->
      {
        Nx.take(t, ids, axis: 0),
        Nx.gather(t, coords),
        Nx.indexed_add(t, coords, upd),
        Nx.sort(t, axis: 1, direction: :desc),
        Nx.pad(t, 0.0, [{1, 0, 0}, {0, 1, 0}])
      }
    end

    args = [t, ids, coords, upd]
    assert_close(apply(amd(fun), args), apply(fun, args), atol: 1.0e-4, rtol: 1.0e-4)
  end

  test "conv + window pooling parity on GPU" do
    {input, _} = Nx.Random.normal(Nx.Random.key(1), shape: {2, 3, 8, 8}, type: :f32)
    {kernel, _} = Nx.Random.normal(Nx.Random.key(2), shape: {4, 3, 3, 3}, type: :f32)

    fun = fn i, k ->
      c = Nx.conv(i, k, strides: [1, 1], padding: [{1, 1}, {1, 1}])
      {c, Nx.window_max(c, {1, 1, 2, 2}, strides: [1, 1, 2, 2])}
    end

    assert_close(apply(amd(fun), [input, kernel]), apply(fun, [input, kernel]), atol: 1.0e-3, rtol: 1.0e-3)
  end

  defn cond_and_topk(t) do
    top =
      t
      |> Nx.top_k(k: 3)
      |> elem(0)

    branched =
      cond do
        Nx.all(Nx.greater(t, 0)) -> Nx.negate(top)
        true -> Nx.multiply(top, 2)
      end

    branched
  end

  test "cond + top_k control-flow parity on GPU" do
    x = Nx.tensor([3.0, 1.0, 4.0, 1.5, 5.0, 9.0], type: :f32)
    assert_close(amd(&cond_and_topk/1).(x), cond_and_topk(x), atol: 1.0e-4, rtol: 1.0e-4)
  end
end
