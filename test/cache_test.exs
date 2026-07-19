defmodule ExTinygrad.CacheTest do
  @moduledoc "Executable caching and TinyJit replay."
  use ExUnit.Case, async: false

  import ExTinygrad.TestGraphs
  alias ExTinygrad.{ExecutableCache, TestGraphs}

  setup do
    ExecutableCache.clear()
    :ok
  end

  defp compile_count, do: ExTinygrad.worker_stats()["compile_count"]

  test "identical graphs compile once, then hit the cache" do
    x1 = Nx.iota({5, 7}, type: :f32)
    x2 = Nx.iota({5, 7}, type: :f32) |> Nx.add(1.0)

    before = compile_count()
    r1 = ExTinygrad.jit(&TestGraphs.reduction/1).(x1)
    r2 = ExTinygrad.jit(&TestGraphs.reduction/1).(x2)

    assert compile_count() - before == 1
    assert_close(r1, TestGraphs.reduction(x1))
    assert_close(r2, TestGraphs.reduction(x2))
  end

  test "cache: false recompiles every time" do
    x = Nx.iota({6, 3}, type: :f32)
    before = compile_count()
    ExTinygrad.jit(&TestGraphs.reduction/1, cache: false).(x)
    ExTinygrad.jit(&TestGraphs.reduction/1, cache: false).(x)
    assert compile_count() - before == 2
  end

  test "a compiled function replays on new same-shaped inputs" do
    f = ExTinygrad.jit(&TestGraphs.elementwise/2)
    x1 = Nx.tensor([[1.0, 2.0]])
    y1 = Nx.tensor([[3.0, 4.0]])
    x2 = Nx.tensor([[5.0, 6.0]])
    y2 = Nx.tensor([[7.0, 8.0]])

    assert_close(f.(x1, y1), TestGraphs.elementwise(x1, y1))
    assert_close(f.(x2, y2), TestGraphs.elementwise(x2, y2))
  end

  test "exactly one execute RPC per compiled-function invocation" do
    x = Nx.iota({2, 3}, type: :f32)
    w = Nx.iota({3, 2}, type: :f32)
    b = Nx.tensor([1.0, 2.0])
    f = ExTinygrad.jit(&TestGraphs.matmul/3)

    before = ExTinygrad.worker_stats()["execute_count"]
    f.(x, w, b)
    assert ExTinygrad.worker_stats()["execute_count"] - before == 1
  end

  test "replayed results are independent (no cross-call contamination)" do
    f = ExTinygrad.jit(&TestGraphs.reduction/1)
    a = f.(Nx.iota({2, 3}, type: :f32))
    _b = f.(Nx.broadcast(9.0, {2, 3}))
    # `a` must not have changed because of the later call.
    assert_close(a, TestGraphs.reduction(Nx.iota({2, 3}, type: :f32)))
  end
end
