defmodule NxTinygrad.CacheTest do
  @moduledoc "Executable caching and TinyJit replay."
  use ExUnit.Case, async: false

  import NxTinygrad.TestGraphs
  alias NxTinygrad.{ExecutableCache, TestGraphs}

  setup do
    ExecutableCache.clear()
    :ok
  end

  defp compile_count, do: NxTinygrad.worker_stats()["compile_count"]

  test "identical graphs compile once, then hit the cache" do
    x1 = Nx.iota({5, 7}, type: :f32)
    x2 = Nx.iota({5, 7}, type: :f32) |> Nx.add(1.0)

    before = compile_count()
    r1 = NxTinygrad.jit(&TestGraphs.reduction/1).(x1)
    r2 = NxTinygrad.jit(&TestGraphs.reduction/1).(x2)

    assert compile_count() - before == 1
    assert_close(r1, TestGraphs.reduction(x1))
    assert_close(r2, TestGraphs.reduction(x2))
  end

  test "cache: false recompiles every time" do
    x = Nx.iota({6, 3}, type: :f32)
    before = compile_count()
    NxTinygrad.jit(&TestGraphs.reduction/1, cache: false).(x)
    NxTinygrad.jit(&TestGraphs.reduction/1, cache: false).(x)
    assert compile_count() - before == 2
  end

  test "a compiled function replays on new same-shaped inputs" do
    f = NxTinygrad.jit(&TestGraphs.elementwise/2)
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
    f = NxTinygrad.jit(&TestGraphs.matmul/3)

    before = NxTinygrad.worker_stats()["execute_count"]
    f.(x, w, b)
    assert NxTinygrad.worker_stats()["execute_count"] - before == 1
  end

  test "replayed results are independent (no cross-call contamination)" do
    f = NxTinygrad.jit(&TestGraphs.reduction/1)
    a = f.(Nx.iota({2, 3}, type: :f32))
    _b = f.(Nx.broadcast(9.0, {2, 3}))
    # `a` must not have changed because of the later call.
    assert_close(a, TestGraphs.reduction(Nx.iota({2, 3}, type: :f32)))
  end

  test "same-shaped inline tensor constants do not collide" do
    x = Nx.tensor([0.0, 0.0])
    add_small = NxTinygrad.jit(fn t -> Nx.add(t, Nx.tensor([1.0, 2.0])) end)
    add_large = NxTinygrad.jit(fn t -> Nx.add(t, Nx.tensor([10.0, 20.0])) end)

    assert_close(add_small.(x), Nx.tensor([1.0, 2.0]))
    assert_close(add_large.(x), Nx.tensor([10.0, 20.0]))
  end

  test "lookup cache is bounded" do
    limit = NxTinygrad.Config.executable_cache_size()

    for key <- 1..(limit + 50) do
      ExecutableCache.put({:test, key}, %{key: key})
    end

    assert ExecutableCache.size() == limit
  end

  test "uncached executables are released after their compiled closure is collected" do
    baseline = NxTinygrad.worker_stats()["executable_count"]
    run_uncached_graph()

    final =
      Enum.reduce_while(1..50, nil, fn _, _ ->
        :erlang.garbage_collect()
        NxTinygrad.ReleaseReaper.drain_now()
        count = NxTinygrad.worker_stats()["executable_count"]

        if count <= baseline,
          do: {:halt, count},
          else:
            (
              Process.sleep(20)
              {:cont, count}
            )
      end)

    assert final <= baseline
  end

  defp run_uncached_graph do
    compiled = NxTinygrad.jit(fn t -> Nx.negate(t) end, cache: false, output: :host)
    _result = compiled.(Nx.tensor([1.0, 2.0]))
    :ok
  end
end
