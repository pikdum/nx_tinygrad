defmodule NxTinygrad.LeakTest do
  @moduledoc "GC-triggered release keeps worker buffers bounded."
  use ExUnit.Case, async: false

  alias NxTinygrad.{Backend, ReleaseReaper}

  defp settled_buffer_count do
    ReleaseReaper.drain_now()
    NxTinygrad.synchronize()
    NxTinygrad.worker_stats()["buffer_count"]
  end

  test "dropped device tensors are released; buffer count returns near baseline" do
    baseline = settled_buffer_count()
    iterations = 1000

    Enum.each(1..iterations, fn i ->
      _t = Nx.tensor([i * 1.0, i * 2.0, i * 3.0]) |> Nx.backend_transfer({Backend, worker: :default})
      :ok
    end)

    final =
      Enum.reduce_while(1..50, nil, fn _, _ ->
        :erlang.garbage_collect()
        count = settled_buffer_count()

        if count - baseline <= 20,
          do: {:halt, count},
          else:
            (
              Process.sleep(20)
              {:cont, count}
            )
      end)

    assert final - baseline <= 20,
           "buffers leaked after #{iterations} iterations: baseline=#{baseline}, final=#{final}"
  end

  test "explicit release is immediate and GC does not double-release" do
    t = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.backend_transfer({Backend, worker: :default})
    assert NxTinygrad.release(t) == :ok
    # take/1 already claimed the ref, so a later dealloc is a no-op.
    assert Nx.backend_deallocate(t) == :already_deallocated
  end
end
