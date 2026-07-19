defmodule ExTinygrad.GPU.LeakTest do
  @moduledoc "Ten-thousand-iteration GPU memory lifecycle test."
  use ExUnit.Case, async: false
  @moduletag :gpu
  @moduletag timeout: 600_000

  alias ExTinygrad.{Backend, ReleaseReaper}

  setup_all do
    ExTinygrad.GPUHelpers.ensure_amd_worker()
    :ok
  end

  defp settled_buffer_count do
    ReleaseReaper.drain_now()
    {:ok, %{}, []} = ExTinygrad.Worker.request(:amd, "synchronize", %{})
    {:ok, stats, []} = ExTinygrad.Worker.request(:amd, "stats", %{})
    stats["buffer_count"]
  end

  test "10k created-and-dropped device tensors do not grow GPU buffers" do
    baseline = settled_buffer_count()

    Enum.each(1..10_000, fn i ->
      _t = Nx.tensor([i * 1.0, i * 2.0, i * 3.0, i * 4.0]) |> Nx.backend_transfer({Backend, worker: :amd})
      :ok
    end)

    final =
      Enum.reduce_while(1..100, nil, fn _, _ ->
        :erlang.garbage_collect()
        count = settled_buffer_count()

        if count - baseline <= 50,
          do: {:halt, count},
          else:
            (
              Process.sleep(20)
              {:cont, count}
            )
      end)

    assert final - baseline <= 50, "GPU buffers leaked: baseline=#{baseline}, final=#{final}"
  end
end
