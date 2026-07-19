defmodule ExTinygrad.WorkerRestartTest do
  @moduledoc "The worker is restartable and a crash does not take down the BEAM."
  use ExUnit.Case, async: false

  alias ExTinygrad.Worker

  test "killing the worker restarts it with a higher generation; BEAM stays up" do
    pid1 = Worker.whereis(:default)
    assert is_pid(pid1)
    gen1 = Worker.generation(:default)

    ref = Process.monitor(pid1)
    Process.exit(pid1, :kill)
    assert_receive {:DOWN, ^ref, :process, ^pid1, _}, 5_000

    # Supervisor restarts it; wait for a fresh, responsive worker.
    pid2 = wait_for_new_worker(pid1)
    assert pid2 != pid1
    assert Process.alive?(pid2)

    gen2 = Worker.generation(:default)
    assert gen2 > gen1

    # And it works again.
    assert {:ok, %{}, []} = Worker.request(:default, "synchronize", %{})
  end

  test "a device tensor from an old generation is stale after restart; graph recompiles" do
    x = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.backend_transfer({ExTinygrad.Backend, worker: :default})
    gen1 = Worker.generation(:default)

    pid = Worker.whereis(:default)
    ref = Process.monitor(pid)
    Process.exit(pid, :kill)
    assert_receive {:DOWN, ^ref, :process, ^pid, _}, 5_000
    wait_for_new_worker(pid)

    assert Worker.generation(:default) > gen1
    assert Process.whereis(ExTinygrad.Supervisor) != nil, "BEAM/app supervisor must stay alive"

    # The old device tensor's data cannot be recovered.
    assert_raise ExTinygrad.StaleTensorError, fn -> Nx.to_binary(x) end

    # A fresh compiled call transparently recompiles on the new generation.
    y = ExTinygrad.jit(fn t -> Nx.multiply(t, 2.0) end).(Nx.tensor([1.0, 2.0]))
    assert Nx.to_flat_list(Nx.backend_transfer(y)) == [2.0, 4.0]
  end

  defp wait_for_new_worker(old_pid, tries \\ 100)
  defp wait_for_new_worker(_old_pid, 0), do: flunk("worker did not restart in time")

  defp wait_for_new_worker(old_pid, tries) do
    case Worker.whereis(:default) do
      pid when is_pid(pid) and pid != old_pid ->
        try do
          Worker.info(:default)
          pid
        catch
          :exit, _ ->
            Process.sleep(50)
            wait_for_new_worker(old_pid, tries - 1)
        end

      _ ->
        Process.sleep(50)
        wait_for_new_worker(old_pid, tries - 1)
    end
  end
end
