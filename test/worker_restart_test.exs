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
