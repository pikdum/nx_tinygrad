defmodule NxTinygrad.GPUHelpers do
  @moduledoc "Helpers for GPU-gated tests: bring up a worker on KFD+AMD:LLVM."

  @device "KFD+AMD:LLVM"

  def device, do: @device

  @doc "Ensure a `:amd` worker is running on KFD+AMD:LLVM and responsive."
  def ensure_amd_worker do
    case NxTinygrad.WorkerSupervisor.start_worker(:amd, @device) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
      other -> raise "could not start AMD worker: #{inspect(other)}"
    end

    # Force device initialization + validate the smoke computation once.
    {:ok, info, []} = NxTinygrad.Worker.request(:amd, "device_info", %{})
    unless info["usable"], do: raise("AMD device not usable: #{inspect(info)}")
    info
  end
end
