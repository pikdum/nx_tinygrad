defmodule NxTinygrad.ReleaseReaper do
  @moduledoc """
  Periodically drains the native release queue (populated when `TensorRef`
  resources are garbage-collected) and sends batched `release` requests to the
  owning workers.

  Releases whose generation no longer matches the worker's current generation
  are discarded: that worker restarted, so the buffer is already gone.
  """
  use GenServer

  alias NxTinygrad.{TensorRef, Worker, WorkerIds}

  @interval 50

  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @doc "Force an immediate drain; returns the number of releases processed."
  def drain_now, do: GenServer.call(__MODULE__, :drain)

  @impl true
  def init(_opts) do
    schedule()
    {:ok, %{}}
  end

  @impl true
  def handle_call(:drain, _from, state), do: {:reply, drain(), state}

  @impl true
  def handle_info(:tick, state) do
    drain()
    schedule()
    {:noreply, state}
  end

  defp schedule, do: Process.send_after(self(), :tick, @interval)

  defp drain do
    tensor_releases = TensorRef.drain_releases()
    executable_releases = TensorRef.drain_executable_releases()

    tensor_releases
    |> Enum.group_by(fn {worker_id, _gen, _handle} -> worker_id end)
    |> Enum.each(&release_group(&1, "release"))

    executable_releases
    |> Enum.group_by(fn {worker_id, _gen, _handle} -> worker_id end)
    |> Enum.each(&release_group(&1, "release_executable"))

    length(tensor_releases) + length(executable_releases)
  end

  defp release_group({worker_id, entries}, command) do
    with name when not is_nil(name) <- WorkerIds.name_for(worker_id),
         pid when is_pid(pid) <- Worker.whereis(name) do
      current = Worker.generation(name)
      handles = for {_wid, gen, handle} <- entries, gen == current, do: handle

      if handles != [] do
        # Best-effort. Once a command reaches a live worker it remains queued even
        # if the reply times out; a worker restart frees all of its old resources.
        _ = Worker.request(name, command, %{"ids" => handles}, [], timeout: 5_000)
      end
    else
      _ -> :ok
    end
  end
end
