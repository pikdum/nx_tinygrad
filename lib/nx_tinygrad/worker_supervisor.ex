defmodule NxTinygrad.WorkerSupervisor do
  @moduledoc """
  Supervises the lazy default worker and any additional device/named workers.
  The tree is `one_for_one` so each worker can restart independently.
  """
  use Supervisor

  alias NxTinygrad.Config

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children =
      if Config.start_default_worker?() do
        [worker_spec(:default, Config.device())]
      else
        []
      end

    Supervisor.init(children, strategy: :one_for_one)
  end

  @doc "Start an additional named worker at runtime."
  def start_worker(name, device) do
    Supervisor.start_child(__MODULE__, worker_spec(name, device))
  end

  @doc """
  Resolve a device string to a running worker name, starting a worker for it if
  necessary. The `:default` worker serves its configured device; other devices
  get their own auto-started worker.
  """
  def worker_for_device(device) do
    name = worker_name(device)
    ensure_started(name, device)
    name
  end

  @doc false
  def ensure_default_started do
    case NxTinygrad.Worker.whereis(:default) do
      pid when is_pid(pid) -> :ok
      nil -> ensure_started(:default, Config.device())
    end
  end

  @doc false
  def worker_name(device), do: if(device == Config.device(), do: :default, else: {:device, device})

  defp ensure_started(name, device) do
    case start_worker(name, device) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
      other -> raise "could not start worker #{inspect(name)} for #{device}: #{inspect(other)}"
    end
  end

  defp worker_spec(name, device) do
    Supervisor.child_spec({NxTinygrad.Worker, [name: name, device: device]}, id: {NxTinygrad.Worker, name})
  end
end
