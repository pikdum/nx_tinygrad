defmodule ExTinygrad.WorkerSupervisor do
  @moduledoc """
  Supervises worker processes. For v0.1 there is a single `:default` worker per
  GPU; the tree is kept `one_for_one` so a worker can restart independently.
  """
  use Supervisor

  alias ExTinygrad.Config

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

  defp worker_spec(name, device) do
    Supervisor.child_spec({ExTinygrad.Worker, [name: name, device: device]}, id: {ExTinygrad.Worker, name})
  end
end
