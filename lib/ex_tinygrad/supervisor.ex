defmodule ExTinygrad.Supervisor do
  @moduledoc """
  Top-level supervisor.

  Children are added as milestones land:

      ExTinygrad.Supervisor
      ├── ExTinygrad.ExecutableCache      (M3)
      ├── ExTinygrad.WorkerSupervisor     (M1)
      │   └── ExTinygrad.Worker (:default)
      └── ExTinygrad.ReleaseReaper        (M5)
  """
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children = [
      {Registry, keys: :unique, name: ExTinygrad.WorkerRegistry},
      ExTinygrad.WorkerIds,
      ExTinygrad.ExecutableCache,
      ExTinygrad.WorkerSupervisor,
      ExTinygrad.ReleaseReaper
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
