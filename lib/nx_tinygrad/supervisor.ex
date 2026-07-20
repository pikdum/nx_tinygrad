defmodule NxTinygrad.Supervisor do
  @moduledoc """
  Top-level supervisor.

  Children are added as milestones land:

      NxTinygrad.Supervisor
      ├── NxTinygrad.ExecutableCache      (M3)
      ├── NxTinygrad.WorkerSupervisor     (M1)
      │   └── NxTinygrad.Worker (:default)
      └── NxTinygrad.ReleaseReaper        (M5)
  """
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children = [
      {Registry, keys: :unique, name: NxTinygrad.WorkerRegistry},
      NxTinygrad.WorkerIds,
      NxTinygrad.ExecutableCache,
      NxTinygrad.WorkerSupervisor,
      NxTinygrad.ReleaseReaper
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
