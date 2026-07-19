defmodule ExTinygrad.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    ExTinygrad.Generation.init()
    ExTinygrad.Supervisor.start_link([])
  end
end
