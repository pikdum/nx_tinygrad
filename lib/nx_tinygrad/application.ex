defmodule NxTinygrad.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    NxTinygrad.Generation.init()
    NxTinygrad.Supervisor.start_link([])
  end
end
