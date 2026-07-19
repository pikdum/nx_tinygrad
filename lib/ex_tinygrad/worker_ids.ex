defmodule ExTinygrad.WorkerIds do
  @moduledoc """
  Stable bidirectional mapping between worker names (atoms) and small integer
  ids. The integer id is what the Rust `TensorRef` resource stores, since it
  cannot hold an Elixir atom; the reaper maps it back to a worker name.
  """
  use GenServer

  @table __MODULE__

  def start_link(_opts), do: GenServer.start_link(__MODULE__, [], name: __MODULE__)

  @impl true
  def init(_) do
    :ets.new(@table, [:named_table, :public, :set, read_concurrency: true])
    {:ok, %{next: 0}}
  end

  @doc "Return (assigning if needed) the integer id for a worker name."
  def id_for(name) do
    case :ets.lookup(@table, {:name, name}) do
      [{_, id}] -> id
      [] -> GenServer.call(__MODULE__, {:assign, name})
    end
  end

  @doc "Return the worker name for an integer id, or `nil`."
  def name_for(id) do
    case :ets.lookup(@table, {:id, id}) do
      [{_, name}] -> name
      [] -> nil
    end
  end

  @impl true
  def handle_call({:assign, name}, _from, state) do
    case :ets.lookup(@table, {:name, name}) do
      [{_, id}] ->
        {:reply, id, state}

      [] ->
        id = state.next
        :ets.insert(@table, {{:name, name}, id})
        :ets.insert(@table, {{:id, id}, name})
        {:reply, id, %{state | next: id + 1}}
    end
  end
end
