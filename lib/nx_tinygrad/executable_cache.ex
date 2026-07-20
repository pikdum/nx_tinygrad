defmodule NxTinygrad.ExecutableCache do
  @moduledoc """
  In-memory cache mapping a graph cache key to a compiled executable in a worker.

  A cached entry records the worker generation it was compiled for; the compiler
  treats an entry from a different generation as a miss (the worker restarted and
  its executables are gone). The cache is a public ETS table for lock-free reads.
  """
  use GenServer

  @table __MODULE__

  def start_link(_opts), do: GenServer.start_link(__MODULE__, [], name: __MODULE__)

  @impl true
  def init(_) do
    :ets.new(@table, [:named_table, :public, :set, read_concurrency: true])
    {:ok, %{}}
  end

  @doc "Fetch a cached entry (a map) by key, or `nil`."
  def get(key) do
    case :ets.lookup(@table, key) do
      [{^key, value}] -> value
      [] -> nil
    end
  end

  @doc "Store a cache entry."
  def put(key, value) do
    :ets.insert(@table, {key, value})
    :ok
  end

  def delete(key) do
    :ets.delete(@table, key)
    :ok
  end

  @doc "Drop all cached entries (used in tests)."
  def clear do
    :ets.delete_all_objects(@table)
    :ok
  end
end
