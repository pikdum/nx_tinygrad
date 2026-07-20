defmodule NxTinygrad.Generation do
  @moduledoc """
  Monotonically increasing worker generation numbers.

  Every worker startup (including restarts) gets a fresh, strictly increasing
  generation. All executable and tensor references carry the generation of the
  worker that produced them, so a reference from an old generation is never sent
  to a restarted worker.
  """

  @key {__MODULE__, :counter}

  @doc "Initialize the global counter. Idempotent; call once at application start."
  def init do
    unless :persistent_term.get(@key, nil) do
      :persistent_term.put(@key, :atomics.new(1, signed: false))
    end

    :ok
  end

  @doc "Return the next generation number."
  @spec next() :: pos_integer()
  def next do
    :atomics.add_get(counter(), 1, 1)
  end

  defp counter do
    case :persistent_term.get(@key, nil) do
      nil ->
        init()
        :persistent_term.get(@key)

      ref ->
        ref
    end
  end
end
