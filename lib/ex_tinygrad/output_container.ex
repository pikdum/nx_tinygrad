defmodule ExTinygrad.OutputContainer do
  @moduledoc """
  Flattening and reconstruction of arbitrary `Nx.Container` output structures.

  The worker only ever sees and returns a flat list of tensors. Elixir keeps the
  original container structure (as a template of expression leaves) and rebuilds
  it, swapping each leaf's backend data for the computed result.
  """
  alias Nx.Defn.Composite

  @doc "Flatten a container into its tensor leaves, in traversal order."
  def flatten(container) do
    container
    |> Composite.reduce([], fn tensor, acc -> [tensor | acc] end)
    |> Enum.reverse()
  end

  @doc """
  Rebuild `container`, replacing each leaf tensor with the corresponding computed
  tensor from `tensors` (matched positionally in traversal order). Leaf metadata
  (shape/type/names) is preserved from the template; only `data` is swapped.
  """
  def reconstruct(container, tensors) do
    {result, []} =
      Composite.traverse(container, tensors, fn template, [tensor | rest] ->
        {%{template | data: tensor.data}, rest}
      end)

    result
  end
end
