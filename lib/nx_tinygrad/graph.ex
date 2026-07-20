defmodule NxTinygrad.Graph do
  @moduledoc """
  The versioned, deterministic tensor graph IR sent to the worker.

  Fields hold wire-ready maps with string keys:

    * `inputs`    — `%{"id", "index", "shape", "dtype"}`
    * `constants` — `%{"id", "shape", "dtype"}` plus either `"value"` (scalar) or
                    `"data_index"` (index into the compile request's blob list)
    * `nodes`     — `%{"id", "op", "inputs", "attrs", "shape", "dtype"}`, topological
    * `outputs`   — `%{"node", "shape", "dtype"}`
    * `blobs`     — inline constant tensor bytes, in `data_index` order (not serialized
                    into the canonical JSON; shipped alongside as protocol blobs)

  Ids form a single sequential space across inputs, constants, and nodes.
  """

  @semantics_version 1

  defstruct version: @semantics_version, inputs: [], constants: [], nodes: [], outputs: [], blobs: []

  @type t :: %__MODULE__{}

  def semantics_version, do: @semantics_version

  @doc "The wire map (string keys), excluding blobs (shipped separately)."
  def to_wire(%__MODULE__{} = g) do
    %{
      "version" => g.version,
      "inputs" => g.inputs,
      "constants" => g.constants,
      "nodes" => g.nodes,
      "outputs" => g.outputs
    }
  end

  @doc "Inline constant blobs, ordered by `data_index`."
  def blobs(%__MODULE__{blobs: blobs}), do: blobs

  @doc """
  Deterministic, canonical JSON encoding (object keys sorted recursively). Used
  for the cache key so structurally identical graphs hash identically regardless
  of map ordering.
  """
  @spec canonical_json(term()) :: iodata()
  def canonical_json(%__MODULE__{} = g), do: canonical_json(to_wire(g))

  def canonical_json(map) when is_map(map) and not is_struct(map) do
    inner =
      map
      |> Enum.sort_by(fn {k, _} -> to_string(k) end)
      |> Enum.map(fn {k, v} -> [JSON.encode!(to_string(k)), ":", canonical_json(v)] end)
      |> Enum.intersperse(",")

    ["{", inner, "}"]
  end

  def canonical_json(list) when is_list(list) do
    ["[", list |> Enum.map(&canonical_json/1) |> Enum.intersperse(","), "]"]
  end

  def canonical_json(value), do: JSON.encode!(value)
end
