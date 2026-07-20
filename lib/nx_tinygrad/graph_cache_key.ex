defmodule NxTinygrad.GraphCacheKey do
  @moduledoc """
  Computes the cache key for a compiled graph.

  The key mixes everything that can change the compiled result: the graph
  semantics version, the canonical graph JSON (which already encodes input and
  output shapes/dtypes), the Nx version, the tinygrad commit/version, the
  protocol version, the device string, and any compile options that affect
  compilation.
  """
  alias NxTinygrad.Graph

  @protocol_version 1

  @spec compute(Graph.t(), keyword()) :: binary()
  def compute(%Graph{} = graph, opts) do
    parts = [
      "gsv:",
      Integer.to_string(Graph.semantics_version()),
      "\ngraph:",
      Graph.canonical_json(graph),
      "\nnx:",
      to_string(Application.spec(:nx, :vsn) || "unknown"),
      "\ntg:",
      to_string(Keyword.get(opts, :tinygrad_commit, "unknown")),
      "\npv:",
      Integer.to_string(@protocol_version),
      "\ndev:",
      to_string(Keyword.get(opts, :device, "CPU")),
      "\ncc:",
      canonical_compile_opts(opts)
    ]

    :crypto.hash(:sha256, parts) |> Base.encode16(case: :lower)
  end

  # Only options that change the compiled artifact belong here (not runtime-only
  # options like output mode or debug).
  defp canonical_compile_opts(opts) do
    opts
    |> Keyword.take([:device])
    |> Enum.sort()
    |> inspect()
  end
end
