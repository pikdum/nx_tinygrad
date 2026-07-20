defmodule NxTinygrad.GraphCacheKey do
  @moduledoc """
  Computes the cache key for a compiled graph.

  The key mixes everything that can change the compiled result: the graph
  semantics version, the canonical graph JSON (which already encodes input and
  output shapes/dtypes), inline constant contents, the Nx version, the tinygrad
  commit/version, the protocol version, the worker/device identity, and any
  compile options that affect compilation.
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
      "\nblobs:",
      blob_fingerprints(graph.blobs),
      "\nnx:",
      to_string(Application.spec(:nx, :vsn) || "unknown"),
      "\ntg:",
      to_string(Keyword.get(opts, :tinygrad_commit, "unknown")),
      "\npv:",
      Integer.to_string(@protocol_version),
      "\ndev:",
      to_string(Keyword.get(opts, :device, "CPU")),
      "\nworker:",
      :erlang.term_to_binary(Keyword.get(opts, :worker, :default)),
      "\ncc:",
      canonical_compile_opts(opts)
    ]

    :crypto.hash(:sha256, parts) |> Base.encode16(case: :lower)
  end

  # Hash blob contents separately so the outer hash does not need to copy large
  # tensor constants into an intermediate binary. Length delimiters make the
  # ordered sequence unambiguous.
  defp blob_fingerprints(blobs) do
    [
      Integer.to_string(length(blobs)),
      Enum.map(blobs, fn blob ->
        [":", Integer.to_string(byte_size(blob)), ":", :crypto.hash(:sha256, blob)]
      end)
    ]
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
