defmodule NxTinygrad.LoweringTest do
  use ExUnit.Case, async: true

  alias NxTinygrad.{Graph, GraphCacheKey, Lowering}

  # Lower a defn function to a graph without touching Python.
  defp lower(fun, args) do
    fun
    |> Nx.Defn.debug_expr()
    |> apply(args)
    |> flatten_roots()
    |> Lowering.to_graph()
  end

  defp flatten_roots(container) do
    container
    |> Nx.Defn.Composite.reduce([], fn t, acc -> [t | acc] end)
    |> Enum.reverse()
  end

  defp ops(graph), do: Enum.map(graph.nodes, & &1["op"])

  test "matmul lowers to dot/add/tanh with three inputs and one output" do
    x = Nx.iota({2, 3}, type: :f32)
    w = Nx.iota({3, 2}, type: :f32)
    b = Nx.tensor([1.0, 2.0])
    graph = lower(&NxTinygrad.TestGraphs.matmul/3, [x, w, b])

    assert length(graph.inputs) == 3
    assert ops(graph) == ["dot", "add", "tanh"]
    assert [%{"shape" => [2, 2], "dtype" => "f32"}] = graph.outputs
  end

  test "ids are sequential and topological (children before parents)" do
    graph = lower(&NxTinygrad.TestGraphs.chain/1, [Nx.iota({3}, type: :f32)])

    all_ids =
      (Enum.map(graph.inputs, & &1["id"]) ++
         Enum.map(graph.constants, & &1["id"]) ++
         Enum.map(graph.nodes, & &1["id"]))
      |> Enum.sort()

    assert all_ids == Enum.to_list(0..(length(all_ids) - 1))

    for node <- graph.nodes, ref <- node["inputs"] do
      assert ref < node["id"], "input #{ref} should precede node #{node["id"]}"
    end
  end

  test "comparison produces a u8 node feeding select" do
    a = Nx.iota({2, 2}, type: :f32)
    graph = lower(&NxTinygrad.TestGraphs.comparison/2, [a, a])
    greater = Enum.find(graph.nodes, &(&1["op"] == "greater"))
    assert greater["dtype"] == "u8"
    assert Enum.any?(graph.nodes, &(&1["op"] == "select"))
  end

  test "reduction normalizes axes and records keep_axes" do
    graph = lower(&NxTinygrad.TestGraphs.reduce_keep/1, [Nx.iota({2, 3}, type: :f32)])
    sum = Enum.find(graph.nodes, &(&1["op"] == "sum"))
    assert sum["attrs"] == %{"axes" => [1], "keep_axes" => true}
  end

  test "a repeated subexpression is emitted once (DAG dedup)" do
    graph = lower(&NxTinygrad.TestGraphs.repeated_input/1, [Nx.iota({4}, type: :f32)])
    # x + x: one input, one add node referencing the same id twice.
    assert length(graph.inputs) == 1
    [add] = graph.nodes
    assert add["op"] == "add"
    assert [id, id] = add["inputs"]
  end

  test "multiple outputs are preserved in order" do
    graph = lower(&NxTinygrad.TestGraphs.multi_output/1, [Nx.iota({2, 3}, type: :f32)])
    assert length(graph.outputs) == 2
  end

  test "canonical JSON and cache key are deterministic for identical graphs" do
    x = Nx.iota({2, 3}, type: :f32)
    w = Nx.iota({3, 2}, type: :f32)
    b = Nx.tensor([1.0, 2.0])
    g1 = lower(&NxTinygrad.TestGraphs.matmul/3, [x, w, b])
    g2 = lower(&NxTinygrad.TestGraphs.matmul/3, [x, w, b])

    assert IO.iodata_to_binary(Graph.canonical_json(g1)) == IO.iodata_to_binary(Graph.canonical_json(g2))
    assert GraphCacheKey.compute(g1, device: "CPU") == GraphCacheKey.compute(g2, device: "CPU")
  end

  test "different input shapes produce different cache keys" do
    b = Nx.tensor([1.0, 2.0])
    g1 = lower(&NxTinygrad.TestGraphs.matmul/3, [Nx.iota({2, 3}, type: :f32), Nx.iota({3, 2}, type: :f32), b])
    g2 = lower(&NxTinygrad.TestGraphs.matmul/3, [Nx.iota({4, 3}, type: :f32), Nx.iota({3, 2}, type: :f32), b])
    refute GraphCacheKey.compute(g1, device: "CPU") == GraphCacheKey.compute(g2, device: "CPU")
  end

  test "unsupported operations raise a compile error before Python" do
    assert_raise NxTinygrad.CompileError, ~r/unsupported Nx operation/, fn ->
      lower(fn x -> Nx.fft(x) end, [Nx.iota({4}, type: {:c, 64})])
    end
  end
end
