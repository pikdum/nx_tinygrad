defmodule NxTinygrad.OutputContainerTest do
  @moduledoc "Container flattening and reconstruction, exercised end-to-end."
  use ExUnit.Case, async: false

  import Nx.Defn
  import NxTinygrad.TestGraphs, only: [assert_close: 2]

  defn(single(x), do: Nx.multiply(x, 2.0))
  defn(tuple2(x), do: {Nx.negate(x), Nx.add(x, 1.0)})
  defn(nested(x), do: {Nx.negate(x), {Nx.add(x, 1.0), Nx.subtract(x, 1.0)}})
  defn(as_map(x), do: %{a: Nx.negate(x), b: Nx.add(x, 1.0)})
  defn(empty_tuple(_x), do: {})
  defn(repeated(x), do: {Nx.negate(x), Nx.negate(x)})

  @x Nx.tensor([1.0, 2.0, 3.0])

  test "single tensor" do
    assert_close(NxTinygrad.jit(&single/1).(@x), single(@x))
  end

  test "tuple" do
    {a, b} = NxTinygrad.jit(&tuple2/1).(@x)
    {ea, eb} = tuple2(@x)
    assert_close(a, ea)
    assert_close(b, eb)
  end

  test "nested tuple" do
    {a, {b, c}} = NxTinygrad.jit(&nested/1).(@x)
    {ea, {eb, ec}} = nested(@x)
    assert_close(a, ea)
    assert_close(b, eb)
    assert_close(c, ec)
  end

  test "map" do
    got = NxTinygrad.jit(&as_map/1).(@x)
    exp = as_map(@x)
    assert_close(got.a, exp.a)
    assert_close(got.b, exp.b)
  end

  test "empty tuple" do
    assert NxTinygrad.jit(&empty_tuple/1).(@x) == {}
  end

  test "repeated output expression" do
    {a, b} = NxTinygrad.jit(&repeated/1).(@x)
    assert_close(a, Nx.negate(@x))
    assert_close(b, Nx.negate(@x))
  end

  test "struct implementing Nx.Container (Nx.Batch-like via Complex? use a tuple-of-map)" do
    # A map with tensor values is a container; verify a heterogeneous nested shape.
    got = NxTinygrad.jit(fn x -> %{outer: {Nx.negate(x)}, flat: Nx.add(x, 2.0)} end).(@x)
    assert_close(got.flat, Nx.add(@x, 2.0))
    {inner} = got.outer
    assert_close(inner, Nx.negate(@x))
  end
end
