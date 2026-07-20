defmodule NxTinygrad.CompilerTest do
  @moduledoc "End-to-end CPU parity against Nx.BinaryBackend through the full worker path."
  use ExUnit.Case, async: false

  import NxTinygrad.TestGraphs
  alias NxTinygrad.TestGraphs, as: G

  setup do
    assert NxTinygrad.Worker.whereis(:default) != nil
    :ok
  end

  test "elementwise chain" do
    x = Nx.tensor([[1.0, -2.0], [3.0, 0.5]])
    y = Nx.tensor([[0.5, 4.0], [-1.0, 2.0]])
    assert_close(NxTinygrad.jit(&G.elementwise/2).(x, y), G.elementwise(x, y))
  end

  test "broadcasting" do
    x = Nx.iota({2, 3}, type: :f32)
    b = Nx.tensor([10.0, 20.0, 30.0])
    assert_close(NxTinygrad.jit(&G.broadcasting/2).(x, b), G.broadcasting(x, b))
  end

  test "reduction (with and without keep_axes)" do
    x = Nx.iota({3, 4}, type: :f32)
    assert_close(NxTinygrad.jit(&G.reduction/1).(x), G.reduction(x))
    assert_close(NxTinygrad.jit(&G.reduce_keep/1).(x), G.reduce_keep(x))
  end

  test "matrix multiplication" do
    x = Nx.iota({2, 3}, type: :f32)
    w = Nx.iota({3, 4}, type: :f32)
    b = Nx.tensor([1.0, 2.0, 3.0, 4.0])
    assert_close(NxTinygrad.jit(&G.matmul/3).(x, w, b), G.matmul(x, w, b))
  end

  test "comparison + select" do
    x = Nx.tensor([[1.0, 5.0], [3.0, 2.0]])
    y = Nx.tensor([[4.0, 4.0], [4.0, 4.0]])
    assert_close(NxTinygrad.jit(&G.comparison/2).(x, y), G.comparison(x, y))
  end

  test "shape ops (transpose/reshape)" do
    x = Nx.iota({2, 3}, type: :f32)
    assert_close(NxTinygrad.jit(&G.shapes/1).(x), G.shapes(x))
  end

  test "softmax (reduce_max, exp, sub, sum, divide)" do
    x = Nx.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    assert_close(NxTinygrad.jit(&G.softmax/1).(x), G.softmax(x))
  end

  test "multiple outputs (tuple)" do
    x = Nx.iota({2, 3}, type: :f32)
    {a, b} = NxTinygrad.jit(&G.multi_output/1).(x)
    {ea, eb} = G.multi_output(x)
    assert_close(a, ea)
    assert_close(b, eb)
  end

  test "nested output container (map + tuple + list)" do
    x = Nx.iota({2, 2}, type: :f32)
    y = Nx.iota({2, 2}, type: :f32) |> Nx.add(1.0)
    got = NxTinygrad.jit(&G.nested_container/2).(x, y)
    exp = G.nested_container(x, y)
    assert_close(got.sum, exp.sum)
    {gt, {gl}} = got.parts
    {et, {el}} = exp.parts
    assert_close(gt, et)
    assert_close(gl, el)
  end

  test "repeated input tensor passed to both operands" do
    x = Nx.tensor([1.0, 2.0, 3.0])
    assert_close(NxTinygrad.jit(&G.repeated_input/1).(x), G.repeated_input(x))
  end

  test "device: option routes to a worker for that device" do
    x = Nx.iota({2, 3}, type: :f32)
    result = NxTinygrad.jit(&G.reduction/1, device: "CPU").(x)
    assert_close(result, G.reduction(x))
  end

  test "compiling twice yields equal results (idempotent)" do
    x = Nx.iota({2, 3}, type: :f32)
    r1 = NxTinygrad.jit(&G.reduction/1).(x)
    r2 = NxTinygrad.jit(&G.reduction/1).(x)
    assert_close(r1, r2)
  end
end
