defmodule ExTinygrad.GradientTest do
  @moduledoc """
  Autograd works because Nx rewrites gradients (`value_and_grad`) before the
  graph reaches our compiler — the compiler just lowers the larger forward graph.
  Everything is validated against `Nx.BinaryBackend`.
  """
  use ExUnit.Case, async: false

  import ExTinygrad.TestGraphs
  alias ExTinygrad.TestGraphs, as: G

  # Gradients through matmuls accumulate a little more error than a single op.
  @tol [atol: 1.0e-4, rtol: 1.0e-3]

  defp params do
    {
      Nx.tensor([[0.1, -0.2, 0.3, 0.05], [-0.1, 0.2, 0.0, 0.15], [0.25, -0.05, 0.1, -0.2]]),
      Nx.tensor([0.01, -0.02, 0.03, 0.0]),
      Nx.tensor([[0.2, -0.1], [0.05, 0.15], [-0.2, 0.1], [0.1, -0.05]]),
      Nx.tensor([0.0, 0.1])
    }
  end

  defp inputs do
    x = Nx.tensor([[0.5, -0.3, 0.8], [0.1, 0.2, -0.4], [-0.6, 0.9, 0.3], [0.2, -0.1, 0.5]])
    y = Nx.tensor([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2], [0.0, 0.1]])
    {x, y}
  end

  test "linear+tanh value_and_grad matches BinaryBackend" do
    w = Nx.tensor([[0.1, -0.2], [0.3, 0.05], [-0.1, 0.2]])
    x = Nx.tensor([[0.5, -0.3, 0.8], [0.1, 0.2, -0.4]])
    targets = Nx.tensor([[0.3, -0.2], [0.1, 0.4]])

    {value, grad} = ExTinygrad.jit(&G.linear_value_and_grad/3).(w, x, targets)
    {ev, eg} = G.linear_value_and_grad(w, x, targets)

    assert_close(value, ev, @tol)
    assert_close(grad, eg, @tol)
  end

  test "MLP inference matches BinaryBackend" do
    p = params()
    {x, _y} = inputs()
    assert_close(ExTinygrad.jit(&G.mlp_predict/2).(p, x), G.mlp_predict(p, x), @tol)
  end

  test "MLP training step: value_and_grad over all parameters matches BinaryBackend" do
    p = params()
    {x, y} = inputs()

    {value, grads} = ExTinygrad.jit(&G.mlp_value_and_grad/3).(p, x, y)
    {ev, egrads} = G.mlp_value_and_grad(p, x, y)

    assert_close(value, ev, @tol)
    # grads is a 4-tuple matching params; assert_close flattens and compares each.
    assert_close(grads, egrads, @tol)
  end

  test "a gradient-descent step reduces the loss" do
    p = params()
    {x, y} = inputs()
    lr = 0.1

    vg = ExTinygrad.jit(&G.mlp_value_and_grad/3)
    {loss0, {g1, g2, g3, g4}} = vg.(p, x, y)
    {w1, b1, w2, b2} = p

    updated = {
      Nx.subtract(w1, Nx.multiply(lr, Nx.backend_transfer(g1))),
      Nx.subtract(b1, Nx.multiply(lr, Nx.backend_transfer(g2))),
      Nx.subtract(w2, Nx.multiply(lr, Nx.backend_transfer(g3))),
      Nx.subtract(b2, Nx.multiply(lr, Nx.backend_transfer(g4)))
    }

    {loss1, _} = vg.(updated, x, y)

    assert Nx.to_number(Nx.backend_transfer(loss1)) < Nx.to_number(Nx.backend_transfer(loss0))
  end
end
