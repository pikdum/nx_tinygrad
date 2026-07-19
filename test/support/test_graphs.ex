defmodule ExTinygrad.TestGraphs do
  @moduledoc "Shared defn functions and assertion helpers for tests."
  import Nx.Defn
  import ExUnit.Assertions

  defn(elementwise(x, y), do: Nx.tanh(x * y + 1))
  defn(broadcasting(x, b), do: Nx.add(x, b))
  defn(reduction(x), do: Nx.sum(x, axes: [1]))
  defn(reduce_keep(x), do: Nx.sum(x, axes: [1], keep_axes: true))
  defn(matmul(x, w, b), do: x |> Nx.dot(w) |> Nx.add(b) |> Nx.tanh())
  defn(comparison(x, y), do: Nx.select(Nx.greater(x, y), x, y))
  defn(chain(x), do: x |> Nx.negate() |> Nx.exp() |> Nx.add(1.0) |> Nx.log())
  defn(shapes(x), do: x |> Nx.transpose() |> Nx.reshape({6}) |> Nx.multiply(2.0))

  defn softmax(x) do
    m = Nx.reduce_max(x, axes: [1], keep_axes: true)
    e = Nx.exp(x - m)
    e / Nx.sum(e, axes: [1], keep_axes: true)
  end

  defn(multi_output(x), do: {Nx.sum(x), Nx.reduce_max(x)})
  defn(nested_container(x, y), do: %{sum: Nx.add(x, y), parts: {Nx.subtract(x, y), {Nx.multiply(x, y)}}})
  defn(repeated_input(x), do: Nx.add(x, x))

  @doc "Assert two tensors (or containers of tensors) are numerically close."
  def assert_close(actual, expected, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    rtol = Keyword.get(opts, :rtol, 1.0e-4)

    actual_leaves = flatten(actual)
    expected_leaves = flatten(expected)

    assert length(actual_leaves) == length(expected_leaves),
           "container arity mismatch: #{length(actual_leaves)} vs #{length(expected_leaves)}"

    Enum.zip(actual_leaves, expected_leaves)
    |> Enum.each(fn {a, e} ->
      assert Nx.shape(a) == Nx.shape(e), "shape mismatch: #{inspect(Nx.shape(a))} vs #{inspect(Nx.shape(e))}"
      assert Nx.type(a) == Nx.type(e), "type mismatch: #{inspect(Nx.type(a))} vs #{inspect(Nx.type(e))}"

      close = Nx.all_close(a, e, atol: atol, rtol: rtol)

      assert Nx.to_number(close) == 1,
             "tensors not close:\n  actual=#{inspect(a)}\n  expected=#{inspect(e)}"
    end)
  end

  defp flatten(container) do
    container
    |> Nx.Defn.Composite.reduce([], fn t, acc -> [t | acc] end)
    |> Enum.reverse()
  end
end
