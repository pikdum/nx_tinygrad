# Basic elementwise example.
#
#   mix run examples/basic.exs
#
# Uses the :default worker (CPU per config/config.exs). To run on the GPU, start
# a worker and pass `worker:`/`device:` — see examples/mlp_training.exs.

defmodule Basic do
  import Nx.Defn
  defn(predict(x, weights, bias), do: x |> Nx.dot(weights) |> Nx.add(bias) |> Nx.tanh())
end

x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
weights = Nx.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
bias = Nx.tensor([0.01, 0.02])

predict = NxTinygrad.jit(&Basic.predict/3)
result = predict.(x, weights, bias)

IO.inspect(Nx.backend_transfer(result), label: "nx_tinygrad")
IO.inspect(Basic.predict(x, weights, bias), label: "reference (BinaryBackend)")
