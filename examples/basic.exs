# Basic elementwise example.
#
#   mix run examples/basic.exs                                  # CPU
#   NX_TINYGRAD_DEVICE="KFD+AMD:LLVM" mix run examples/basic.exs # AMD GPU

defmodule Basic do
  import Nx.Defn
  defn(predict(x, weights, bias), do: x |> Nx.dot(weights) |> Nx.add(bias) |> Nx.tanh())
end

device = System.get_env("NX_TINYGRAD_DEVICE", "CPU")

x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
weights = Nx.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
bias = Nx.tensor([0.01, 0.02])

predict = NxTinygrad.jit(&Basic.predict/3, device: device)
result = predict.(x, weights, bias)

IO.inspect(Nx.backend_transfer(result), label: "nx_tinygrad")
IO.inspect(Basic.predict(x, weights, bias), label: "reference (BinaryBackend)")
