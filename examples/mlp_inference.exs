# Two-layer MLP inference.
#
#   mix run examples/mlp_inference.exs                                  # CPU
#   NX_TINYGRAD_DEVICE="KFD+AMD:LLVM" mix run examples/mlp_inference.exs # AMD GPU

defmodule MLP do
  import Nx.Defn

  defn predict({w1, b1, w2, b2}, x) do
    h = Nx.tanh(Nx.dot(x, w1) + b1)
    Nx.dot(h, w2) + b2
  end
end

params = {
  Nx.tensor([[0.1, -0.2, 0.3, 0.05], [-0.1, 0.2, 0.0, 0.15], [0.25, -0.05, 0.1, -0.2]]),
  Nx.tensor([0.01, -0.02, 0.03, 0.0]),
  Nx.tensor([[0.2, -0.1], [0.05, 0.15], [-0.2, 0.1], [0.1, -0.05]]),
  Nx.tensor([0.0, 0.1])
}

x = Nx.tensor([[0.5, -0.3, 0.8], [0.1, 0.2, -0.4]])

device = System.get_env("NX_TINYGRAD_DEVICE", "CPU")
predict = NxTinygrad.jit(&MLP.predict/2, device: device)
IO.inspect(Nx.backend_transfer(predict.(params, x)), label: "nx_tinygrad prediction")
IO.inspect(MLP.predict(params, x), label: "reference")
