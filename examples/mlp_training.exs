# MLP training with Nx autograd through nx_tinygrad.
#
#   mix run examples/mlp_training.exs
#
# To run on the AMD GPU, start a worker first and pass worker:/device: to jit:
#
#   {:ok, _} = NxTinygrad.WorkerSupervisor.start_worker(:amd, "KFD+AMD:LLVM")
#   vg = NxTinygrad.jit(&MLP.value_and_grad/3, worker: :amd)

defmodule MLP do
  import Nx.Defn

  defn predict({w1, b1, w2, b2}, x) do
    h = Nx.tanh(Nx.dot(x, w1) + b1)
    Nx.dot(h, w2) + b2
  end

  defn(loss(params, x, y), do: Nx.mean(Nx.pow(predict(params, x) - y, 2)))

  defn value_and_grad(params, x, y) do
    Nx.Defn.value_and_grad(params, fn p -> loss(p, x, y) end)
  end
end

key = Nx.Random.key(42)
{w1, key} = Nx.Random.normal(key, shape: {3, 4}, type: :f32)
{w2, key} = Nx.Random.normal(key, shape: {4, 2}, type: :f32)
params = {Nx.multiply(w1, 0.3), Nx.broadcast(0.0, {4}), Nx.multiply(w2, 0.3), Nx.broadcast(0.0, {2})}

{x, key} = Nx.Random.normal(key, shape: {16, 3}, type: :f32)
{y, _key} = Nx.Random.normal(key, shape: {16, 2}, type: :f32)

vg = NxTinygrad.jit(&MLP.value_and_grad/3)
lr = 0.1

params =
  Enum.reduce(1..50, params, fn step, {w1, b1, w2, b2} = p ->
    {loss, {g1, g2, g3, g4}} = vg.(p, x, y)

    if rem(step, 10) == 1 or step == 50 do
      IO.puts("step #{step}  loss=#{Nx.to_number(Nx.backend_transfer(loss))}")
    end

    {
      Nx.subtract(w1, Nx.multiply(lr, Nx.backend_transfer(g1))),
      Nx.subtract(b1, Nx.multiply(lr, Nx.backend_transfer(g2))),
      Nx.subtract(w2, Nx.multiply(lr, Nx.backend_transfer(g3))),
      Nx.subtract(b2, Nx.multiply(lr, Nx.backend_transfer(g4)))
    }
  end)

{final_loss, _} = vg.(params, x, y)
IO.puts("final loss=#{Nx.to_number(Nx.backend_transfer(final_loss))}")
