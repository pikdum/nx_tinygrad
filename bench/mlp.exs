# MLP inference + training-step benchmark.
#
#   mix run bench/mlp.exs

alias ExTinygrad.Backend

defmodule BenchMLP do
  import Nx.Defn

  defn predict({w1, b1, w2, b2}, x) do
    h = Nx.tanh(Nx.dot(x, w1) + b1)
    Nx.dot(h, w2) + b2
  end

  defn(loss(p, x, y), do: Nx.mean(Nx.pow(predict(p, x) - y, 2)))
  defn(value_and_grad(p, x, y), do: Nx.Defn.value_and_grad(p, fn p -> loss(p, x, y) end))
end

worker = :default
{batch, din, dh, dout} = {128, 256, 256, 64}

key = Nx.Random.key(1)
{w1, key} = Nx.Random.normal(key, shape: {din, dh}, type: :f32)
{w2, key} = Nx.Random.normal(key, shape: {dh, dout}, type: :f32)
params = {w1, Nx.broadcast(0.0, {dh}), w2, Nx.broadcast(0.0, {dout})}
{x, key} = Nx.Random.normal(key, shape: {batch, din}, type: :f32)
{y, _} = Nx.Random.normal(key, shape: {batch, dout}, type: :f32)

pdev =
  params
  |> Tuple.to_list()
  |> Enum.map(&Nx.backend_transfer(&1, {Backend, worker: worker}))
  |> List.to_tuple()

xdev = Nx.backend_transfer(x, {Backend, worker: worker})
ydev = Nx.backend_transfer(y, {Backend, worker: worker})

avg = fn f, iters ->
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: f.() end)
  us / iters / 1000.0
end

infer = ExTinygrad.jit(&BenchMLP.predict/2, worker: worker, output: :device)
infer.(pdev, xdev)
IO.puts("== MLP (#{batch}x#{din} -> #{dh} -> #{dout}) ==")
IO.puts("inference warm       : #{Float.round(avg.(fn -> infer.(pdev, xdev) end, 20), 3)} ms/call")

vg = ExTinygrad.jit(&BenchMLP.value_and_grad/3, worker: worker, output: :device)
vg.(pdev, xdev, ydev)
IO.puts("value_and_grad warm  : #{Float.round(avg.(fn -> vg.(pdev, xdev, ydev) end, 20), 3)} ms/call")
