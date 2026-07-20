# One-shot integration test: train a small MLP with Axon, compiled end-to-end by
# nx_tinygrad (forward + autograd backward + optimizer updates) on the configured
# device. No model download required.
#
#   elixir examples/axon_mlp_training.exs                 # CPU
#   NX_TINYGRAD_DEVICE="KFD+AMD:LLVM" elixir examples/axon_mlp_training.exs   # AMD GPU
#
# Exercises: Axon.Loop training, dense layers, tanh, MSE loss, Adam — the whole
# step lowered through NxTinygrad.Compiler.

Mix.install([
  {:nx_tinygrad, path: Path.expand("..", __DIR__)},
  {:axon, "~> 0.7"}
])

# Host-side tensor bookkeeping stays on the binary backend; every defn (the
# train step) is compiled by nx_tinygrad.
Nx.global_default_backend(Nx.BinaryBackend)

device = System.get_env("NX_TINYGRAD_DEVICE", "CPU")
# output: :host so Axon's between-step eager tensor ops run on the binary backend.
Nx.Defn.default_options(compiler: NxTinygrad.Compiler, device: device, output: :host)
IO.puts("Training through nx_tinygrad on device=#{device}\n")

# Target: a nonlinear function of two inputs. The MLP must learn it.
target = fn x ->
  x1 = x[[.., 0]]
  x2 = x[[.., 1]]
  Nx.sin(Nx.multiply(x1, 3.0)) |> Nx.add(Nx.multiply(x2, x2)) |> Nx.reshape({:auto, 1})
end

batch = fn key ->
  {x, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {64, 2}, type: :f32)
  {{x, target.(x)}, key}
end

data =
  Stream.unfold(Nx.Random.key(1), fn key ->
    {sample, key} = batch.(key)
    {sample, key}
  end)

model =
  Axon.input("x", shape: {nil, 2})
  |> Axon.dense(64, activation: :tanh)
  |> Axon.dense(64, activation: :tanh)
  |> Axon.dense(1)

params =
  model
  |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam(learning_rate: 1.0e-2))
  |> Axon.Loop.run(data, %{}, epochs: 5, iterations: 100)

# Evaluate the trained model on fresh points.
{x_test, _} = Nx.Random.uniform(Nx.Random.key(99), -1.0, 1.0, shape: {8, 2}, type: :f32)
pred = Axon.predict(model, params, %{"x" => x_test})
expected = target.(x_test)
mae = pred |> Nx.subtract(expected) |> Nx.abs() |> Nx.mean() |> Nx.to_number()

IO.puts("\nFinal test MAE: #{Float.round(mae, 4)}")

if mae < 0.25,
  do: IO.puts("PASS ✅  MLP trained through nx_tinygrad"),
  else: IO.puts("HIGH ERROR ⚠️  (train longer)")
