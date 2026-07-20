# Three-way comparison: plain Nx (BinaryBackend) vs Nx -> tinygrad (CPU) vs
# Nx -> tinygrad (AMD GPU), using Benchee.
#
#   mix run bench/nx_backends.exs
#
# Methodology: the SAME Nx computation is run three ways. For the tinygrad
# backends the graph is compiled + captured once (warmup), inputs are resident on
# that device, and outputs stay on-device (output: :device) — so each measured
# call is a warm replay + one execute RPC, the steady-state a training loop sees.
# Plain Nx runs eagerly on the host BinaryBackend. Device outputs are released in
# an untimed after_each hook so buffers don't accumulate.
#
# The GPU row appears only if an AMD device is available.

alias NxTinygrad.Backend

# ---------------------------------------------------------------- workers ----
cpu = :default

gpu =
  with {:started, _} <-
         (case NxTinygrad.WorkerSupervisor.start_worker(:amd, "KFD+AMD:LLVM") do
            {:ok, pid} -> {:started, pid}
            {:error, {:already_started, pid}} -> {:started, pid}
            other -> other
          end),
       {:ok, %{"usable" => true} = info, []} <- NxTinygrad.Worker.request(:amd, "device_info", %{}) do
    IO.puts("GPU: #{info["architecture"]} via #{info["interface"]}+AMD:#{info["renderer"]}")
    :amd
  else
    _ ->
      IO.puts("GPU: not available — skipping GPU rows")
      nil
  end

# -------------------------------------------------------------- workloads ----
defmodule W do
  def elementwise(x) do
    x
    |> Nx.multiply(1.001)
    |> Nx.add(0.5)
    |> Nx.tanh()
    |> Nx.negate()
    |> Nx.exp()
    |> Nx.add(1.0)
    |> Nx.log()
    |> Nx.multiply(2.0)
    |> Nx.subtract(0.25)
    |> Nx.sigmoid()
  end

  def predict({w1, b1, w2, b2}, x) do
    h = Nx.tanh(Nx.add(Nx.dot(x, w1), b1))
    Nx.add(Nx.dot(h, w2), b2)
  end

  def loss(p, x, y), do: Nx.mean(Nx.pow(Nx.subtract(predict(p, x), y), 2))
  def grad(p, x, y), do: Nx.Defn.value_and_grad(p, fn pp -> loss(pp, x, y) end)
end

# ------------------------------------------------------------------ inputs ----
key = Nx.Random.key(0)
{mm_a, key} = Nx.Random.normal(key, shape: {64, 64}, type: :f32)
{mm_b, key} = Nx.Random.normal(key, shape: {64, 64}, type: :f32)
{big_a, key} = Nx.Random.normal(key, shape: {1024, 1024}, type: :f32)
{big_b, key} = Nx.Random.normal(key, shape: {1024, 1024}, type: :f32)
{ew_x, key} = Nx.Random.normal(key, shape: {512, 512}, type: :f32)
{ew_big, key} = Nx.Random.normal(key, shape: {4096, 4096}, type: :f32)
{w1, key} = Nx.Random.normal(key, shape: {128, 128}, type: :f32)
{w2, key} = Nx.Random.normal(key, shape: {128, 32}, type: :f32)
{mx, key} = Nx.Random.normal(key, shape: {64, 128}, type: :f32)
{my, _key} = Nx.Random.normal(key, shape: {64, 32}, type: :f32)
params = {Nx.multiply(w1, 0.1), Nx.broadcast(0.0, {128}), Nx.multiply(w2, 0.1), Nx.broadcast(0.0, {32})}

# ------------------------------------------------------------------ runner ----
run = fn title, fun, host_args ->
  IO.puts("\n============================================================")
  IO.puts("## #{title}")
  IO.puts("============================================================")

  to_dev = fn worker -> Enum.map(host_args, &Nx.backend_copy(&1, {Backend, worker: worker})) end
  cpu_args = to_dev.(cpu)
  gpu_args = if gpu, do: to_dev.(gpu)

  f_cpu = NxTinygrad.jit(fun, worker: cpu, output: :device)
  f_gpu = if gpu, do: NxTinygrad.jit(fun, worker: gpu, output: :device)

  # warm up (compile + capture)
  NxTinygrad.release(apply(f_cpu, cpu_args))
  if gpu, do: NxTinygrad.release(apply(f_gpu, gpu_args))

  # Probe BinaryBackend feasibility with a timeout so a 60s matmul doesn't stall
  # the benchmark just to be skipped.
  task = Task.async(fn -> apply(fun, host_args) end)

  bin_ok? =
    case Task.yield(task, 2_500) || Task.shutdown(task, :brutal_kill) do
      {:ok, _} -> true
      _ -> false
    end

  unless bin_ok?, do: IO.puts("   (plain Nx BinaryBackend skipped: > 2.5 s per call)")

  # Tinygrad realize() only *enqueues* work; we synchronize the device (wait for
  # compute, without downloading) so the measurement reflects real compute plus
  # the Elixir<->Python bridge round trips — not lazy scheduling.
  tg = fn f, args, worker ->
    fn ->
      r = apply(f, args)
      NxTinygrad.synchronize(worker: worker)
      r
    end
  end

  jobs =
    %{}
    |> then(&if(bin_ok?, do: Map.put(&1, "nx (binary, cpu)", fn -> apply(fun, host_args) end), else: &1))
    |> Map.put("nx→tinygrad (cpu)", tg.(f_cpu, cpu_args, cpu))
    |> then(&if(gpu, do: Map.put(&1, "nx→tinygrad (gpu)", tg.(f_gpu, gpu_args, gpu)), else: &1))

  Benchee.run(jobs,
    time: 2,
    warmup: 1,
    memory_time: 0,
    after_each: fn result -> NxTinygrad.release(result) end,
    print: [benchmarking: false, fast_warning: false, configuration: false]
  )

  Enum.each(cpu_args, &NxTinygrad.release/1)
  if gpu, do: Enum.each(gpu_args, &NxTinygrad.release/1)
end

# ------------------------------------------------------------------- runs ----
run.("matmul 64×64 (small — bridge-bound)", fn a, b -> Nx.dot(a, b) end, [mm_a, mm_b])
run.("elementwise fusion, 512×512 (10 ops → 1 kernel)", &W.elementwise/1, [ew_x])
run.("elementwise fusion, 4096×4096 (10 ops → 1 kernel)", &W.elementwise/1, [ew_big])
run.("MLP inference, batch 64 (128→128→32)", &W.predict/2, [params, mx])
run.("MLP value_and_grad, batch 64 (training step)", &W.grad/3, [params, mx, my])
run.("matmul 1024×1024 (compute-heavy — GPU wins)", fn a, b -> Nx.dot(a, b) end, [big_a, big_b])
