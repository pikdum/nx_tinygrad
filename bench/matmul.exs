# f32 matmul benchmark: BinaryBackend vs ex_tinygrad (compile vs warm replay).
#
#   mix run bench/matmul.exs
#   EX_TINYGRAD_BENCH_DEVICE="KFD+AMD:LLVM" mix run bench/matmul.exs   # on GPU

alias ExTinygrad.Backend

n = String.to_integer(System.get_env("N", "1024"))
device = System.get_env("EX_TINYGRAD_BENCH_DEVICE", "CPU")

worker =
  if device == "CPU" do
    :default
  else
    {:ok, _} = ExTinygrad.WorkerSupervisor.start_worker(:bench, device)
    :bench
  end

a = Nx.iota({n, n}, type: :f32) |> Nx.divide(n)
b = Nx.iota({n, n}, type: :f32) |> Nx.divide(n)

avg = fn f, iters ->
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: f.() end)
  us / iters / 1000.0
end

IO.puts("== #{n}x#{n} f32 matmul on #{device} ==")

# BinaryBackend reference
bin_ms = avg.(fn -> Nx.dot(a, b) end, 3)
IO.puts("BinaryBackend        : #{Float.round(bin_ms, 2)} ms/call")

# ex_tinygrad, device-resident inputs and outputs
adev = Nx.backend_transfer(a, {Backend, worker: worker})
bdev = Nx.backend_transfer(b, {Backend, worker: worker})
mm = ExTinygrad.jit(fn x, y -> Nx.dot(x, y) end, worker: worker, output: :device)

{compile_us, _} = :timer.tc(fn -> mm.(adev, bdev) end)
IO.puts("ex_tinygrad compile  : #{Float.round(compile_us / 1000.0, 2)} ms (first call)")

warm_ms = avg.(fn -> mm.(adev, bdev) end, 20)
IO.puts("ex_tinygrad warm     : #{Float.round(warm_ms, 3)} ms/call (resident in+out, replay)")
