# Bridge overhead: how much does the Elixir<->Python round trip cost per call?
#
#   mix run bench/bridge_overhead.exs
#
# Uses a trivial graph so the measured time is dominated by the framing + RPC,
# not by compute. Also confirms one execute RPC per invocation.

alias ExTinygrad.Backend

worker = :default
x = Nx.tensor([1.0, 2.0, 3.0, 4.0]) |> Nx.backend_transfer({Backend, worker: worker})
f = ExTinygrad.jit(fn t -> Nx.add(t, 1.0) end, worker: worker, output: :device)

# warm up (compile + capture)
f.(x)

iters = 2000
before = ExTinygrad.worker_stats()["execute_count"]
{us, _} = :timer.tc(fn -> for _ <- 1..iters, do: f.(x) end)
after_ = ExTinygrad.worker_stats()["execute_count"]

IO.puts("== bridge overhead (trivial graph, resident input/output) ==")
IO.puts("per-call round trip : #{Float.round(us / iters, 1)} us")
IO.puts("execute RPCs        : #{after_ - before} for #{iters} calls (expect #{iters})")
