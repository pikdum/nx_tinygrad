# Per-iteration cost of an eager `while` loop.
#
#   mix run bench/while_loop_overhead.exs
#   NX_TINYGRAD_BENCH_DEVICE="KFD+AMD:LLVM" mix run bench/while_loop_overhead.exs  # GPU
#
# A `while` body that contains a dynamic slice (a runtime index — here a
# scheduler-style `coeffs[i]`) can't be JIT-captured, so the worker interprets
# the WHOLE body node-by-node on every iteration (see docs/performance.md, the
# Stable Diffusion breakdown). This is the shape of SD's denoise loop: a big
# static compute (the UNet, independent of the runtime index) followed by cheap
# dynamic glue. The per-iter time here is ~linear in the static body size, which
# is why SD's thousands-of-op UNet costs seconds/step while the GPU sits idle.
#
# Use this to measure any future acceleration of the eager `while` path (e.g.
# JIT-capturing the static sub-body): per-iter time should drop toward the pure
# device compute, and `close?` must stay true.

Nx.global_default_backend(Nx.BinaryBackend)

defmodule WhileBench do
  import Nx.Defn

  # Big STATIC "UNet-like" compute — depends only on the carried state + weights,
  # not on the runtime index, exactly like SD's UNet vs the scheduler.
  defnp block(acc, w) do
    acc
    |> Nx.dot(w)
    |> Nx.tanh()
    |> Nx.dot(w)
    |> Nx.tanh()
    |> Nx.dot(w)
    |> Nx.tanh()
    |> Nx.dot(w)
    |> Nx.tanh()
    |> Nx.dot(w)
    |> Nx.tanh()
    |> Nx.dot(w)
    |> Nx.tanh()
    |> Nx.dot(w)
    |> Nx.tanh()
    |> Nx.dot(w)
    |> Nx.tanh()
  end

  defn run(w, coeffs) do
    {out, _i, _w, _c} =
      while {acc = Nx.broadcast(0.5, {1, 256}), i = 0, ww = w, cc = coeffs}, i < 20 do
        pred = block(acc, ww)
        # Dynamic slice by the runtime loop counter -> forces the eager path.
        c = Nx.reshape(Nx.slice(cc, [i], [1]), {})
        acc = pred |> Nx.multiply(c) |> Nx.add(0.001)
        {acc, i + 1, ww, cc}
      end

    out
  end
end

device = System.get_env("NX_TINYGRAD_BENCH_DEVICE", "CPU")
w = Nx.iota({256, 256}, type: :f32) |> Nx.divide(65_536)
coeffs = Nx.iota({20}, type: :f32) |> Nx.divide(20)

f = NxTinygrad.jit(&WhileBench.run/2, device: device)
_ = f.(w, coeffs) |> Nx.backend_transfer()

t0 = System.monotonic_time(:millisecond)
r = f.(w, coeffs) |> Nx.backend_transfer()
elapsed = System.monotonic_time(:millisecond) - t0

close = Nx.all_close(r, WhileBench.run(w, coeffs), atol: 1.0e-3) |> Nx.to_number()

IO.puts(
  "device=#{device}  20-iter while (16 static dots/iter): #{elapsed} ms  " <>
    "(#{Float.round(elapsed / 20, 1)} ms/iter)  close?=#{close == 1}"
)
