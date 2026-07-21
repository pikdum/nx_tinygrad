# Synthetic repro of SD's denoise loop: a `while` with a dynamic slice (scheduler
# indexing by the step counter) that forces the eager path, carrying weights and
# doing real matmul work per step. Runs via `mix run` (live priv/, fast iterate).

Nx.global_default_backend(Nx.BinaryBackend)

defmodule W do
  import Nx.Defn

  # Big STATIC "UNet-like" compute — independent of the dynamic coeff, like SD's
  # UNet depends only on latents+weights, not on alpha[step].
  defnp unet(acc, ww) do
    acc
    |> Nx.dot(ww) |> Nx.tanh() |> Nx.dot(ww) |> Nx.tanh()
    |> Nx.dot(ww) |> Nx.tanh() |> Nx.dot(ww) |> Nx.tanh()
    |> Nx.dot(ww) |> Nx.tanh() |> Nx.dot(ww) |> Nx.tanh()
    |> Nx.dot(ww) |> Nx.tanh() |> Nx.dot(ww) |> Nx.tanh()
    |> Nx.dot(ww) |> Nx.tanh() |> Nx.dot(ww) |> Nx.tanh()
    |> Nx.dot(ww) |> Nx.tanh() |> Nx.dot(ww) |> Nx.tanh()
    |> Nx.dot(ww) |> Nx.tanh() |> Nx.dot(ww) |> Nx.tanh()
    |> Nx.dot(ww) |> Nx.tanh()
  end

  defn run(w, coeffs) do
    {out, _i, _w, _c} =
      while {acc = Nx.broadcast(0.5, {1, 256}), i = 0, ww = w, cc = coeffs}, i < 20 do
        pred = unet(acc, ww)

        # Dynamic slice: index coeffs by the runtime loop counter -> eager taint,
        # exactly like Bumblebee's scheduler indexing alphas/timesteps by step.
        c = Nx.reshape(Nx.slice(cc, [i], [1]), {})

        # Cheap scheduler-like glue that uses the dynamic coeff on the UNet output.
        acc = pred |> Nx.multiply(c) |> Nx.add(0.001)

        {acc, i + 1, ww, cc}
      end

    out
  end
end

device = System.get_env("NX_TINYGRAD_DEVICE", "CPU")
w = Nx.iota({256, 256}, type: :f32) |> Nx.divide(65_536)
coeffs = Nx.iota({20}, type: :f32) |> Nx.divide(20)

f = NxTinygrad.jit(&W.run/2, device: device)

# Warm (first call compiles), then time steady-state.
_ = f.(w, coeffs) |> Nx.backend_transfer()

t0 = System.monotonic_time(:millisecond)
r = f.(w, coeffs) |> Nx.backend_transfer()
t1 = System.monotonic_time(:millisecond)

ref = W.run(w, coeffs)
close = Nx.all_close(r, ref, atol: 1.0e-3) |> Nx.to_number()
IO.puts("device=#{device}  20-iter while: #{t1 - t0} ms  (#{Float.round((t1 - t0) / 20, 1)} ms/iter)  close?=#{close}")
