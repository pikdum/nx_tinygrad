# Benchmarks

```sh
# Three-way: plain Nx (BinaryBackend) vs Nx→tinygrad CPU vs Nx→tinygrad GPU.
NX_TINYGRAD_GPU_TESTS=1 mix run bench/nx_backends.exs   # GPU row needs an AMD device

# Focused scripts.
mix run bench/matmul.exs
mix run bench/mlp.exs
mix run bench/bridge_overhead.exs
DEV=CPU python bench/direct_tinygrad.py
```

## Three-way comparison (`bench/nx_backends.exs`)

Reference machine: **Ryzen 5 5600X** (CPU) + **Radeon RX 7900 XT / gfx1100** (GPU),
tinygrad 0.13, via Benchee. The **same Nx computation** is run three ways:

- **nx (binary, cpu)** — plain Nx on `Nx.BinaryBackend`, eager on the host.
- **nx→tinygrad (cpu)** — `NxTinygrad.jit`, CPU worker.
- **nx→tinygrad (gpu)** — `NxTinygrad.jit`, AMD worker (`KFD+AMD:LLVM`).

Methodology: graphs are compiled + captured once (warmup); inputs are resident on
the target device; each call is a warm replay followed by a device `synchronize`
so the number reflects **real compute + the Elixir↔Python bridge**, not tinygrad's
lazy scheduling. (Measuring via `.numpy()` instead would drown everything in the
result download; measuring without a sync would only time enqueuing.)

Average time per call (lower is better):

| Workload                                   | plain Nx (binary) | tinygrad CPU | tinygrad GPU |
| ------------------------------------------ | ----------------: | -----------: | -----------: |
| matmul 64×64 (tiny)                        |          27.1 ms  |     1.20 ms  |     1.28 ms  |
| elementwise ×10 fused, 512×512             |           203 ms  |     4.80 ms  |     1.25 ms  |
| elementwise ×10 fused, 4096×4096           |    (too slow)     |    85.1 ms   |     1.45 ms  |
| MLP inference, batch 64 (128→128→32)       |           137 ms  |     1.86 ms  |     1.41 ms  |
| MLP value_and_grad, batch 64               |           308 ms  |     5.87 ms  |     5.05 ms  |
| matmul 1024×1024                           |    (too slow)     |    36.9 ms   |     1.32 ms  |

## What the numbers say

- **Going through tinygrad beats plain Nx by 22–160×** wherever `BinaryBackend`
  is fast enough to measure. `Nx.BinaryBackend` is a pure-Elixir interpreter with
  per-operation overhead; tinygrad fuses the whole graph into compiled kernels.
- **The GPU wins big on compute-heavy work** — 28× over tinygrad-CPU on a 1024²
  matmul, 59× on a 4096² fused elementwise chain.
- **Small graphs are bridge-bound.** There is a ~1.2 ms floor per call — one
  `execute` and one `synchronize` round trip across the Erlang Port to Python.
  For tiny ops (64² matmul, small MLP) that floor dominates, so CPU and GPU look
  the same and the GPU's compute advantage is hidden. Keep tensors resident and
  make the graphs meaty to amortize it.

The takeaway matches the project's thesis: Nx supplies the API and autograd,
tinygrad supplies fused compiled kernels and the accelerator — and the win grows
with the amount of work per `execute`.
