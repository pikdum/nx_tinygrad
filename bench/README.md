# Benchmarks

```sh
mix run bench/bridge_overhead.exs
N=512 mix run bench/matmul.exs
mix run bench/mlp.exs
DEV=CPU python bench/direct_tinygrad.py

# On the AMD GPU:
EX_TINYGRAD_BENCH_DEVICE="KFD+AMD:LLVM" N=1024 mix run bench/matmul.exs
DEV=AMD AMD_LLVM=1 AMD_IFACE=KFD python bench/direct_tinygrad.py
```

Representative numbers on the reference machine (Ryzen 5 5600X, **CPU device**,
tinygrad 0.12.0). Absolute values vary; the point is the *shape* of the results.

| Workload (CPU)                         | Result |
| -------------------------------------- | ------ |
| 512×512 matmul — `Nx.BinaryBackend`    | ~57,000 ms/call |
| 512×512 matmul — ex_tinygrad compile   | ~160 ms (first call) |
| 512×512 matmul — ex_tinygrad warm      | ~1.3 ms/call (resident in+out, replay) |
| 512×512 matmul — direct tinygrad warm  | ~3.1 ms/call |
| MLP 128×256→256→64 inference warm      | ~1.7 ms/call |
| MLP value_and_grad warm                | ~6.4 ms/call |
| Bridge overhead (trivial graph)        | one execute RPC per call, ~3 ms round trip |

Notes:

- Warm ex_tinygrad replay is in the same ballpark as direct tinygrad for the same
  captured graph (the architecture acceptance target is within ~20%).
- The huge BinaryBackend matmul time is expected — it is a pure-Elixir O(n³)
  reference, not an optimized kernel.
- The first call includes compile + `TinyJit` capture; every later call replays
  with a single `execute` RPC.
