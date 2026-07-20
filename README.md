# nx_tinygrad

An [Elixir](https://elixir-lang.org) [Nx](https://github.com/elixir-nx/nx)
compiler and tensor backend that uses [tinygrad](https://tinygrad.org) as the
optimizing compiler and accelerator runtime.

```text
Elixir / Nx / Axon
        │
        ▼
NxTinygrad.Compiler ──► versioned serialized tensor graph
        │
        ▼
supervised Python worker (Erlang Port)
        │
        ▼
tinygrad Tensor graph + TinyJit
        │
        ▼
tinygrad AMD runtime ──► Linux amdgpu driver via KFD
```

The whole `Nx.Defn` expression is transferred to tinygrad as **one graph** —
there is no Python RPC per Nx operation. Nx provides the Elixir tensor API,
containers, and autograd; tinygrad provides scheduling, fusion, kernel
generation, memory planning, JIT replay, and AMD execution.

The default installation requires **no ROCm, HIP, comgr, rocBLAS, MIOpen, XLA,
TensorFlow, or PyTorch**. The AMD GPU is driven through tinygrad's native KFD
interface, and kernels are compiled with libLLVM (`AMD_LLVM=1`).

## Status

Early development. See [STATUS.md](STATUS.md) for milestone progress and pinned
versions, and [SPEC.md](SPEC.md) for the design.

## Quickstart (NixOS)

```sh
# Enter the dev shell (Elixir 1.20 / OTP 29, Rust, ROCm-free tinygrad worker).
nix develop

mix deps.get
mix test                       # Elixir unit + CPU integration tests

# Probe the GPU (native KFD + LLVM, no ROCm):
python priv/worker/device.py "KFD+AMD:LLVM"

# Prove the worker closure has no ROCm:
nix build .#checks.x86_64-linux.no-rocm-closure
```

## Target usage

```elixir
defmodule Example do
  import Nx.Defn

  defn predict(x, weights, bias) do
    x |> Nx.dot(weights) |> Nx.add(bias) |> Nx.tanh()
  end
end

predict = NxTinygrad.jit(&Example.predict/3, device: "KFD+AMD:LLVM")
result = predict.(x, weights, bias)
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — the stack, supervision tree, and lifecycle
- [docs/protocol.md](docs/protocol.md) — the XTG1 worker protocol
- [docs/operation-coverage.md](docs/operation-coverage.md) — supported Nx operations
- [docs/amd-nixos.md](docs/amd-nixos.md) — running on AMD without ROCm
- [docs/troubleshooting.md](docs/troubleshooting.md)
- [docs/performance.md](docs/performance.md) — cost breakdown + optimization experiments
- [examples](https://github.com/pikdum/nx_tinygrad/tree/master/examples) and
  [benchmarks](https://github.com/pikdum/nx_tinygrad/tree/master/bench)

## Telemetry

Emits `[:nx_tinygrad, :compile | :execute, :start | :stop]` spans,
`[:nx_tinygrad, :transfer, :upload | :download]`, and
`[:nx_tinygrad, :worker, :restart]`.

## License

MIT — see [LICENSE](LICENSE).
