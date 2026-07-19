# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-07-19

First release: an `Nx.Defn` compiler and tensor backend that runs whole Nx
graphs on tinygrad, verified end-to-end on an AMD RX 7900 XT through
`KFD+AMD:LLVM` with no ROCm in the closure.

### Added

- **M0** — Nix flake providing Elixir 1.20 / OTP 29, Rust, and a ROCm-free
  tinygrad worker Python environment. Project scaffolding, device-string parsing
  (`ExTinygrad.Device`), dtype mapping (`ExTinygrad.Dtype`), a standalone Python
  device probe (`priv/worker/device.py`), and a `no-rocm-closure` flake check.
- **M1** — XTG1 framed wire protocol (`ExTinygrad.Protocol` / `priv/worker/protocol.py`),
  a supervised Erlang Port worker (`ExTinygrad.Worker`) with monotonic generation
  tracking and crash isolation, worker-side buffer registry and stats, and the
  `hello`, `device_info`, `upload`, `download`, `release`, `stats`, `synchronize`,
  and `shutdown` commands.
- **M2** — versioned deterministic graph IR (`ExTinygrad.Graph`) with canonical
  JSON + cache key, Nx `Expr` lowering (`ExTinygrad.Lowering`) covering
  elementwise/comparison/select/shape/reduction/dot ops, an `Nx.Defn.Compiler`
  (`ExTinygrad.Compiler`) running the whole graph through the worker in one
  execute RPC, worker-side graph validation/operations/executable, and the
  `ExTinygrad.jit/2`, `jit_apply/3`, `device_info/1`, `worker_stats/1`,
  `synchronize/1` API. CPU results validated against `Nx.BinaryBackend`.
- **M3** — TinyJit-backed executables: the graph function is captured
  (warmup/capture/validate) at compile time and replayed on execute. Adds an
  in-memory `ExTinygrad.ExecutableCache` keyed by graph + device + versions (so
  identical graphs compile once per worker generation), duplicate-input cloning,
  and output cloning for immutability. One execute RPC per invocation.
- **M4** — `ExTinygrad.Backend` (`Nx.Backend`) keeps tensors resident as worker
  buffers: `from_binary`/`to_binary`/`backend_copy`/`backend_transfer`/
  `backend_deallocate`/`inspect` work, all other ops raise (no silent fallback).
  The compiler defaults to `output: :device`, passes device-resident inputs by
  handle, and reconstructs arbitrary containers (`ExTinygrad.OutputContainer`).
  Tensors carry a worker generation; a restart makes them stale
  (`ExTinygrad.StaleTensorError`). Adds `ExTinygrad.release/1`.
- **M5** — Rustler NIF (`native/ex_tinygrad_ref`) providing a `TensorRef`
  resource that owns only reference metadata. Its `Drop` pushes releases onto a
  native queue; `ExTinygrad.ReleaseReaper` drains it and sends batched releases
  to workers, discarding stale generations. Explicit release uses `take/1` so GC
  cannot double-free. Verified by a leak test over 1000 dropped device tensors.
- **M7** — autograd via Nx: `Nx.Defn.value_and_grad` graphs lower and execute
  with the existing op set; validated against `Nx.BinaryBackend` for a
  linear+tanh loss and a 2-layer MLP (inference, gradients, and a loss-reducing
  gradient step).
- **M6** — AMD `KFD+AMD:LLVM` path verified end-to-end on an RX 7900 XT
  (gfx1100): device_info, f32 elementwise/matmul/softmax parity, MLP
  value_and_grad parity, device-resident persistence + output immutability, a
  10k-iteration buffer-lifecycle test, and a `/proc/self/maps` check that no
  ROCm/HIP/comgr library is loaded. GPU tests are gated behind
  `EX_TINYGRAD_GPU_TESTS=1`.
- **M8** — telemetry spans (`compile`/`execute`) and events
  (`transfer.upload`/`transfer.download`, `worker.restart`); docs
  (architecture, protocol, operation coverage, AMD-on-NixOS, troubleshooting);
  runnable examples; and benchmarks (matmul, MLP, bridge overhead, direct
  tinygrad baseline).
