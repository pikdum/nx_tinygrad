# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
