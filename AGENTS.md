# AGENTS.md

Guidance for coding agents working on `nx_tinygrad` — an Elixir `Nx` compiler and
tensor backend that runs whole Nx graphs on [tinygrad](https://tinygrad.org)
(AMD via native KFD + LLVM, no ROCm).

## Dev environment

- NixOS + flake. `direnv` auto-loads the dev shell (`.envrc` = `use flake`), so run
  `mix`, `elixir`, `python`, `cargo` directly in the repo — no `nix develop` wrapper.
- The dev shell provides Elixir 1.20 / OTP 29, Rust, and a ROCm-free tinygrad
  Python (`NX_TINYGRAD_PYTHON`). `MIX_REBAR3` points at a working rebar3 (the
  OTP-29 build is broken upstream).
- Toolchain is pinned via `flake.lock` (nixos-unstable). See `STATUS.md` for versions.

## Structure

- `lib/nx_tinygrad/` — Elixir: `compiler.ex` (`Nx.Defn.Compiler`), `lowering.ex`
  (Expr → graph IR), `graph.ex`, `backend.ex` (`Nx.Backend`, device tensors),
  `worker.ex` (Port GenServer), `protocol.ex` (XTG1 frames), `executable_cache.ex`,
  `release_reaper.ex`, `tensor_ref.ex` (Rustler NIF stub), supervision tree.
- `priv/worker/` — Python worker: `main.py` (loop/dispatch), `protocol.py`,
  `graph.py` (validation), `operations.py` (op table), `executable.py` (TinyJit),
  `device.py`, `dtype.py`.
- `native/nx_tinygrad_ref/` — Rust NIF: GC-triggered buffer release queue.
- `test/` (CPU + pure), `test/gpu/` (`@moduletag :gpu`), `worker_tests/` (Python),
  `bench/`, `examples/`, `docs/`, `SPEC.md`.

## Common commands

- Tests (CPU + pure): `mix test`
- GPU tests (needs an AMD device): `NX_TINYGRAD_GPU_TESTS=1 mix test test/gpu`
- Python worker tests: `python -m pytest -q worker_tests`
- Format: `mix format`
- Benchmarks: `NX_TINYGRAD_GPU_TESTS=1 mix run bench/nx_backends.exs`
- Device probe: `python priv/worker/device.py "KFD+AMD:LLVM"`
- Flake checks: `nix flake check` (incl. `no-rocm-closure`, `python-tests`)

## Conventions

- Conventional Commits. Keep the tree runnable and tests green per change.
- **Never silently fall back to another backend** inside a compiled graph —
  unsupported ops raise `NxTinygrad.CompileError` at compile time.
- Keep the Elixir and Python sides of the **dtype** and **device** mappings in
  sync (`lib/nx_tinygrad/dtype.ex` ↔ `priv/worker/dtype.py`, `device.ex` ↔
  `device.py`).
- Every external handle carries a worker **generation**; a stale handle raises
  `NxTinygrad.StaleTensorError`, never wrong data.
- Add a regression test for any correctness bug found.

## Gotchas

- **Device string:** on tinygrad 0.13, `KFD+AMD:LLVM` is a native `DEV` string
  (interface prefix + renderer suffix). We pass it through; the old
  `AMD_IFACE`/`AMD_LLVM` env vars are deprecated. Tensors live on backend `AMD`.
- **No ROCm:** the default worker closure must stay ROCm-free (enforced by the
  `no-rocm-closure` flake check + a `/proc/self/maps` runtime check). Don't enable
  nixpkgs `rocmSupport`.
- **Perf:** the Erlang↔Python transport is ~44 µs — not the bottleneck. Costs are
  Python/tinygrad-side (see `docs/performance.md`). When benchmarking device work,
  `synchronize()` (don't rely on lazy `realize()`, and note `.numpy()` includes a
  download).
- The worker uses tinygrad **internals** in a few hot spots (`immutable_copy`,
  JIT capture) — these shift between tinygrad versions; re-verify on bumps.

## Change checklist

- `mix format`, then `mix test` (and `python -m pytest -q worker_tests`).
- If the change touches the GPU/AMD path, run `NX_TINYGRAD_GPU_TESTS=1 mix test test/gpu`.
- If it touches the worker closure or flake, run `nix flake check`.
- Update `STATUS.md` / `CHANGELOG.md` / `docs/` when behavior or versions change.
