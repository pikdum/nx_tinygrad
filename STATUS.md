# STATUS

Living record of pinned versions and milestone progress. Updated as work lands.

## Pinned versions

| Thing                        | Value                                             |
| ---------------------------- | ------------------------------------------------- |
| Nx version                   | 0.13.0                                            |
| tinygrad version             | 0.13.0 (nixpkgs `python3Packages.tinygrad`)       |
| Python version               | 3.14.6                                            |
| LLVM version                 | nixpkgs `llvmPackages.llvm` (AMD LLVM renderer)  |
| Elixir version               | 1.20.2 (`beam29Packages.elixir_1_20`)             |
| OTP version                  | 29.0.3 (`beam29Packages.erlang`)                  |
| Rust version                 | 1.96.1 (nixpkgs `rustc`)                          |
| nixpkgs                      | nixos-unstable (locked in flake.lock)             |
| Tested GPU                   | AMD Radeon RX 7900 XT (gfx1100, RDNA3)            |
| Tested tinygrad device       | `DEV=KFD+AMD:LLVM` (native)                       |

## Device string note

On tinygrad 0.13, `KFD+AMD:LLVM` is a **native** `DEV` string: the interface
prefix (`KFD+`) and renderer suffix (`:LLVM`) are part of it. nx_tinygrad passes
it straight through as `DEV` and creates tensors on the backend (`AMD`). The old
`AMD_IFACE`/`AMD_LLVM` environment variables are deprecated in 0.13. See
`NxTinygrad.Device` / `priv/worker/device.py`.

## Milestones

- [x] **M0** — flake, scaffolding, device probe, no-ROCm closure check.
- [x] **M1** — framed protocol + supervised CPU worker (24 Elixir + 11 Python tests).
- [x] **M2** — graph IR + Nx lowering, CPU end-to-end, BinaryBackend parity (44 Elixir + 23 Python tests).
- [x] **M3** — TinyJit capture/replay, executable cache, one execute RPC/call (53 Elixir + 25 Python tests).
- [x] **M4** — device tensor backend, device-resident I/O, output containers, stale-tensor errors (66 Elixir tests).
- [x] **M5** — Rustler tensor-ref resource, GC-triggered release queue, reaper, leak test (68 Elixir tests).
- [x] **M7** — Nx `value_and_grad` parity, MLP inference + training step (CPU, 4 tests).
- [x] **M6** — AMD `KFD+AMD:LLVM` path: smoke, parity, persistence, no-ROCm-loaded (9 GPU tests on RX 7900 XT).
- [x] **M8** — telemetry, docs, examples, benchmarks; 73 Elixir + 25 Python tests, flake checks green. Release 0.1.0.

## Commands / results

Recorded per milestone as they run (see git history for details).

### M0

- `nix build .#checks.x86_64-linux.no-rocm-closure` — verifies the worker
  closure contains no ROCm/HIP/comgr paths.
- `python priv/worker/device.py CPU` — device probe smoke test.
