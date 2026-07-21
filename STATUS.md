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
- [x] **M1** — framed protocol + supervised CPU worker.
- [x] **M2** — graph IR + Nx lowering, CPU end-to-end, BinaryBackend parity.
- [x] **M3** — TinyJit capture/replay, executable cache, one execute RPC/call.
- [x] **M4** — device tensor backend, device-resident I/O, output containers, stale-tensor errors.
- [x] **M5** — Rustler tensor/executable references, GC-triggered release queues, reaper, leak tests.
- [x] **M7** — Nx `value_and_grad` parity, MLP inference, and a verified parameter update.
- [x] **M6** — AMD `KFD+AMD:LLVM` path: smoke, parity, persistence, and no ROCm loaded on RX 7900 XT.
- [x] **M8** — telemetry, docs, examples, benchmarks, and flake checks. Release 0.1.0.
- [x] **M9** — op-coverage march (full `Nx.Defn` primitive surface, ~97 worker
  ops) + Bumblebee integration suite verified on CPU and the RX 7900 XT: text
  classification, image classification (ResNet-50), Axon MLP training, and
  **Stable Diffusion v1.4 text-to-image** (CLIP + UNet + VAE). Large models are
  made runnable by weight residency: `preallocate_params: true` uploads weights
  once and passes them by handle, rather than re-shipping them every call.
- [x] **M10** — large-model performance: eager device-side backend ops
  (`run_node`) make Bumblebee weight loads land resident (~386 s → ~30 s for
  SD v1.4); symbolic-`Variable` while-body JIT captures dynamic-slice loops
  (denoise runs as TinyJit replays at ~1 s/step, 0 fallbacks; bench 27 →
  ~3.5 ms/iter); top-level segment JIT captures the static regions around
  `while` nodes (text encoder / VAE). SD v1.4 warm generation: ~10+ min →
  **~19 s**, safety checker included — its former ~143 s/image was the
  featurizer's `NxImage.resize` running under `Nx.Defn.Evaluator` on
  `BinaryBackend`; routing plain defn calls through the compiler
  (`Nx.Defn.global_default_options`) makes it ~free. GPU now ~100 % busy
  during generation. See `docs/performance.md`.

## Commands / results

Recorded per milestone as they run (see git history for details).

### M0

- `nix build .#checks.x86_64-linux.no-rocm-closure` — verifies the worker
  closure contains no ROCm/HIP/comgr paths.
- `python priv/worker/device.py CPU` — device probe smoke test.
