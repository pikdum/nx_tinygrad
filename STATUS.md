# STATUS

Living record of pinned versions and milestone progress. Updated as work lands.

## Pinned versions

| Thing                        | Value                                             |
| ---------------------------- | ------------------------------------------------- |
| Nx version                   | 0.13.0                                            |
| tinygrad version             | 0.12.0 (nixpkgs `python3Packages.tinygrad`)       |
| tinygrad commit              | v0.12.0 tag (via nixpkgs)                          |
| Python version               | 3.13.12                                           |
| LLVM version                 | 21.1.x (nixpkgs `llvmPackages.llvm`, AMD_LLVM)    |
| Elixir version               | 1.20.0-rc.4 (`beam29Packages.elixir_1_20`)        |
| OTP version                  | 29.0-rc3 (`beam29Packages.erlang`)                |
| Rust version                 | 1.94.1 (nixpkgs `rustc`)                          |
| nixpkgs rev                  | 07800bee2b362f6c73fe17cb1593a260c5e183c6          |
| Tested GPU                   | AMD Radeon RX 7900 XT (gfx1100, RDNA3)            |
| Tested tinygrad device       | `KFD+AMD:LLVM` -> tinygrad `AMD` + AMD_LLVM=1     |

## Device string note

tinygrad 0.12.0's `Device[...]` does not accept `"KFD+AMD:LLVM"` literally (it
splits on `:` and treats the prefix as a device class). ex_tinygrad translates
the logical string into `tinygrad_device = "AMD"` plus env
`AMD_IFACE=KFD, AMD_LLVM=1`. See `ExTinygrad.Device` / `priv/worker/device.py`.

## Milestones

- [x] **M0** — flake, scaffolding, device probe, no-ROCm closure check.
- [x] **M1** — framed protocol + supervised CPU worker (24 Elixir + 11 Python tests).
- [x] **M2** — graph IR + Nx lowering, CPU end-to-end, BinaryBackend parity (44 Elixir + 23 Python tests).
- [x] **M3** — TinyJit capture/replay, executable cache, one execute RPC/call (53 Elixir + 25 Python tests).
- [ ] **M4** — device tensor backend + output containers.
- [ ] **M5** — Rustler tensor-ref + release reaper.
- [ ] **M6** — AMD GPU path.
- [ ] **M7** — Nx autograd + MLP.
- [ ] **M8** — docs, benchmarks, release 0.1.0.

## Commands / results

Recorded per milestone as they run (see git history for details).

### M0

- `nix build .#checks.x86_64-linux.no-rocm-closure` — verifies the worker
  closure contains no ROCm/HIP/comgr paths.
- `python priv/worker/device.py CPU` — device probe smoke test.
