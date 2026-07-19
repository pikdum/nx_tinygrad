# AMD on NixOS without ROCm

ex_tinygrad drives AMD GPUs through tinygrad's **native KFD interface** and
compiles kernels with **libLLVM** (`AMD_LLVM=1`). It needs none of ROCm, HIP,
comgr, rocBLAS, or MIOpen.

## Requirements

- Linux `amdgpu` kernel driver
- Accessible `/dev/kfd`
- Accessible `/dev/dri/renderD*`
- LLVM ≥ 18 with the AMDGPU target (provided by nixpkgs `llvmPackages.llvm`)

Verified on an **AMD Radeon RX 7900 XT (gfx1100, RDNA3)**.

## Device string

The spec's device string is `KFD+AMD:LLVM`. tinygrad 0.12.0's `Device[...]` does
**not** accept that literally (it splits on `:` and treats the prefix as a device
class). `ExTinygrad.Device` translates it:

```text
KFD+AMD:LLVM  ->  tinygrad device "AMD"
                  env AMD_IFACE=KFD   (force KFD; never PCI/USB, which can unbind amdgpu)
                  env AMD_LLVM=1      (compile with libLLVM instead of comgr)
```

These env vars are read by tinygrad as ContextVars at import time, so the worker
Port is started with them already set.

`HSA_OVERRIDE_GFX_VERSION` is deliberately **not** set.

## Verifying no ROCm is loaded

`device_info` reports `rocm_libraries_loaded`, produced by scanning
`/proc/self/maps` for `libamdhip64`, `libhsa-runtime64`, `libamd_comgr`,
`librocblas`, and `libMIOpen`. The GPU smoke test passes only when all are
absent.

The Nix flake also ships a build-time check:

```sh
nix build .#checks.x86_64-linux.no-rocm-closure
```

which fails if any ROCm/HIP/comgr path appears in the worker's runtime closure.

## Running the GPU tests

```sh
EX_TINYGRAD_GPU_TESTS=1 mix test test/gpu
```

These start a `:amd` worker on `KFD+AMD:LLVM` and cover device info, f32 parity,
device-resident persistence, output immutability, and a 10k-iteration buffer
lifecycle test.

## Note on the LLVM vs comgr path

nixpkgs' `python3Packages.tinygrad` with `rocmSupport = true` (the comgr path) is
currently broken: its patch fixes `comgr.py` but not `comgr_3.py`, which
tinygrad 0.12.0 actually uses. The LLVM path (`AMD_LLVM=1`) sidesteps this
entirely and is what ex_tinygrad uses by default.
