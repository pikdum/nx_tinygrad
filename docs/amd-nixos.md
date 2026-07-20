# AMD on NixOS without ROCm

nx_tinygrad drives AMD GPUs through tinygrad's **native KFD interface** and
compiles kernels with **libLLVM** (`AMD_LLVM=1`). It needs none of ROCm, HIP,
comgr, rocBLAS, or MIOpen.

## Requirements

- Linux `amdgpu` kernel driver
- Accessible `/dev/kfd`
- Accessible `/dev/dri/renderD*`
- LLVM ≥ 18 with the AMDGPU target (provided by nixpkgs `llvmPackages.llvm`)

Verified on an **AMD Radeon RX 7900 XT (gfx1100, RDNA3)**.

## Device string

On tinygrad 0.13, `KFD+AMD:LLVM` is a **native** `DEV` string — the interface
prefix (`KFD+`) and renderer suffix (`:LLVM`) are part of it:

```text
KFD+AMD:LLVM  ->  DEV=KFD+AMD:LLVM   (interface KFD, backend AMD, renderer LLVM)
                  tensors are created on backend "AMD"
```

`NxTinygrad.Device` passes the string through as `DEV` (defaulting a bare `AMD`
to KFD + LLVM; never PCI/USB, which can unbind amdgpu). `DEV` is read at import
time, so the worker Port is started with it already set. The old `AMD_IFACE` /
`AMD_LLVM` environment variables are deprecated in tinygrad 0.13.

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
NX_TINYGRAD_GPU_TESTS=1 mix test test/gpu
```

These start a `:amd` worker on `KFD+AMD:LLVM` and cover device info, f32 parity,
device-resident persistence, output immutability, and a 10k-iteration buffer
lifecycle test.

## Note on the LLVM vs comgr path

nx_tinygrad uses the LLVM renderer (the `:LLVM` in `KFD+AMD:LLVM`), so kernels are
compiled with libLLVM and the entire ROCm/HIP/comgr stack is unnecessary. We
build plain `python3Packages.tinygrad` (no `rocmSupport`), keeping the closure
ROCm-free.
