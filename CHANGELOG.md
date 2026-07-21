# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Operation-coverage march toward a general-purpose, swap-in `Nx.Backend`. Each new
primitive is verified against `Nx.BinaryBackend` in `test/differential_test.exs`.

### Added

- **Symbolic while-body JIT** — `while` bodies whose only runtime scalars are
  dynamic `slice`/`put_slice` starts (SD's denoise loop, iterative linalg) are
  now TinyJit-captured whole: each start becomes a bound tinygrad `Variable`
  passed as a jit argument (so replay rebinds it via `var_vals`), with the
  scalar index chain evaluated eagerly and clamped host-side per iteration.
  Loop-invariant vars (how Nx carries closed-over weights through `while`)
  bypass the jit — no per-iteration multi-GB clone of model weights. Any
  capture/replay failure falls back to node-by-node interpretation (never
  wrong data), counted in worker stats (`while_steps_*`,
  `while_jit_fallbacks`) so tinygrad API drift can't silently degrade the
  fast path.

- **Control flow & multi-output**: `while` (dynamic loops, eager worker-side
  execution — unblocks generation and Axon training loops), `cond` (predicated
  selects), `elem` + tuple-valued blocks (unlocks `top_k` and non-iterative
  `Nx.LinAlg` composites like `determinant`).
- **Convolution backward**: permuted/dilated `conv`, so conv gradients (w.r.t.
  kernel and input) lower — unblocks CNN training.
- **`window_scatter_max`/`window_scatter_min`** (select-and-scatter) — completes
  **max-pool backward**, so full CNN training (conv + maxpool gradients) works.
- **`cholesky`** — iterative linalg via the `while` path.
- **`bf16` dtype** — rides a uint16 transport carrier, bitcast to tinygrad
  bfloat16 in the worker (HF bfloat16 checkpoints load without host conversion).
- **Complex numbers (`c64`/`c128`) + `fft`/`ifft`** — complex tensors are held
  as paired reals (`S+[2]`); covers complex arithmetic, `conjugate`, `real`,
  `imag`, `abs`, `exp`, shape ops, `dot`, and `fft`/`ifft` (DFT matmul).
- **`triangular_solve`** — unrolled forward/back substitution.
- **Full `Nx.LinAlg` family** — `cholesky`, `qr`, `lu`, `svd`, `eigh`,
  `determinant` (via while/cond/dynamic-put_slice + tuple projection).
- **Custom-function `reduce` and `window_reduce`** — fold the traced reducer body
  over the (statically unrolled) reduced axes / windows.
- **Dynamic slice**: `slice` with runtime (tensor) start indices.
- Elementwise: `erf_inv`, `count_leading_zeros`, `population_count`,
  `conjugate` (real), `bitcast`.
- Integration examples under `examples/` (Mix.install): Axon MLP training,
  Bumblebee text classification (DistilRoBERTa), Bumblebee image classification
  (ResNet-50), and **Stable Diffusion v1.4 text-to-image** — the CLIP text
  encoder, the UNet denoiser, and the VAE decoder all compiled by nx_tinygrad,
  verified end-to-end on CPU and the AMD RX 7900 XT (a coherent 512×512 image).
- **Weight residency for large models** — `NxTinygrad.Compiler.__to_backend__/1`
  now resolves the backend to the same worker the executable runs on (honoring
  `:device`/`:worker`), so weights preallocated via `Nx.backend_copy` /
  Bumblebee's `preallocate_params: true` upload to the device **once** and are
  passed by handle every call, instead of being re-shipped as a multi-GB frame
  per invocation. This is what makes Stable Diffusion (~4 GB of UNet weights,
  called once per denoise step) runnable at all.

- Elementwise unary ops: `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`,
  `acosh`, `atanh`, `erf`, `cbrt`, `sign`, `is_nan`, `is_infinity`, `bitwise_not`.
  Most map directly to tinygrad primitives; `cbrt` is composed (magnitude root +
  sign) since tinygrad has none.
- Elementwise binary ops: `bitwise_and`, `bitwise_or`, `bitwise_xor`,
  `left_shift`, `right_shift`, `logical_and`, `logical_or`, `logical_xor`,
  `remainder`, `quotient`, `atan2` (`remainder`/`quotient` composed to Nx's
  truncated-division semantics; `atan2` reconstructed from `atan` with quadrant
  correction).
- Reductions: `product`, `argmax`, `argmin` (`argmax`/`argmin` honor `axis`,
  `keep_axis`, and both `:low`/`:high` tie-breaks), and windowed reductions
  `window_sum`/`window_max`/`window_min`/`window_product` (strides, per-edge
  padding with each reduction's identity, and window dilation, via tinygrad
  `_pool`).
- Shape / indexing: `reverse`, `pad` (edge padding with finite or ±infinity
  fill; interior/negative padding raise for now), `sort`, `argsort`, `iota`,
  `gather` (Nx coordinate-gather semantics), `clip`, `stack`, `eye`. `take`,
  `take_along_axis`, and `tile` come from existing primitives (block path /
  reshape+broadcast).
- Elementwise: `round` (composed to Nx's half-away-from-zero rounding, vs
  tinygrad's half-to-even), `erfc`.
- Elementwise: `erf_inv` (Giles rational approximation), `count_leading_zeros`,
  `population_count` (bit-width-aware), and `conjugate` (identity on real inputs).
- **Multi-output `:elem` + tuple blocks** — projecting an element out of a
  tuple-returning block/op. Unlocks `top_k` and any tuple-valued composite.
- **`cond`** — pure clauses lowered to a right-folded chain of predicated
  `select`s.
- Non-iterative `Nx.LinAlg` composites that decompose to the supported op set
  (e.g. `determinant`) now lower.
- New-op parity is verified on the real AMD RX 7900 XT in
  `test/gpu/amd_ops_test.exs` (conv, pooling, indexing, scatter, sort,
  cumulative, `top_k`, `cond`, extended elementwise).
- `conv` — maps Nx's default-layout convolution onto tinygrad `conv2d` (general
  over spatial rank), honoring strides, per-edge (asymmetric) padding, kernel
  dilation, and feature groups. Input dilation (transposed conv), non-identity
  tensor permutations, and batch grouping raise `NxTinygrad.CompileError`.
- Scatter: `put_slice` (composed from `pad` + `select`; compile-time start
  offsets), `indexed_add`, `indexed_put` (coordinate scatter over `:axes` via
  tinygrad `scatter_reduce`/`scatter`; duplicate indices accumulate for
  `indexed_add`).
- **Generic `:block` lowering** — optional Nx ops that carry a pre-traced pure
  default expression (e.g. `cumulative_sum`, `cumulative_product`,
  `cumulative_max`, `cumulative_min`) are lowered by binding their inputs to the
  default expression and lowering that, composing them from existing primitives.
  Blocks without a pure default raise `NxTinygrad.CompileError`; the impure
  callback is never executed.

### Fixed

- Relaxed the `nx` requirement from `~> 0.13.0` to `~> 0.12` so the Bumblebee
  `examples/*.exs` (`Mix.install`) can resolve — Bumblebee 0.7 requires
  `nx ~> 0.12.0`, which the single-line `0.13` pin excluded, making those
  examples unresolvable since they were added. `mix test` still runs on the
  newest permitted nx (0.13); the examples resolve nx 0.12.
- All `examples/` now honor `NX_TINYGRAD_DEVICE` (default `CPU`), so every
  example runs on either CPU or the AMD GPU (`KFD+AMD:LLVM`). Previously
  `basic`, `matmul`, `mlp_inference`, and `mlp_training` were hardwired to the
  default CPU worker and could not target the GPU.
- `bumblebee_text_classification` now uses
  `j-hartmann/emotion-english-distilroberta-base` — the previous
  `finiteautomata/bertweet-base-sentiment-analysis` ships no Rust-compatible
  `tokenizer.json`, so Bumblebee 0.7 failed at `load_tokenizer` before the
  compiler ran.
- Verified end-to-end on both CPU and the AMD RX 7900 XT: all seven examples
  (basic, matmul, MLP inference/training, Axon MLP training, ResNet-50 image
  classification, DistilRoBERTa emotion classification) run through the compiler
  with matching numerics on both devices.

## [0.1.0] - 2026-07-19

First release: an `Nx.Defn` compiler and tensor backend that runs whole Nx
graphs on tinygrad, verified end-to-end on an AMD RX 7900 XT through
`KFD+AMD:LLVM` with no ROCm in the closure.

Toolchain (nixos-unstable): Elixir 1.20.2 / OTP 29.0.3, tinygrad 0.13.0,
Python 3.14, Rust 1.96, Nx 0.13. On tinygrad 0.13 `KFD+AMD:LLVM` is a native
`DEV` string (interface prefix + renderer suffix), so `NxTinygrad.Device` passes
it straight through — the deprecated `AMD_IFACE`/`AMD_LLVM` env vars are gone.

### Added

- **M0** — Nix flake providing Elixir 1.20 / OTP 29, Rust, and a ROCm-free
  tinygrad worker Python environment. Project scaffolding, device-string parsing
  (`NxTinygrad.Device`), dtype mapping (`NxTinygrad.Dtype`), a standalone Python
  device probe (`priv/worker/device.py`), and a `no-rocm-closure` flake check.
- **M1** — XTG1 framed wire protocol (`NxTinygrad.Protocol` / `priv/worker/protocol.py`),
  a supervised Erlang Port worker (`NxTinygrad.Worker`) with monotonic generation
  tracking and crash isolation, worker-side buffer registry and stats, and the
  `hello`, `device_info`, `upload`, `download`, `release`, `stats`, `synchronize`,
  and `shutdown` commands. The default worker starts lazily and can run either a
  configured Python interpreter or the flake's packaged worker executable.
- **M2** — versioned deterministic graph IR (`NxTinygrad.Graph`) with canonical
  JSON + cache key, Nx `Expr` lowering (`NxTinygrad.Lowering`) covering
  elementwise/comparison/select/shape/reduction/dot ops, an `Nx.Defn.Compiler`
  (`NxTinygrad.Compiler`) running the whole graph through the worker in one
  execute RPC, worker-side graph validation/operations/executable, and the
  `NxTinygrad.jit/2`, `jit_apply/3`, `device_info/1`, `worker_stats/1`,
  `synchronize/1` API. CPU results validated against `Nx.BinaryBackend`.
- **M3** — TinyJit-backed executables: the graph function is captured
  (warmup/capture/value validation) at compile time and replayed on execute. Adds
  a bounded in-memory `NxTinygrad.ExecutableCache` keyed by graph, inline constant
  contents, worker, device, and versions, duplicate-input cloning, and output
  cloning for immutability. One execute RPC per invocation.
- **M4** — `NxTinygrad.Backend` (`Nx.Backend`) keeps tensors resident as worker
  buffers: `from_binary`/`to_binary`/`backend_copy`/`backend_transfer`/
  `backend_deallocate`/`inspect` work, all other ops raise (no silent fallback).
  The compiler defaults to `output: :device`, passes device-resident inputs by
  handle, and reconstructs arbitrary containers (`NxTinygrad.OutputContainer`).
  Tensors carry a worker generation; a restart makes them stale
  (`NxTinygrad.StaleTensorError`). Adds `NxTinygrad.release/1`.
- **M5** — Rustler NIF (`native/nx_tinygrad_ref`) providing a `TensorRef`
  resource that owns only reference metadata. Tensor and compiled-executable
  resources push releases onto native queues; `NxTinygrad.ReleaseReaper` drains
  them and sends batched releases to workers, discarding stale generations.
  Explicit tensor release uses `take/1` so GC cannot double-free.
- **M7** — autograd via Nx: `Nx.Defn.value_and_grad` graphs lower and execute
  with the existing op set; validated against `Nx.BinaryBackend` for a
  linear+tanh loss and a 2-layer MLP (inference, gradients, and a loss-reducing
  gradient step).
- **M6** — AMD `KFD+AMD:LLVM` path verified end-to-end on an RX 7900 XT
  (gfx1100): device_info, f32 elementwise/matmul/softmax parity, MLP
  value_and_grad parity, device-resident persistence + output immutability, a
  10k-iteration buffer-lifecycle test, and a `/proc/self/maps` check that no
  ROCm/HIP/comgr library is loaded. GPU tests are gated behind
  `NX_TINYGRAD_GPU_TESTS=1`.
- **M8** — telemetry spans (`compile`/`execute`) and events
  (`transfer.upload`/`transfer.download`, `worker.restart`); docs
  (architecture, protocol, operation coverage, AMD-on-NixOS, troubleshooting);
  runnable examples; benchmarks (matmul, MLP, bridge overhead, direct tinygrad
  baseline); release packaging checks; and public CI.
