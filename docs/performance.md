# Performance notes

Where the time goes, and what we did about it. All numbers are on the reference
box (Ryzen 5 5600X + Radeon RX 7900 XT / gfx1100, tinygrad 0.13), for a warm
compiled graph with device-resident inputs.

## Cost breakdown of one `execute`

Measured with tiny graphs so fixed overheads dominate:

| Segment | Cost |
| --- | ---: |
| Bare Port round-trip (`stats`/`synchronize`, no compute) | **~44 µs** |
| tinygrad `TinyJit` replay (`ex.run`) | ~112 µs |
| output immutability copy, old `clone().realize()` | ~660–1000 µs |
| output immutability copy, raw SDMA transfer (new) | ~330 µs |
| full `execute`, `output: :host` (tiny) | ~376 µs |
| full `execute`, `output: :device` (before → after exp. 1) | ~1060 → ~620 µs |

The headline: **the Erlang↔Python transport is ~44 µs — a few percent.** The
per-call floor is Python/tinygrad-side work plus Elixir tensor marshaling, not
the pipe.

## Experiment 1 — cheaper device-output copy ✅ (kept)

`output: :device` must copy each output off the buffer TinyJit reuses across
replays, or a returned handle would mutate on the next call. That copy was
`clone().realize()` — a full non-JIT realize (~1 ms on GPU, size-independent),
which dominated device-mode `execute`.

Replaced with `immutable_copy/1`: a raw device-to-device buffer transfer via the
allocator (HCQ/AMD SDMA), ~3× cheaper (~330 µs) and size-independent. The HCQ
transfer signals the device timeline, so later kernels that read the copy wait
for it — ordering is correct. Two supporting details:

- Outputs are made `contiguous()` **inside the captured graph** (replay speed,
  no-op when already contiguous). A raw buffer copy preserves physical layout, so
  strided outputs — e.g. gradients from `dot`/`transpose` — must be row-major
  first. This was a real bug caught in testing (permuted gradient elements).
- Falls back to `clone().realize()` on devices without `_transfer` (e.g. CPU).

Result: **GPU `output: :device` execute ~1060 µs → ~620 µs.** All parity and
immutability tests still pass.

## Experiment 2 — Pythonx transport instead of a Port ❌ (not adopted)

[Pythonx](https://github.com/livebook-dev/pythonx) embeds CPython in the BEAM via
a NIF, replacing the Port's pipe IPC with in-process calls. It runs on this
NixOS box (uv fetches a standalone CPython) and a bare round-trip measured
**~10 µs vs the Port's ~44 µs**.

But that ~34 µs saving is ~5% of the ~620 µs `execute` — the transport was never
the bottleneck (see the breakdown). Against that small gain:

- **Crash isolation is lost.** With the Port, a tinygrad/KFD/driver segfault
  kills one OS process and the supervisor restarts it (spec §9). With CPython
  embedded, the same segfault takes down the whole BEAM.
- Pythonx manages its **own** uv/pip Python, so the AMD path wouldn't get the
  nix-patched `libLLVM` our worker relies on — it'd be effectively CPU-only
  without more Nix work.

Verdict: not worth it as *the* transport. It could be a nice **opt-in** transport
for CPU-only / tight-Nx↔numpy workflows (a pluggable `NxTinygrad.Transport`
behaviour would slot it in), but it does not move the number people care about.

## What's left on the table

- The immutability copy (~330 µs) is still the largest single execute cost. For
  multi-output graphs it's paid per output; batching the copies (one timeline
  wait for all) would help gradient-heavy graphs. Worker stats expose
  `immutable_copy_fast` and `immutable_copy_fallback` so tinygrad API drift cannot
  silently hide a lost fast path.
- Elixir-side marshaling (~220 µs of the host-mode path) — tensor construction,
  `TensorRef` NIF calls per input/output.
- Nothing here is transport-bound, so effort should stay on the Python/tinygrad
  and marshaling sides.

## Large-model / Stable Diffusion breakdown (2026-07-21)

Running `examples/stable_diffusion.exs` (SD v1.4, 20 steps, 1 image) on the RX
7900 XT is ~50–100× slower end-to-end than ComfyUI-on-ROCm. It is **not one
bottleneck** — measured per phase (GPU, 512×512, weights preallocated resident):

| Phase | Cost | Where it lives |
| --- | --- | --- |
| **Weight load** (`Bumblebee.load_model`) | **~386 s** | Upstream: Bumblebee/Nx `BinaryBackend` |
| Preallocate (per-tensor upload RPCs) | ~15 s (1022 tensors, 4.13 GB) | `backend.ex` `from_binary` |
| Compile (first UNet execute, kernel JIT) | ~10–70 s one-time | tinygrad |
| **Denoise loop** | **~7 s/step, GPU ~94 % idle** | worker `_run_while` eager interpretation |

### 1. Weight load (~386 s) — the dominant cost, and it's upstream

`Bumblebee.load_model` reads the safetensors and remaps params (transpose /
reshape / f16→f32 upcast of ~1 B parameters) on the pure-Elixir `BinaryBackend`,
**single-core** (observed: one beam process pegged at 107 % CPU for 6+ minutes,
no swap). This is inherent to loading a large model without a JIT host backend —
with EXLA those transforms run compiled/vectorized. nx_tinygrad is not involved
(the load happens before the compiler). **f16 does not help** — loading with
`type: :f16` was *slower* (~574 s), since the upcast/cast happens regardless.
Real fixes are upstream-shaped: a compiled host backend for load, or making
nx_tinygrad's backend support the eager movement ops Bumblebee needs at load
(`transpose`/`reshape`/`as_type`) so params load resident on-device.

### 2. Denoise loop (~7 s/step, GPU idle) — the compute cost we own

SD's whole graph runs **eagerly interpreted** (node-by-node Python dispatch, no
TinyJit capture) because the denoise `while` body contains a **dynamic slice**
(the scheduler indexes its alpha/sigma/timestep arrays by the step counter),
which `_requires_eager` taints — see `executable.py` `_run_while`. So the entire
UNet (thousands of ops) is re-interpreted every step. Measured overhead is
~0.9 ms per body op per iteration (synthetic repro), which is why the GPU sits
~94 % idle: the cost is host-side dispatch, not kernels.

**Prototype fix (not landed):** partition the loop body — JIT-capture the static
sub-body (the UNet, which depends only on carried latents + invariant weights,
not the runtime index) and interpret only the dynamic-slice-tainted glue (the
cheap scheduler math). A synthetic SD-shaped loop (big static compute, then
dynamic scheduler glue) went 27 → 9 ms/iter (~3×) with exact parity, and SD's
far larger body should gain much more. **Why it's not landed:** it produced
*silent wrong results* for the iterative-linalg `while`s (cholesky/svd/eigh,
which read *and* update carried state — a matrix — across iterations by a dynamic
index). Restricting to read-only dynamic slices (excluding `put_slice`) did NOT
fix it — those loops use read slices too — so the partition is unsound for any
loop that carries evolving state read via a dynamic index, not just a narrow
op-based subset. A cheap 2-iteration replay-validation can't catch divergence
that first appears at iteration 3+, so it can't be used as the safety net either.
Landing it needs a correctness-robust design: e.g. a full static-region JIT with
proper multi-input capture validation (like `_capture`'s `_dummy`-seeded replay
checks), or tinygrad symbolic `Variable` indexing so the whole body JIT-captures
(works eagerly here but currently breaks under `TinyJit` — `KeyError`). The
working prototype (partition + snapshot-safe validation) is preserved for review.

Attempted and rejected: making the dynamic slice itself JIT-capturable via a
tinygrad symbolic `Variable` — `t.shrink(((v, v+len),))` binds correctly eagerly,
but `TinyJit` fails to propagate the binding (`KeyError`) in this tinygrad.

### 3. Preallocate (~15 s) — minor, byte-bound

`Nx.backend_copy` uploads each of ~1022 param tensors as its own `from_binary`
RPC. Per-RPC overhead is ~44 µs (~45 ms total), so the ~15 s is the 4.13 GB byte
movement + per-tensor `realize`, not the round-trip count — batching RPCs would
not help much.
