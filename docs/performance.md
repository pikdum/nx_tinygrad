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
