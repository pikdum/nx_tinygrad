# ex_tinygrad — Design Specification (condensed)

This is the authoritative design summary. It condenses the full working spec into
the invariants the implementation must hold. Treat it as guidelines that sharpen
as the problem is understood.

## Goal

An `Nx.Defn.Compiler` + `Nx.Backend` that lowers an entire Nx defn expression to
a versioned serialized graph, sends it once to a supervised Python worker, which
builds a tinygrad `Tensor` graph, wraps it in `TinyJit`, captures it, and replays
it. One `execute` RPC per compiled-function invocation. AMD via native KFD +
libLLVM, **no ROCm** in the default closure.

## Hard invariants

1. The whole defn expression becomes **one** graph; never one Python RPC per op.
2. Unsupported ops **fail at compile time** with a useful error — never a silent
   `Nx.BinaryBackend` fallback inside a compiled graph.
3. Every external handle carries a worker **generation**; a handle from an old
   generation is never sent to a restarted worker (→ `StaleTensorError`).
4. Worker crashes never crash the BEAM (isolated OS process behind a Port).
5. Returned Nx tensors are immutable: JIT output buffers are cloned before being
   exposed as persistent handles.
6. The default Nix closure contains no ROCm/HIP/comgr; `/proc/self/maps` shows
   none loaded after device init.
7. Results validate against `Nx.BinaryBackend` within documented f32 tolerances.
8. Dropped tensors are eventually released; no unbounded GPU-buffer growth.

## Architecture

```
ExTinygrad.Supervisor
├── ExTinygrad.ExecutableCache
├── ExTinygrad.WorkerSupervisor
│   └── ExTinygrad.Worker (:default)   # owns one Python Port
└── ExTinygrad.ReleaseReaper
```

Port: `{:spawn_executable, python}` with `[:binary, :exit_status, packet: 4,
args:, env:]`. Python stdout is protocol frames only; all logging is stderr.

## Wire protocol

Outer framing is Port `packet: 4`. Each frame payload:

```
4  bytes  magic "XTG1"
8  bytes  request id (u64, big endian)
4  bytes  JSON metadata length (u32, big endian)
2  bytes  blob count (u16, big endian)
2  bytes  reserved
8*N       blob lengths (u64, big endian)
M         UTF-8 JSON metadata
...       concatenated blob bytes
```

Commands: `hello`, `device_info`, `compile`, `upload`, `execute`, `download`,
`release`, `synchronize`, `stats`, `shutdown`. Every request → exactly one
response with the same id. Success: `{"ok": true, "result": {...}}`. Failure:
`{"ok": false, "error": {"class","message","details","python_traceback"}}`.

No pickle, no base64. Tensor bytes travel as raw little-endian contiguous blobs.

## Graph IR (versioned, deterministic)

```json
{
  "version": 1,
  "inputs":    [{"id", "index", "shape", "dtype"}],
  "constants": [{"id", "value"|"data", "shape", "dtype"}],
  "nodes":     [{"id", "op", "inputs":[ids], "attrs":{}, "shape", "dtype"}],
  "outputs":   [{"node", "shape", "dtype"}]
}
```

- Topological order; sequential deterministic ids (not Nx expression ids).
- Negative axes normalized; attrs use sorted canonical keys.
- Canonicalize JSON before hashing. No executable code or module names.

## Graph cache key

`hash(graph semantics version, canonical graph JSON, input specs, output specs,
Nx version, tinygrad commit, protocol version, device string, compile options)`.

## Dtypes (stable wire names)

v0.1 required: `f32 <-> {:f,32}`, `s32 <-> {:s,32}`, `u8 <-> {:u,8}`. Nx
determines output types; tinygrad results are explicitly cast to satisfy the
serialized output spec. Comparisons produce `u8`. Downloads are little-endian
contiguous.

## Operation coverage (v0.1)

Syntax: parameter, constant, tensor, metadata, elem. Elementwise: add, subtract,
multiply, divide, pow, negate, abs, max, min, exp, expm1, log, log1p, sqrt,
rsqrt, tanh, sigmoid, sin, cos, floor, ceil. Comparison/select: equal, not_equal,
less, less_equal, greater, greater_equal, select. Shape: reshape, squeeze,
broadcast, transpose, concatenate, slice, as_type. Reductions: sum, reduce_max,
reduce_min, all, any. Linear algebra: 2-D matmul, vec-mat, mat-vec, batched
matmul where tinygrad expresses it. Unsupported `Nx.dot` axis configs → detailed
compile error.

## Autograd

No tinygrad autograd in v0.1. `Nx.Defn.value_and_grad` rewrites gradients before
the graph reaches the compiler, so the compiler sees a larger forward graph.

## AMD without ROCm

`DEV=KFD+AMD:LLVM` → tinygrad `AMD` device, env `AMD_IFACE=KFD, AMD_LLVM=1`.
Requires amdgpu, `/dev/kfd`, `/dev/dri/renderD*`, LLVM ≥ 18. No HIP/HSA/comgr.
Do not set `HSA_OVERRIDE_GFX_VERSION`. Do not use PCI/USB interfaces by default.

## Milestones

M0 flake+probe · M1 protocol+CPU worker · M2 graph IR+lowering · M3 TinyJit+cache
· M4 device backend+containers · M5 Rustler ref+reaper · M6 AMD · M7 autograd+MLP
· M8 docs+bench+release.

## Definition of done (0.1.0)

`ExTinygrad.jit(Nx.Defn.value_and_grad(&loss/1))` for a small MLP runs on the RX
7900 XT through `KFD+AMD:LLVM`, agrees with `Nx.BinaryBackend` within f32
tolerance, replays a cached captured graph with one execute RPC per call, keeps
intermediates on-device, releases dropped tensors, survives worker restart (stale
tensors raise), and loads no ROCm libraries.
