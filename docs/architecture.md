# Architecture

```text
Elixir / Nx / Axon
        │  Nx.Defn expression
        ▼
NxTinygrad.Compiler  (Nx.Defn.Compiler)
        │  NxTinygrad.Lowering
        ▼
NxTinygrad.Graph  — versioned, deterministic tensor-graph IR (canonical JSON)
        │  XTG1 framed protocol over an Erlang Port (packet: 4)
        ▼
priv/worker/main.py  — supervised Python worker (one OS process)
        │  graph.py → operations.py → executable.py
        ▼
tinygrad Tensor graph wrapped in TinyJit (capture once, replay)
        │
        ▼
tinygrad AMD runtime → Linux amdgpu driver via /dev/kfd  (or CPU / other devices)
```

## Why a separate OS process

The Python worker runs as its own process behind an Erlang Port, not embedded in
the BEAM:

- A Python or native-library crash cannot take down the BEAM (the supervisor
  restarts the worker).
- The GIL and interpreter lifecycle stay isolated.
- The Python dependency closure (tinygrad, numpy, LLVM) is separate from the
  Elixir release.
- GPU state has one clear owner.

The Port is opened with `packet: 4` binary framing; the worker's stdout carries
only protocol frames, and all logging goes to stderr.

## Supervision tree

```text
NxTinygrad.Supervisor
├── Registry (NxTinygrad.WorkerRegistry)
├── NxTinygrad.WorkerIds        # worker name <-> integer id (for the NIF)
├── NxTinygrad.ExecutableCache  # graph cache key -> {generation, executable_id}
├── NxTinygrad.WorkerSupervisor
│   └── NxTinygrad.Worker (:default)   # owns one Python Port
└── NxTinygrad.ReleaseReaper    # drains the native release queue
```

Each worker startup gets a strictly increasing **generation**. Every backend
tensor and executable carries the generation of the worker that produced it, so
a reference from a dead generation is never sent to a restarted worker — it
raises `NxTinygrad.StaleTensorError` instead.

## Compilation flow

1. The compiler callback calls the defn function with parameter templates to
   obtain the output expression container.
2. `NxTinygrad.Lowering` walks the expression DAG post-order into the graph IR.
   Unsupported operations raise `NxTinygrad.CompileError` here — before any
   Python is contacted, with no silent host fallback.
3. `NxTinygrad.GraphCacheKey` hashes the canonical graph JSON plus device, Nx and
   tinygrad versions, protocol version, and compile options.
4. On a cache miss, the graph is sent to the worker (`compile`), which validates
   it, builds a tinygrad graph function, and captures it with `TinyJit`.
5. A runtime closure is returned. Each invocation issues exactly one `execute`
   RPC (inputs by handle when already device-resident, else as blobs) and
   reconstructs the original output container.

## Memory lifecycle

Device tensors are backed by `NxTinygrad.TensorRef`, a Rustler resource holding
only `{worker_id, generation, handle}`. When the resource is garbage-collected,
its Rust `Drop` pushes a release onto a native queue (never blocking, never
touching the Port). `NxTinygrad.ReleaseReaper` drains the queue and sends batched
`release` requests. Explicit release uses `take/1` so GC cannot double-free.
