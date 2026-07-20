# Troubleshooting

## The worker won't start

- `NxTinygrad.Config.python_executable/0` resolves the interpreter from
  `NX_TINYGRAD_PYTHON` (set by the Nix devshell), falling back to `python3` on
  `PATH`. Inside `nix develop` this is set automatically.
- Run the probe directly: `python priv/worker/device.py CPU` (or `KFD+AMD:LLVM`).
  It prints a device_info JSON, or an error.

## `NxTinygrad.CompileError: unsupported Nx operation: ...`

The operation is not lowered in v0.1 (see [operation-coverage.md](operation-coverage.md)).
This is intentional — there is no silent host fallback. Restructure the defn to
use supported operations, or add the op to `NxTinygrad.Lowering` +
`priv/worker/operations.py`.

## `NxTinygrad.StaleTensorError`

A device tensor outlived its worker (the worker restarted, so its buffers are
gone). Its data cannot be recovered. Recompute from host-resident inputs; cached
graphs recompile transparently on the new generation.

## Eager op raises `NxTinygrad eager operations are not supported`

`NxTinygrad.Backend` only moves data; it does not run eager math. Wrap the
computation in `NxTinygrad.jit/2` (or `Nx.Defn.jit/2`), or transfer the tensor to
`Nx.BinaryBackend` first with `Nx.backend_transfer/1`.

## AMD device not found / not usable

- Check `/dev/kfd` and `/dev/dri/renderD*` exist and are accessible.
- Confirm the `amdgpu` kernel driver is loaded.
- `rocminfo` (from your system, not required at runtime) should list the GPU.
- The first AMD run downloads register-definition headers into
  `~/.cache/tinygrad`; it needs network once, then works offline.

## First AMD run prints a flood of URLs

That's tinygrad's `tqdm` download progress for the AMD register headers (piped
output concatenates the progress line). Set `CI=1` to silence it; subsequent runs
are cached and quiet.

## Rust NIF fails to build

The `nx_tinygrad_ref` crate needs `cargo`/`rustc` and a C toolchain, all provided
by the devshell. The first `mix compile` fetches Rust crates from crates.io
(needs network once). `MIX_REBAR3` in the devshell points at a working rebar3
(the OTP-29 build is broken upstream).
