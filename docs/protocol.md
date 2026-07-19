# Worker protocol

The Elixir worker (`ExTinygrad.Worker`) talks to the Python worker
(`priv/worker/main.py`) over an Erlang Port opened with `packet: 4`. Every
message is length-prefixed by the Port; inside each message is an **XTG1 frame**.
There is no pickle and no base64 — tensor data travels as raw little-endian
contiguous bytes.

## Frame layout

All integers are big-endian:

```text
4  bytes  magic "XTG1"
8  bytes  request id (u64)
4  bytes  JSON metadata length (u32)
2  bytes  blob count (u16)
2  bytes  reserved
8*N       blob lengths (u64 each)
M         UTF-8 JSON metadata
...       concatenated blob bytes
```

Every request gets exactly one response with the same request id.

- Request metadata: `{"command": "...", "args": {...}}`
- Success: `{"ok": true, "result": {...}}`
- Failure: `{"ok": false, "error": {"class", "message", "details", "python_traceback"}}`

Implemented in `ExTinygrad.Protocol` (Elixir) and `priv/worker/protocol.py`.

## Commands

| Command        | Args                                              | Result |
| -------------- | ------------------------------------------------- | ------ |
| `hello`        | `protocol_version`                                | versions, `device`, `generation` |
| `device_info`  | —                                                 | probe: `selected`, `interface`, `renderer`, `architecture`, `usable`, `rocm_libraries_loaded` |
| `compile`      | `graph`, `validate_capture` (+ constant blobs)    | `executable_id`, `input_specs`, `output_specs`, `compile_ms`, `kernel_count` |
| `execute`      | `executable_id`, `inputs`, `output` (+ blobs)     | `outputs` (specs, and handles or blobs) |
| `upload`       | `shape`, `dtype` (+ 1 blob)                        | buffer `id` |
| `download`     | `id`                                              | tensor spec (+ 1 blob) |
| `release`      | `ids`                                             | `released` count (idempotent) |
| `synchronize`  | —                                                 | `{}` |
| `stats`        | —                                                 | buffer/executable counts, byte counters |
| `shutdown`     | —                                                 | `{}` then the worker exits |

`execute` inputs are either `{"kind": "handle", "id": n}` (device-resident) or
`{"kind": "blob", "blob_index": k, "shape", "dtype"}`. `output` is `"device"`
(returns handles) or `"host"` (returns blobs).

## Security boundary

The worker treats the graph as untrusted compiler input: it validates the graph
version, checks every operation name against a fixed allowlist, validates dtypes
and shapes, requires node inputs to reference already-defined ids, and bounds the
node count. It never evaluates Python source or imports modules based on graph
content. See `priv/worker/graph.py`.
