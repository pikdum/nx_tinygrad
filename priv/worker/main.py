#!/usr/bin/env python3
"""nx_tinygrad Python worker.

Runs as an OS process behind an Erlang Port. Reads framed protocol requests from
stdin, dispatches them against tinygrad, and writes framed responses to stdout.
stdout carries protocol frames only; all logging goes to stderr.

The DEV device string is configured before tinygrad is imported because tinygrad
reads it as a ContextVar at import time.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

# Make sibling modules importable when run directly as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import device as device_mod  # noqa: E402
from errors import WorkerError, ProtocolError  # noqa: E402

PROTOCOL_VERSION = 1
WORKER_VERSION = "0.1.0"


def log(msg: str) -> None:
    print(f"[nx_tinygrad.worker] {msg}", file=sys.stderr, flush=True)


def configure_environment() -> dict:
    spec = os.environ.get("NX_TINYGRAD_DEVICE") or os.environ.get("DEV") or "CPU"
    parsed = device_mod.parse_device(spec)
    device_mod.apply_env(parsed)
    return parsed


class Handler:
    """Dispatches protocol commands. Holds all worker state."""

    def __init__(self, parsed_device: dict):
        self.parsed_device = parsed_device
        self.tg_device = parsed_device["tinygrad_device"]
        self.generation = int(os.environ.get("NX_TINYGRAD_GENERATION", "1"))
        self.should_stop = False
        self._device_info = None

        # Imports that pull in tinygrad happen only after env configuration.
        import numpy as np  # noqa: E402
        from tinygrad import Tensor, Device  # noqa: E402

        from registry import TensorRegistry
        from executable import ExecutableRegistry
        from stats import Stats

        self.np = np
        self.Tensor = Tensor
        self.Device = Device
        self.registry = TensorRegistry()
        self.exec_registry = ExecutableRegistry()
        self.stats = Stats()

        self._dispatch = {
            "hello": self.cmd_hello,
            "device_info": self.cmd_device_info,
            "compile": self.cmd_compile,
            "execute": self.cmd_execute,
            "upload": self.cmd_upload,
            "download": self.cmd_download,
            "release": self.cmd_release,
            "release_executable": self.cmd_release_executable,
            "stats": self.cmd_stats,
            "synchronize": self.cmd_synchronize,
            "shutdown": self.cmd_shutdown,
        }

    # -- dispatch -----------------------------------------------------------

    def handle(self, meta: dict, blobs: list[bytes]) -> tuple[dict, list[bytes]]:
        command = meta.get("command")
        args = meta.get("args") or {}
        fn = self._dispatch.get(command)
        if fn is None:
            return self._error(ProtocolError(f"unknown command: {command!r}", details={"command": command})), []
        try:
            result, out_blobs = fn(args, blobs)
            return {"ok": True, "result": result}, out_blobs
        except WorkerError as exc:
            return self._error(exc), []
        except Exception as exc:  # noqa: BLE001
            return self._error(exc), []

    def _error(self, exc: Exception) -> dict:
        if isinstance(exc, WorkerError):
            klass, message, details = exc.error_class, exc.message, exc.details
        else:
            klass, message, details = type(exc).__name__, str(exc), {}
        tb = traceback.format_exc() if os.environ.get("NX_TINYGRAD_DEBUG_TRACEBACK") == "1" else None
        return {"ok": False, "error": {"class": klass, "message": message, "details": details, "python_traceback": tb}}

    # -- commands -----------------------------------------------------------

    def cmd_hello(self, args, _blobs):
        client_version = args.get("protocol_version")
        if client_version is not None and client_version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"protocol version mismatch: worker={PROTOCOL_VERSION}, client={client_version}",
                details={"worker": PROTOCOL_VERSION, "client": client_version},
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "worker_version": WORKER_VERSION,
            "python_version": sys.version.split()[0],
            "tinygrad_version": self._tinygrad_version(),
            "tinygrad_commit": self._tinygrad_version(),
            "device": self.parsed_device["spec"],
            "generation": self.generation,
        }, []

    def cmd_device_info(self, _args, _blobs):
        if self._device_info is None:
            self._device_info = device_mod.probe(self.parsed_device["spec"])
        return self._device_info, []

    def cmd_compile(self, args, blobs):
        from compiler import compile_graph

        graph = args["graph"]
        validate_capture = bool(args.get("validate_capture", True))
        exec_id = self.exec_registry.allocate_id()
        t0 = time.perf_counter()
        executable = compile_graph(exec_id, graph, blobs, self.tg_device, validate_capture=validate_capture)
        self.exec_registry.put(executable)
        self.stats.compile_count += 1
        return {
            "executable_id": exec_id,
            "input_specs": executable.input_specs,
            "output_specs": executable.output_specs,
            "compile_ms": (time.perf_counter() - t0) * 1000.0,
            "kernel_count": executable.kernel_count,
        }, []

    def cmd_execute(self, args, blobs):
        from dtype import numpy_dtype

        executable = self.exec_registry.get(args["executable_id"])
        output_mode = args.get("output", "device")

        input_tensors = []
        for spec in args["inputs"]:
            kind = spec.get("kind")
            if kind == "handle":
                input_tensors.append(self.registry.get(spec["id"]).tensor)
            elif kind == "blob":
                arr = (
                    self.np.frombuffer(blobs[spec["blob_index"]], dtype=numpy_dtype(spec["dtype"]))
                    .reshape(spec["shape"])
                    .copy()
                )
                tensor = self.Tensor(arr, device=self.tg_device)
                if arr.shape == ():
                    tensor = tensor.clone()
                input_tensors.append(tensor.realize())
            else:
                raise ProtocolError(f"unknown input kind: {kind!r}")

        clones_before = executable.duplicate_input_clones
        outputs = executable.run(input_tensors)
        self.stats.duplicate_input_clones += executable.duplicate_input_clones - clones_before
        self.stats.execute_count += 1

        if output_mode == "host":
            # numpy() copies to the host immediately, detaching from the reused
            # JIT output buffer — safe without an extra device clone.
            specs, out_blobs = [], []
            for tensor, ospec in zip(outputs, executable.output_specs):
                arr = self.np.ascontiguousarray(tensor.numpy(), dtype=numpy_dtype(ospec["dtype"]))
                data = arr.tobytes()
                self.stats.download_bytes += len(data)
                out_blobs.append(data)
                specs.append({"shape": ospec["shape"], "dtype": ospec["dtype"]})
            return {"outputs": specs}, out_blobs

        # Device mode: copy each output to a fresh buffer so the returned handle
        # is immutable across later executions (JIT reuses output buffers).
        from executable import immutable_copy

        specs = []
        for tensor, ospec in zip(outputs, executable.output_specs):
            cloned = immutable_copy(tensor, self.stats)
            nbytes = int(self.np.prod(ospec["shape"], dtype="int64")) * numpy_dtype(ospec["dtype"]).itemsize
            bid = self.registry.put(cloned, ospec["shape"], ospec["dtype"], nbytes)
            specs.append({"id": bid, "shape": ospec["shape"], "dtype": ospec["dtype"]})
        return {"outputs": specs}, []

    def cmd_upload(self, args, blobs):
        from dtype import numpy_dtype

        if not blobs:
            raise ProtocolError("upload requires a tensor blob")
        shape = tuple(args["shape"])
        dtype = args["dtype"]
        arr = self.np.frombuffer(blobs[0], dtype=numpy_dtype(dtype)).reshape(shape).copy()
        tensor = self.Tensor(arr, device=self.tg_device)
        if arr.shape == ():
            tensor = tensor.clone()
        tensor = tensor.realize()
        nbytes = arr.nbytes
        buffer_id = self.registry.put(tensor, shape, dtype, nbytes)
        self.stats.upload_bytes += nbytes
        return {"id": buffer_id, "shape": list(shape), "dtype": dtype}, []

    def cmd_download(self, args, _blobs):
        from dtype import numpy_dtype

        buf = self.registry.get(args["id"])
        arr = buf.tensor.numpy()
        arr = self.np.ascontiguousarray(arr, dtype=numpy_dtype(buf.dtype))
        data = arr.tobytes()
        self.stats.download_bytes += len(data)
        return {"shape": list(buf.shape), "dtype": buf.dtype}, [data]

    def cmd_release(self, args, _blobs):
        ids = args.get("ids", [])
        removed = self.registry.release(ids)
        return {"released": removed}, []

    def cmd_release_executable(self, args, _blobs):
        ids = args.get("ids", [])
        removed = self.exec_registry.release(ids)
        return {"released": removed}, []

    def cmd_stats(self, _args, _blobs):
        s = self.stats.as_dict()
        s.update(
            {
                "buffer_count": self.registry.count(),
                "buffer_bytes": self.registry.total_bytes(),
                "executable_count": self.exec_registry.count(),
                "generation": self.generation,
            }
        )
        return s, []

    def cmd_synchronize(self, _args, _blobs):
        dev = self.Device[self.tg_device]
        if hasattr(dev, "synchronize"):
            dev.synchronize()
        return {}, []

    def cmd_shutdown(self, _args, _blobs):
        self.should_stop = True
        return {}, []

    # -- helpers ------------------------------------------------------------

    def _tinygrad_version(self) -> str:
        try:
            import importlib.metadata as md

            return md.version("tinygrad")
        except Exception:  # noqa: BLE001
            return "unknown"


def main() -> int:
    parsed = configure_environment()
    log(f"starting: device={parsed['spec']} -> tinygrad {parsed['tinygrad_device']} "
        f"(generation {os.environ.get('NX_TINYGRAD_GENERATION', '1')})")

    from protocol import read_packet, write_packet, decode_frame, encode_frame

    handler = Handler(parsed)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        try:
            payload = read_packet(stdin)
        except ProtocolError as exc:
            log(f"fatal protocol error: {exc}")
            return 2
        if payload is None:
            log("stdin closed, exiting")
            return 0

        try:
            req_id, meta, blobs = decode_frame(payload)
        except ProtocolError as exc:
            log(f"frame decode error: {exc}")
            return 2

        response_meta, response_blobs = handler.handle(meta, blobs)
        write_packet(stdout, encode_frame(req_id, response_meta, response_blobs))

        if handler.should_stop:
            log("shutdown requested, exiting")
            return 0


if __name__ == "__main__":
    sys.exit(main())
