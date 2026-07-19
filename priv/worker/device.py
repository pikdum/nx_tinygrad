"""Device-string parsing, environment configuration, probing, and ROCm-load
detection for the ex_tinygrad worker.

Logical device strings follow the spec: ``[<IFACE>+]<BACKEND>[:<RENDERER>]``,
e.g. ``KFD+AMD:LLVM``, ``CPU``, ``AMD``.

IMPORTANT: tinygrad 0.12.0's ``Device[...]`` does NOT accept ``KFD+AMD:LLVM``
literally — it splits on ``:`` and treats the prefix (``KFD+AMD``) as the device
class name, which does not exist. So we translate the logical string into a
concrete tinygrad device name plus environment variables here.

Some of those environment variables (notably ``AMD_LLVM`` and ``AMD_IFACE``) are
read by tinygrad as ContextVars *at import time*, so they must be applied BEFORE
tinygrad is imported. Callers must therefore call :func:`apply_env` before the
first ``import tinygrad``.
"""
from __future__ import annotations

import os

from errors import DeviceUnavailable

# Libraries that would indicate a ROCm/HIP/comgr runtime got loaded.
ROCM_LIBS = (
    "libamdhip64",
    "libhsa-runtime64",
    "libamd_comgr",
    "librocblas",
    "libMIOpen",
)


def parse_device(spec: str | None) -> dict:
    """Parse a logical device string into a concrete tinygrad configuration."""
    spec = (spec or "").strip() or "CPU"

    rest = spec
    iface = None
    if "+" in rest:
        iface, rest = rest.split("+", 1)

    renderer = None
    backend = rest
    if ":" in rest:
        backend, renderer = rest.split(":", 1)

    backend = backend.upper()
    iface = iface.upper() if iface else None
    renderer = renderer.upper() if renderer else None

    env: dict[str, str] = {}
    if backend == "AMD":
        # Never default to PCI/USB — those interfaces can unbind the card from
        # the amdgpu driver. Force KFD unless an interface was requested.
        env["AMD_IFACE"] = iface or "KFD"
        iface = env["AMD_IFACE"]
        if renderer in (None, "LLVM"):
            env["AMD_LLVM"] = "1"
            renderer = "LLVM"

    return {
        "spec": spec,
        "backend": backend,
        "interface": iface,
        "renderer": renderer,
        "tinygrad_device": backend,
        "env": env,
    }


def apply_env(parsed: dict) -> None:
    """Apply the device's environment. Existing values win (Elixir may have set
    them on the Port already)."""
    for key, value in parsed["env"].items():
        os.environ.setdefault(key, value)


def loaded_rocm_libraries() -> dict:
    """Report which ROCm/HIP/comgr shared libraries are mapped into this
    process, by scanning /proc/self/maps."""
    loaded = {name: False for name in ROCM_LIBS}
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as fh:
            maps = fh.read()
    except OSError:
        return loaded
    for name in ROCM_LIBS:
        if name in maps:
            loaded[name] = True
    return loaded


def probe(spec: str | None) -> dict:
    """Initialize the requested device and run a tiny smoke computation.

    Returns a device_info dict (see spec §20) augmented with a report of any
    ROCm libraries that got loaded.
    """
    parsed = parse_device(spec)
    apply_env(parsed)

    # Import only after the environment is configured.
    from tinygrad import Tensor, Device  # noqa: E402

    tg_device = parsed["tinygrad_device"]

    try:
        dev = Device[tg_device]
    except Exception as exc:  # noqa: BLE001
        raise DeviceUnavailable(
            f"could not open tinygrad device {tg_device!r} for {parsed['spec']!r}: {exc}",
            details={"requested": parsed["spec"]},
        ) from exc

    # Smoke computation: x * 2 == [2, 4, 6].
    try:
        x = Tensor([1, 2, 3], device=tg_device)
        y = (x * 2).realize()
        usable = y.tolist() == [2, 4, 6]
    except Exception as exc:  # noqa: BLE001
        raise DeviceUnavailable(
            f"smoke computation failed on {parsed['spec']!r}: {exc}",
            details={"requested": parsed["spec"]},
        ) from exc

    arch = getattr(dev, "arch", None)
    selected = getattr(dev, "device", tg_device)

    return {
        "requested": parsed["spec"],
        "selected": selected,
        "renderer": parsed["renderer"],
        "interface": parsed["interface"],
        "architecture": arch,
        "device_name": arch or selected,
        "usable": usable,
        "rocm_libraries_loaded": loaded_rocm_libraries(),
    }


if __name__ == "__main__":
    # Standalone probe: `python device.py [DEVICE]` prints device_info JSON.
    import json
    import sys

    requested = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DEV", "CPU")
    try:
        info = probe(requested)
        print(json.dumps(info, indent=2))
    except DeviceUnavailable as err:
        print(json.dumps({"error": err.error_class, "message": err.message}, indent=2))
        sys.exit(1)
