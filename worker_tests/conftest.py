"""Make the worker modules importable from either the repo layout
(``priv/worker``) or the Nix check layout (``worker``)."""
import os
import sys

# Worker unit tests run on CPU. Without this, tinygrad's device auto-detection
# could pick the AMD GPU on this machine.
os.environ.setdefault("DEV", "CPU")

_here = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    os.path.join(_here, "..", "priv", "worker"),
    os.path.join(_here, "..", "worker"),
):
    if os.path.isdir(candidate):
        sys.path.insert(0, os.path.abspath(candidate))
        break
