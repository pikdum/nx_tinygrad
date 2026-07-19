#!/usr/bin/env python3
"""Direct tinygrad matmul baseline, to compare against ex_tinygrad warm replay.

    DEV=CPU python bench/direct_tinygrad.py
    DEV=AMD AMD_LLVM=1 AMD_IFACE=KFD python bench/direct_tinygrad.py   # RX 7900 XT

The architecture acceptance target: warm ex_tinygrad replay within ~20% of this
for the same captured graph, when inputs and outputs are device-resident.
"""
import os
import time

import numpy as np
from tinygrad import Tensor, TinyJit

N = int(os.environ.get("N", "1024"))
DEV = os.environ.get("DEV", "CPU")


def mk(v):
    return Tensor(np.full((N, N), v, dtype=np.float32)).realize()


@TinyJit
def matmul(a, b):
    return (a @ b).realize()


a, b = mk(0.001), mk(0.002)
# warmup: normal, capture, replay
for _ in range(3):
    matmul(mk(0.001), mk(0.002))

iters = 20
t0 = time.perf_counter()
for _ in range(iters):
    out = matmul(a, b)
out.numpy()  # force completion
elapsed_ms = (time.perf_counter() - t0) / iters * 1000.0

print(f"== direct tinygrad {N}x{N} f32 matmul on {DEV} ==")
print(f"warm replay: {elapsed_ms:.3f} ms/call")
