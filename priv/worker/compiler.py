"""Turns a compile request (graph IR + constant blobs) into an Executable."""
from __future__ import annotations

import graph as graph_mod
from executable import Executable


def compile_graph(exec_id: int, graph: dict, blobs: list[bytes], device: str) -> Executable:
    graph_mod.validate(graph, blob_count=len(blobs))
    return Executable(exec_id, graph, blobs, device)
