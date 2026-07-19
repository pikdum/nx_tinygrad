"""Graph IR parsing and validation.

The worker treats the graph as untrusted input from a compiler service: every
operation name is checked against a fixed allowlist, dtypes and shapes are
validated, node inputs must reference already-defined ids, and the node count is
bounded. No Python source is ever evaluated and no modules are imported based on
graph content.
"""
from __future__ import annotations

from dtype import is_supported
from errors import GraphValidationError
from operations import SUPPORTED_OPS

MAX_NODES = 200_000
MAX_RANK = 16
GRAPH_VERSION = 1


def _check(cond, message, **details):
    if not cond:
        raise GraphValidationError(message, details=details)


def _valid_shape(shape):
    return (
        isinstance(shape, list)
        and len(shape) <= MAX_RANK
        and all(isinstance(d, int) and d >= 0 for d in shape)
    )


def validate(graph: dict, blob_count: int, max_nodes: int = MAX_NODES) -> None:
    _check(isinstance(graph, dict), "graph must be an object")
    _check(graph.get("version") == GRAPH_VERSION, f"unsupported graph version: {graph.get('version')}")

    for key in ("inputs", "constants", "nodes", "outputs"):
        _check(isinstance(graph.get(key), list), f"graph.{key} must be a list")

    _check(len(graph["nodes"]) <= max_nodes, "graph exceeds maximum node count", limit=max_nodes)

    defined: set[int] = set()

    def define(entity, ident):
        _check(isinstance(ident, int) and ident >= 0, f"invalid id in {entity}", id=ident)
        _check(ident not in defined, f"duplicate id {ident} in {entity}")
        defined.add(ident)

    for inp in graph["inputs"]:
        define("input", inp.get("id"))
        _check(isinstance(inp.get("index"), int) and inp["index"] >= 0, "input index must be a non-negative int")
        _check(_valid_shape(inp.get("shape")), "invalid input shape", shape=inp.get("shape"))
        _check(is_supported(inp.get("dtype")), f"unsupported input dtype: {inp.get('dtype')}")

    for const in graph["constants"]:
        define("constant", const.get("id"))
        _check(_valid_shape(const.get("shape")), "invalid constant shape", shape=const.get("shape"))
        _check(is_supported(const.get("dtype")), f"unsupported constant dtype: {const.get('dtype')}")
        has_value = "value" in const
        has_data = "data_index" in const
        _check(has_value != has_data, "constant must have exactly one of value/data_index")
        if has_data:
            _check(
                isinstance(const["data_index"], int) and 0 <= const["data_index"] < blob_count,
                "constant data_index out of range",
                data_index=const.get("data_index"),
            )

    for node in graph["nodes"]:
        define("node", node.get("id"))
        _check(node.get("op") in SUPPORTED_OPS, f"unsupported Nx operation: {node.get('op')}", op=node.get("op"))
        _check(isinstance(node.get("inputs"), list), "node inputs must be a list")
        for ref in node["inputs"]:
            _check(ref in defined, f"node {node['id']} references undefined id {ref}")
        _check(isinstance(node.get("attrs"), dict), "node attrs must be an object")
        _check(_valid_shape(node.get("shape")), "invalid node shape", shape=node.get("shape"))
        _check(is_supported(node.get("dtype")), f"unsupported node dtype: {node.get('dtype')}")

    _check(len(graph["outputs"]) >= 1, "graph must have at least one output")
    for out in graph["outputs"]:
        _check(out.get("node") in defined, f"output references undefined id {out.get('node')}")
