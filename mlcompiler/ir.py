"""Value-oriented, inference-only common IR. No frontend objects are retained."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import json
from math import prod
from typing import Any

import torch


class Diagnostic(ValueError):
    """A user-facing unsupported-workload or graph-invariant diagnostic."""


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int | None, ...] | None = None
    dtype: torch.dtype | None = None

    def check(self, value: torch.Tensor, name: str) -> None:
        if self.dtype is not None and value.dtype != self.dtype:
            raise Diagnostic(f"%{name}: expected {self.dtype}, got {value.dtype}")
        if self.shape is not None and (len(self.shape) != value.ndim or any(
            dim is not None and dim != actual for dim, actual in zip(self.shape, value.shape)
        )):
            raise Diagnostic(f"%{name}: expected shape {self.shape}, got {tuple(value.shape)}")

    @classmethod
    def of(cls, value: torch.Tensor) -> TensorSpec:
        return cls(tuple(value.shape), value.dtype)


@dataclass(frozen=True)
class Node:
    output: str
    kind: str
    inputs: tuple[str, ...] = ()
    spec: TensorSpec = field(default_factory=TensorSpec)
    attrs: dict[str, Any] = field(default_factory=dict)
    value: torch.Tensor | None = field(default=None, repr=False, compare=False)
    source: str = "native"
    metadata: dict[str, Any] = field(default_factory=dict)


# Strict arities; attributes are validated separately. Pure tensor operations only.
ARITY = {"input": 0, "constant": 0, "add": 2, "multiply": 2, "subtract": 2,
         "matmul": 2, "relu": 1, "identity": 1, "reshape": 1, "transpose": 1,
         "sum": 1, "fused_affine_relu": 3}
DTYPES = {"float32": torch.float32, "float64": torch.float64, "int64": torch.int64,
          "int32": torch.int32, "bool": torch.bool}


@dataclass(frozen=True)
class Graph:
    ops: tuple[Node, ...]
    outputs: tuple[str, ...]
    name: str = "workload"

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(node.output for node in self.ops if node.kind == "input")

    @property
    def compute(self) -> tuple[Node, ...]:
        return tuple(n for n in self.ops if n.kind not in ("input", "constant"))

    def validate(self) -> Graph:
        seen: set[str] = set()
        for node in self.ops:
            if not node.output or node.output in seen:
                raise Diagnostic(f"Duplicate or empty value name: %{node.output}")
            if node.kind not in ARITY:
                raise Diagnostic(f"%{node.output}: unsupported IR operator {node.kind!r}")
            if len(node.inputs) != ARITY[node.kind]:
                raise Diagnostic(f"%{node.output}: {node.kind} expects {ARITY[node.kind]} operands")
            missing = set(node.inputs) - seen
            if missing:
                raise Diagnostic(f"%{node.output}: undefined/forward dependency {sorted(missing)}; graph must be topological")
            if node.spec.dtype is not None and node.spec.dtype not in DTYPES.values():
                raise Diagnostic(f"%{node.output}: unsupported dtype {node.spec.dtype}")
            if node.spec.shape is not None and any(d is not None and (type(d) is not int or d < 0)
                                                   for d in node.spec.shape):
                raise Diagnostic(f"%{node.output}: invalid tensor shape {node.spec.shape}")
            allowed = {"reshape": {"shape"}, "transpose": {"perm"},
                       "sum": {"dims", "keepdim"}}.get(node.kind, set())
            if set(node.attrs) - allowed:
                raise Diagnostic(f"%{node.output}: unsupported attributes {set(node.attrs) - allowed}")
            if node.kind == "reshape":
                shape = node.attrs.get("shape")
                if not isinstance(shape, (list, tuple)) or any(type(d) is not int or d < -1 for d in shape) or list(shape).count(-1) > 1:
                    raise Diagnostic(f"%{node.output}: reshape requires a static shape with at most one -1")
            if node.kind == "transpose":
                perm = node.attrs.get("perm")
                if not isinstance(perm, (list, tuple)) or sorted(perm) != list(range(len(perm))):
                    raise Diagnostic(f"%{node.output}: transpose requires a complete permutation")
                if node.spec.shape is not None and len(perm) != len(node.spec.shape):
                    raise Diagnostic(f"%{node.output}: permutation rank mismatch")
            if node.kind == "sum":
                dims = node.attrs.get("dims")
                if dims is not None and (not isinstance(dims, (list, tuple)) or any(type(d) is not int for d in dims) or len(set(dims)) != len(dims)):
                    raise Diagnostic(f"%{node.output}: sum dims must be distinct integer axes")
                if type(node.attrs.get("keepdim", False)) is not bool:
                    raise Diagnostic(f"%{node.output}: keepdim must be boolean")
            if node.kind == "constant":
                if node.value is None or node.value.device.type != "cpu" or node.value.requires_grad:
                    raise Diagnostic(f"%{node.output}: constant must be a detached CPU tensor")
                node.spec.check(node.value, node.output)
            elif node.value is not None:
                raise Diagnostic(f"%{node.output}: only constants may store a value")
            seen.add(node.output)
        if not self.outputs or set(self.outputs) - seen:
            raise Diagnostic(f"Missing or undefined graph outputs: {self.outputs}")
        return self

    def topological(self) -> tuple[Node, ...]:
        self.validate()
        return self.ops

    def fingerprint(self) -> str:
        """Structural progress/dedup key, including constant bytes and metadata."""
        digest = hashlib.sha256()
        for node in self.ops:
            digest.update(repr((node.output, node.kind, node.inputs, node.spec,
                                sorted(node.attrs.items()))).encode())
            if node.value is not None:
                digest.update(node.value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(repr(self.outputs).encode())
        return digest.hexdigest()

    def __str__(self) -> str:
        rows = [f"graph {self.name}({', '.join('%' + n for n in self.inputs)}):"]
        for node in self.ops:
            spec = f"{node.spec.shape} / {node.spec.dtype}"
            attrs = f" {node.attrs}" if node.attrs else ""
            literal = ""
            if node.value is not None:
                literal = f" value={node.value.flatten()[:6].tolist()}" + ("…" if node.value.numel() > 6 else "")
            rows.append(f"  %{node.output} = {node.kind}({', '.join('%' + n for n in node.inputs)}){attrs} : {spec}{literal}")
        return "\n".join(rows + ["  return " + ", ".join("%" + n for n in self.outputs)])


def analysis(graph: Graph) -> dict[str, Any]:
    graph.validate()
    sizes = []
    for node in graph.compute:
        shape, dtype = node.spec.shape, node.spec.dtype
        sizes.append(None if shape is None or None in shape or dtype is None else
                     prod(shape) * torch.empty((), dtype=dtype).element_size())
    uses = Counter(name for node in graph.ops for name in node.inputs)
    return {"compute_ops": len(graph.compute), "operators": dict(Counter(n.kind for n in graph.compute)),
            "inputs": len(graph.inputs), "constants": sum(n.kind == "constant" for n in graph.ops),
            "shared_values": sum(count > 1 for count in uses.values()),
            "shapes": [n.spec.shape for n in graph.ops if n.kind == "input"],
            "dtypes": sorted({str(n.spec.dtype) for n in graph.ops}),
            "estimated_compute_output_bytes": None if None in sizes else sum(sizes),
            "memory_note": "Sum of logical compute output sizes; not measured peak memory or allocation count"}


def from_legacy(graph: Any) -> Graph:
    """Adapt the preserved demo IR without changing its public API."""
    return Graph(tuple(Node(op.output, op.kind, op.inputs, TensorSpec(op.shape, op.dtype),
                            source="legacy FX") for op in graph.ops), graph.outputs, "legacy_affine").validate()
