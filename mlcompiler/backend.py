"""CPU execution plans and explicit sample-based numerical validation."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
from typing import Callable, Protocol
import torch
from .ir import DTYPES, Diagnostic, Graph, Node, TensorSpec


def evaluate(node: Node, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    kind = node.kind
    if kind == "add": return args[0] + args[1]
    if kind == "subtract": return args[0] - args[1]
    if kind == "multiply": return args[0] * args[1]
    if kind == "matmul": return torch.matmul(*args)
    if kind == "relu": return torch.relu(args[0])
    if kind == "identity": return args[0]
    if kind == "reshape": return args[0].reshape(tuple(node.attrs["shape"]))
    if kind == "transpose": return args[0].permute(tuple(node.attrs["perm"]))
    if kind == "sum":
        dims = node.attrs.get("dims")
        return args[0].sum(dim=None if dims is None else tuple(dims), keepdim=node.attrs.get("keepdim", False))
    if kind == "fused_affine_relu":
        # Dispatch grouping only on this backend. PyTorch still materializes intermediates.
        return torch.relu(args[0] * args[1] + args[2])
    raise Diagnostic(f"CPU backend does not implement {kind}")


class Backend(Protocol):
    name: str
    def compile(self, graph: Graph) -> Callable[..., tuple[torch.Tensor, ...]]: ...


class CPUPlan:
    def __init__(self, graph: Graph):
        self.graph = graph.validate()
        self.input_nodes = tuple(n for n in graph.ops if n.kind == "input")
        self.constants = {n.output: n.value for n in graph.ops if n.kind == "constant"}
        self.steps = graph.compute

    @torch.no_grad()
    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(args) != len(self.input_nodes):
            raise Diagnostic(f"Expected {len(self.input_nodes)} inputs, got {len(args)}")
        values = dict(self.constants)
        for node, arg in zip(self.input_nodes, args):
            if not isinstance(arg, torch.Tensor) or arg.device.type != "cpu" or arg.requires_grad:
                raise Diagnostic(f"%{node.output}: expected a CPU inference tensor without gradients")
            if arg.dtype not in DTYPES.values():
                raise Diagnostic(f'%{node.output}: unsupported runtime dtype {arg.dtype}')
            node.spec.check(arg, node.output)
            values[node.output] = arg
        for node in self.steps:
            try:
                values[node.output] = evaluate(node, tuple(values[n] for n in node.inputs))
                node.spec.check(values[node.output], node.output)
            except (RuntimeError, IndexError, TypeError) as exc:
                raise Diagnostic(f"%{node.output} ({node.kind}): {exc}") from exc
        return tuple(values[name] for name in self.graph.outputs)

    def describe(self) -> str:
        rows = ["CPU / PyTorch tensor execution plan (inference)"]
        for i, node in enumerate(self.steps):
            rows.append(f"{i:02d} %{node.output} ← {node.kind}({', '.join(node.inputs)})")
        return "\n".join(rows + ["Fused affine nodes group dispatch; CPU intermediates remain.",
                                 "No Triton execution or GPU launch counts are inferred."])


class CPUBackend:
    name = "cpu-torch-interpreter"
    def compile(self, graph: Graph) -> CPUPlan:
        return CPUPlan(infer_shapes(graph))


def infer_shapes(graph: Graph) -> Graph:
    """Static meta-tensor propagation where input shapes are fully known.

    No concrete input values are evaluated. This is not a numerical check.
    """
    graph.validate()
    values, nodes = {}, []
    for node in graph.ops:
        if node.kind == "constant":
            value = torch.empty_like(node.value, device="meta")
        elif node.kind == "input":
            spec = node.spec
            if spec.shape is None or None in spec.shape or spec.dtype is None:
                nodes.append(node)
                continue
            value = torch.empty(spec.shape, dtype=spec.dtype, device="meta")
        else:
            if any(name not in values for name in node.inputs):
                nodes.append(node)
                continue
            try:
                value = evaluate(node, tuple(values[name] for name in node.inputs))
            except (RuntimeError, IndexError, TypeError) as exc:
                raise Diagnostic(f"Static shape analysis failed at %{node.output}: {exc}") from exc
        node.spec.check(value, node.output)
        values[node.output] = value
        nodes.append(replace(node, spec=TensorSpec.of(value)))
    return replace(graph, ops=tuple(nodes)).validate()


@dataclass(frozen=True)
class Validation:
    passed: bool
    samples: int
    rtol: float
    atol: float
    message: str


def compare_outputs(expected, actual, *, rtol=1e-5, atol=1e-6) -> None:
    if not all(math.isfinite(t) and t >= 0 for t in (rtol, atol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if len(expected) != len(actual):
        raise AssertionError("Output arity differs")
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol, equal_nan=True)
        # Signed zeros are observable under the documented strict semantics.
        zeros = (left == 0) & (right == 0)
        if left.is_floating_point() and not torch.equal(torch.signbit(left[zeros]), torch.signbit(right[zeros])):
            raise AssertionError("Signed-zero outputs differ")


def validate(original: Graph, candidate: Graph, samples, backend: Backend | None = None,
             *, rtol=1e-5, atol=1e-6) -> Validation:
    backend = backend or CPUBackend()
    if not samples:
        return Validation(False, 0, rtol, atol, "Not tested: no input samples supplied")
    try:
        left, right = backend.compile(original), backend.compile(candidate)
        for args in samples:
            compare_outputs(left(*args), right(*args), rtol=rtol, atol=atol)
    except (AssertionError, ValueError, RuntimeError) as exc:
        return Validation(False, len(samples), rtol, atol, str(exc))
    return Validation(True, len(samples), rtol, atol, "Sample-based runtime equivalence passed; not a proof")
