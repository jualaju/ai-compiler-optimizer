"""Pure graph rewrites with explicit applicability and guarded fixed-point execution."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Protocol
import torch
from .backend import evaluate, infer_shapes, validate
from .ir import Graph, Node, TensorSpec


@dataclass(frozen=True)
class PassResult:
    graph: Graph
    changes: int
    reason: str


@dataclass(frozen=True)
class Applicability:
    applicable: bool
    reason: str


class OptimizationPass(Protocol):
    name: str
    def check(self, graph: Graph) -> Applicability: ...
    def apply(self, graph: Graph) -> PassResult: ...


class SafePass:
    def check(self, graph: Graph) -> Applicability:
        """Conservative preconditions; apply resolves exact dependency patterns."""
        kinds = Counter(n.kind for n in graph.ops)
        predicates = {
            'constant_folding': kinds['constant'] > 0 and bool(graph.compute),
            'dead_code_elimination': bool(graph.compute) or kinds['constant'] > 0,
            'algebraic_simplification': kinds['identity'] > 0 or kinds['relu'] > 1,
            'layout_simplification': kinds['transpose'] > 0,
            'affine_relu_fusion': all(kinds[k] > 0 for k in ('multiply', 'add', 'relu')),
        }
        eligible = predicates[self.name]
        return Applicability(eligible, 'Operator preconditions met' if eligible else
                             'Required operator preconditions absent')


def substitute(graph: Graph, aliases: dict[str, str], removed: set[str]) -> Graph:
    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name
    return replace(graph, ops=tuple(replace(n, inputs=tuple(resolve(x) for x in n.inputs))
                                    for n in graph.ops if n.output not in removed),
                   outputs=tuple(resolve(x) for x in graph.outputs))


class ConstantFolding(SafePass):
    name = "constant_folding"
    def __init__(self, max_bytes: int = 1024 * 1024):
        self.max_bytes = max_bytes

    def apply(self, graph: Graph) -> PassResult:
        constants, nodes, changes, limited = {}, [], 0, 0
        for node in graph.ops:
            if node.kind == "constant":
                constants[node.output] = node.value
            elif node.kind != "input" and all(n in constants for n in node.inputs):
                # Analyze result size before evaluation to bound expansion allocations.
                from math import prod
                shape, dtype = node.spec.shape, node.spec.dtype
                if shape is None or None in shape or dtype is None or prod(shape) * torch.empty((), dtype=dtype).element_size() > self.max_bytes:
                    limited += 1
                else:
                    with torch.no_grad():
                        value = evaluate(node, tuple(constants[n] for n in node.inputs)).detach().clone()
                    node = Node(node.output, "constant", spec=TensorSpec.of(value), value=value, source=node.source,
                                metadata={'storage': 'folded_constant', 'role': 'constant', 'folded_from': node.kind})
                    constants[node.output] = value
                    changes += 1
            nodes.append(node)
        return PassResult(replace(graph, ops=tuple(nodes)), changes,
                          f"Folded {changes} static expressions; {limited} skipped (unknown/over-budget size)")


class DeadCodeElimination(SafePass):
    name = "dead_code_elimination"
    def apply(self, graph: Graph) -> PassResult:
        live = set(graph.outputs)
        for node in reversed(graph.ops):
            if node.output in live:
                live.update(node.inputs)
        # Preserve input signature even when an argument becomes unused.
        nodes = tuple(n for n in graph.ops if n.output in live or n.kind == "input")
        changes = len(graph.ops) - len(nodes)
        return PassResult(replace(graph, ops=nodes), changes, f"Removed {changes} unreachable pure values; input signature preserved")


class AlgebraicSimplification(SafePass):
    name = "algebraic_simplification"
    def apply(self, graph: Graph) -> PassResult:
        by_name = {n.output: n for n in graph.ops}
        aliases = {}
        for node in graph.compute:
            parent = by_name[node.inputs[0]]
            if node.kind == "identity" or (node.kind == "relu" and parent.kind == "relu"):
                aliases[node.output] = node.inputs[0]
        result = substitute(graph, aliases, set(aliases))
        return PassResult(result, len(aliases),
                          f"Removed {len(aliases)} identity/repeated-ReLU operations; no floating-point reassociation or x+0 rewrite")


class LayoutSimplification(SafePass):
    name = "layout_simplification"
    def apply(self, graph: Graph) -> PassResult:
        by_name = {n.output: n for n in graph.ops}
        aliases = {}
        for node in graph.compute:
            parent = by_name[node.inputs[0]]
            if node.kind == "transpose":
                perm = tuple(node.attrs["perm"])
                if perm == tuple(range(len(perm))):
                    aliases[node.output] = node.inputs[0]
                elif parent.kind == "transpose":
                    old = parent.attrs["perm"]
                    if tuple(old[i] for i in perm) == tuple(range(len(perm))):
                        aliases[node.output] = parent.inputs[0]
        return PassResult(substitute(graph, aliases, set(aliases)), len(aliases),
                          f"Cancelled {len(aliases)} identity/inverse permutations; arbitrary layout conversion is not enabled")


class AffineReLUFusion(SafePass):
    name = "affine_relu_fusion"
    def apply(self, graph: Graph) -> PassResult:
        by_name = {n.output: n for n in graph.ops}
        uses = Counter(x for n in graph.ops for x in n.inputs)
        uses.update(graph.outputs)
        removed, replacements = set(), {}
        for relu in graph.compute:
            if relu.kind != "relu":
                continue
            add = by_name[relu.inputs[0]]
            if add.kind != "add" or uses[add.output] != 1:
                continue
            # Preserve operand order exactly. Commuting NaNs/rounding is unnecessary.
            mul = by_name[add.inputs[0]]
            if mul.kind != "multiply" or uses[mul.output] != 1:
                continue
            replacements[relu.output] = replace(relu, kind="fused_affine_relu", inputs=mul.inputs + (add.inputs[1],))
            removed.update((mul.output, add.output))
        result = replace(graph, ops=tuple(replacements.get(n.output, n) for n in graph.ops if n.output not in removed))
        return PassResult(result, len(replacements),
                          f"Fused {len(replacements)} single-use multiply → add → ReLU chains; "
                          "shared or graph-output intermediates are preserved; operand order must match. "
                          "CPU groups dispatch, not tensor allocation")


PASS_TYPES = (ConstantFolding, DeadCodeElimination, AlgebraicSimplification,
              LayoutSimplification, AffineReLUFusion)
PASS_NAMES = tuple(cls.name for cls in PASS_TYPES)


@dataclass(frozen=True)
class PassRecord:
    name: str
    iteration: int
    status: str
    changes: int
    reason: str
    validated: bool
    before_ops: int
    after_ops: int
    applicable: bool = False
    candidate: str = ''


@dataclass(frozen=True)
class PassSnapshot:
    record_index: int
    before: Graph
    after: Graph
    candidate: str = ''


@dataclass(frozen=True)
class OptimizationResult:
    graph: Graph
    records: tuple[PassRecord, ...]
    converged: bool
    snapshots: tuple[PassSnapshot, ...] = ()


class PassManager:
    def __init__(self, passes: tuple[OptimizationPass, ...] | None = None, max_iterations=8):
        self.passes = passes if passes is not None else tuple(cls() for cls in PASS_TYPES)
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.max_iterations = max_iterations

    def run(self, graph: Graph, samples=(), *, backend=None, rtol=1e-5, atol=1e-6,
            capture_trace=False) -> OptimizationResult:
        graph = infer_shapes(graph)
        current, records, seen = graph, [], {graph.fingerprint()}
        snapshots = []
        for iteration in range(1, self.max_iterations + 1):
            progress = False
            for optimization in self.passes:
                previous = current
                before_ops = len(current.compute)
                try:
                    precondition = optimization.check(current) if hasattr(optimization, 'check') else Applicability(True, 'Custom pass')
                    if not precondition.applicable:
                        records.append(PassRecord(optimization.name, iteration, 'skipped', 0,
                            precondition.reason, False, before_ops, before_ops))
                        if capture_trace:
                            snapshots.append(PassSnapshot(len(records)-1, previous, current))
                        continue
                    result = optimization.apply(current)
                    candidate = infer_shapes(result.graph)
                    changed = candidate.fingerprint() != current.fingerprint()
                    if not changed:
                        status, reason, checked = "skipped", result.reason + "; no structural progress", False
                    elif not samples:
                        status, reason, checked = "skipped", "Applicable, but no samples: transformation not accepted", False
                    elif candidate.fingerprint() in seen:
                        status, reason, checked = "failed", "Repeated graph detected; transformation rolled back", False
                    else:
                        check = validate(current, candidate, samples, backend, rtol=rtol, atol=atol)
                        checked = check.passed
                        status = "applied" if checked else "failed"
                        reason = result.reason if checked else "Rolled back: " + check.message
                        if checked:
                            current = candidate
                            seen.add(current.fingerprint())
                            progress = True
                    records.append(PassRecord(optimization.name, iteration, status,
                        result.changes if status == "applied" else 0, reason, checked,
                        before_ops, len(current.compute), changed))
                except (ValueError, RuntimeError, AssertionError, IndexError, TypeError) as exc:
                    records.append(PassRecord(optimization.name, iteration, "failed", 0,
                                              f"Rolled back: {exc}", False, before_ops, before_ops))
                if capture_trace:
                    snapshots.append(PassSnapshot(len(records)-1, previous, current))
            if not progress:
                return OptimizationResult(current, tuple(records), True, tuple(snapshots))
        records.append(PassRecord("fixed_point", self.max_iterations, "failed", 0,
                                  "Iteration bound reached; convergence not established", False,
                                  len(current.compute), len(current.compute)))
        return OptimizationResult(current, tuple(records), False, tuple(snapshots))


def make_passes(names) -> tuple[OptimizationPass, ...]:
    registry = {cls.name: cls for cls in PASS_TYPES}
    unknown = set(names) - set(registry)
    if unknown:
        raise ValueError(f"Unknown passes {sorted(unknown)}; available: {', '.join(PASS_NAMES)}")
    return tuple(registry[name]() for name in names)
