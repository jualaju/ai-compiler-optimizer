"""A deliberately small FX -> custom IR -> fusion -> Triton compiler."""
from collections import Counter
from dataclasses import dataclass, replace
import operator
from pathlib import Path

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


class AffineReLU(torch.nn.Module):
    def forward(self, x, scale, bias):
        return torch.relu(x * scale + bias)


@dataclass(frozen=True)
class Op:
    output: str
    kind: str
    inputs: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class Graph:
    ops: list[Op]
    outputs: tuple[str, ...]

    @property
    def inputs(self):
        return tuple(op.output for op in self.ops if op.kind == "input")

    def __str__(self):
        lines = [f"inputs: {', '.join(self.inputs)}"]
        for op in self.ops:
            lines.append(f"  %{op.output} = {op.kind}({', '.join('%' + x for x in op.inputs)})"
                         f" : shape={op.shape}, dtype={op.dtype}")
        return "\n".join(lines + [f"outputs: {', '.join('%' + x for x in self.outputs)}"])


def capture(model, examples):
    """Translate only supported FX nodes; no FX nodes remain in our IR."""
    fx = symbolic_trace(model)
    ShapeProp(fx).propagate(*examples)
    kinds = {operator.mul: "multiply", operator.add: "add", torch.relu: "relu"}
    ops, outputs = [], ()
    for node in fx.graph.nodes:
        if node.op == "output":
            values = node.args[0]
            values = values if isinstance(values, tuple) else (values,)
            outputs = tuple(value.name for value in values)
            continue
        if node.op == "placeholder":
            kind = "input"
        elif node.op == "call_function" and node.target in kinds and not node.kwargs:
            kind = kinds[node.target]
        else:
            raise NotImplementedError(f"Unsupported FX node: {node.format_node()}")
        if any(not isinstance(arg, torch.fx.Node) for arg in node.args):
            raise NotImplementedError("Use tensor inputs for scale and bias, not literals")
        meta = node.meta["tensor_meta"]
        ops.append(Op(node.name, kind, tuple(arg.name for arg in node.args),
                      tuple(meta.shape), meta.dtype))
    return Graph(ops, outputs)


def fuse(graph):
    """Fuse single-use multiply -> add -> relu chains; preserve shared values."""
    by_name = {op.output: op for op in graph.ops}
    uses = Counter(name for op in graph.ops for name in op.inputs)
    uses.update(graph.outputs)
    removed, replacements = set(), {}
    for relu in graph.ops:
        if relu.kind != "relu":
            continue
        add = by_name[relu.inputs[0]]
        if add.kind != "add" or uses[add.output] != 1:
            continue
        for index, name in enumerate(add.inputs):
            mul = by_name[name]
            if mul.kind == "multiply" and uses[mul.output] == 1:
                replacements[relu.output] = replace(
                    relu, kind="fused_affine_relu",
                    inputs=mul.inputs + (add.inputs[1 - index],))
                removed.update((mul.output, add.output))
                break
    return Graph([replacements.get(op.output, op) for op in graph.ops
                  if op.output not in removed], graph.outputs)


def interpret(graph, *args):
    """CPU-capable IR interpreter for testing the optimization's semantics."""
    if len(args) != len(graph.inputs):
        raise ValueError("Wrong number of inputs")
    values = dict(zip(graph.inputs, args))
    functions = {"multiply": operator.mul, "add": operator.add,
                 "relu": torch.relu,
                 "fused_affine_relu": lambda x, s, b: torch.relu(x * s + b)}
    for op in graph.ops:
        if op.kind != "input":
            values[op.output] = functions[op.kind](*(values[x] for x in op.inputs))
    return tuple(values[x] for x in graph.outputs)


def generate_triton(graph, path="generated/affine_relu_kernel.py"):
    """Lower one fused op. Support contiguous float32, scalar or same-shape operands."""
    compute = [op for op in graph.ops if op.kind != "input"]
    if len(compute) != 1 or compute[0].kind != "fused_affine_relu":
        raise ValueError("Codegen requires exactly one fused_affine_relu operation")
    op = compute[0]
    specs = {item.output: item for item in graph.ops if item.kind == "input"}
    if set(op.inputs) != set(graph.inputs) or len(graph.inputs) != 3:
        raise ValueError("Codegen requires three distinct tensor inputs")
    if graph.outputs != (op.output,):
        raise ValueError("Codegen requires the fused result as the sole output")
    x, scale, bias = (specs[name] for name in op.inputs)
    if any(item.dtype != torch.float32 for item in (*specs.values(), op)):
        raise ValueError("Codegen supports float32 only")
    if x.shape != op.shape or any(item.shape not in ((), x.shape) for item in (scale, bias)):
        raise ValueError("Scale/bias must be scalar (0-D) or the same shape as x")
    scale_load = "tl.load(scale_ptr)" if scale.shape == () else "tl.load(scale_ptr + offsets, mask=mask, other=0.0)"
    bias_load = "tl.load(bias_ptr)" if bias.shape == () else "tl.load(bias_ptr + offsets, mask=mask, other=0.0)"
    source = f'''# Generated from our custom IR by compiler.generate_triton.
# Operation: %{op.output} = fused_affine_relu({", ".join(op.inputs)})
import triton
import triton.language as tl


@triton.jit
def affine_relu_kernel(x_ptr, scale_ptr, bias_ptr, out_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = {scale_load}
    bias = {bias_load}
    affine = x * scale + bias
    # Preserve NaNs, as torch.relu does.
    result = tl.where(affine < 0.0, 0.0, affine)
    tl.store(out_ptr + offsets, result, mask=mask)
'''
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path
