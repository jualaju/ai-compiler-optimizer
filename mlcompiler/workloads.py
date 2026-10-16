"""Deterministic inference workloads and disjoint parameter/input seeds."""
from __future__ import annotations
from dataclasses import dataclass
import time
import torch
from torch import nn
from .backend import CPUBackend, compare_outputs
from .frontends import native, pytorch
from .ir import Graph


@dataclass
class Workload:
    name: str
    family: str
    graph: Graph
    samples: tuple[tuple[torch.Tensor, ...], ...]
    evaluation_inputs: tuple[torch.Tensor, ...]
    frontend_ms: float
    reference_checked: bool
    seed: int
    frontend: str = 'native'


NAMES = ("affine", "constant", "dead_branch", "redundant_relu", "layout", "residual", "mlp")


def build(name="affine", size=256, seed=7, source="native") -> Workload:
    if name not in NAMES or size < 1 or source not in ("native", "torch"):
        raise ValueError(f"Choose workload {NAMES}, positive size, source native/torch")
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    xshape = (size, 16) if name == "mlp" else (size,)
    examples = (torch.randn(xshape, generator=generator),)
    calibration2 = (torch.randn(xshape, generator=generator),)
    heldout = (torch.randn(xshape, generator=generator),)
    dtype = "float32"
    nodes = [{"name": "x", "op": "input", "shape": list(xshape), "dtype": dtype}]
    def c(name, value, role='constant'):
        nodes.append({"name": name, "op": "constant", "dtype": dtype, "value": value,
                      'metadata': {'storage': 'constant', 'role': role}})
    def n(name, op, *inputs, **attrs):
        nodes.append({"name": name, "op": op, "inputs": list(inputs), "attrs": attrs})
        return name
    if name in ("affine", "constant", "dead_branch", "residual"):
        c("scale", 1.5)
        c("bias", 0.5, 'bias')
    if name in ("affine", "constant", "dead_branch"):
        scale = "scale"
        if name == "constant":
            c("extra", 2.0)
            scale = n("foldable", "multiply", "scale", "extra")
        output = n("output", "relu", n("shift", "add", n("product", "multiply", "x", scale), "bias"))
        if name == "dead_branch":
            n("unused", "matmul", "x", "x")
    elif name == "redundant_relu":
        output = n("output", "relu", n("first", "relu", "x"))
    elif name == "layout":
        reshaped = n("matrix", "reshape", "x", shape=[size, 1])
        transposed = n("transposed", "transpose", reshaped, perm=[1, 0])
        output = n("output", "transpose", transposed, perm=[1, 0])
    elif name == "residual":
        first = n("product", "multiply", "x", "scale")
        relu = n("relu", "relu", n("shift", "add", first, "bias"))
        output = n("output", "add", relu, first)
    else:
        w1 = torch.randn(16, 32, generator=generator) * 0.1
        w2 = torch.randn(32, 4, generator=generator) * 0.1
        c("w1", w1.tolist(), 'weight'); c("w2", w2.tolist(), 'weight')
        c("b1", torch.randn(32, generator=generator).tolist(), 'bias'); c("b2", [0.1] * 4, 'bias')
        hidden = n("hidden", "relu", n("affine", "add", n("dense", "matmul", "x", "w1"), "b1"))
        output = n("output", "add", n("readout", "matmul", hidden, "w2"), "b2")
    graph = native({"version": 1, "name": name, "nodes": nodes, "outputs": [output]})
    constants = {n.output: n.value for n in graph.ops if n.kind == "constant"}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            for key, value in constants.items(): self.register_buffer(key, value.clone())
        def forward(self, x):
            if name == "affine": return torch.relu(x * self.scale + self.bias)
            if name == "constant": return torch.relu(x * (self.scale * self.extra) + self.bias)
            if name == "dead_branch":
                unused = x @ x
                return torch.relu(x * self.scale + self.bias)
            if name == "redundant_relu": return torch.relu(torch.relu(x))
            if name == "layout": return x.reshape(size, 1).permute(1, 0).permute(1, 0)
            if name == "residual":
                product = x * self.scale
                return torch.relu(product + self.bias) + product
            return torch.relu(x @ self.w1 + self.b1) @ self.w2 + self.b2

    model = Model().eval()
    if source == "torch": graph = pytorch(model, examples)
    plan = CPUBackend().compile(graph)
    with torch.no_grad():
        for sample in (examples, calibration2, heldout):
            compare_outputs((model(*sample),), plan(*sample))
    return Workload(name, name, graph, (examples, calibration2), heldout,
                    (time.perf_counter() - started) * 1000, True, seed, source)


def suite(source='native', seed=7):
    """Family-level split is recorded for future learning; no model is trained."""
    for name in NAMES:
        for size in (32, 2048):
            yield build(name, size=size, seed=seed, source=source), ("development" if name in
                ("affine", "constant", "redundant_relu", "layout") else "heldout_family")
