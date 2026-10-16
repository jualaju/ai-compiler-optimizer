"""Explicitly bounded frontend conversions to the common IR."""
from __future__ import annotations
from dataclasses import replace
import json
import operator
from pathlib import Path
from typing import Any
import torch
from torch import nn
from .backend import infer_shapes
from .ir import Diagnostic, DTYPES, Graph, Node, TensorSpec


def native(data: dict | str | Path) -> Graph:
    """Load a version-1 JSON graph (or dictionary); constants use JSON arrays."""
    if not isinstance(data, dict):
        data = json.loads(Path(data).read_text())
    if data.get("version") != 1:
        raise Diagnostic("Native format requires version: 1")
    nodes = []
    try:
        for entry in data["nodes"]:
            dtype = DTYPES[entry["dtype"]] if "dtype" in entry else None
            shape = tuple(entry["shape"]) if entry.get("shape") is not None else None
            value = None
            if entry["op"] == "constant":
                if dtype is None:
                    raise Diagnostic("Native constants require an explicit dtype")
                value = torch.tensor(entry["value"], dtype=dtype)
                if shape is not None:
                    value = value.reshape(shape)
                shape = tuple(value.shape)
            nodes.append(Node(entry["name"], entry["op"], tuple(entry.get("inputs", ())),
                              TensorSpec(shape, dtype), dict(entry.get("attrs", {})), value, "native",
                              dict(entry.get('metadata', {}))))
        return infer_shapes(Graph(tuple(nodes), tuple(data["outputs"]), data.get("name", "native")))
    except (KeyError, TypeError) as exc:
        raise Diagnostic(f"Invalid native graph field: {exc}") from exc


def pytorch(model: nn.Module, examples: tuple[torch.Tensor, ...]) -> Graph:
    """FX inference frontend; tensors/parameters become CPU snapshots.

    Supported modules: Linear, ReLU, Identity. No training/dropout,
    mutation, data-dependent control flow, or general Python execution.
    """
    if any(n.training for n in model.modules() if isinstance(n, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d))):
        raise Diagnostic("Training/stateful modules are unsupported; use inference eval mode")
    try:
        fx = torch.fx.symbolic_trace(model)
    except Exception as exc:
        raise Diagnostic(f"FX tracing failed (only static tensor graphs supported): {exc}") from exc
    nodes, names = [], set()
    inputs = iter(examples)

    def emit(name, kind, args=(), attrs=None, value=None, spec=None, source="FX", metadata=None):
        if name in names:
            raise Diagnostic(f"FX generated duplicate name {name}")
        names.add(name)
        nodes.append(Node(name, kind, tuple(args), spec or TensorSpec(), attrs or {}, value, source, metadata or {}))
        return name

    def constant(name, value, *, source='FX literal', metadata=None):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        value = value.detach().cpu().clone()
        return emit(name, "constant", value=value, spec=TensorSpec.of(value), source=source,
                    metadata=metadata or {'storage': 'constant', 'role': 'constant'})

    def operand(arg, name):
        if isinstance(arg, torch.fx.Node): return arg.name
        if type(arg) in (int, float, bool): return constant(name, arg)
        raise Diagnostic(f"FX operand {arg!r} is unsupported")

    def linear(name, x, weight, bias):
        weight_t = emit(name + "__weight_t", "transpose", (weight,), {"perm": [1, 0]})
        product = emit(name + "__matmul" if bias else name, "matmul", (x, weight_t))
        if bias: emit(name, "add", (product, bias))

    kinds = {operator.add: "add", torch.add: "add", operator.mul: "multiply", torch.mul: "multiply",
             operator.sub: "subtract", torch.sub: "subtract", operator.matmul: "matmul",
             torch.matmul: "matmul", torch.relu: "relu", torch.nn.functional.relu: "relu"}
    outputs = None
    for node in fx.graph.nodes:
        source = f"FX {node.op}:{node.target}"
        if node.op == "placeholder":
            try: value = next(inputs)
            except StopIteration as exc: raise Diagnostic("Not enough FX example inputs") from exc
            emit(node.name, "input", spec=TensorSpec.of(value), source=source)
        elif node.op == "get_attr":
            value = fx
            for part in str(node.target).split("."):
                value = getattr(value, part)
            if not isinstance(value, torch.Tensor):
                raise Diagnostic(f"{source}: only tensor attributes supported")
            parameter = str(node.target) in dict(fx.named_parameters())
            role = str(node.target).split('.')[-1]
            constant(node.name, value, source=source, metadata={
                'storage': 'parameter' if parameter else 'buffer',
                'role': role if role in ('weight', 'bias') else 'unknown',
                'parameter_name': str(node.target), 'snapshot': True})
        elif node.op == "output":
            items = node.args[0]
            items = items if isinstance(items, (tuple, list)) else (items,)
            if not all(isinstance(item, torch.fx.Node) for item in items):
                raise Diagnostic("FX outputs must be tensors or a flat tensor sequence")
            outputs = tuple(item.name for item in items)
        elif node.op == "call_module":
            module = fx.get_submodule(node.target)
            if node.kwargs or len(node.args) != 1:
                raise Diagnostic(f"{source}: expected one positional tensor")
            x = operand(node.args[0], node.name + "__x")
            if isinstance(module, nn.Linear):
                weight = constant(node.name + "__weight", module.weight, source=source,
                    metadata={'storage': 'parameter', 'role': 'weight', 'parameter_name': str(node.target)+'.weight', 'snapshot': True})
                bias = constant(node.name + "__bias", module.bias, source=source,
                    metadata={'storage': 'parameter', 'role': 'bias', 'parameter_name': str(node.target)+'.bias', 'snapshot': True}) if module.bias is not None else None
                linear(node.name, x, weight, bias)
            elif isinstance(module, nn.ReLU) and not module.inplace:
                emit(node.name, "relu", (x,), source=source)
            elif isinstance(module, nn.Identity):
                emit(node.name, "identity", (x,), source=source)
            else:
                raise Diagnostic(f"{source}: unsupported module {type(module).__name__}; see docs/OPERATORS.md")
        elif node.op == "call_function" and node.target in kinds:
            kwargs = dict(node.kwargs)
            allowed = {"inplace"} if node.target is torch.nn.functional.relu else {"alpha"} if node.target in (torch.add, torch.sub) else set()
            if set(kwargs) - allowed or kwargs.get("inplace", False) or kwargs.get("alpha", 1) != 1:
                raise Diagnostic(f"{source}: unsupported kwargs {node.kwargs}")
            emit(node.name, kinds[node.target], tuple(operand(arg, node.name + f"__arg{i}") for i, arg in enumerate(node.args)), source=source)
        elif node.op == "call_function" and node.target is torch.nn.functional.linear:
            if node.kwargs or len(node.args) not in (2, 3):
                raise Diagnostic(f"{source}: linear expects positional input, weight, optional bias")
            operands = [operand(a, node.name + f"__arg{i}") for i, a in enumerate(node.args[:2])]
            bias = operand(node.args[2], node.name + "__bias") if len(node.args) == 3 and node.args[2] is not None else None
            linear(node.name, *operands, bias)
        elif node.op == "call_method" and node.target in ("relu", "sum", "reshape", "permute", "t", "transpose"):
            x = operand(node.args[0], node.name + "__x")
            args, kwargs = node.args[1:], dict(node.kwargs)
            if node.target == "relu" and not args and not kwargs:
                emit(node.name, "relu", (x,), source=source)
            elif node.target == "sum":
                if len(args) > 2 or set(kwargs) - {"dim", "keepdim"}:
                    raise Diagnostic(f"{source}: unsupported reduction arguments")
                dim = args[0] if args else kwargs.get("dim")
                keep = args[1] if len(args) > 1 else kwargs.get("keepdim", False)
                dims = None if dim is None else [dim] if isinstance(dim, int) else list(dim)
                emit(node.name, "sum", (x,), {"dims": dims, "keepdim": keep}, source=source)
            elif node.target == "reshape" and not kwargs:
                shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
                if not all(type(d) is int for d in shape): raise Diagnostic("FX reshape must use a static shape")
                emit(node.name, "reshape", (x,), {"shape": list(shape)}, source=source)
            elif node.target == "permute" and not kwargs:
                perm = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
                emit(node.name, "transpose", (x,), {"perm": list(perm)}, source=source)
            elif node.target == "t" and not args and not kwargs:
                emit(node.name, "transpose", (x,), {"perm": [1, 0]}, source=source)
            else:
                raise Diagnostic(f"{source}: unsupported method arguments (use permute with a complete permutation)")
        else:
            raise Diagnostic(f"{source}: unsupported node {node.format_node()}; see docs/OPERATORS.md")
    try:
        next(inputs)
        raise Diagnostic("Too many FX example inputs")
    except StopIteration:
        pass
    if outputs is None: raise Diagnostic("FX graph has no output")
    return infer_shapes(Graph(tuple(nodes), outputs, type(model).__name__))


def onnx(path: str | Path, input_specs: dict[str, TensorSpec] | None = None) -> Graph:
    """ONNX default-domain opset 13–21 subset. Import is optional and lazy."""
    try:
        import onnx as ox
        from onnx import numpy_helper, helper
    except ImportError as exc:
        raise Diagnostic("ONNX frontend requires optional dependencies: pip install -r requirements-onnx.txt") from exc
    model = ox.load(str(path), load_external_data=False)
    ox.checker.check_model(model)
    versions = {item.domain: item.version for item in model.opset_import}
    if not 13 <= versions.get("", 0) <= 21:
        raise Diagnostic("ONNX frontend supports default-domain opsets 13–21 only")
    if model.functions or model.training_info:
        raise Diagnostic("ONNX local functions/training graphs are unsupported")
    dtype_map = {ox.TensorProto.FLOAT: torch.float32, ox.TensorProto.DOUBLE: torch.float64,
                 ox.TensorProto.INT64: torch.int64, ox.TensorProto.INT32: torch.int32, ox.TensorProto.BOOL: torch.bool}
    initializers = {v.name: v for v in model.graph.initializer}
    nodes, constants, specs = [], {}, input_specs or {}
    def tensor(proto):
        if proto.data_location == ox.TensorProto.EXTERNAL:
            raise Diagnostic(f"ONNX external tensor data unsupported: {proto.name}")
        if proto.data_type not in dtype_map:
            raise Diagnostic(f"ONNX tensor dtype {proto.data_type} unsupported")
        return torch.from_numpy(numpy_helper.to_array(proto).copy())
    for item in model.graph.input:
        if item.name in initializers: continue
        if item.type.WhichOneof("value") != "tensor_type":
            raise Diagnostic(f"ONNX %{item.name}: tensor inputs required")
        typ = item.type.tensor_type
        if typ.elem_type not in dtype_map: raise Diagnostic(f"ONNX input dtype {typ.elem_type} unsupported")
        shape = tuple(d.dim_value if d.HasField("dim_value") else None for d in typ.shape.dim) if typ.HasField("shape") else None
        declared = TensorSpec(shape, dtype_map[typ.elem_type])
        spec = specs.get(item.name, declared)
        if item.name in specs:
            if declared.dtype != spec.dtype or (shape is not None and (spec.shape is None or len(shape) != len(spec.shape) or any(a is not None and a != b for a, b in zip(shape, spec.shape)))):
                raise Diagnostic(f"ONNX %{item.name}: input override conflicts with declared metadata")
        nodes.append(Node(item.name, "input", spec=spec, source="ONNX input"))
    for name, proto in initializers.items():
        value = tensor(proto)
        constants[name] = value
        nodes.append(Node(name, "constant", spec=TensorSpec.of(value), value=value, source="ONNX initializer"))
    mapping = {"Add": "add", "Mul": "multiply", "Sub": "subtract", "MatMul": "matmul",
               "Relu": "relu", "Identity": "identity"}
    for item in model.graph.node:
        source = f"ONNX {item.name or item.op_type}"
        if item.domain not in ("",) or len(item.output) != 1:
            raise Diagnostic(f"{source}: custom domains/multiple outputs unsupported")
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in item.attribute}
        output, inputs = item.output[0], tuple(item.input)
        if item.op_type == "Constant":
            if set(attrs) != {"value"}: raise Diagnostic(f"{source}: only tensor-valued Constant supported")
            value = tensor(attrs["value"])
            constants[output] = value
            nodes.append(Node(output, "constant", spec=TensorSpec.of(value), value=value, source=source))
            continue
        if item.op_type in mapping and not attrs:
            kind = mapping[item.op_type]
        elif item.op_type == "Transpose" and set(attrs) <= {"perm"}:
            kind = "transpose"
            if "perm" not in attrs:
                # ONNX default is reverse axes. Obtain rank through common analysis.
                partial = infer_shapes(Graph(tuple(nodes), (inputs[0],), "onnx_partial"))
                shape = next(n.spec.shape for n in partial.ops if n.output == inputs[0])
                if shape is None: raise Diagnostic(f"{source}: Transpose needs known rank or explicit perm")
                attrs["perm"] = list(reversed(range(len(shape))))
        elif item.op_type == "Reshape" and set(attrs) <= {"allowzero"} and len(inputs) == 2 and inputs[1] in constants:
            shape_tensor = constants[inputs[1]]
            if shape_tensor.dtype != torch.int64 or shape_tensor.ndim != 1:
                raise Diagnostic(f"{source}: reshape shape must be a 1-D int64 constant")
            shape = shape_tensor.tolist()
            if 0 in shape and not attrs.get("allowzero", 0):
                raise Diagnostic(f"{source}: ONNX zero-copy reshape dimensions unsupported; use explicit dimensions")
            kind, inputs, attrs = "reshape", inputs[:1], {"shape": shape}
        else:
            raise Diagnostic(f"{source}: unsupported operator/attributes {item.op_type} {list(attrs)}; see docs/OPERATORS.md")
        nodes.append(Node(output, kind, inputs, attrs=attrs, source=source))
    # Preserve declared tensor metadata on intermediate/output values too.
    declared_specs = {}
    for item in (*model.graph.value_info, *model.graph.output):
        if item.type.WhichOneof("value") != "tensor_type":
            raise Diagnostic(f"ONNX %{item.name}: only tensor values are supported")
        typ = item.type.tensor_type
        if typ.elem_type not in dtype_map:
            raise Diagnostic(f"ONNX %{item.name}: unsupported dtype {typ.elem_type}")
        shape = tuple(d.dim_value if d.HasField("dim_value") else None for d in typ.shape.dim) if typ.HasField("shape") else None
        declared_specs[item.name] = TensorSpec(shape, dtype_map[typ.elem_type])
    nodes = [replace(node, spec=declared_specs.get(node.output, node.spec)) for node in nodes]
    return infer_shapes(Graph(tuple(nodes), tuple(v.name for v in model.graph.output), model.graph.name or "onnx"))
