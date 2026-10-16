"""Lossless dependencies and bounded metadata previews from the actual common IR."""
from __future__ import annotations
from dataclasses import asdict
import math
from typing import Any
import torch
from mlcompiler.ir import Graph, analysis


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Strict JSON; values remain explicitly nan/inf, never zero.
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [json_safe(v) for v in value]
    return value


def tensor_metadata(value: torch.Tensor | None) -> dict | None:
    if value is None: return None
    return json_safe({'shape': list(value.shape), 'dtype': str(value.dtype), 'elements': value.numel(),
        'bytes': value.numel() * value.element_size(), 'preview': value.flatten()[:12].tolist(),
        'preview_truncated': value.numel() > 12, 'device': str(value.device),
        'requires_grad': value.requires_grad})


def spec_metadata(spec) -> dict:
    shape = None if spec.shape is None else list(spec.shape)
    return {'shape': shape, 'dtype': str(spec.dtype) if spec.dtype is not None else 'unknown',
            'shape_label': 'unknown' if shape is None else '[' + ', '.join('?' if d is None else str(d) for d in shape) + ']'}


def node_signature(node) -> tuple:
    return (node.kind, node.inputs, node.spec, repr(node.attrs),
            None if node.value is None else node.value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())


def graph_diff(before: Graph, after: Graph) -> dict:
    old, new = ({n.output: n for n in g.ops} for g in (before, after))
    common = old.keys() & new.keys()
    modified = sorted(name for name in common if node_signature(old[name]) != node_signature(new[name]))
    return {'added': sorted(new.keys()-old.keys()), 'removed': sorted(old.keys()-new.keys()),
            'modified': modified, 'unchanged': sorted(common-set(modified)),
            'outputs_before': list(before.outputs), 'outputs_after': list(after.outputs),
            'identical': before.fingerprint() == after.fingerprint()}


def serialize_graph(graph: Graph, peer: Graph | None = None, *, side='original') -> dict:
    graph.validate()
    by_name = {n.output: n for n in graph.ops}
    peer_nodes = {n.output: n for n in peer.ops} if peer is not None else {}
    live = set(graph.outputs)
    for node in reversed(graph.ops):
        if node.output in live: live.update(node.inputs)
    nodes, edges = [], []
    for node in graph.ops:
        role = node.metadata.get('role', 'constant' if node.kind == 'constant' else 'unknown')
        node_type = 'input' if node.kind == 'input' else 'constant' if node.kind == 'constant' else 'operation'
        if node.kind == 'constant' and role in ('weight', 'bias'): node_type = role
        other = peer_nodes.get(node.output)
        change = 'unchanged' if other and node_signature(node) == node_signature(other) else 'modified' if other else ('removed' if side == 'original' else 'added')
        if peer is None: change = 'unchanged'
        spec = spec_metadata(node.spec)
        data = {'id': 'v:'+node.output, 'value_id': node.output, 'operation': node.kind,
                'node_type': node_type, 'inputs': list(node.inputs), 'outputs': [node.output],
                'attributes': node.attrs, 'source': node.source or 'unknown',
                'metadata': node.metadata, 'constant': tensor_metadata(node.value),
                'contributes_to_output': node.output in live, 'is_graph_output': node.output in graph.outputs,
                'change': change, 'correspondence_id': 'v:'+node.output if other else None, **spec}
        attributes = ' · '.join(f'{key}={value}' for key,value in node.attrs.items())
        data['label'] = f'{node.output}\n{node.kind}' + (f' · {role}' if node_type in ('weight','bias') else '') + f'\n{spec["shape_label"]} {spec["dtype"]}' + (f'\n{attributes}' if attributes else '')
        nodes.append({'data': data})
        for index, operand in enumerate(node.inputs):
            tensor = spec_metadata(by_name[operand].spec)
            edges.append({'data': {'id': f'e:{node.output}:{index}', 'source': 'v:'+operand,
                'target': 'v:'+node.output, 'tensor_id': operand, 'input_index': index,
                'label': f'{operand} · {tensor["shape_label"]} {tensor["dtype"]}', **tensor}})
    for index, name in enumerate(graph.outputs):
        spec = spec_metadata(by_name[name].spec)
        peer_output = peer.outputs[index] if peer is not None and index < len(peer.outputs) else None
        nodes.append({'data': {'id': f'out:{index}', 'value_id': name, 'node_type': 'output',
            'operation': 'graph_output', 'inputs': [name], 'outputs': [], 'attributes': {},
            'source': 'Graph.outputs', 'metadata': {'output_index': index}, 'constant': None,
            'contributes_to_output': True, 'is_graph_output': True,
            'change': 'modified' if peer_output is not None and peer_output != name else 'unchanged',
            'correspondence_id': f'out:{index}' if peer is not None and index < len(peer.outputs) else None,
            'label': f'OUTPUT {index}\n{name}\n{spec["shape_label"]} {spec["dtype"]}', **spec}})
        edges.append({'data': {'id': f'output-edge:{index}', 'source': 'v:'+name, 'target': f'out:{index}',
            'tensor_id': name, 'input_index': 0, 'label': f'{name} · {spec["shape_label"]} {spec["dtype"]}', **spec}})
    return json_safe({'name': graph.name, 'inputs': list(graph.inputs), 'outputs': list(graph.outputs),
                     'nodes': nodes, 'edges': edges, 'ir': str(graph), 'analysis': analysis(graph),
                     'fingerprint': graph.fingerprint()})
