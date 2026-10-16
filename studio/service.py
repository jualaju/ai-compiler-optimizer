"""Session application service. No compiler rewrites or benchmark substitutes here."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import math
import threading
import time
import uuid

import torch
from mlcompiler import CPUBackend, Diagnostic, compile_graph, native, validate
from mlcompiler.__main__ import load_file
from mlcompiler.ir import ARITY
from mlcompiler.measurement import MeasureConfig, environment, measure_many
from mlcompiler.passes import PASS_NAMES
from mlcompiler.workloads import NAMES, Workload, build
from runtime import gpu_unavailable_reason
from .serialization import graph_diff, json_safe, serialize_graph, tensor_metadata

DESCRIPTIONS = {
    'affine': ('Fusion opportunity', 'A single-use multiply → add → ReLU chain can be fused. CPU fusion groups dispatch; intermediate allocations remain.'),
    'constant': ('Constant folding', 'The native graph contains scale × extra, a real static expression. FX may fold that expression during tracing; inspect the captured IR.'),
    'dead_branch': ('Dead computation', 'An unused dot product is present in the original graph and can be removed by liveness analysis.'),
    'redundant_relu': ('Safe algebraic simplification', 'Repeated ReLU can be removed without reassociation or unsafe x + 0 identities.'),
    'layout': ('Layout simplification', 'Two inverse permutations can be cancelled; the now-unused view is removed by DCE.'),
    'residual': ('Correctness guard', 'The multiply result contributes to both the residual output and the affine branch. Fusion must preserve it; no single-use fusion is legal.'),
    'mlp': ('No applicable rewrite', 'This native MLP already has five necessary matmul/add/ReLU operations. There is no elementwise affine chain, static expression, dead result or redundant view. An unchanged graph is correct.'),
}


def now(): return datetime.now(timezone.utc).isoformat()


def integer(data, key, default, low, high):
    value = data.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise Diagnostic(f'{key} must be an integer from {low} to {high}')
    return value


def tolerances(data):
    result = {}
    for name, default in (('rtol', 1e-5), ('atol', 1e-6)):
        value = data.get(name, default)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise Diagnostic(f'{name} must be finite and nonnegative')
        result[name] = float(value)
    return result


def measurement(data):
    return MeasureConfig(integer(data, 'warmup', 3, 0, 100), integer(data, 'batches', 7, 3, 31),
                         integer(data, 'iterations', 20, 1, 1000))


@dataclass
class Run:
    id: str
    workload: Workload
    request: dict
    created: str = field(default_factory=now)
    compilation: object | None = None
    validation: dict | None = None
    benchmark: dict | None = None
    events: list = field(default_factory=list)


class StudioService:
    def __init__(self):
        self.runs: dict[str, Run] = {}
        self.events: list[dict] = []
        self.lock = threading.RLock()  # Serial CPU work avoids overlapping timing experiments.

    def event(self, action, status, *, run=None, message='', details=None):
        entry = {'id': len(self.events)+1, 'at': now(), 'action': action, 'status': status,
                 'run_id': run.id if run else None, 'message': message, 'details': details or {}}
        self.events.append(entry)
        if run: run.events.append(entry)

    def catalog(self):
        return {'workloads': [{'id': name, 'title': DESCRIPTIONS[name][0], 'description': DESCRIPTIONS[name][1]} for name in NAMES],
                'passes': list(PASS_NAMES), 'frontends': ['native', 'torch', 'native_json', 'onnx'],
                'backend': 'cpu-torch-interpreter', 'gpu': gpu_unavailable_reason() or 'Available; not executed by Studio',
                'default_workload': 'constant'}

    def support(self):
        root = Path(__file__).resolve().parents[1]
        return {'operators': [{'operation': op, 'arity': arity, 'cpu': True} for op, arity in ARITY.items()],
            'frontends': [{'id': 'native', 'status': 'supported', 'detail': 'Version-1 JSON/dictionary common IR'},
                          {'id': 'torch', 'status': 'supported subset', 'detail': 'Static FX tensor inference; Linear/ReLU/Identity; explicit operator subset'},
                          {'id': 'onnx', 'status': 'optional subset', 'detail': 'Default domain opsets 13–21; see exact matrix below'}],
            'backends': [{'id': 'CPU', 'status': 'available', 'detail': 'PyTorch interpreter; no native CPU code generation'},
                         {'id': 'Legacy Triton affine', 'status': 'unavailable' if gpu_unavailable_reason() else 'not requested',
                          'detail': gpu_unavailable_reason() or 'Studio performs CPU execution only'}],
            'unsupported': ['General Python execution/control flow', 'Arbitrary ONNX/PyTorch operators',
                            'Training/autograd', 'General GPU backend', 'Measured peak memory', 'Measured GPU latency'],
            'documentation': root.joinpath('docs/OPERATORS.md').read_text()}

    def create(self, request):
        with self.lock:
            try:
                if len(self.runs) >= 40: raise Diagnostic('Session limit of 40 runs reached; restart Studio for a new session')
                source = request.get('frontend', 'native')
                size = integer(request, 'size', 32, 1, 65536)
                seed = integer(request, 'seed', 7, 0, 2**31-1)
                if source in ('native', 'torch'):
                    workload = build(request.get('workload', 'constant'), size, seed, source)
                elif source == 'native_json':
                    started = time.perf_counter()
                    data = request.get('graph')
                    if not isinstance(data, dict) or len(data.get('nodes', [])) > 3000:
                        raise Diagnostic('Provide a native graph object with at most 3000 nodes')
                    graph = native(data)
                    values = request.get('inputs')
                    if not isinstance(values, dict) or set(values) != set(graph.inputs):
                        raise Diagnostic('Input JSON keys must match the graph input names')
                    args = tuple(torch.tensor(values[n.output], dtype=n.spec.dtype) for n in graph.ops if n.kind == 'input')
                    CPUBackend().compile(graph)(*args)
                    workload = Workload(graph.name, 'external', graph, (args,), args,
                        (time.perf_counter()-started)*1000, False, seed, 'native_json')
                elif source == 'onnx':
                    workload = load_file('onnx', request.get('path'), request.get('inputs_path'), seed)
                else: raise Diagnostic('Unsupported frontend; choose native, torch, native_json or onnx')
                run = Run(uuid.uuid4().hex[:12], workload, request)
                self.runs[run.id] = run
                self.event('capture', 'success', run=run, message='Original graph captured and CPU-executed; optimization not yet requested',
                           details={'compute_ops': len(workload.graph.compute), 'reference_checked': workload.reference_checked})
                return self.snapshot(run)
            except Exception as exc:
                self.event('capture', 'failed', message=str(exc))
                raise

    def get(self, run_id):
        if run_id not in self.runs: raise KeyError('Run not found in this server session')
        return self.runs[run_id]

    def action(self, run_id, action, data):
        with self.lock:
            run = self.get(run_id)
            try:
                w = run.workload
                if action == 'optimize':
                    strategy = data.get('strategy', 'conventional')
                    passes = data.get('passes') if strategy == 'manual' else None
                    if strategy == 'manual' and (not isinstance(passes, list) or any(not isinstance(n, str) for n in passes)):
                        raise Diagnostic('Manual policy requires a list of pass names; an empty list runs no passes')
                    result = compile_graph(w.graph, w.samples, strategy=strategy, passes=passes,
                        capture_trace=True, measurement=measurement(data), **tolerances(data))
                    # Hold the last good compilation until the complete action succeeds.
                    run.compilation, run.validation, run.benchmark = result, None, None
                    self.event('optimize', 'success', run=run, message=result.decision,
                        details={'strategy': strategy, 'candidate': result.candidate,
                                 'before_ops': len(w.graph.compute), 'after_ops': len(result.graph.compute),
                                 'sample_validation': asdict(result.validation)})
                elif action == 'validate':
                    if run.compilation is None: raise Diagnostic('Run optimization or baseline compilation before validating')
                    settings = tolerances(data)
                    started = time.perf_counter()
                    checked = validate(w.graph, run.compilation.graph, (w.evaluation_inputs,), **settings)
                    result = {**asdict(checked), 'elapsed_ms': (time.perf_counter()-started)*1000,
                              'input_scope': 'separate built-in evaluation input' if w.family != 'external' else 'provided external input (not held out)'}
                    if checked.passed:
                        original = CPUBackend().compile(w.graph)(*w.evaluation_inputs)
                        optimized = run.compilation.plan(*w.evaluation_inputs)
                        result['outputs'] = [{'original': tensor_metadata(a), 'optimized': tensor_metadata(b)}
                                             for a,b in zip(original, optimized)]
                    run.validation = result
                    self.event('validate', 'success' if checked.passed else 'failed', run=run, message=checked.message, details=result)
                elif action == 'benchmark':
                    if run.compilation is None: raise Diagnostic('Compile a policy before benchmarking its selected graph')
                    settings = tolerances(data)
                    checked = validate(w.graph, run.compilation.graph, (w.evaluation_inputs,), **settings)
                    if not checked.passed: raise Diagnostic('Benchmark rejected: '+checked.message)
                    config = measurement(data)
                    started = time.perf_counter()
                    baseline = CPUBackend().compile(w.graph)
                    baseline_compile_ms = (time.perf_counter()-started)*1000
                    timings = measure_many({'original': lambda: baseline(*w.evaluation_inputs),
                                            'optimized': lambda: run.compilation.plan(*w.evaluation_inputs)}, config)
                    result = {**timings, 'environment': environment(), 'correctness': asdict(checked),
                        'selected_policy': run.compilation.strategy, 'selected_candidate': run.compilation.candidate,
                        'original_fingerprint': w.graph.fingerprint(), 'optimized_fingerprint': run.compilation.graph.fingerprint(),
                        'original_backend_preparation_ms': baseline_compile_ms,
                        'input_scope': 'held-out built-in input' if w.family != 'external' else 'same provided external input',
                        'limitations': ['CPU interpreter timings include dispatch, input checks and allocation',
                                       'No measured memory or GPU timings', 'Single host/session; timing noise applies',
                                       'CPU fusion still materializes intermediate tensors']}
                    run.benchmark = result
                    self.event('benchmark', 'success', run=run, message='CPU timing batches completed after numerical checks',
                               details={'config': asdict(config), 'timings': timings['timings']})
                else: raise Diagnostic('Unknown action; choose optimize, validate or benchmark')
                return self.snapshot(run)
            except Exception as exc:
                self.event(action, 'failed', run=run, message=str(exc))
                raise

    def snapshot(self, run):
        w, c = run.workload, run.compilation
        after = c.graph if c else None
        steps = []
        if c:
            for snapshot in c.snapshots:
                record = c.records[snapshot.record_index]
                steps.append({'record_index': snapshot.record_index, 'candidate': snapshot.candidate,
                    'selected_candidate': snapshot.candidate == c.candidate, 'record': asdict(record),
                    'before': serialize_graph(snapshot.before, snapshot.after, side='original'),
                    'after': serialize_graph(snapshot.after, snapshot.before, side='optimized'),
                    'diff': graph_diff(snapshot.before, snapshot.after)})
        return json_safe({'id': run.id, 'created': run.created, 'workload': w.name, 'frontend': w.frontend,
            'configuration': run.request,
            'description': DESCRIPTIONS.get(w.name, ('External graph', 'User-supplied graph; support is checked during conversion.'))[1],
            'reference_checked': w.reference_checked, 'backend': 'cpu-torch-interpreter',
            'input_scope': 'separate calibration/evaluation samples' if w.family != 'external' else 'provided external input; no input holdout',
            'inputs': [tensor_metadata(value) for value in w.samples[0]],
            'original': serialize_graph(w.graph, after, side='original'),
            'optimized': serialize_graph(after, w.graph, side='optimized') if after else None,
            'diff': graph_diff(w.graph, after) if after else None,
            'compilation': None if c is None else {'strategy': c.strategy, 'candidate': c.candidate,
                'decision': c.decision, 'plan': c.plan.describe(), 'planning_ms': c.planning_ms,
                'optimization_ms': c.optimization_ms, 'backend_preparation_ms': c.backend_compile_ms,
                'final_validation_ms': c.validation_ms, 'total_compile_ms': c.total_compile_ms,
                'sample_validation': asdict(c.validation), 'passes': [asdict(r) for r in c.records],
                'calibration': c.calibration}, 'frontend_preparation_ms': w.frontend_ms,
            'steps': steps, 'validation': run.validation, 'benchmark': run.benchmark, 'events': run.events,
            'gpu': gpu_unavailable_reason() or 'Available, not executed by Studio'})

    def history(self):
        with self.lock:
            return {'runs': [{'id': r.id, 'workload': r.workload.name, 'frontend': r.workload.frontend,
                'created': r.created, 'compiled': r.compilation is not None, 'validated': r.validation,
                'benchmarked': r.benchmark is not None} for r in reversed(list(self.runs.values()))],
                'events': list(reversed(self.events))}
