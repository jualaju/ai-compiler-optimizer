"""Common-pipeline dashboard, sharing the existing Rich widgets and colors."""
from __future__ import annotations
from pathlib import Path
import os
import select
import sys
import torch
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from demo_ui import panel, tensor_values
from . import CPUBackend, analysis, compile_graph, validate
from .experiment import markdown, run_suite, save
from .measurement import MeasureConfig
from .passes import PASS_NAMES
from .workloads import NAMES, build


def prompt(ui, message, default, choices=None):
    if choices is not None:
        from terminal_controls import choice
        return choice(ui.console, message, choices, default)
    try:
        return ui.console.input(f'[cyan]{message}[/] [dim]({default})[/]: ').strip() or default
    except EOFError:
        return default


def _run_dashboard(ui, args):
    """Interactive controls plus scriptable options; no compiler logic in widgets."""
    from .__main__ import load_file
    torch.set_num_threads(1)
    source = args.source or prompt(ui, 'Source: native / torch / native-file / onnx', 'native', ('native', 'torch', 'native-file', 'onnx'))
    path = args.path
    if source == 'native-file':
        source = 'native'
        path = path or prompt(ui, 'Native graph path', 'examples/native_affine.json')
    if source not in ('native', 'torch', 'onnx'):
        raise ValueError('Choose native, torch or onnx')
    if source == 'onnx' and not path:
        path = prompt(ui, 'ONNX model path', 'model.onnx')
    if path:
        if source == 'torch':
            raise ValueError('PyTorch files are not loaded as code. Choose a built-in workload or use the Python API.')
        input_path = args.inputs or prompt(ui, 'Input JSON path', 'inputs.json')
        workload = load_file(source, path, input_path, 7)
    else:
        name = args.workload or prompt(ui, 'Workload: ' + ' / '.join(NAMES), 'constant', NAMES)
        size = args.size
        while size is None:
            try:
                size = int(prompt(ui, 'Size / batch dimension', '32'))
                if size < 1: raise ValueError
            except ValueError:
                ui.console.print('Size must be a positive integer.', style='yellow')
                size = None
        workload = build(name, size=size, source=source)
    ui.console.print(panel('ORIGINAL COMMON IR', Text(str(workload.graph))))
    ui.console.print(Text('Supported workload operators: ' + ', '.join(sorted(analysis(workload.graph)['operators'])) +
                          '\nFull support matrix: S in the dashboard / docs/OPERATORS.md', style='dim'))
    strategy = args.strategy or prompt(ui, 'Policy: intelligent / conventional / manual / baseline / compare', 'conventional', ('intelligent', 'conventional', 'manual', 'baseline', 'compare'))
    compare_requested = args.compare or strategy == 'compare'
    if strategy == 'compare': strategy = 'intelligent'
    names = None
    if strategy == 'manual':
        if args.passes is None:
            ui.console.print('An empty policy can be entered as none; unknown pass names are rejected.', style='dim')
        ui.console.print(Text('Available passes: ' + ', '.join(PASS_NAMES), style='dim'))
        text = args.passes if args.passes is not None else prompt(ui, 'Comma-separated passes (none for empty policy)', 'dead_code_elimination')
        names = tuple(n.strip() for n in text.split(',') if n.strip() and n.strip() != 'none')
    config = MeasureConfig()
    result = compile_graph(workload.graph, workload.samples, strategy=strategy,
                           passes=names, measurement=config)
    check = validate(workload.graph, result.graph, (workload.evaluation_inputs,))
    if not check.passed: raise ValueError('Runtime validation failed: ' + check.message)
    original = CPUBackend().compile(workload.graph)(*workload.evaluation_inputs)
    optimized = result.plan(*workload.evaluation_inputs)
    report = None
    report_paths = None

    def benchmark():
        nonlocal report, report_paths
        report = run_suite(((workload, 'external_same_input' if path else 'evaluation'),), config)
        report_paths = save(report, args.output)

    if compare_requested: benchmark()
    before_stats, after_stats = analysis(workload.graph), analysis(result.graph)
    from runtime import gpu_unavailable_reason
    reason = gpu_unavailable_reason() or 'GPU available; not requested by this CPU pipeline'

    def pass_text():
        if not result.records: return 'No optimization passes requested.'
        return '\n'.join(f'{r.iteration:02d} [{r.status.upper()}] {r.candidate}/{r.name}\n  {r.reason}\n  applicable={r.applicable}, changes={r.changes}, sample validated={r.validated}' for r in result.records)

    def support():
        return Path(__file__).resolve().parents[1].joinpath('docs/OPERATORS.md').read_text()

    def text_view(mode):
        return {'o': str(workload.graph), 'g': str(result.graph), 'p': result.plan.describe(),
                'd': pass_text(), 's': support(),
                'b': markdown(report) + '\nArtifacts: ' + str(report_paths) if report else 'Press B to collect real CPU measurements.'}[mode]

    def overview():
        info = (f'{source.upper()} / {workload.name}  •  CPU / PyTorch inference\n'
                f'{len(workload.graph.inputs)} inputs  •  {len(workload.graph.outputs)} outputs  •  selected: {result.candidate}\n'
                f'Compute operations {before_stats["compute_ops"]} → {after_stats["compute_ops"]} (structural)\n'
                f'Planning {result.planning_ms:.3f} ms  •  backend preparation {result.backend_compile_ms:.3f} ms')
        statuses = ', '.join(f'{status}: {sum(r.status == status for r in result.records)}'
                             for status in ('applied','skipped','failed'))
        output = '\n'.join(f'Output {i}: {tensor_values(a)} → {tensor_values(b)}'
                           for i,(a,b) in enumerate(zip(original,optimized)))
        nodes = lambda graph: '\n'.join(f'%{n.output} = {n.kind}({", ".join(n.inputs)})' for n in graph.compute)
        limit = max(2, min(6, ui.console.height - 22))
        def preview(graph):
            lines = nodes(graph).splitlines()
            return Text('\n'.join(lines[:limit]) + ('\n… full graph: O / G' if len(lines)>limit else ''))
        graph_panels = [panel('ORIGINAL', preview(workload.graph)), panel('OPTIMIZED', preview(result.graph), 'magenta')]
        graphs = Columns(graph_panels, equal=True, expand=True) if ui.console.width >= 80 else Group(*graph_panels)
        if ui.console.is_terminal and sys.stdin.isatty() and ui.console.height < 38:
            # Fixed panel sizes keep controls visible in the alternate screen.
            layout = Layout()
            layout.split_column(
                Layout(panel('COMMON COMPILER / CPU', Text(f'{source} / {workload.name} / {result.candidate}')), size=3),
                Layout(panel('ANALYSIS', Text(f'Compute ops: {before_stats["compute_ops"]} → {after_stats["compute_ops"]}\n'
                    f'Planning: {result.planning_ms:.2f} ms\n{statuses}')), size=5),
                Layout(panel('OPTIMIZED GRAPH / G: FULL', preview(result.graph)), ratio=1, minimum_size=3),
                Layout(panel('VALIDATION', Text(f'✓ {check.samples} evaluation sample checked\n{tensor_values(optimized[0], limit=4)}'), 'green'), size=4),
                Layout(Text('CPU execution only; no GPU timings\nO/G graphs • D passes • P plan • S support\nB benchmark • A overview • N/U scroll • Q quit', style='cyan'), size=3))
            return layout
        return Group(panel('INPUT → COMMON IR → ANALYSIS → PLANNER → CPU', Text(info)), graphs,
            panel('VALIDATION / PASSES', Text(f'✓ Sample-based equivalence passed; {len(workload.samples)} calibration inputs + evaluation check\n'
                f'Frontend reference: {"checked" if workload.reference_checked else "not independently checked"}\n'
                f'{statuses}\n{output}\nCPU fusion groups dispatch; intermediate tensors remain.'), 'green'),
            panel('MEASUREMENTS', Text('B: run/inspect CPU strategy comparison' if report is None else
                'CPU measurements collected; B: inspect timings and limitations.\nArtifacts: ' + str(report_paths))),
            Text(reason, style='yellow'),
            Text('[O] Original [G] Optimized [D] Passes [P] Plan [S] Support [B] Benchmark\n[A] Overview [N/U] Scroll [Q] Quit', style='cyan'))

    if not (ui.console.is_terminal and sys.stdin.isatty()):
        ui.console.print(overview())
        ui.console.print(panel('OPTIMIZATION PASSES', Text(pass_text())))
        ui.console.print(panel('CPU EXECUTION PLAN', Text(result.plan.describe())))
        if report: ui.console.print(Text(markdown(report)))
        return
    from terminal_controls import browse
    def benchmark_view():
        if report is None:
            benchmark()
        return text_view('b')
    browse(ui.console, overview,
           {'o': lambda: text_view('o'), 'g': lambda: text_view('g'),
            'd': lambda: text_view('d'), 'p': lambda: text_view('p'),
            's': lambda: text_view('s'), 'b': benchmark_view},
           titles={'o': 'Original', 'g': 'Optimized', 'd': 'Passes',
                   'p': 'Plan', 's': 'Support', 'b': 'Benchmark'},
           initial='d' if getattr(args, 'debug', False) else 'overview')


def run_dashboard(ui, args):
    from argparse import Namespace
    while True:
        try:
            return _run_dashboard(ui, args)
        except (ValueError, RuntimeError, OSError, KeyError) as exc:
            if not (ui.console.is_terminal and sys.stdin.isatty()):
                raise
            ui.console.print(f'Compiler action failed: {exc}', style='red', markup=False)
            if prompt(ui, 'Retry with new inputs? y / n', 'y', ('y', 'n')) == 'n':
                return
            values = vars(args).copy()
            values.update(source=None, workload=None, path=None, inputs=None, size=None,
                          strategy=None, passes=None, compare=False)
            args = Namespace(**values)
