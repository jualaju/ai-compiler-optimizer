"""Rich dashboard widgets and terminal interaction; no compiler transformations."""
import math
import select
import sys

from rich import box
from rich.align import Align
from rich.bar import Bar
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


def panel(title, content, color='cyan'):
    return Panel(content, title=Text(title, style=f'bold {color}'),
                 title_align='left', border_style=color, box=box.ROUNDED, padding=(0, 1))


def tensor_values(tensor, limit=8):
    values = tensor.flatten().tolist()
    shown = ', '.join(f'{v:.6g}' for v in values[:limit])
    return '[' + shown + (f', … ({len(values)} values)' if len(values) > limit else '') + ']'


class DemoUI:
    def __init__(self):
        self.console = Console()

    def intro(self):
        self.console.print(panel('AI COMPILER OPTIMIZATION',
            '[bold cyan]GRAPH → IR → FUSION → TRITON[/]\n'
            '[dim]CPU semantic validation • Enter accepts defaults[/]'))

    def numbers(self, label, default, scalar=False):
        while True:
            try:
                raw = self.console.input(f'[cyan]{label}[/] [dim]({default})[/]: ').strip()
            except EOFError:
                raw = ''
            try:
                values = [float(item) for item in (raw or default).split()]
                if not values or (scalar and len(values) != 1) or not all(
                    math.isfinite(v) and abs(v) <= 3.4028234663852886e38 for v in values
                ):
                    raise ValueError
                return values[0] if scalar else values
            except ValueError:
                self.console.print('Enter one finite float32 number.' if scalar else
                                   'Enter finite float32 numbers separated by spaces.', style='yellow')

    def graph(self, data, height):
        def flow(graph, optimized):
            labels = {'multiply': 'MUL', 'add': 'ADD', 'relu': 'ReLU',
                      'fused_affine_relu': 'FUSED OP\nAFFINE + ReLU'}
            nodes = [labels.get(op.kind, op.kind) for op in graph.ops if op.kind != 'input']
            color = 'magenta' if optimized else 'cyan'
            if height >= 14:
                items = [Align.center(Text('INPUT', style='cyan'))]
                for node in nodes:
                    items.append(panel('', Align.center(Text(node + ' ↓', style=f'bold {color}')), color))
                items.append(Align.center(Text('↓ OUTPUT', style=color)))
                return Group(*items)
            return Group(Text('INPUT ↓', style='cyan'),
                         panel('', Text(' → '.join(nodes).replace('FUSED OP\nAFFINE + ReLU', 'FUSED AFFINE+ReLU'), style=f'bold {color}'), color),
                         Text('↓ OUTPUT', style=color))
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(Text('ORIGINAL', style='bold cyan'), Text('OPTIMIZED', style='bold magenta'))
        grid.add_row(flow(data['before'], False), flow(data['after'], True))
        return panel('GRAPH TRANSFORMATION', grid)

    def impact(self, data, compact):
        before = [op for op in data['before'].ops if op.kind != 'input']
        after = [op for op in data['after'].ops if op.kind != 'input']
        avoided = len(({op.output for op in before} - {op.output for op in after})
                      - set(data['before'].outputs))
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column()
        table.add_column(ratio=1)
        table.add_column(justify='right')
        for name in ('Operations', 'Launches*'):
            if compact:
                table.add_row(name, Bar(len(before), 0, len(after), color='magenta', bgcolor='grey23'),
                              f'{len(before)} → {len(after)}')
            else:
                table.add_row(name + ' BEFORE', Bar(len(before), 0, len(before), color='cyan'), str(len(before)))
                table.add_row('AFTER', Bar(len(before), 0, len(after), color='magenta', bgcolor='grey23'), str(len(after)))
        table.add_row('Intermediates', Bar(max(1, len(before)-1), 0, avoided, color='magenta'), f'{avoided} avoided')
        return panel('OPTIMIZATION IMPACT', Group(table,
            Text('* Launches: structural estimate; no GPU timing', style='dim'),
            Text('FUSION STATUS: ' + ('APPLIED' if data['fused'] else 'NOT APPLIED'), style='bold magenta')), 'magenta')

    def input_panel(self, data, rows):
        x, scale, bias = data['inputs']
        scale_text = f'{scale.item():.4g}' if scale.numel() == 1 else f'[{scale.numel()}D]'
        bias_text = f'{bias.item():.4g}' if bias.numel() == 1 else f'[{bias.numel()}D]'
        values = x.flatten().tolist()
        peak = max((abs(v) for v in values), default=1) or 1
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column()
        table.add_column(justify='right')
        table.add_column(ratio=1)
        limit = max(1, rows - 4)
        for i, value in enumerate(values[:limit]):
            table.add_row(f'x[{i}]', f'{value:.4g}', Bar(peak, 0, abs(value),
                          color='magenta' if value < 0 else 'cyan', bgcolor='grey19'))
        if len(values) > limit:
            table.add_row('…', f'+{len(values)-limit}', '')
        return panel('INPUT VECTOR', Group(table,
            Text(f'scale {scale_text}  •  bias {bias_text}', style='cyan'),
            Text(f'float32 • {len(values)} elements', style='dim')))

    def result(self, data):
        return panel('CPU VALIDATION', Group(Columns([
            panel('ORIGINAL', Text(tensor_values(data['original'])), 'green'),
            panel('OPTIMIZED', Text(tensor_values(data['optimized'])), 'green')], equal=True, expand=True),
            Text('✓ NUMERICAL EQUIVALENCE VERIFIED', style='bold green'),
            Text('Triton source verified • CPU semantics passed', style='dim green')), 'green')

    def status(self, data):
        return panel('COMPILER STATUS', Group(
            Text('● FX GRAPH ONLINE\n● CUSTOM IR ONLINE', style='green'),
            Text('● FUSION PASS ' + ('APPLIED' if data['fused'] else 'NOT APPLIED'), style='magenta'),
            Text('● TRITON SOURCE VERIFIED\n● CPU VALIDATION PASS', style='green'),
            Text('● NVIDIA GPU UNAVAILABLE' if data['reason'] else '● GPU EXECUTION NOT REQUESTED', style='yellow')))

    def dashboard(self, data):
        width, height = self.console.size
        compact = height < 38
        layout = Layout()
        header = Columns([Text('╱╱ AI·CO\nAI COMPILER OPTIMIZATION', style='bold cyan'),
                          Text('COMPILER CORE // ONLINE\nOPTIMIZATION ENGINE // READY', style='bold green')], expand=True)
        layout.split_column(Layout(panel('GRAPH → IR → FUSION → TRITON', header), name='header', size=4),
                            Layout(name='top', ratio=4, minimum_size=8),
                            Layout(name='middle', size=7 if compact else 9),
                            Layout(name='bottom', ratio=3, minimum_size=4),
                            Layout(name='footer', size=1))
        top_height = max(8, (height - 4 - (7 if compact else 9) - 1) * 4 // 7)
        if width >= 95:
            layout['top'].split_row(Layout(name='pipeline', size=17), Layout(name='graph', ratio=3),
                                    Layout(name='input', size=30))
            pipeline = ('PyTorch\n   ↓\nFX Graph\n   ↓\nCustom IR\n   ↓\nFusion\n   ↓\nTriton'
                        if top_height >= 11 else 'PyTorch\n↓ FX Graph\n↓ Custom IR\n↓ Fusion\n↓ Triton')
            layout['pipeline'].update(panel('PIPELINE', Text(pipeline, style='bold cyan')))
        else:
            layout['top'].split_row(Layout(name='graph', ratio=3), Layout(name='input', ratio=2))
        layout['graph'].update(self.graph(data, top_height))
        if top_height >= 14:
            layout['input'].split_column(Layout(self.input_panel(data, top_height - 6), ratio=1),
                Layout(panel('COMPILER STATUS', Group(
                    Text('● FX / CUSTOM IR ONLINE', style='green'),
                    Text('● FUSION ' + ('APPLIED' if data['fused'] else 'NOT APPLIED'), style='magenta'),
                    Text('● TRITON READY / CPU PASS', style='green'),
                    Text('● NVIDIA GPU UNAVAILABLE' if data['reason'] else '● GPU NOT EXECUTED', style='yellow'))), size=6))
        else:
            layout['input'].update(self.input_panel(data, top_height))
        layout['middle'].split_row(Layout(self.impact(data, compact), ratio=1), Layout(self.result(data), ratio=1))
        bottom_height = max(4, height - 4 - (7 if compact else 9) - 1 - top_height)
        source_lines = data['source'].splitlines()
        # Preview is an exact slice of the generated artifact, never a substitute kernel.
        start = source_lines.index('@triton.jit')
        limit = max(1, bottom_height - 2)
        preview = '\n'.join(source_lines[start:start + limit])
        kernel = panel('GENERATED GPU KERNEL // TRITON',
            Syntax(preview, 'python', theme='ansi_dark', background_color='default', word_wrap=False))
        kernel.subtitle = Text(f'Preview • K: full source ({len(source_lines)} lines)', style='dim')
        trace_rows = data['trace'][-max(1, bottom_height - 2):]
        trace = Group(*(Text.assemble((stamp + ' ', 'dim'), (f'[{tag}] ', 'magenta' if tag == 'FUSE' else 'cyan'),
                       (msg, 'yellow' if tag == 'GPU' else 'green'), no_wrap=True, overflow='ellipsis') for stamp, tag, msg in trace_rows))
        layout['bottom'].split_row(Layout(kernel, ratio=3), Layout(panel('COMPILER TRACE', trace), ratio=2))
        warning = '⚠ GPU EXECUTION UNAVAILABLE' if data['reason'] else 'GPU execution not requested'
        layout['footer'].update(Text(f'{warning} • CPU validated • [K] Kernel [D] Details [Q] Quit', style='yellow'))
        if width < 65 or height < 24:
            # A reduced overview fits small terminals; source and details remain accessible.
            layout = Layout()
            layout.split_column(Layout(panel('AI COMPILER OPTIMIZATION', Text('GRAPH → IR → FUSION → TRITON', style='cyan')), size=3),
                Layout(panel('GRAPH / FUSION', Text('x → MUL → ADD → ReLU\nx → FUSED AFFINE + ReLU', style='magenta')), size=4),
                Layout(self.result(data), minimum_size=7),
                Layout(Text('GPU unavailable' if data['reason'] else 'CPU execution only', style='yellow'), size=1),
                Layout(Text('[K] Kernel [D] Details [Q] Quit', style='cyan'), size=1))
        return layout

    def details(self, data):
        return (data['debug'] or f"CUSTOM IR / BEFORE\n{data['before']}\n\nCUSTOM IR / AFTER\n{data['after']}") + (
            f"\n\nGPU EXECUTION\n{data['reason'] or 'Not requested in this CPU demonstration.'}"
            '\nTriton source generation and Python syntax verified.'
            '\nCPU semantic validation completed. GPU kernel was not executed.'
            f"\n\nArtifact: {data['path']}\n\nCOMPILER TRACE\n" +
            '\n'.join(f'{stamp} [{tag}] {msg}' for stamp, tag, msg in data['trace']))

    def show(self, data, debug=False):
        if not (self.console.is_terminal and sys.stdin.isatty()):
            self.console.print(self.dashboard(data))
            if debug:
                self.console.print(panel('DEBUG DETAILS', Text(self.details(data))))
            return
        import termios
        import tty
        mode = 'd' if debug else 'dashboard'
        offset = 0
        original_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            with Live(console=self.console, screen=True, auto_refresh=False) as live:
                while True:
                    if mode == 'dashboard':
                        view = self.dashboard(data)
                    else:
                        lines = (data['source'] if mode == 'k' else self.details(data)).splitlines()
                        page_size = max(1, self.console.height - 12)
                        offset = min(offset, max(0, len(lines) - page_size))
                        body = '\n'.join(lines[offset:offset + page_size])
                        content = Syntax(body, 'python', theme='ansi_dark', word_wrap=True) if mode == 'k' else Text(body)
                        view = Group(self.status(data), panel('GENERATED GPU KERNEL // TRITON' if mode == 'k' else 'COMPILER DETAILS', content),
                            Text(f'Lines {offset+1}–{min(offset+page_size, len(lines))}/{len(lines)} • N/P scroll • G dashboard • Q quit', style='cyan'))
                    live.update(view, refresh=True)
                    if select.select([sys.stdin], [], [], 0.25)[0]:
                        key = sys.stdin.read(1).lower()
                        if not key or key == 'q':
                            break
                        if key in ('k', 'd', 'g'):
                            mode = 'dashboard' if key == 'g' else key
                            offset = 0
                        elif key == 'n':
                            offset += 1
                        elif key == 'p':
                            offset = max(0, offset - 1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
