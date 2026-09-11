"""Full-screen presentation of the existing compiler; CPU validation only."""
import argparse
import ast
from datetime import datetime

from demo_ui import DemoUI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--debug', action='store_true', help='open raw compiler details')
    args = parser.parse_args()
    ui = DemoUI()
    ui.intro()
    x = ui.numbers('Tensor', '2 -1 3 4')
    scale = ui.numbers('Scale', '1.5', scalar=True)
    bias = ui.numbers('Bias', '0.5', scalar=True)
    ui.console.clear()
    trace = []

    def record(tag, message):
        trace.append((datetime.now().strftime('%H:%M:%S'), tag, message))

    with ui.console.status('[cyan]Compiling graph and validating CPU semantics…', spinner='dots'):
        import torch
        from compiler import AffineReLU, capture, fuse, generate_triton, interpret
        from runtime import gpu_unavailable_reason

        inputs = tuple(torch.tensor(v, dtype=torch.float32) for v in (x, scale, bias))
        model = AffineReLU()
        before = capture(model, inputs)
        record('GRAPH', 'FX graph captured')
        record('IR', 'Lowered to custom IR')
        after = fuse(before)
        fused = any(op.kind == 'fused_affine_relu' for op in after.ops)
        record('FUSE', 'MUL → ADD → ReLU pattern detected' if fused else 'No fusion pattern found')
        count = lambda graph: sum(op.kind != 'input' for op in graph.ops)
        record('FUSE', f'{count(before)} ops → {count(after)} fused op' if fused else 'Graph preserved')
        path = generate_triton(after)
        source = path.read_text()
        ast.parse(source)  # Verify Python syntax without importing or executing Triton.
        record('CODE', 'Triton source generated; syntax verified')
        expected = model(*inputs)
        original = interpret(before, *inputs)[0]
        optimized = interpret(after, *inputs)[0]
        torch.testing.assert_close(original, expected)
        torch.testing.assert_close(optimized, expected)
        torch.testing.assert_close(original, optimized)
        record('CHECK', 'CPU numerical equivalence verified ✓')
        reason = gpu_unavailable_reason()
        record('GPU', 'Execution unavailable' if reason else 'Execution not requested; CPU demo')
        debug = (f'FX GRAPH\n{torch.fx.symbolic_trace(model).graph}\n\n'
                 f'CUSTOM IR / BEFORE\n{before}\n\nCUSTOM IR / AFTER\n{after}\n\n'
                 f'GENERATED SOURCE\n{path}\n{source}') if args.debug else None
    ui.show(dict(before=before, after=after, inputs=inputs, original=original,
                 optimized=optimized, source=source, path=path, trace=trace,
                 reason=reason, fused=fused, debug=debug), debug=args.debug)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:
        from rich.console import Console
        Console(stderr=True).print(f'Compiler demo failed: {exc}', style='bold red', markup=False)
        raise SystemExit(1)
