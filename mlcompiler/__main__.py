"""Independent CPU compiler CLI: python -m mlcompiler --help."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import torch
from . import CPUBackend, Diagnostic, TensorSpec, analysis, compile_graph, native, onnx
from .experiment import run_suite, save
from .measurement import MeasureConfig
from .passes import PASS_NAMES
from .workloads import NAMES, Workload, build, suite


def load_file(source, path, input_path, seed):
    if not path or not input_path:
        raise Diagnostic("File frontends require --path GRAPH and --inputs INPUTS.json")
    started = time.perf_counter()
    graph = native(path) if source == "native" else onnx(path)
    # JSON input mapping; dtype comes from IR input specification, never guessed.
    data = json.loads(Path(input_path).read_text())
    args = []
    for node in graph.ops:
        if node.kind == "input":
            if node.output not in data or node.spec.dtype is None:
                raise Diagnostic(f"Missing input or dtype for %{node.output}")
            value = torch.tensor(data[node.output], dtype=node.spec.dtype)
            node.spec.check(value, node.output)
            args.append(value)
    if set(data) != set(graph.inputs): raise Diagnostic("Input JSON keys must match graph input names")
    args = tuple(args)
    CPUBackend().compile(graph)(*args)
    reference_checked = False
    if source == "onnx":
        from onnx.reference import ReferenceEvaluator
        evaluator = ReferenceEvaluator(str(path))
        from .backend import compare_outputs
        expected = tuple(torch.from_numpy(v) for v in evaluator.run(None, {n: t.numpy() for n, t in zip(graph.inputs, args)}))
        compare_outputs(expected, CPUBackend().compile(graph)(*args))
        reference_checked = True
    return Workload(Path(path).stem, "external", graph, (args,), args,
                    (time.perf_counter() - started) * 1000, reference_checked, seed, source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("native", "torch", "onnx"), default="native")
    parser.add_argument("--workload", choices=NAMES, default="affine")
    parser.add_argument("--path", help="Native JSON or ONNX file; otherwise use a built-in workload")
    parser.add_argument("--inputs", help="JSON name→tensor values for file input")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--strategy", choices=("baseline", "conventional", "intelligent", "manual"), default="conventional")
    parser.add_argument("--passes", default=None, help="Comma-separated manual pass names: " + ",".join(PASS_NAMES))
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--compare", action="store_true", help="Measure all three strategies")
    parser.add_argument("--suite", action="store_true", help="Run the 14-workload CPU evaluation suite")
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="results/cpu_comparison.json")
    args = parser.parse_args(argv)
    if args.size < 1 or args.threads < 1 or args.rtol < 0 or args.atol < 0:
        parser.error("size/threads must be positive and tolerances nonnegative")
    if args.passes is not None and args.strategy != "manual": parser.error("--passes requires --strategy manual")
    if args.source == "onnx" and not args.path: parser.error("ONNX requires --path and --inputs")
    if args.source == "torch" and args.path: parser.error("Custom PyTorch models use the Python API; CLI torch selects built-ins")
    if args.suite and (args.path or args.source == 'onnx'):
        parser.error('--suite uses native/torch built-ins; file inputs use --compare')
    torch.set_num_threads(args.threads)
    try:
        config = MeasureConfig(args.warmup, args.batches, args.iterations)
        if args.suite:
            report = run_suite(suite(args.source, args.seed), config, rtol=args.rtol, atol=args.atol)
        else:
            workload = load_file(args.source, args.path, args.inputs, args.seed) if args.path else build(args.workload, args.size, args.seed, args.source)
            if args.compare:
                report = run_suite(((workload, "external_same_input" if args.path else "evaluation"),), config, rtol=args.rtol, atol=args.atol)
            else:
                passes = tuple(n.strip() for n in args.passes.split(",") if n.strip()) if args.passes is not None else None
                result = compile_graph(workload.graph, workload.samples, strategy=args.strategy,
                                       passes=passes, measurement=config, rtol=args.rtol, atol=args.atol)
                print("ORIGINAL\n" + str(workload.graph))
                print("\nOPTIMIZED\n" + str(result.graph))
                print("\n" + result.plan.describe())
                for record in result.records:
                    print(f"[{record.status}] {record.name}: {record.reason}")
                print(f"\nValidation: {result.validation.message}; samples={result.validation.samples}")
                print(f"Selected: {result.candidate}; {result.decision}")
                print(f"Planning {result.planning_ms:.3f} ms; backend compile {result.backend_compile_ms:.3f} ms")
                print("Outputs:", tuple(t.tolist() if t.numel() <= 16 else f"shape={tuple(t.shape)} (preview omitted)" for t in result.plan(*workload.evaluation_inputs)))
                from runtime import gpu_unavailable_reason
                print("GPU:", gpu_unavailable_reason() or "available, not requested; CPU execution only")
                return 0
        paths = save(report, args.output)
        from .experiment import markdown
        print(markdown(report))
        print("Artifacts:", *paths)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Compiler diagnostic: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
