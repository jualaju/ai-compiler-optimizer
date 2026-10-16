"""Auditable CPU strategy comparison; calibration and evaluation are separate."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import time
from .backend import CPUBackend, compare_outputs, validate
from .ir import analysis
from .measurement import MeasureConfig, environment, measure_many
from .planner import compile_graph


def compare(workload, config=None, *, rtol=1e-5, atol=1e-6) -> dict:
    config = config or MeasureConfig()
    compiled = {}
    for strategy in ("baseline", "conventional", "intelligent"):
        compiled[strategy] = compile_graph(workload.graph, workload.samples,
            strategy=strategy, measurement=config, rtol=rtol, atol=atol)
    # The held-out input was never used for calibration or candidate acceptance.
    checks = {}
    for name, result in compiled.items():
        check = validate(workload.graph, result.graph, (workload.evaluation_inputs,), rtol=rtol, atol=atol)
        checks[name] = asdict(check)
        if not check.passed:
            raise ValueError(f"Held-out runtime validation failed for {name}: {check.message}")
    functions = {name: (lambda result=result: result.plan(*workload.evaluation_inputs))
                 for name, result in compiled.items()}
    # Evaluate all calibrated distinct candidates too, so selection regret has an
    # explicit candidate set. These observations are never fed back into selection.
    selected = compiled["intelligent"]
    unique, seen = {}, set()
    for name, graph in selected.candidates.items():
        key = graph.fingerprint()
        if key not in seen:
            check = validate(workload.graph, graph, (workload.evaluation_inputs,), rtol=rtol, atol=atol)
            if not check.passed: raise ValueError(f"Held-out candidate {name} failed: {check.message}")
            seen.add(key)
            plan = CPUBackend().compile(graph)
            unique[name] = graph
            functions["candidate/" + name] = lambda plan=plan: plan(*workload.evaluation_inputs)
    measured = measure_many(functions, config)
    timings = measured["timings"]
    base = timings["baseline"]["median_ms"]
    conv = timings["conventional"]["median_ms"]
    intel = timings["intelligent"]["median_ms"]
    oracle_name = min(unique, key=lambda n: timings["candidate/" + n]["median_ms"])
    oracle = timings["candidate/" + oracle_name]["median_ms"]
    planning_extra = max(0, selected.planning_ms - compiled["baseline"].planning_ms)
    compile_extra = max(0, selected.total_compile_ms - compiled["baseline"].total_compile_ms)
    saving = base - intel
    rows = {}
    for strategy, result in compiled.items():
        rows[strategy] = {"candidate": result.candidate, "decision": result.decision,
            "planning_ms": result.planning_ms, "optimization_ms": result.optimization_ms,
            "backend_compile_ms": result.backend_compile_ms,
            "final_validation_ms": result.validation_ms, "total_compile_ms": result.total_compile_ms,
            "compile_time_invariants": "topology, arity, metadata, static shapes checked",
            "sample_validation": asdict(result.validation), "heldout_validation": checks[strategy],
            "graph": str(result.graph), "analysis": analysis(result.graph),
            "pass_records": [asdict(record) for record in result.records],
            "calibration": result.calibration, "execution": timings[strategy]}
    return {"name": workload.name, "family": workload.family, "seed": workload.seed, "frontend": workload.frontend,
        "frontend_preparation_ms": workload.frontend_ms,
        "frontend_reference_checked": workload.reference_checked,
        "input_parameters": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in workload.evaluation_inputs],
        "original_graph": str(workload.graph), "features": analysis(workload.graph), "strategies": rows,
        "original_graph_sha256": workload.graph.fingerprint(),
        "evaluation": measured,
        "selection": {"candidate_set": list(unique), "evaluation_oracle": oracle_name,
            "regret_ms": intel - oracle, "baseline_speedup_ratio": base / intel,
            "conventional_speedup_ratio": base / conv,
            "regression_vs_baseline": intel > base * 1.05,
            "regression_threshold_fraction": 0.05,
            "selected_within_5pct_of_observed_oracle": intel <= oracle * 1.05,
            "extra_planning_ms": planning_extra,
            "extra_total_compile_ms": compile_extra,
            "break_even_calls_vs_baseline": None if saving <= 0 else compile_extra / saving,
            "note": "Oracle/regret are noisy held-out observations over enumerated candidates, not ground truth or a trained-model accuracy claim"}}


def run_suite(workloads, config=None, *, rtol=1e-5, atol=1e-6) -> dict:
    config = config or MeasureConfig()
    rows = []
    for workload, split in workloads:
        row = compare(workload, config, rtol=rtol, atol=atol)
        row["split"] = split
        rows.append(row)
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment(), "measurement": asdict(config),
            "learning": "No fitted model. Online workload-local calibration only. Family splits reserved for future model training; evaluation observations never influence decisions.",
            "limitations": ["CPU interpreter timings include Python overhead; not native CPU code generation",
                "Fused CPU nodes still materialize PyTorch intermediates", "No GPU execution or timings",
                "No reliable memory-use measurements collected", "Small noisy sample; does not establish novelty or general performance gains"],
            "workloads": rows}


def markdown(report) -> str:
    env = report["environment"]
    lines = ["# CPU compiler strategy experiment", "", f"Recorded: {report['created_utc']}",
             f"Machine: {env['platform']}; PyTorch {env['torch']}; threads={env['torch_threads']}", "",
             report["learning"], "", "All execution columns are median milliseconds per call.", "",
             "| Workload / shape | Baseline | Conventional | Intelligent | Selected | Extra planning ms |",
             "|---|---:|---:|---:|---|---:|"]
    for row in report["workloads"]:
        s = row["strategies"]
        lines.append(f"| {row['name']} / {row['input_parameters'][0]['shape']} | " +
            " | ".join(f"{s[n]['execution']['median_ms']:.6f}" for n in ("baseline", "conventional", "intelligent")) +
            f" | {s['intelligent']['candidate']} | {row['selection']['extra_planning_ms']:.3f} |")
    rows = report["workloads"]
    count = sum(r["selection"]["regression_vs_baseline"] for r in rows)
    within = sum(r["selection"]["selected_within_5pct_of_observed_oracle"] for r in rows)
    lines += ["", f"Intelligent regressions >5% versus baseline: {count}/{len(rows)}.",
              f"Within 5% of the observed candidate oracle: {within}/{len(rows)} (timing noise applies).", "",
              "Planning includes candidate transformations, sample checks, candidate compilation and calibration. "
              "Optimization time is a subset of planning, not an additive cost. Backend compilation is timed separately; "
              "final sample validation is timed separately. Total compilation includes these stages. "
              "Break-even estimates use extra total compilation cost. Frontend preparation includes workload setup, "
              "conversion, shape analysis and independent reference checks. "
              "Built-ins use a separate input and fresh timing batches after selection; "
              "external_same_input files reuse the supplied input, with fresh timing batches only.", "",
              "Raw batch observations, randomized orders, candidate calibration, variability, graph text, "
              "pass records, tolerances and software configuration are in the accompanying JSON.", "",
              "## Limitations", ""]
    lines.extend("- " + limitation for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def save(report: dict, output: str | Path) -> tuple[Path, Path]:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    human = path.with_suffix(".md")
    human.write_text(markdown(report))
    return path, human
