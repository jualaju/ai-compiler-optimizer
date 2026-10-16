"""Transparent measured candidate selection; no fabricated training labels."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import time
from typing import Any
from .backend import Backend, CPUBackend, Validation, validate
from .ir import Graph, analysis
from .measurement import MeasureConfig, measure_many
from .passes import PASS_NAMES, PassManager, PassRecord, PassSnapshot, make_passes


@dataclass
class Compilation:
    original: Graph
    graph: Graph
    plan: Any
    strategy: str
    candidate: str
    records: tuple[PassRecord, ...]
    validation: Validation
    optimization_ms: float
    planning_ms: float
    backend_compile_ms: float
    decision: str
    features: dict
    calibration: dict = field(default_factory=dict)
    candidates: dict[str, Graph] = field(default_factory=dict, repr=False)
    validation_ms: float = 0.0
    total_compile_ms: float = 0.0
    snapshots: tuple[PassSnapshot, ...] = ()


def compile_graph(graph: Graph, samples=(), *, strategy="conventional", passes=None,
                  backend: Backend | None = None, measurement: MeasureConfig | None = None,
                  rtol=1e-5, atol=1e-6, capture_trace=False) -> Compilation:
    """Compile a pure inference graph. Changed candidates require runtime samples.

    Intelligent selection tunes on the first supplied sample; all samples validate
    candidates. Its wall time (including tuning) is reported as planning_ms.
    """
    if strategy not in ("baseline", "conventional", "intelligent", "manual"):
        raise ValueError("Unknown strategy; use baseline/conventional/intelligent/manual")
    if passes is not None and strategy != "manual":
        raise ValueError("Explicit pass lists require strategy='manual'")
    backend = backend or CPUBackend()
    started = time.perf_counter()
    features = analysis(graph)
    candidates = {"baseline": graph}
    records_by_name = {"baseline": ()}
    snapshots = []
    optimization_started = time.perf_counter()
    policies = {}
    if strategy == "manual":
        policies["manual"] = tuple(passes or ())
    elif strategy in ("conventional", "intelligent"):
        policies["conventional"] = PASS_NAMES
        if strategy == "intelligent":
            # Workload-aware candidate construction from operator features.
            kinds = features["operators"]
            if "fused_affine_relu" not in kinds and all(k in kinds for k in ("multiply", "add", "relu")):
                policies["cleanup_only"] = tuple(n for n in PASS_NAMES if n != "affine_relu_fusion")
                policies["fusion_only"] = ("affine_relu_fusion",)
    for name, names in policies.items():
        result = PassManager(make_passes(names)).run(graph, samples, backend=backend, rtol=rtol, atol=atol,
                                                   capture_trace=capture_trace)
        candidates[name] = result.graph
        offset = sum(len(records) for records in records_by_name.values())
        records_by_name[name] = tuple(replace(record, candidate=name) for record in result.records)
        snapshots.extend(replace(snapshot, candidate=name, record_index=snapshot.record_index+offset)
                         for snapshot in result.snapshots)
    optimization_ms = (time.perf_counter() - optimization_started) * 1000
    calibration = {}
    decision = "No optimization requested"
    chosen = "baseline"
    if strategy in ("conventional", "manual"):
        chosen = strategy
        decision = "Deterministic bounded fixed-point pass policy; failed rewrites roll back"
    elif strategy == "intelligent":
        if not samples:
            decision = "Insufficient evidence: no validation/calibration samples; baseline fallback"
        else:
            # Deduplicate structurally identical candidates; baseline wins ties.
            unique, seen = {}, set()
            compile_started = time.perf_counter()
            for name, candidate in candidates.items():
                key = candidate.fingerprint()
                if key not in seen:
                    seen.add(key)
                    check = validate(graph, candidate, samples, backend, rtol=rtol, atol=atol)
                    if check.passed:
                        plan = backend.compile(candidate)
                        unique[name] = lambda plan=plan: plan(*samples[0])
            if len(unique) > 1:
                calibration = measure_many(unique, measurement or MeasureConfig())
                calibration["scope"] = "Same-workload online calibration; not a trained predictor or held-out performance estimate"
                timings = calibration["timings"]
                best = min(timings, key=lambda n: timings[n]["median_ms"])
                baseline = timings["baseline"]
                winner = timings[best]
                margin = max(0.05 * baseline["median_ms"],
                             2 * (baseline["mad_ms"] + winner["mad_ms"]))
                gain = baseline["median_ms"] - winner["median_ms"]
                chosen = best if best != "baseline" and gain > margin else "baseline"
                decision = (f"Calibration best={best}; gain={gain:.6f} ms, required margin={margin:.6f} ms "
                            f"(max of 5% baseline and 2× summed MAD); selected {chosen}. Heuristic, not a confidence interval.")
                calibration["candidate_preparation_and_measurement_ms"] = (time.perf_counter() - compile_started) * 1000
            else:
                decision = "No distinct validated optimization candidate; baseline fallback without timing search"
    # Include optimization and candidate timing in planning overhead, not execution latency.
    planning_ms = (time.perf_counter() - started) * 1000
    validation_started = time.perf_counter()
    check = validate(graph, candidates[chosen], samples, backend, rtol=rtol, atol=atol)
    validation_ms = (time.perf_counter() - validation_started) * 1000
    if samples and not check.passed:
        raise ValueError("Final candidate failed numerical validation: " + check.message)
    backend_started = time.perf_counter()
    plan = backend.compile(candidates[chosen])
    backend_ms = (time.perf_counter() - backend_started) * 1000
    records = records_by_name.get(chosen, ())
    if strategy == "intelligent":
        records = tuple(record for name in policies for record in records_by_name[name])
    return Compilation(graph, candidates[chosen], plan, strategy, chosen, records, check,
                       optimization_ms, planning_ms, backend_ms, decision, features,
                       calibration, candidates, validation_ms, (time.perf_counter() - started) * 1000, tuple(snapshots))
