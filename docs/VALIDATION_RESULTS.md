# Implementation validation and observed results

Verified on 2026-10-16 using the repository `.venv`: Python 3.12, CPU PyTorch
2.14.0+cpu, ONNX 1.23.2. See JSON environment sections for exact machine,
package versions, thread counts, PyTorch build configuration and source hashes.
System Python lacks PyTorch, so the environment was activated for tests/CLI;
`python3 demo.py` independently reuses that environment as before.

## Commands actually executed

- `python3 -m unittest -v`: **33 tests run, 32 passed, 1 GPU test skipped**.
  Includes three optional ONNX tests against reference execution/conversion,
  full FX MLP capture, parameter snapshots, shape/IR rejection, every pass,
  shared/multiple outputs, broadcasting/empty shapes, special floating-point
  values, signed-zero comparison, rollback, cycles/iteration limits, backend
  injection, policy margins, all seven workloads through both native/FX frontends,
  and the complete strategy/report pipeline. Policy-margin unit tests use mocked
  observations; these are isolated tests, never benchmark/training observations.
- `python3 -m mlcompiler --suite --output results/cpu_suite.json`: 14 real CPU
  workload comparisons, 7 timing batches × 20 iterations after 3 warm-ups,
  single CPU thread, two calibration inputs and a different evaluation input.
- `python3 -m mlcompiler --source torch --workload mlp --size 32 --compare
  --output results/fx_mlp.json`: full FX MLP strategy comparison.
- Native file CLI: `examples/native_affine.json` with
  `examples/native_inputs.json`, compilation and execution succeeded.
- ONNX file CLI: a locally generated opset-18 affine model, native input JSON,
  strategy comparison with `--rtol 0 --atol 0`. Independent ONNX reference and
  all original/optimized checks passed. Raw output was written to
  `/tmp/onnx-comparison.json`; this was a file-path integration smoke test.
- `python3 demo.py`: existing default trained-MLP dashboard completed using EOF
  defaults. Existing manual mode also completed with Enter defaults.
- `python3 demo.py --pipeline ...`: native and FX/manual modes, intelligent
  comparison/report generation, noninteractive rendering and 48-column rendering
  completed. A PTY run exercised S/D/G/P inspection, N/U scrolling, B real CPU
  benchmark/report generation and Q quit. The rapid-key buffering issue found
  during this check was fixed and rechecked.
- `python3 examples/trained_mlp.py`: original 250-step training demo passed;
  original/fused hidden outputs and full readout predictions matched on 66
  validation inputs; reported maximum absolute error **0.0**.
- `python3 benchmark.py`: explicitly skipped all GPU benchmarks; no GPU timings.
- `python3 -m compileall -q mlcompiler demo.py test_pipeline.py` and
  `git diff --check`: passed. `pip check`: no broken requirements.

## Final suite observations

[Human report](../results/cpu_suite.md) and [raw observations](../results/cpu_suite.json).
All sample and evaluation numerical comparisons passed. The selector chose the
conventional graph in 10 cases, and baseline in the 4 residual/MLP cases.
There was **1/14 measured regressions >5% versus baseline**. All 14 selected
executions were within 5% of the separately measured candidate oracle. These
counts are observations from one process/run; they are not statistical guarantees.

Example medians, milliseconds/call:

| Workload | Baseline | Conventional | Intelligent |
|---|---:|---:|---:|
| affine, 32 elements | 0.021525 | 0.016923 | 0.017182 |
| shared residual, 32 elements | 0.025724 | 0.025534 | 0.025832 |
| MLP, batch 32 / input width 16 | 0.036590 | 0.039282 | 0.038537 |

The intelligent MLP execution was about 5.3% slower than the separately measured
baseline even though baseline was selected. This is timing variability over
identical graphs, illustrating why repeated runs and uncertainty matter.
Extra planning was approximately 7–60 ms; measured savings in beneficial cases
suggest roughly 800–6900 calls to amortize extra total compilation versus
baseline. This does not show an advantage over conventional optimization:
most selected candidates are exactly the conventional graph.

Earlier observations are retained as `results/cpu_suite_run1.{json,md}` and
`results/cpu_suite_pilot.{json,md}`. The pilot predates the final runtime metadata
checks and workload cleanup; it must not be pooled as if it used the final code.
The FX report is an additional smoke-sized experiment, not a second independent
research dataset. Reports include their actual raw timing batches.

## Limits and next experiment

There is no fitted selector, GPU execution, native CPU code generator, reliable
memory measurement, general Triton backend or established novelty. ONNX and FX
support are deliberately bounded (see OPERATORS.md). Per-pass sample checks
protect the tested inputs; they do not prove every input correct.

Freeze the current policy and collect multiple fresh-process repetitions across
additional independent graph families and CPUs. Compare total cost at deployment
call budgets, not only steady-state latency. Randomize compilation order as well
as timing order, quantify regressions/uncertainty, and test whether selection adds
anything over the fixed policy. Only then consider fitting an interpretable model
using family-grouped training/held-out splits.
