# CPU compiler strategy experiment

Recorded: 2026-10-16T03:52:29.203442+00:00
Machine: Linux-6.17.0-29-generic-x86_64-with-glibc2.39; PyTorch 2.14.0+cpu; threads=1

No fitted model. Online workload-local calibration only. Family splits reserved for future model training; evaluation observations never influence decisions.

All execution columns are median milliseconds per call.

| Workload / shape | Baseline | Conventional | Intelligent | Selected | Extra planning ms |
|---|---:|---:|---:|---|---:|
| affine / [32] | 0.017636 | 0.016246 | 0.015499 | conventional | 29.518 |
| affine / [2048] | 0.019697 | 0.018242 | 0.017411 | conventional | 28.140 |
| constant / [32] | 0.020162 | 0.016474 | 0.015766 | conventional | 54.495 |
| constant / [2048] | 0.022463 | 0.017419 | 0.017061 | conventional | 53.966 |
| dead_branch / [32] | 0.025366 | 0.019578 | 0.015991 | conventional | 48.168 |
| dead_branch / [2048] | 0.024221 | 0.017519 | 0.017318 | conventional | 51.523 |
| redundant_relu / [32] | 0.013723 | 0.010502 | 0.010613 | conventional | 21.740 |
| redundant_relu / [2048] | 0.016154 | 0.013812 | 0.012990 | conventional | 15.752 |
| layout / [32] | 0.017706 | 0.010461 | 0.010258 | conventional | 10.560 |
| layout / [2048] | 0.017650 | 0.010118 | 0.010184 | conventional | 14.808 |
| residual / [32] | 0.020444 | 0.020740 | 0.020633 | baseline | 19.166 |
| residual / [2048] | 0.028949 | 0.026824 | 0.026837 | baseline | 11.161 |
| mlp / [32, 16] | 0.042398 | 0.032304 | 0.031627 | baseline | 30.320 |
| mlp / [2048, 16] | 0.125245 | 0.130392 | 0.127266 | baseline | 70.241 |

Intelligent regressions >5% versus baseline: 0/14.
Within 5% of the observed candidate oracle: 13/14 (timing noise applies).

Planning includes candidate transformations, sample checks, candidate compilation and calibration. Optimization time is a subset of planning, not an additive cost. Backend compilation is timed separately; final sample validation is timed separately. Total compilation includes these stages. Break-even estimates use extra total compilation cost. Frontend preparation includes workload setup, conversion, shape analysis and independent reference checks. Execution uses a separate input and fresh timing batches after selection.

Raw batch observations, randomized orders, candidate calibration, variability, graph text, pass records, tolerances and software configuration are in the accompanying JSON.

## Limitations

- CPU interpreter timings include Python overhead; not native CPU code generation
- Fused CPU nodes still materialize PyTorch intermediates
- No GPU execution or timings
- No reliable memory-use measurements collected
- Small noisy sample; does not establish novelty or general performance gains
