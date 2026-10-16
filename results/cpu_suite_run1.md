# CPU compiler strategy experiment

Recorded: 2026-10-16T03:56:18.770628+00:00
Machine: Linux-6.17.0-29-generic-x86_64-with-glibc2.39; PyTorch 2.14.0+cpu; threads=1

No fitted model. Online workload-local calibration only. Family splits reserved for future model training; evaluation observations never influence decisions.

All execution columns are median milliseconds per call.

| Workload / shape | Baseline | Conventional | Intelligent | Selected | Extra planning ms |
|---|---:|---:|---:|---|---:|
| affine / [32] | 0.025714 | 0.017059 | 0.016945 | conventional | 23.349 |
| affine / [2048] | 0.024689 | 0.020531 | 0.018806 | conventional | 31.030 |
| constant / [32] | 0.027584 | 0.018530 | 0.018465 | conventional | 53.883 |
| constant / [2048] | 0.029262 | 0.020023 | 0.019629 | conventional | 63.278 |
| dead_branch / [32] | 0.032815 | 0.018536 | 0.017607 | conventional | 50.424 |
| dead_branch / [2048] | 0.030547 | 0.019077 | 0.019563 | conventional | 54.083 |
| redundant_relu / [32] | 0.016324 | 0.012251 | 0.011932 | conventional | 18.493 |
| redundant_relu / [2048] | 0.016461 | 0.012171 | 0.011919 | conventional | 12.710 |
| layout / [32] | 0.023843 | 0.012053 | 0.012296 | conventional | 8.652 |
| layout / [2048] | 0.021705 | 0.012118 | 0.011945 | conventional | 13.797 |
| residual / [32] | 0.025935 | 0.026236 | 0.026579 | baseline | 11.411 |
| residual / [2048] | 0.028838 | 0.028996 | 0.028902 | baseline | 15.927 |
| mlp / [32, 16] | 0.036911 | 0.038201 | 0.036994 | baseline | 7.071 |
| mlp / [2048, 16] | 0.126878 | 0.124562 | 0.126725 | baseline | 8.081 |

Intelligent regressions >5% versus baseline: 0/14.
Within 5% of the observed candidate oracle: 14/14 (timing noise applies).

Planning includes candidate transformations, sample checks, candidate compilation and calibration. Optimization time is a subset of planning, not an additive cost. Backend compilation is timed separately; final sample validation is timed separately. Total compilation includes these stages. Break-even estimates use extra total compilation cost. Frontend preparation includes workload setup, conversion, shape analysis and independent reference checks. Built-ins use a separate input and fresh timing batches after selection; external_same_input files reuse the supplied input, with fresh timing batches only.

Raw batch observations, randomized orders, candidate calibration, variability, graph text, pass records, tolerances and software configuration are in the accompanying JSON.

## Limitations

- CPU interpreter timings include Python overhead; not native CPU code generation
- Fused CPU nodes still materialize PyTorch intermediates
- No GPU execution or timings
- No reliable memory-use measurements collected
- Small noisy sample; does not establish novelty or general performance gains
