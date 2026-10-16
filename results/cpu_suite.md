# CPU compiler strategy experiment

Recorded: 2026-10-16T03:58:18.965534+00:00
Machine: Linux-6.17.0-29-generic-x86_64-with-glibc2.39; PyTorch 2.14.0+cpu; threads=1

No fitted model. Online workload-local calibration only. Family splits reserved for future model training; evaluation observations never influence decisions.

All execution columns are median milliseconds per call.

| Workload / shape | Baseline | Conventional | Intelligent | Selected | Extra planning ms |
|---|---:|---:|---:|---|---:|
| affine / [32] | 0.021525 | 0.016923 | 0.017182 | conventional | 31.042 |
| affine / [2048] | 0.023755 | 0.018950 | 0.018985 | conventional | 30.963 |
| constant / [32] | 0.026160 | 0.017720 | 0.017606 | conventional | 54.180 |
| constant / [2048] | 0.027842 | 0.019088 | 0.018943 | conventional | 59.977 |
| dead_branch / [32] | 0.027751 | 0.017815 | 0.017822 | conventional | 48.149 |
| dead_branch / [2048] | 0.029829 | 0.019420 | 0.019096 | conventional | 53.987 |
| redundant_relu / [32] | 0.015911 | 0.013085 | 0.011664 | conventional | 12.810 |
| redundant_relu / [2048] | 0.016790 | 0.012347 | 0.012141 | conventional | 12.894 |
| layout / [32] | 0.021889 | 0.011970 | 0.011884 | conventional | 8.488 |
| layout / [2048] | 0.022379 | 0.014416 | 0.012109 | conventional | 8.686 |
| residual / [32] | 0.025724 | 0.025534 | 0.025832 | baseline | 11.421 |
| residual / [2048] | 0.028459 | 0.028696 | 0.028608 | baseline | 12.485 |
| mlp / [32, 16] | 0.036590 | 0.039282 | 0.038537 | baseline | 6.968 |
| mlp / [2048, 16] | 0.122435 | 0.123387 | 0.122147 | baseline | 12.358 |

Intelligent regressions >5% versus baseline: 1/14.
Within 5% of the observed candidate oracle: 14/14 (timing noise applies).

Planning includes candidate transformations, sample checks, candidate compilation and calibration. Optimization time is a subset of planning, not an additive cost. Backend compilation is timed separately; final sample validation is timed separately. Total compilation includes these stages. Break-even estimates use extra total compilation cost. Frontend preparation includes workload setup, conversion, shape analysis and independent reference checks. Built-ins use a separate input and fresh timing batches after selection; external_same_input files reuse the supplied input, with fresh timing batches only.

Raw batch observations, randomized orders, candidate calibration, variability, graph text, pass records, tolerances and software configuration are in the accompanying JSON.

## Limitations

- CPU interpreter timings include Python overhead; not native CPU code generation
- Fused CPU nodes still materialize PyTorch intermediates
- No GPU execution or timings
- No reliable memory-use measurements collected
- Small noisy sample; does not establish novelty or general performance gains
