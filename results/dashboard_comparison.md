# CPU compiler strategy experiment

Recorded: 2026-10-16T05:18:07.561738+00:00
Machine: Linux-6.17.0-29-generic-x86_64-with-glibc2.39; PyTorch 2.14.0+cpu; threads=1

No fitted model. Online workload-local calibration only. Family splits reserved for future model training; evaluation observations never influence decisions.

All execution columns are median milliseconds per call.

| Workload / shape | Baseline | Conventional | Intelligent | Selected | Extra planning ms |
|---|---:|---:|---:|---|---:|
| constant / [32] | 0.025794 | 0.017232 | 0.025716 | baseline | 185.448 |

Intelligent regressions >5% versus baseline: 0/1.
Within 5% of the observed candidate oracle: 0/1 (timing noise applies).

Planning includes candidate transformations, sample checks, candidate compilation and calibration. Optimization time is a subset of planning, not an additive cost. Backend compilation is timed separately; final sample validation is timed separately. Total compilation includes these stages. Break-even estimates use extra total compilation cost. Frontend preparation includes workload setup, conversion, shape analysis and independent reference checks. Built-ins use a separate input and fresh timing batches after selection; external_same_input files reuse the supplied input, with fresh timing batches only.

Raw batch observations, randomized orders, candidate calibration, variability, graph text, pass records, tolerances and software configuration are in the accompanying JSON.

## Limitations

- CPU interpreter timings include Python overhead; not native CPU code generation
- Fused CPU nodes still materialize PyTorch intermediates
- No GPU execution or timings
- No reliable memory-use measurements collected
- Small noisy sample; does not establish novelty or general performance gains
