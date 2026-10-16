# Research notes: optimization selection

Novelty is **not established**. Constant folding, liveness DCE, safe identity
rewrites, fusion, phase ordering and autotuning are established compiler ideas.
This repository implements a small experimental substrate, not a first-of-kind
learning-guided compiler.

## Verified starting references

These primary publication pages were checked during implementation (2026-10-16).
This is a starting reading list, not a systematic literature review.

- Chen et al., **TVM: An Automated End-to-End Optimizing Compiler for Deep
  Learning**, OSDI 2018. Graph/operator optimization and learning-based cost
  modeling are already part of established tensor compiler research.
  [USENIX publication](https://www.usenix.org/conference/osdi18/presentation/chen)
- Zheng et al., **Ansor: Generating High-Performance Tensor Programs for Deep
  Learning**, OSDI 2020. Explores tensor-program schedules with search and a
  learned cost model. This project's graph-pass candidates are a much smaller
  search space and currently use direct measurements rather than a fitted model.
  [USENIX publication](https://www.usenix.org/conference/osdi20/presentation/zheng)
- Trofin et al., **MLGO: a Machine Learning Guided Compiler Optimizations
  Framework**, 2021. Demonstrates learned compiler decisions in LLVM; the initial
  case study concerns inlining for size, a different domain/objective from our
  CPU graph latency selection.
  [Author publication page](https://research.google/pubs/mlgo-a-machine-learning-guided-compiler-optimizations-framework/)

Implementation references: [ONNX reference evaluator](https://onnx.ai/onnx/api/reference.html),
[ONNX Reshape semantics](https://onnx.ai/onnx/operators/onnx__Reshape.html).

## Falsifiable question

Does conservative measured selection reduce held-out latency or regression rate
relative to a fixed safe policy **after including decision cost**, on a defined
CPU/workload distribution? A null/negative result is useful: fixed policies may
already dominate this small candidate space, and Python bookkeeping may dwarf
any kernel benefit. CPU affine fusion has no allocation-free kernel here.

## What the implementation can test now

The suite contains beneficial cleanup cases, shared-output graphs that block
fusion, a no-rewrite dense MLP, and shape variants where tensor cost can dwarf
small dispatch savings. These are deterministic workloads, not fabricated timing
labels. The selector sees calibration measurements only. Reports retain all
candidate observations and fresh evaluation measurements, regressions, observed
candidate regret and compile-cost amortization estimates. Shape/model family
splits are recorded for future learning; no training uses them yet.

## Next experiment and rejection criteria

1. Freeze candidates and threshold before collecting test observations. Run at
   least 20 separate-process repetitions on one controlled CPU, then another CPU.
   Randomize strategy compile order too; current suite fixes that order and
   records only randomized execution batch order.
2. Expand workload families (MLP widths/depths, genuine branching, reductions,
   broadcasts, transposed matmuls), shape scales, and independent seeds. Avoid
   treating variants of a single graph as independent research evidence.
3. Compare no optimization, fixed policy, conservative online selection and
   always-baseline fallback. Include total time at 1/10/100/1000/10000 calls,
   selection overhead, p50/p95 regressions and uncertainty across runs.
4. Reject the practical hypothesis for a deployment call budget if selection's
   total cost is not better, or if its regression rate exceeds the fixed policy.
   Report failures and no-benefit workloads; never remove inconvenient results.
5. Only with enough independently measured families, fit a small decision tree
   or nearest-neighbor cost model on development families. Use nested/grouped
   validation to choose features/thresholds and leave test families untouched.
   Fit preprocessing only on training data; measure prediction overhead and
   compare against online tuning. Uncertain/unseen shapes should fall back.

## Literature work still required before novelty claims

Read full papers and their related work; review tensor compiler auto-scheduling,
compiler pass-order selection, graph fusion profitability, measurement-guided
search, contextual bandits and compiler cost models. Check subsequent work,
reproducibility artifacts and differences in target hardware/search spaces.
Identify a precise unaddressed question and independent baselines. A combination
of known passes with a selector, or a dashboard, alone does not establish novelty.
