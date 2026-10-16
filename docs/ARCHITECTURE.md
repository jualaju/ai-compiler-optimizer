# Common inference compiler architecture

The main local interface is now [Compiler Studio](STUDIO.md), launched with
`python3 studio.py`. The terminal dashboard remains a debugging interface.
Both call the same compiler APIs. Studio's optional `capture_trace=True` retains
accepted per-pass graph snapshots; it does not implement separate transformations.

```
 native JSON/dict ─┐
 PyTorch FX ───────┼─> Graph / Node / TensorSpec
 optional ONNX ───┘          │
                   topology + static shape analysis
                            │
                 candidate planner / pass manager
                            │
              sample checks / rollback / fixed point
                            │
              Backend.compile(graph) -> callable
                            │
                CPUPlan / PyTorch tensor operations
                            │
               fresh evaluation timings -> JSON + Markdown
```

## Migration and boundaries

The original `compiler.py`, `runtime.py`, generated artifacts, GPU benchmark and
trained example are retained. `from_legacy` bridges the old Graph/Op types to
the common IR. `demo.py` adds a third input mode and `--pipeline`; the first two
modes remain intact. `demo_ui.py` retains the existing dashboard widgets.

`mlcompiler.ir` contains no FX nodes or ONNX protos. Shapes and operation meaning
are common across frontends. Tensor constants and dtype descriptors currently
use torch objects: framework independence means multiple source frontends, not
a runtime without PyTorch. The required CPU backend is a planned interpreter,
not a native machine-code compiler. `Backend` is a small injected protocol;
there is no plugin framework. A general new-IR GPU backend is future work.

- `ir.py`: tensor metadata, nodes, topology/invariants, fingerprints, statistics.
- `frontends.py`: bounded native, FX, ONNX lowering, optional import isolation.
- `backend.py`: static meta-tensor inference, CPU execution plans, comparisons.
- `passes.py`: pure graph transformations and validated bounded iteration.
- `planner.py`: policy candidates and conservative workload-local calibration.
- `measurement.py`: raw interleaved timing batches and environment information.
- `workloads.py`: seeds, input samples and independent PyTorch references.
- `experiment.py`: strategy comparison, held-out checks, reports and raw data.
- `dashboard.py`, `__main__.py`: presentation and independent CLI entry points.

## IR invariants and semantics

Nodes have unique value names, a single result, operand names, optional shape and
dtype, attributes, constant storage and source diagnostics. Graph outputs can be
multiple tensor values. Node storage is topological: `validate` rejects dangling
uses, cycles/forward references, invalid arity/attributes, unsupported dtypes,
invalid constants and missing outputs. `topological()` validates and returns
that order. Static shape inference uses meta tensors and checks known metadata.
It does not evaluate real input values and does not prove numerical equivalence.

All operations are pure inference operations. Mutation, gradient execution,
randomness, I/O and state updates cannot be represented. Values are observable;
alias/stride identity and NaN payload bits are not. Signed zeros, NaNs as values,
Inf, shape and dtype are checked. No fast-math, reassociation, x+0 or x*0 rewrite.
Parameters are detached CPU snapshots at capture. Graph objects are frozen, but
attribute dictionaries and tensors must be treated as immutable by API callers.
Do not mutate returned constants or input/output storage between executions.

Broadcasting follows the CPU tensor operators. A fused affine node preserves
operand and arithmetic order. The CPU evaluator still executes multiply, add,
and ReLU separately, allocating intermediates. Fusion reduces IR traversal and
dispatch bookkeeping only. Removed IR values are not reported as measured
CPU tensor allocation savings. The estimated byte feature sums logical IR
compute outputs; it is neither peak memory nor a measured allocation count.

## Pass contract and acceptance

Each pass has a name, conservative `check(graph)` precondition, and an `apply`
returning graph, number of rewrites, and reason. Exact dependency checks happen
in apply. Records mark exact applicability (whether a change was proposed),
applied/skipped/failed status, iteration, before/after compute counts, candidate
policy and whether sample validation ran and passed. A skip can indicate absent
patterns, size limits, or no supplied validation samples.

Implemented rules:

1. Constant folding: all operands constant; statically known result at most 1 MiB.
   No input-dependent folding, RNG or arbitrary Python evaluation.
2. DCE: reverse liveness from **all** graph outputs; preserve input signature.
3. Algebraic simplification: identity elimination and ReLU(ReLU(x)).
4. Layout simplification: identity permutation or inverse permutation pairs;
   earlier shared views survive until dead-code analysis removes only unused ones.
5. Affine-ReLU fusion: ordered multiply -> add -> relu, single-use internal
   results and no observable intermediate outputs, broadcasting allowed.

Default ordering is the list above. Later rewrites can expose dead nodes or new
patterns, so the manager repeats to a fixed point. Fingerprints include constant
bytes and metadata. A repeated graph is rejected; a maximum of eight rounds
bounds execution and reports nonconvergence. Every changed graph passes invariant
and shape checks and numerical comparison on the supplied samples before
acceptance. Failed transformations roll back, retaining diagnostics. With no
samples, applicable rewrites are skipped and validation says **not tested**.
Input samples are finite tests, never a proof for all possible inputs.

Tolerances are explicit (`rtol=1e-5`, `atol=1e-6` defaults). Equal NaNs are
permitted; signed-zero signs must agree. Original/optimized checks share CPU
operator semantics, so frontend lowering is tested separately against original
PyTorch models and ONNX's independent reference evaluator.

## Selection and measurement

- Baseline preserves the unoptimized graph (same CPU executor and checks).
- Conventional applies the fixed safe policy until convergence.
- Manual applies only named passes, using the same checks.
- Intelligent enumerates baseline and conventional; operator features can add
  cleanup-only and fusion-only candidates. It deduplicates structural equivalents,
  validates candidates, and times them on one calibration input. The minimum
  observed median is accepted only if its gain exceeds both 5% of baseline and
  twice the sum of baseline/winner MADs. Otherwise baseline wins.

This is interpretable online autotuning, not a trained predictor. The threshold
is a conservative heuristic, not a statistical confidence interval. Features
include operators, shapes, dtypes, shared uses, constants and estimated logical
output bytes. Only operator features gate candidate families currently; timing
evidence determines the final choice. Richer shape-based candidate generation,
caching, learned models and formal uncertainty estimates remain research work.

Built-ins use two calibration samples and a third different evaluation sample.
Benchmarking evaluates selection only on fresh timing batches after the decision
is frozen. File-based workloads currently reuse the one supplied input, explicitly
labelled `external_same_input`; they do not claim input holdout. Future learning
must split whole families/model identities, never variants of one model across
train/test. The suite records family splits but fits no model.

Compilation stages are separately reported: shared frontend/workload preparation,
planning (including per-pass checks and online calibration), backend plan
preparation, final validation, and total compile cost. Optimization is a subset
of planning; adding both double-counts. Raw calibration and evaluation batches,
randomized batch order, median, MAD, standard deviation, range, tolerances,
decisions and environment configuration are retained. Cold import, CPU frequency,
background load and allocator state can affect observations; do repeated process
runs before conclusions. CPU threads default to one for CLI comparisons.

Regret is measured against the best **enumerated** candidate in held-out batches,
not an unattainable global optimum. Negative observed regret is possible from
noise because selected/candidate functions are measured separately. Regressions
use a disclosed 5% threshold; no significance is implied. Break-even calls divide
extra total compile cost by observed per-call savings; no estimate is given
when there is no saving. Memory is not measured. GPU timing is never inferred.
