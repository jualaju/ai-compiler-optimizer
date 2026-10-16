# A Framework-Independent ML Compiler with Intelligent Optimization Selection

A small inference compiler with native graph, PyTorch FX and optional ONNX
frontends, one typed common IR, validated graph optimizations and a CPU execution
backend. It also preserves the original affine-ReLU Triton demonstration and
trained hidden-layer example.

The research question is whether workload-aware selection improves measured
execution or reduces regressions enough to justify decision overhead. The
current selector uses transparent online calibration, **not a trained model**.
No general performance or novelty claim is established.

```text
Native / FX / ONNX → common IR → analysis → planner → safe passes
                                                      ↓
                    JSON + Markdown ← CPU timings ← validation + execution plan
```

## Setup (Linux / Python 3.12)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt
python3 -m unittest -v
```

The core needs CPU PyTorch, NumPy and Rich. ONNX is optional:

```bash
python3 -m pip install -r requirements-onnx.txt
```

A missing ONNX installation does not prevent native/FX use; optional tests skip
and ONNX input produces an installation diagnostic. No CUDA dependency is needed.
Activate `.venv` for APIs, tests and CLI; `demo.py` also reuses the local `.venv`
when launched with system Python lacking project dependencies.

## Local Compiler Studio (main interface)

After setup, run from this repository:

```bash
python3 studio.py
```

Open **http://127.0.0.1:8765** in a local browser. Use `--port 8766` if the
default port is occupied. The launcher reuses `.venv` when system Python lacks
PyTorch. No account, paid service, Node.js build, or internet connection is
required at runtime. Cytoscape.js 3.30.4 and its MIT license are vendored locally.

1. Choose a native/FX built-in, native JSON graph, or supported ONNX file.
2. **Capture workload** executes the original workload and exposes its actual IR.
3. Inspect **Graph Explorer**: pan, wheel zoom, Zoom +/−, fit, reset layout,
   search/select nodes, and compare original/optimized graphs side by side.
4. Choose conventional, intelligent, baseline, or manual passes, then
   **Run optimization**. **Optimization Passes → Inspect step** shows the actual
   accepted graph before/after that pass. Rejected/skipped transformations leave
   the graph unchanged. Intelligent candidate traces are labelled separately
   from the final selected graph.
5. **Validate outputs** checks evaluation inputs with explicit tolerances and
   shows bounded previews of real original/optimized output tensors.
6. **Compare → Benchmark original vs selected** validates first, then collects
   genuine CPU timing batches for the exact selected graph, including manual
   policies. Raw batches, variability, method, and environment are inspectable
   and downloadable as JSON. Planning/backend preparation stay separate.
7. **IR Inspector**, **Support Matrix**, and **Run History** expose metadata,
   restrictions and completed/failed actions. History is in memory for this
   server session; download runs you want to retain. Each captured run opens
   its latest successful compilation; action logs retain earlier results.

Graph nodes distinguish inputs, operations, constants, known weight/bias roles,
and graph outputs. Edges come from every IR operand (including shared/duplicate
uses), with shapes/dtypes. Node details include attributes, tensor previews,
parameter provenance and source metadata. Unknown metadata is labelled unknown.
Correspondence uses preserved IR value IDs; arbitrary renamed values are not
guessed to correspond. Selecting a graph output highlights contributing values.

Start with the **native constant** example: it exercises constant folding,
dead-value cleanup and fusion. Other controlled examples:

| Workload | What to inspect |
|---|---|
| `constant` | Native `scale × extra` becomes a computed constant; FX may fold it during tracing already |
| `dead_branch` | An actual unused dot product is removed |
| `affine` | Single-use multiply/add/ReLU becomes a fused IR node |
| `mlp` | Five necessary matmul/add/ReLU operations remain unchanged; no implemented rewrite is applicable |
| `residual` | Shared multiply result blocks fusion for correctness |
| `redundant_relu`, `layout` | Safe repeated-ReLU and inverse-permutation cleanup |

ONNX mode reads user-entered paths on the **server machine** and needs the
optional requirements. Native JSON mode provides an editable example; graph
construction and all transformations still run through the compiler APIs.
Workload edits require recapture, so controls cannot silently optimize stale data.
Failed operations preserve the last good graph and display their real diagnostic.

This is a single-user loopback development server, not a hosted multiuser service.
CPU fusion still allocates PyTorch intermediates; no GPU execution or memory
measurements are claimed. Large graphs need zoom/pan/search. See
[Studio architecture and audit](docs/STUDIO.md).

## Terminal dashboard (debugging interface)

```bash
python3 demo.py
```

Modes 1 and 2 preserve manual affine-ReLU and trained hidden-layer workflows.
Mode 3 opens the common compiler: choose native/FX built-ins or an ONNX file,
then baseline, conventional, manual, intelligent or comparison policy. Native
files are also selectable with `--path`. Passes show applicability, changes,
skips/failures, reasons and sample-validation status. The original panel style
and interactive navigation remain; small terminals use a compact overview.

```bash
python3 demo.py --pipeline --source torch --workload mlp --size 32 --strategy conventional
python3 demo.py --pipeline --source native --workload constant --size 256 --strategy intelligent --compare
python3 demo.py --pipeline --source native --path examples/native_affine.json --inputs examples/native_inputs.json --strategy manual --passes affine_relu_fusion
```

Common dashboard keys: O original graph, G optimized graph, D pass diagnostics,
P execution plan, S support matrix, B collect/inspect CPU strategy measurements,
A overview, N/U scroll details, Q quit. Reports default to
`results/dashboard_comparison.json` and `.md`; `--output` changes the path.
Noninteractive input renders the overview, pass records and execution plan.
Both terminal modes share unbuffered key handling. Unknown keys show feedback,
invalid menu choices are retried, and failed view/benchmark actions retain the
last view. Press `?` for help. Legacy modes use K kernel, D details, G overview,
N/P scroll, Q quit; common mode uses the keys above. The common mode defaults
to the constant-folding example rather than the no-rewrite MLP.

## Studio verification

Core and HTTP tests need only the project dependencies:

```bash
source .venv/bin/activate
python3 -m unittest -v
```

Optional real Chromium interaction tests (not needed to run Studio):

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m playwright install chromium
python3 tests/studio_browser.py
```

On a minimal Linux install, Playwright may also need its documented browser
system dependencies (`python3 -m playwright install --with-deps chromium`).
Browser tests start an isolated ephemeral local server and check graph canvas
selection, correspondence, zoom, pass snapshots, tabs/keyboard, manual policies,
real timing, JSON downloads, no-op graphs, errors, history, and mobile layout.

## CPU CLI and API

```bash
python3 -m mlcompiler --source torch --workload mlp --size 32
python3 -m mlcompiler --workload affine --strategy intelligent
python3 -m mlcompiler --workload constant --strategy manual --passes constant_folding,dead_code_elimination
python3 -m mlcompiler --source native --path examples/native_affine.json --inputs examples/native_inputs.json
python3 -m mlcompiler --source onnx --path model.onnx --inputs inputs.json
python3 -m mlcompiler --workload residual --size 2048 --compare --output results/residual.json
python3 -m mlcompiler --suite --threads 1 --warmup 3 --batches 7 --iterations 20 --output results/cpu_suite.json
```

`--rtol` and `--atol` set numerical tolerances. Baseline, conventional and
intelligent comparisons use the same executor, inputs and measurement procedure.
The ONNX file path also checks converted outputs with ONNX ReferenceEvaluator.
File-based experiments currently reuse the supplied input for calibration and
evaluation (labelled `external_same_input`); built-in experiments use separate
calibration and held-out evaluation inputs.

```python
import torch
from mlcompiler import pytorch, compile_graph

model = torch.nn.Sequential(
    torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
).eval()
samples = ((torch.randn(4, 3),), (torch.randn(4, 3),))
graph = pytorch(model, samples[0])
compiled = compile_graph(graph, samples, strategy="conventional")
outputs = compiled.plan(*samples[0])  # always a tuple of tensor outputs
print(compiled.graph)
print(compiled.validation)
```

Parameters are detached snapshots. The common IR contains no FX/ONNX objects;
it currently uses torch tensors/dtypes for constant storage and execution.
Framework-independent input does not mean independence from the PyTorch runtime.
This is a graph compiler with a CPU interpreter backend, not CPU machine-code
or general-purpose program compilation.

## Implemented passes and policies

- Constant folding of bounded, fully static expressions.
- Dead-code elimination preserving all outputs and the input signature.
- Identity and repeated-ReLU simplification; no unsafe floating-point identities.
- Affine-ReLU fusion of single-use chains, with broadcasting and order preserved.
- Identity/inverse-permutation elimination after shape analysis.

Passes run to a bounded fixed point. Changed candidates require invariant checks
and runtime sample comparison before acceptance. A failure rolls back and is
reported. No input samples means no accepted rewrites and **not tested** status.
Static metadata validation is distinct from sample-based numerical testing.

Intelligent selection enumerates a small applicable candidate set, measures it
on the host, and uses a disclosed median/MAD margin to fall back to baseline
when evidence is weak. The complete decision and calibration observations are
recorded. It can lose to a fixed policy, especially after including planning cost.
There is no fitted model, fabricated training data, performance cache, or claim
of arbitrary pass-order search.

## Evaluation and interpretation

Seven families at two sizes: affine, constant expression, dead branch, redundant
ReLU, inverse layout permutations, shared residual branch and dense MLP. The MLP
compiles end to end through matmul/add/ReLU; the original trained demo remains
available separately. Seeds and family splits are recorded. No workload variants
are used to train a predictor; splits prepare future group-held-out experiments.

Reports retain raw calibration/evaluation timing batches, randomized order,
median, MAD, standard deviation, range, environment/software, graph text, pass
records, decisions, correctness checks and tolerances. Timings include Python
interpreter dispatch, tensor allocation and input checks; cold import and model
preparation are outside execution measurements. Planning includes optimization,
per-pass checks and calibration. Final validation and backend preparation are
reported separately, with total compilation and observed break-even estimates.
Do not add optimization time to planning time: it is already included.

Structural compute counts and logical output-byte estimates are separate from
measured latency. **CPU affine fusion still materializes intermediate tensors.**
There are no measured memory, GPU latency, bandwidth or GPU speedup results.
Fresh-process repetitions and more diverse families are needed for conclusions.

## Preserved GPU demonstration

```bash
python3 examples/trained_mlp.py
python3 benchmark.py
```

The trained example validates the compiled hidden layer plus PyTorch readout;
it does not claim full-model GPU compilation. `benchmark.py` retains synchronized
GPU timing of eager, generated Triton and torch.compile. On this CPU host it
prints an explicit skip and collects no GPU timings. Generated files under
`generated/` are retained. Source generation/syntax checking is not execution.

For supported NVIDIA hardware, install CUDA PyTorch using the
[official instructions](https://pytorch.org/get-started/locally/) and install
Triton separately. The legacy runtime checks CUDA, compute capability >=7.0 and
Triton availability. The common CPU pipeline does not yet have a general Triton
backend; unsupported operators/models are rejected rather than silently lowered.

## Further documentation

- [Architecture and numerical semantics](docs/ARCHITECTURE.md)
- [Exact operator/frontend support matrix](docs/OPERATORS.md)
- [Research notes and next experiment](docs/RESEARCH_NOTES.md)
- [Implementation checklist](docs/IMPLEMENTATION_CHECKLIST.md)
- [Executed checks and observed results](docs/VALIDATION_RESULTS.md)
- [Latest CPU suite report](results/cpu_suite.md) and [raw observations](results/cpu_suite.json)
- `test_compiler.py`: preserved legacy coverage.
- `test_pipeline.py`: IR, passes, conversion, validation, backend and integration.
