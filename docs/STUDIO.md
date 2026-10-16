# Local Compiler Studio: architecture, audit and verification

## Scope and evidence

Studio runs locally on Python 3.12 and CPU PyTorch, using the standard-library
HTTP server and vendored Cytoscape.js 3.30.4 (MIT). It needs no web framework,
Node build, hosted service, account, or network connection at runtime. It is a
loopback, single-user development tool, not an internet-facing deployment.

The two screenshots referenced in the request were not attached to this session.
The reported problems were reproduced from the repository and local terminal /
browser runs; no claim is made to have inspected those missing screenshots.
Generated browser screenshots used for QA are under `/tmp/compiler-studio-*.png`.

## Root causes and fixes

- The previous UI exposed only terminal pages. It had no browser graph renderer,
  graph serialization API, node inspection or pass-state snapshots. Studio adds
  all of these using the actual common IR, not an independent UI graph model.
- The native MLP has five essential operations: matmul/add/ReLU/matmul/add. The
  implemented fusion requires elementwise multiply/add/ReLU; inputs to dense
  layers are not constant expressions. No dead or redundant values exist.
  **Zero applied passes is correct.** It remains a documented no-op control.
  Native `constant`, `dead_branch`, and `affine` show real transformations;
  `residual` demonstrates a shared-use correctness guard. The common terminal
  and Studio default to `constant` so the first demonstration has eligible work.
- Legacy raw keys used TextIO buffering and polling, losing queued keystrokes.
  Both terminal dashboards now share raw `os.read` navigation and a testable
  key dispatcher. Unknown keys show feedback; invalid menu choices retry.
- Failed terminal callbacks/benchmarks previously escaped the viewer. They now
  report errors and retain the last view; failed initial common-pipeline inputs
  can be reentered. `--debug` opens common pass details as well as legacy details.
- Browser configuration edits cannot silently target an old captured workload:
  capture-affecting edits disable optimize/validate/benchmark until recapture.
  Failed requests retain the previous successful graph. Action buttons show
  pending/completed/error states tied to the actual HTTP operation.

## Separation of responsibilities

```
studio/static/ (HTML + CSS + plain JS + local Cytoscape)
       ↓ JSON requests / authoritative serialized graphs
studio/server.py (loopback HTTP adapter)
       ↓
studio/service.py (session runs, compiler calls, numerical checks, timing)
       ↓
mlcompiler frontends → IR → planner/pass manager → CPU backend
```

`studio/serialization.py` projects real IR nodes/operands to node/edge JSON.
It adds explicit graph-output endpoints (one per output slot, including aliases).
It preserves duplicate operand edges and shared dependencies. Unknown shape,
dtype or roles remain unknown. Constant previews are bounded to 12 elements;
all numerical comparisons still use full tensors. Nonfinite preview values are
explicit strings so the transport remains strict JSON.

Node details include IR ID, opcode, operands/results, attributes, shape/dtype,
constant bytes/elements/preview, parameter metadata and source. FX records real
Linear weight/bias parameter snapshots and get_attr parameter/buffer provenance.
Native workloads supply known tensor roles; ONNX initializers with no explicit
role remain constants rather than guessed trainable parameters.

The renderer uses Cytoscape's directed breadth-first layout, mouse/touch pan,
wheel zoom, fit/reset, search/navigation and node selection. Original and selected
optimized graphs have separate canvases. Preserved IR IDs establish correspondence;
removed/added/modified nodes are marked. Output selection highlights dependency
ancestors. The node navigator is also a readable selection alternative for a
large fitted graph. Layout is responsive; a 34-node graph and 390-pixel viewport
are covered by browser tests. Very large graphs may still require zoom/search.

## Pass snapshots

`compile_graph(..., capture_trace=True)` enables optional pass snapshots.
The default remains false for existing API and benchmark callers. The pass
manager retains the accepted graph before/after each attempt, plus the exact
record index. Failed/rejected changes retain the previous accepted graph.
Intelligent candidates have separate policy tags and globally aligned record
indices. Studio labels candidate traces separately from the final selected graph.

This records the actual compiler execution; the web service does not rerun an
imitation pass or infer transformations from a reduced node count. Snapshot
serialization is presentation overhead, outside reported compiler stage times.

## HTTP and service contract

- `GET /api/catalog`: real built-in workload catalogue and pass names.
- `GET /api/support`: support documentation, frontends and backend limitations.
- `POST /api/runs`: capture built-in native/FX, native JSON or a local ONNX file.
- `GET /api/runs`: session runs and success/failure action records.
- `GET /api/runs/{id}`: latest accepted state, graphs, snapshots and results.
- `POST /api/runs/{id}/optimize`: conventional/intelligent/baseline/manual policy.
- `POST /api/runs/{id}/validate`: full output checks with explicit rtol/atol.
- `POST /api/runs/{id}/benchmark`: check first, then real original/selected timings.

Studio retains at most 40 captured runs per process. History resets on restart;
JSON download exports the selected run, including observations and events. Each
capture is a separate run. Reoptimizing a run replaces its latest compilation
and clears stale validation/benchmark state, while action history retains earlier
results and diagnostics. New capture errors are logged without replacing an
existing run. HTTP errors use useful JSON diagnostics. No arbitrary Python
model files are evaluated. ONNX file paths refer to the server machine and use
the existing optional frontend and independent ONNX reference check.

CPU work is serialized by a service lock so server requests do not benchmark
each other concurrently. The launcher uses one PyTorch thread. Host and Origin
checks keep the API local, and all runtime assets are same-origin. The server
handles disconnected clients without treating a canceled response as a new
compiler failure.

## Numerical and measurement reporting

Original capture/CPU execution is distinct from optimized numerical validation.
The compiler checks transformed candidates on calibration samples; the Validate
button checks the evaluation input and displays actual bounded output previews.
Built-ins have separate calibration/evaluation inputs. External files currently
reuse the provided inputs; this is labelled, never called a holdout.

Benchmarking validates before timing and compares the original graph with the
**exact selected plan**, including a manual pass policy. It does not substitute
conventional optimization for a manually selected graph. Timings use the existing
randomized interleaved warm-up/batch method and retain raw observations, median,
MAD, standard deviation, range, machine/software and config. This is a two-plan
comparison; the existing CLI suite remains the three-strategy research experiment.

Frontend preparation, planning, optimization (a subset of planning), backend
preparation, final sample checks and execution timing are separate. Neither
logical output-byte estimates nor smaller graphs are reported as measured
memory/speedup. CPU fusion still materializes PyTorch intermediates. No GPU
execution, GPU timing, general GPU backend or reliable memory measurement is added.

## Verification performed

- `source .venv/bin/activate; python3 -m unittest -v`: **47 tests, 46 passed,
  1 hardware-gated GPU skip**. Includes the preserved tests, real HTTP API,
  serialization/edges/unknown metadata, parameter roles, pass snapshots, all five
  demonstration cases, exact manual-policy timing, error preservation and PTY
  keyboard/error recovery. ONNX tests ran with the optional package installed.
- `python3 tests/studio_browser.py`: **3 real Chromium tests passed**. Exercises
  every Studio tab, keyboard tab navigation, graph canvas clicks/correspondence,
  zoom/pan/fit/reset/search, pass steps, IR, manual/no-op policies, validation,
  genuine benchmarks, download, history/reopen, API errors and mobile overflow.
  The browser issued no external service/CDN requests and had no unhandled
  JavaScript errors. Browser testing is optional and uses requirements-dev.txt.
- Local `python3 studio.py` served successfully on 127.0.0.1:8765 and returned
  HTTP 200 with its local assets/API. Tests also start isolated ephemeral servers.
- Legacy/manual and common constant/MLP terminal runs completed. Invalid model
  input was visibly rejected. The trained example still validated 66 predictions
  with maximum absolute error 0.0; `benchmark.py` correctly skipped GPU timing.
- `node --check studio/static/app.js`, Python compile checks and `git diff --check`
  passed. No commits were created; prior uncommitted work was retained.

[Recorded Studio observations](../results/studio_validation.json) contain actual
original/optimized graphs, all pass snapshots, numerical checks and raw CPU
measurements for the five required examples. They are integration observations
from one host/session, not research evidence of general performance improvement.
