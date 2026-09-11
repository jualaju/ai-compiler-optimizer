# Tiny affine + ReLU compiler

A presentation-sized proof of concept for `y = relu(x * scale + bias)`.
We implement the IR, pattern-matching fusion, source generation and launch
wrapper ourselves. PyTorch FX supplies tracing and shape propagation; Triton
compiles the emitted kernel for the GPU. `torch.compile` is a benchmark baseline.

```text
PyTorch module -> torch.fx -> our Graph/Op IR -> our fusion pass
                                                |
                           generated Triton Python source -> GPU runtime
```

Each IR operation has an opcode, named input values, a named output, shape and
dtype. The graph lists its inputs and outputs explicitly. Fusion replaces a
single-use `multiply -> add -> relu` chain with `fused_affine_relu`, retaining
the result name and metadata. Shared intermediates and graph outputs prevent
unsafe removal. A small PyTorch-backed IR interpreter checks semantics on CPU.

## Run on this CPU machine

Requires Python 3.10+. No Triton installation is needed for CPU use.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python demo.py
python -m unittest -v
python benchmark.py
```

The demo prints both IR graphs, checks their outputs against eager PyTorch,
and saves and prints `generated/affine_relu_kernel.py`. The source can be
generated and syntax-checked without importing Triton. On a CPU machine,
GPU execution and benchmarks print an explicit skip message; GPU tests are
reported as skipped. CPU interpreter correctness does **not** validate GPU code.

## Run on NVIDIA

In a separate environment, install CUDA-enabled PyTorch using the command from
the [official PyTorch installer](https://pytorch.org/get-started/locally/),
then install Triton:

```bash
python -m pip install triton
python demo.py
python -m unittest -v
python benchmark.py --size 1048576 --warmup 20 --iterations 100
```

Use a Triton-supported NVIDIA setup (the runtime checks CUDA availability,
compute capability >= 7.0, and the Triton package). GPU/compiler installation
errors beyond those checks are surfaced for diagnosis.

The benchmark compares eager PyTorch, our generated Triton kernel and
`torch.compile(..., backend="inductor", fullgraph=True)` on identical inputs.
It validates each available implementation with `torch.testing.assert_close`,
compiles before timing, warms each implementation, and reports the median of
five batches using synchronized wall-clock milliseconds per call. Timings
include Python dispatch and output allocation, exclude transfers and compilation,
and use resident inputs. They are a simple steady-state demo, not a rigorous
performance study. If `torch.compile` fails, its failure is printed and its timing
is omitted. No speedup is assumed or fabricated.

## Files and limitations

- `compiler.py`: model/eager reference, FX translation, custom IR, fusion,
  CPU interpreter and source generation.
- `runtime.py`: hardware checks, generated module loading and kernel launch.
- `demo.py`: presentation workflow.
- `benchmark.py`: GPU correctness and timing comparison.
- `test_compiler.py`: CPU coverage and hardware-gated GPU correctness tests.
- `generated/affine_relu_kernel.py`: readable generated presentation artifact.

Code generation intentionally supports one fused operation, three distinct
tensor inputs, contiguous float32 data, and scale/bias that are either 0-D
scalars or exactly the shape of x. Shapes/dtypes are fixed at capture time and
checked at launch. This is inference-only: no autograd, general broadcasting,
arbitrary FX operators or multi-kernel scheduling. Tail elements use masked
loads/stores; empty tensors do not launch a kernel. Multiply/add contraction is
disabled to match eager float32 rounding. The implementation uses the
[FX API](https://docs.pytorch.org/docs/stable/fx.html) and Triton's documented
[masked vector kernel pattern](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html).

No NVIDIA GPU benchmark results are included; run the commands on an NVIDIA
machine to obtain actual correctness results and timings.

Verified locally with Python 3.12.3 and PyTorch 2.14.0+cpu: four CPU tests
passed, one GPU test skipped, and both demo and benchmark commands exited
successfully with explicit GPU skip messages. A minimal PyTorch installation
may print a missing-NumPy warning; this demo does not require NumPy.
