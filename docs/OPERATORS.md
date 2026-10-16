# Operator support matrix

Support is bounded to pure inference tensor graphs. A cell names the accepted
frontend syntax; a dash means conversion is rejected. The preserved `compiler.py`
frontend still supports only multiply/add/ReLU with tensor operands.

| Common IR | Native JSON | PyTorch FX frontend | ONNX default domain, opsets 13–21 | CPU |
|---|---|---|---|---|
| input / constant | explicit nodes | placeholders, tensor get_attr snapshots, scalar literals | inputs, initializers, tensor-valued Constant | yes |
| add | add | operator.add, torch.add (`alpha=1`) | Add | broadcasting |
| subtract | subtract | operator.sub, torch.sub (`alpha=1`) | Sub | broadcasting |
| multiply | multiply | operator.mul, torch.mul | Mul | broadcasting |
| matmul | matmul | operator.matmul, torch.matmul | MatMul | PyTorch matmul shapes |
| relu | relu | torch.relu, F.relu, x.relu(), nn.ReLU (not inplace) | Relu | yes |
| identity | identity | nn.Identity | Identity | value identity |
| reshape | static shape attribute | x.reshape(static dimensions) | constant int64 shape; explicit dimensions | yes |
| transpose | complete perm attribute | x.permute(complete permutation), x.t() for rank 2 | Transpose (known rank for default perm) | view permutation |
| sum | dims / keepdim | x.sum(dim, keepdim) | — | reduction |
| fused_affine_relu | three operands | created by optimization | — | grouped dispatch; **not** an allocation-free kernel |
| linear layer | matmul + add | nn.Linear / positional F.linear lowered to transpose + matmul + optional add | use MatMul + Add, **not Gemm** | yes |

`torch.add`/`torch.sub` allow `alpha=1` only. No `out=` arguments. Other kwargs
are rejected except `F.relu(inplace=False)`. FX accepts positional tensor
operands, flat tensor tuple/list outputs, static shapes/permutations, and tensor
parameters/buffers. nn.Sequential is traversed by FX. `torch.sum`, `torch.reshape`,
`nn.Flatten`, x.transpose, convolutions, attention, dropout, normalization,
quantization, arbitrary methods, mutation and data-dependent control flow are
not currently converted. Diagnostics identify the failing FX node or ONNX op.

Common storage dtypes: float32, float64, int32, int64, bool. An operator must
also accept its operand dtype under CPU/PyTorch semantics (for example, ReLU on
bool is rejected). Type promotion follows PyTorch; ONNX requires its declared
operator types and metadata. ONNX checking and metadata propagation happen at
import. API users must independently check converted outputs against their
source model; the file CLI/dashboard does this using ONNX ReferenceEvaluator.

The ONNX subset excludes custom domains, functions, training graphs, external
weights, sparse initializers, sequences, multiple-output nodes, dynamic reshape
shape tensors, and zero-copy reshape dimensions (`0` with allowzero=0). No claim
of arbitrary ONNX-model support is made. Symbolic input dimensions can remain
unknown; runtime inputs must match all known dimensions. Equal symbolic names
are not constraint-solved. Known inconsistent shapes are rejected by static meta
tensor propagation; unknown-shape incompatibilities surface at execution.

Native graphs use topologically ordered, single-output nodes with explicit graph
outputs. Multiple graph outputs (including input aliases and shared values) are
supported. JSON constants require a dtype. See `examples/native_affine.json`
and `examples/native_inputs.json` for runnable input. Native API/dictionary
construction uses the same format as file loading.

Legacy Triton codegen still targets exactly one float32 affine-ReLU kernel with
three distinct inputs and scalar or identical-shape contiguous operands. It is
not a general backend for the new IR. GPU execution remains capability-checked.
