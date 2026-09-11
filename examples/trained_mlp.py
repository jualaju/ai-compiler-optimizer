"""Train a tiny MLP and compile its hidden layer with the existing compiler.

Run from the project root: python3 examples/trained_mlp.py
The dense 1 -> 8 hidden layer is expressed as elementwise multiply/add/ReLU.
Learned parameters become explicit inference inputs because capture does not
support parameter get_attr nodes. The 8 -> 1 readout stays in PyTorch: the
existing compiler does not support reductions or whole-MLP code generation.
"""
import ast
from pathlib import Path
import sys
import time

import torch
from torch import nn

# Make the existing project modules importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler import AffineReLU, capture, fuse, generate_triton, interpret
from runtime import gpu_unavailable_reason


class TinyMLP(nn.Module):
    """A fully connected scalar-input MLP with eight trainable hidden neurons."""

    def __init__(self):
        super().__init__()
        self.hidden_weight = nn.Parameter(torch.randn(8))
        self.hidden_bias = nn.Parameter(torch.linspace(-0.5, 0.5, 8))
        self.output_weight = nn.Parameter(torch.randn(8) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(()))
        self.hidden = AffineReLU()

    def readout(self, hidden):
        return (hidden * self.output_weight).sum(dim=-1, keepdim=True) + self.output_bias

    def forward(self, x):
        return self.readout(self.hidden(x, self.hidden_weight, self.hidden_bias))


def main():
    started = time.perf_counter()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    # Entirely local, deterministic regression data; no files or downloads.
    x = torch.linspace(-2.0, 2.0, 128).reshape(-1, 1)
    target = 0.6 * x + 0.8 * torch.relu(x - 0.25) - 0.2
    model = TinyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial_parameters = [p.detach().clone() for p in model.parameters()]
    initial_loss = nn.functional.mse_loss(model(x), target).item()
    model.train()
    for _ in range(250):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
    model.eval()

    with torch.no_grad():
        final_loss = nn.functional.mse_loss(model(x), target).item()
        assert final_loss < initial_loss * 0.1, "Training did not substantially reduce loss"
        assert all(not torch.equal(old, new) for old, new in
                   zip(initial_parameters, model.parameters())), "Parameters did not train"
        print("Trained MLP: 1 -> 8 (ReLU) -> 1; 25 learned parameters")
        print(f"128 synthetic samples, 250 CPU training steps; MSE {initial_loss:.6f} -> {final_loss:.6f}")

        # Held-out inputs include positive/negative values and zero. Materialize
        # broadcasting outside capture to satisfy existing codegen's same-shape,
        # contiguous float32 contract. These are the actual trained parameters.
        validation_x = torch.cat((torch.linspace(-2.25, 2.25, 65), torch.zeros(1))).reshape(-1, 1)
        expanded_x = validation_x.expand(-1, 8).contiguous()
        inputs = (
            expanded_x,
            model.hidden_weight.detach().expand_as(expanded_x).contiguous(),
            model.hidden_bias.detach().expand_as(expanded_x).contiguous(),
        )
        print("\nCompiling the trained hidden layer; output readout remains in PyTorch.")
        print("FX graph:")
        print(torch.fx.symbolic_trace(model.hidden).graph)
        before = capture(model.hidden, inputs)
        after = fuse(before)
        print(f"Custom IR before fusion:\n{before}")
        print(f"Custom IR after fusion:\n{after}")
        assert [op.kind for op in before.ops if op.kind != "input"] == ["multiply", "add", "relu"]
        assert [op.kind for op in after.ops if op.kind != "input"] == ["fused_affine_relu"]
        print("Detected multiply -> add -> ReLU: 3 operations -> 1 fused operation.")

        path = generate_triton(after, ROOT / "generated" / "trained_mlp_kernel.py")
        ast.parse(path.read_text())
        print(f"\nTriton kernel generated; Python syntax verified: {path}")

        expected_hidden = model.hidden(validation_x, model.hidden_weight, model.hidden_bias)
        original_hidden = interpret(before, *inputs)[0]
        fused_hidden = interpret(after, *inputs)[0]
        torch.testing.assert_close(original_hidden, expected_hidden)
        torch.testing.assert_close(fused_hidden, expected_hidden)
        expected = model(validation_x)
        actual = model.readout(fused_hidden)
        torch.testing.assert_close(actual, expected)
        max_error = (actual - expected).abs().max().item()
        print(f"CPU correctness PASS: unfused/fused hidden layer and full MLP predictions ({len(validation_x)} inputs).")
        print(f"Full-model maximum absolute error: {max_error:.3e}")

    reason = gpu_unavailable_reason()
    if reason:
        print(f"\nGPU execution is unavailable on this machine. {reason}")
    else:
        print("\nGPU execution was not requested for this CPU demonstration.")
    print("Validation uses the CPU IR interpreter; the generated Triton kernel was not executed.")
    print(f"Demo completed in {time.perf_counter() - started:.2f}s (excluding imports).")


if __name__ == "__main__":
    main()
