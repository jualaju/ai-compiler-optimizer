"""Presentation entry point: capture, fuse, emit source, check, optionally launch."""
import torch
from compiler import AffineReLU, capture, fuse, generate_triton, interpret
from runtime import TritonRuntime, gpu_unavailable_reason


def main():
    torch.manual_seed(0)
    inputs = (torch.randn(1025), torch.tensor(1.5), torch.tensor(-0.25))
    model = AffineReLU()
    print("=== ORIGINAL MODEL ===")
    print("x → multiply(scale) → add(bias) → relu")
    print("\n=== FX GRAPH ===")
    print(torch.fx.symbolic_trace(model).graph)
    before = capture(model, inputs)
    print("\n=== OUR IR BEFORE OPTIMIZATION ===")
    print(before)
    after = fuse(before)
    print("\n=== APPLYING FUSION ===")
    print("Our fusion pass replaces multiply + add + relu with fused_affine_relu.")
    print("\n=== OUR IR AFTER OPTIMIZATION ===")
    print(after)
    path = generate_triton(after)
    print("\n=== GENERATED TRITON KERNEL ===")
    print(path.read_text())
    print("=== CORRECTNESS ===")
    expected = model(*inputs)
    torch.testing.assert_close(interpret(before, *inputs)[0], expected)
    torch.testing.assert_close(interpret(after, *inputs)[0], expected)
    print("CPU IR correctness: PASS — original and optimized IR match the PyTorch reference.")
    reason = gpu_unavailable_reason()
    if reason:
        print(f"SKIPPED GPU execution: {reason}")
    else:
        actual = TritonRuntime(after, path)(*(x.cuda() for x in inputs))
        torch.testing.assert_close(actual.cpu(), expected)
        print("GPU Triton correctness: PASS")
    original_count = sum(op.kind != "input" for op in before.ops)
    optimized_count = sum(op.kind != "input" for op in after.ops)
    print("\n=== OPTIMIZATION SUMMARY ===")
    print(f"Original operation count: {original_count} (excluding inputs)")
    print(f"Optimized operation count: {optimized_count} (excluding inputs)")
    print("Original kernel-launch count: 3 conceptual elementwise kernels")
    print("Fused kernel-launch count: 1 (generated Triton kernel)")
    print(f"Intermediate tensor materializations avoided: {original_count - optimized_count} "
          "(multiply and add results in the generated kernel)")
    print("CPU interpretation checks semantics; it does not execute the fused GPU kernel.")
    print("Actual GPU latency/speedup requires an NVIDIA GPU and has NOT been measured on this machine.")


if __name__ == "__main__":
    main()
