"""Presentation entry point: capture, fuse, emit source, check, optionally launch."""
import torch
from compiler import AffineReLU, capture, fuse, generate_triton, interpret
from runtime import TritonRuntime, gpu_unavailable_reason


def main():
    torch.manual_seed(0)
    inputs = (torch.randn(1025), torch.tensor(1.5), torch.tensor(-0.25))
    model = AffineReLU()
    before = capture(model, inputs)
    after = fuse(before)
    print("BEFORE FUSION\n", before, sep="")
    print("\nAFTER FUSION\n", after, sep="")
    expected = model(*inputs)
    torch.testing.assert_close(interpret(before, *inputs)[0], expected)
    torch.testing.assert_close(interpret(after, *inputs)[0], expected)
    print("\nCPU IR correctness: PASS")
    path = generate_triton(after)
    print(f"\nGenerated kernel: {path}\n{path.read_text()}")
    reason = gpu_unavailable_reason()
    if reason:
        print(f"SKIPPED GPU execution: {reason}")
        return
    actual = TritonRuntime(after, path)(*(x.cuda() for x in inputs))
    torch.testing.assert_close(actual.cpu(), expected)
    print("GPU Triton correctness: PASS")


if __name__ == "__main__":
    main()
