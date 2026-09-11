"""Warm steady-state CUDA timings; compilation and warmup are excluded."""
import argparse
import statistics
import time
import torch
from compiler import AffineReLU, capture, fuse, generate_triton
from runtime import TritonRuntime, gpu_unavailable_reason


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return statistics.median(samples)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2**20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if min(args.size, args.warmup, args.iterations) < 1:
        parser.error("size, warmup and iterations must be positive")
    reason = gpu_unavailable_reason()
    if reason:
        print(f"SKIPPED all GPU benchmarks: {reason} No timings were collected.")
        return
    torch.manual_seed(0)
    cpu = (torch.randn(args.size), torch.tensor(1.5), torch.tensor(-0.25))
    inputs = tuple(t.cuda() for t in cpu)
    model = AffineReLU().eval()
    graph = fuse(capture(model, cpu))
    runtime = TritonRuntime(graph, generate_triton(graph))
    providers = {"PyTorch eager": lambda: model(*inputs),
                 "Our fused Triton": lambda: runtime(*inputs)}
    try:
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        compiled(*inputs)  # Trigger compilation outside timing.
        providers["torch.compile"] = lambda: compiled(*inputs)
    except Exception as error:
        print(f"torch.compile unavailable/failed: {error}")
    expected = model(*inputs)
    for fn in providers.values():
        torch.testing.assert_close(fn(), expected, rtol=1e-5, atol=1e-6)
    print(f"GPU: {torch.cuda.get_device_name()} | torch={torch.__version__} | n={args.size}")
    print("Correctness: PASS. Median of 5 batches; synchronized wall time, including launch/allocation overhead.")
    for name, fn in providers.items():
        print(f"{name:20s} {measure(fn, args.warmup, args.iterations):.6f} ms/call")


if __name__ == "__main__":
    main()
