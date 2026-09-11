import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from compiler import AffineReLU, capture, fuse, generate_triton, interpret
from runtime import TritonRuntime, gpu_unavailable_reason


class CompilerTests(unittest.TestCase):
    def test_capture_fusion_and_correctness(self):
        for shape in [(0,), (1,), (1025,), (3, 7)]:
            for scalar in [True, False]:
                with self.subTest(shape=shape, scalar=scalar):
                    torch.manual_seed(42)
                    x = torch.randn(shape)
                    s = torch.tensor(-1.5) if scalar else torch.randn(shape)
                    b = torch.tensor(0.25) if scalar else torch.randn(shape)
                    inputs = (x, s, b)
                    graph = capture(AffineReLU(), inputs)
                    self.assertEqual([op.kind for op in graph.ops],
                                     ["input"] * 3 + ["multiply", "add", "relu"])
                    optimized = fuse(graph)
                    self.assertEqual([op.kind for op in optimized.ops],
                                     ["input"] * 3 + ["fused_affine_relu"])
                    self.assertEqual(fuse(optimized), optimized)
                    self.assertEqual(optimized.outputs, graph.outputs)
                    self.assertEqual(optimized.ops[-1].shape, shape)
                    self.assertEqual(optimized.ops[-1].dtype, torch.float32)
                    expected = AffineReLU()(*inputs)
                    for ir in [graph, optimized]:
                        torch.testing.assert_close(interpret(ir, *inputs)[0], expected)
                    with tempfile.TemporaryDirectory() as directory:
                        path = generate_triton(optimized, Path(directory) / "kernel.py")
                        ast.parse(path.read_text())
                        self.assertIn("@triton.jit", path.read_text())
                        self.assertIn("mask=mask", path.read_text())

    def test_shared_intermediates_are_preserved(self):
        class Shared(torch.nn.Module):
            def forward(self, x, scale, bias):
                product = x * scale
                affine = product + bias
                return torch.relu(affine), product, affine

        inputs = (torch.randn(9), torch.tensor(2.0), torch.tensor(-1.0))
        before = capture(Shared(), inputs)
        after = fuse(before)
        self.assertEqual(after, before)
        torch.testing.assert_close(interpret(after, *inputs), Shared()(*inputs))

    def test_unsupported_codegen_and_fx(self):
        inputs = (torch.randn(2, 3), torch.randn(3), torch.randn(3))
        graph = capture(AffineReLU(), inputs)
        with self.assertRaises(ValueError):
            generate_triton(graph)
        with self.assertRaisesRegex(ValueError, "same shape"):
            generate_triton(fuse(graph))
        doubles = tuple(x.double() for x in inputs)
        with self.assertRaisesRegex(ValueError, "float32"):
            generate_triton(fuse(capture(AffineReLU(), doubles)))
        with self.assertRaises(NotImplementedError):
            capture(torch.nn.Sigmoid(), (inputs[0],))

    def test_cpu_guard_does_not_import_triton(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertIn("NVIDIA", gpu_unavailable_reason())
            with self.assertRaisesRegex(RuntimeError, "NVIDIA"):
                TritonRuntime(None, "nonexistent.py")

    @unittest.skipIf(gpu_unavailable_reason() is not None,
                     gpu_unavailable_reason() or "")
    def test_gpu_correctness(self):
        for shape in [(0,), (1,), (1025,), (3, 7)]:
            for scalar in [True, False]:
                with self.subTest(shape=shape, scalar=scalar):
                    cpu = (torch.randn(shape),
                           torch.tensor(-1.5) if scalar else torch.randn(shape),
                           torch.tensor(0.25) if scalar else torch.randn(shape))
                    graph = fuse(capture(AffineReLU(), cpu))
                    with tempfile.TemporaryDirectory() as directory:
                        path = generate_triton(graph, Path(directory) / "kernel.py")
                        runtime = TritonRuntime(graph, path)
                        inputs = tuple(t.cuda() for t in cpu)
                        torch.testing.assert_close(runtime(*inputs), AffineReLU()(*inputs))
                        with self.assertRaises(ValueError):
                            runtime(*cpu)


if __name__ == "__main__":
    unittest.main()
