"""Load the emitted source only when an NVIDIA GPU and Triton are available."""
import importlib.util
import sys
import torch


def gpu_unavailable_reason():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return "GPU benchmarks/execution require an NVIDIA GPU and a CUDA-enabled PyTorch installation."
    if torch.cuda.get_device_capability()[0] < 7:
        return "This demo requires an NVIDIA GPU with compute capability 7.0 or newer."
    if importlib.util.find_spec("triton") is None:
        return "Triton is missing; install it with: pip install triton"
    return None


class TritonRuntime:
    def __init__(self, graph, source_path):
        reason = gpu_unavailable_reason()
        if reason:
            raise RuntimeError(reason)
        spec = importlib.util.spec_from_file_location("generated_affine_relu", source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.kernel = module.affine_relu_kernel
        self.graph = graph
        self.op = next(op for op in graph.ops if op.kind == "fused_affine_relu")

    def __call__(self, *args):
        if len(args) != len(self.graph.inputs):
            raise ValueError("Wrong number of tensor inputs")
        values = dict(zip(self.graph.inputs, args))
        for op in self.graph.ops:
            if op.kind == "input":
                tensor = values[op.output]
                if (not tensor.is_cuda or tensor.device != args[0].device
                        or tensor.dtype != op.dtype or tuple(tensor.shape) != op.shape
                        or not tensor.is_contiguous() or tensor.requires_grad):
                    raise ValueError("Expected same-device CUDA, contiguous tensors matching captured shapes/dtypes, without gradients")
        x, scale, bias = (values[name] for name in self.op.inputs)
        output = torch.empty_like(x)
        n = x.numel()
        if n:
            with torch.cuda.device(x.device):
                # Separate float32 multiply/add rounding matches eager execution.
                self.kernel[((n + 255) // 256,)](
                    x, scale, bias, output, n, BLOCK_SIZE=256, enable_fp_fusion=False)
        return output
