"""Framework-neutral inference graph pipeline with a PyTorch CPU backend.

Legacy compiler.py remains available; new code uses these public APIs.
"""
from .ir import Diagnostic, Graph, Node, TensorSpec, analysis, from_legacy
from .backend import Backend, CPUBackend, compare_outputs, infer_shapes, validate
from .frontends import native, onnx, pytorch
from .planner import Compilation, compile_graph

__all__ = ["Diagnostic", "Graph", "Node", "TensorSpec", "analysis", "from_legacy",
           "Backend", "CPUBackend", "compare_outputs", "infer_shapes", "validate",
           "native", "onnx", "pytorch", "Compilation", "compile_graph"]
