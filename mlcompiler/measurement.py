"""Wall-clock CPU measurements. Raw batches retained; no timing assertions."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import os
from importlib.metadata import PackageNotFoundError, version
import hashlib
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any
import torch


@dataclass(frozen=True)
class MeasureConfig:
    warmup: int = 3
    batches: int = 7
    iterations: int = 20
    seed: int = 19

    def __post_init__(self):
        if self.warmup < 0 or self.batches < 3 or self.iterations < 1:
            raise ValueError("Require warmup >= 0, batches >= 3, iterations >= 1")


def summarize(raw: list[float]) -> dict[str, Any]:
    median = statistics.median(raw)
    return {"raw_ms": raw, "median_ms": median, "min_ms": min(raw), "max_ms": max(raw),
            "stdev_ms": statistics.stdev(raw),
            "mad_ms": statistics.median(abs(v - median) for v in raw)}


def measure_many(functions, config: MeasureConfig) -> dict:
    """Interleave candidates in seeded random order, amortize timer cost in batches."""
    observations = {name: [] for name in functions}
    rng = random.Random(config.seed)
    with torch.no_grad():
        for fn in functions.values():
            for _ in range(config.warmup): fn()
        orders = []
        for _ in range(config.batches):
            order = list(functions)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                started = time.perf_counter_ns()
                for _ in range(config.iterations): functions[name]()
                elapsed = time.perf_counter_ns() - started
                observations[name].append(elapsed / (1e6 * config.iterations))
    return {"config": asdict(config), "orders": orders,
            "timings": {name: summarize(raw) for name, raw in observations.items()},
            "method": "perf_counter_ns; warm CPU inference; randomized interleaved batches; includes Python dispatch/allocation and input checks; excludes compilation and validation"}


def environment() -> dict:
    dependencies = {}
    for package in ('torch', 'numpy', 'rich', 'onnx'):
        try: dependencies[package] = version(package)
        except PackageNotFoundError: dependencies[package] = None
    root = Path(__file__).resolve().parents[1]
    sources = (*root.joinpath('mlcompiler').glob('*.py'), root / 'compiler.py', root / 'runtime.py')
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    cpu_model = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        cpu_model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith('model name')), cpu_model)
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu_model": cpu_model,
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": sys.version, "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(),
            "backend": "cpu-torch-interpreter", "pid": os.getpid(),
            "torch_config": torch.__config__.show(), "dependencies": dependencies,
            "source_sha256": hashes}
