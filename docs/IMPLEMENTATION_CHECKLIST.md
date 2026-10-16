# Implementation checklist

Migration: preserve compiler.py/runtime.py and existing GPU benchmark and demos;
add a common CPU pipeline alongside them, with an explicit legacy IR adapter.
Initial Git state was clean. No commits were created.

- [x] Typed IR, validation, analysis and CPU execution
- [x] Safe passes, explicit preconditions/diagnostics, sample checks and rollback
- [x] Native, FX and optional ONNX frontends with a bounded support matrix
- [x] Baseline, deterministic and conservative measured selection (no fitted model)
- [x] Reproducible suite, retained raw observations and Markdown reports
- [x] Dashboard integration and independent CLI/API
- [x] Unit/integration tests, demonstrations, architecture and research notes

Verification and actual observations: [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md).
Research extensions, arbitrary operator support and a general GPU backend are
explicitly future work, not completed features.
