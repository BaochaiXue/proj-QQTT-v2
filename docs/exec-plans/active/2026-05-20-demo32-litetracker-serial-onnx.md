# Demo 3.2 LiteTracker Serial ONNX Runtime

## Goal

Add a serial-only LiteTracker ONNX-CUDA runtime for Demo 3.2 A/B profiling
against the existing PyTorch serial path.

## Scope

- Add `--litetracker-runtime pytorch|onnx-cuda`.
- Add ONNX directory/export/opset/optimization-level plumbing.
- Treat `--litetracker-onnx-optimization-level 5` as ORT's highest graph
  optimization level when available.
- Keep the requested ONNX opset configurable, but record actual export opset;
  local PyTorch 2.11 validation requires effective opset 18 for a loadable
  `updateformer.onnx`.
- Implement only serial `OnnxLiteTrackerAdapter.initialize()` and `update()`.
- Keep PyTorch `LiteTrackerAdapter` as the default baseline.
- Do not implement ONNX batch-views, TensorRT, renderer changes, marker changes,
  surface filtering changes, or query-manager changes.

## Plan

1. Add point-tracker config fields for LiteTracker runtime selection.
2. Add `qqtt/tracking/backends/litetracker_onnx_adapter.py`.
3. Route `backend=litetracker` + `runtime=onnx-cuda` to the ONNX adapter.
4. Thread runtime fields through Demo 3.1/3.2 CLI, contract, and child process
   JSON config.
5. Add focused fake-wrapper tests for availability, serial yx/xy conversion,
   shape/stats, and batch rejection.
6. Document `onnx` / `onnxruntime-gpu` as Demo 3.1 max optional ONNX runtime
   dependencies.

## Validation

- `python -m py_compile` on the new adapter and touched runtime/config files.
- `python -m unittest -v tests.test_litetracker_onnx_adapter_contract`.
- Real adapter smoke with exported `fnet.onnx`, `corr_mlp.onnx`,
  `updateformer.onnx`, `CUDAExecutionProvider`, 848x480 RGB, 4096 serial
  queries, and two `update()` calls.
- Focused Demo 3.2 contract/process config tests.
- `python scripts/harness/check_all.py`.
- `git diff --check`.
