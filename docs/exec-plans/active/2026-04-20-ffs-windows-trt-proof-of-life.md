# 2026-04-20 FFS Windows TensorRT Proof-of-Life

## Goal

Validate the upstream Fast-FoundationStereo two-stage ONNX / TensorRT path on the local Windows RTX 5090 Laptop GPU setup without changing QQTT production depth-backend behavior.

## Non-Goals

- no QQTT `depth_backend` interface changes
- no viewer / align TRT integration in this pass
- no single-ONNX / single-engine workflow upgrade
- no `848x480` native TensorRT engine validation in this pass

## Files To Touch

- new `scripts/harness/verify_ffs_tensorrt_windows.py`
- new `docs/generated/ffs_tensorrt_windows_validation.md`
- `docs/generated/README.md`

## Implementation Plan

1. add a Windows-only harness that:
   - verifies Python-side TensorRT importability
   - exports the existing two-stage ONNX artifacts at `640x480`
   - builds both engines with `trtexec`
   - runs the TensorRT demo headlessly and writes file outputs
   - compares TensorRT vs PyTorch latency on the same local setup
2. keep the TRT work external-facing only:
   - use the external Fast-FoundationStereo repo and TensorRT SDK outside QQTT
   - write runtime artifacts under `data/ffs_proof_of_life/`
3. document exact commands, versions, outputs, and Windows-specific caveats under `docs/generated/`

## Validation Plan

- `C:\Users\zhang\miniconda3\envs\ffs-standalone\python.exe scripts\harness\verify_ffs_tensorrt_windows.py`
