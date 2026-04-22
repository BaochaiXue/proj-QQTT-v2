## Goal

Generate a single-engine TensorRT proof-of-life artifact for Fast-FoundationStereo on WSL/Linux and record the exact commands and outcomes under `docs/generated/`.

## Scope

- generate one single-engine TensorRT artifact directory under `data/ffs_proof_of_life/`
- record commands and outcomes under `docs/generated/`

## Design

1. use the external Fast-FoundationStereo `make_single_onnx.py` path
2. keep the engine shape viewer-compatible at `864x480`
3. use the same default checkpoint family as the existing two-stage WSL proof-of-life unless blocked
4. since `trtexec` is not available on this machine, build the `.engine` with the TensorRT Python API
5. validate by running the upstream `run_demo_single_trt.py` headlessly on the demo pair

## Validation

- exported `.onnx` and `.yaml` exist
- built `.engine` exists
- `run_demo_single_trt.py` writes `disp_vis.png`, `depth_meter.npy`, and `cloud.ply`
- commands and outcomes are recorded in `docs/generated/`
