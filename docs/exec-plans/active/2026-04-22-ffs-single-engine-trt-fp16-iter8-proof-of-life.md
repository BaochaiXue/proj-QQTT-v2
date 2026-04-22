## Goal

Generate a dedicated FP16 single-engine TensorRT proof-of-life artifact for Fast-FoundationStereo on WSL/Linux with `valid_iters=8`, and record the exact commands and outcomes under `docs/generated/`.

## Scope

- create one standalone FP16 single-engine TensorRT artifact directory under `data/ffs_proof_of_life/`
- keep the existing FP16 iter-4 and FP32 single-engine directories untouched so viewer selection stays explicit
- record commands and validation outcomes under the existing generated single-engine validation note

## Design

1. export a fresh single ONNX + YAML using the existing upstream `make_single_onnx.py` helper with `valid_iters=8`
2. build a new TensorRT engine with `fp16=True` via the existing Python builder helper
3. place the artifact in its own directory with a unique single `.engine`
4. run a minimal smoke through the repo-integrated single-engine runner to confirm the engine loads and produces finite disparity on the upstream demo pair

## Validation

- artifact directory contains `.onnx`, `.yaml`, `.engine`, and build log
- generated config records `valid_iters: 8`
- repo-integrated single-engine runner can load the engine and run on the upstream demo pair
- validation note records the exact command, artifact paths, and observed audit stats
