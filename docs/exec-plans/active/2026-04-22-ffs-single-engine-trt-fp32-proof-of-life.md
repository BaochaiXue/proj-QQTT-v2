## Goal

Generate a dedicated FP32 single-engine TensorRT proof-of-life artifact for Fast-FoundationStereo on WSL/Linux and record the exact commands and outcomes under `docs/generated/`.

## Scope

- create one standalone FP32 single-engine TensorRT artifact directory under `data/ffs_proof_of_life/`
- keep the existing FP16 single-engine directory untouched so viewer auto-discovery semantics do not break
- record commands and validation outcomes under the existing generated single-engine validation note

## Design

1. reuse the existing exported single ONNX + YAML from the current `864x480` single-engine proof-of-life
2. build a new TensorRT engine with `fp16=False` via the existing Python builder helper
3. place the FP32 artifact in its own directory with a unique single `.engine`
4. run a minimal smoke through the repo-integrated single-engine runner to confirm the FP32 engine produces finite full-width disparity

## Validation

- FP32 artifact directory contains `.onnx`, `.yaml`, `.engine`, and build log
- repo-integrated single-engine runner can load the FP32 engine and run on the upstream demo pair
- validation note records the exact command, artifact paths, and observed audit stats
