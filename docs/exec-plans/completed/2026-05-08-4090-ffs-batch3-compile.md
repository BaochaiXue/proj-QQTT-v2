# Demo v0.3 RTX 4090 FFS TensorRT batch=3 compile / validate / profile

## Goal

Add isolated scripts to build, validate, and profile an RTX 4090 batch=3 Fast-FoundationStereo TensorRT engine without changing the existing batch=1 engine or public batch=1 runtime behavior.

## Constraints

- Do not touch running 7001 / 7002 / 5201 services.
- Do not overwrite the batch=1 engine directory.
- Default runtime behavior remains batch=1 unless explicitly using the new batch3 scripts or engine path.
- Validate/profile with 100 real-IR triplet replay kits when data is present.
- Do not commit generated engines, replay data, or large logs.

## Steps

1. Inspect existing repo and Fast-FoundationStereo TensorRT build/runtime paths.
2. Add batch3 build script that reuses existing FFS build machinery when possible and writes metadata.
3. Add replay loading, validation, camera-order checking, and profiling helpers.
4. Add unit tests that do not require TensorRT, GPU, or RealSense.
5. Run py_compile, unit tests, harness, and diff checks.
6. Run local batch1 profile, batch3 build, validation, and profiling if the 100-kit replay folder and builder path are available.
7. Write generated report and push branch.

## Outcome

- Batch3 compile scripts, validation script, profile script, and pure unit tests were added.
- Batch3 TensorRT two-stage engine compiled successfully on RTX 4090 GPU0 with static batch size 3.
- The external Fast-FoundationStereo repo and existing batch=1 engine/runtime behavior were not modified.
- 100-kit validation/profile could not run on this machine because `result/demo_v0_3_ir_triplet_100kits_848x480` is missing.
- Final report: `docs/generated/demo_v03_batch3_compile_validate_profile_4090.md`
