# 2026-05-02 HF EdgeTAM Streaming Proof

## Goal

Validate the Hugging Face EdgeTAMVideo streaming API in an isolated conda environment cloned from `edgetam-max`, without modifying the existing official EdgeTAM patched environment.

## Plan

1. Inspect `edgetam-max` CUDA/PyTorch baseline and clone it to `edgetam-hf-stream` if needed.
2. Install a recent Transformers stack only in `edgetam-hf-stream`.
3. Add a harness smoke script that verifies `EdgeTamVideoModel`, processor/session API availability, and frame-by-frame synthetic streaming.
4. Write generated validation artifacts under `docs/generated/`.
5. Register the new harness script in the harness catalog.
6. Run focused catalog/check validation.

## Validation

- `conda run --no-capture-output -n edgetam-hf-stream python -m pip check`
- `conda run --no-capture-output -n edgetam-hf-stream python scripts/harness/verify_hf_edgetam_streaming.py ...`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_harness_catalog.py`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`

## Outcome

- Created `edgetam-hf-stream` by cloning `edgetam-max`.
- Installed `transformers==5.7.0`, `accelerate==1.13.0`, `safetensors==0.7.0`, and `huggingface_hub==1.13.0` in the cloned environment only.
- Verified `EdgeTamVideoModel`, `Sam2VideoProcessor`, and `EdgeTamVideoInferenceSession` imports.
- Added `scripts/harness/verify_hf_edgetam_streaming.py`.
- Validated synthetic frame-by-frame streaming with direct `EdgeTamVideoInferenceSession`:
  - output: `docs/generated/hf_edgetam_streaming_validation.md`
  - output: `docs/generated/hf_edgetam_streaming_results.json`
- Also validated the HF docs-style `Sam2VideoProcessor.init_video_session()` path:
  - output: `docs/generated/hf_edgetam_streaming_processor_session_validation.md`
  - output: `docs/generated/hf_edgetam_streaming_processor_session_results.json`
- Recorded the `edgetam-hf-stream` environment in `docs/envs.md`.
- Registered the new harness script in `scripts/harness/_catalog.py`.
- Registered the concurrently present `scripts/harness/experiments/probe_edgetam_video_trt_compile.py` experiment in the harness catalog so repo catalog checks remain green; the experiment file itself was not modified.
- Validation passed:
  - `conda run --no-capture-output -n edgetam-hf-stream python -m pip check`
  - `conda run --no-capture-output -n edgetam-hf-stream python scripts/harness/verify_hf_edgetam_streaming.py --json-output docs/generated/hf_edgetam_streaming_results.json --markdown-output docs/generated/hf_edgetam_streaming_validation.md`
  - `conda run --no-capture-output -n edgetam-hf-stream python scripts/harness/verify_hf_edgetam_streaming.py --frames 3 --session-init processor --json-output docs/generated/hf_edgetam_streaming_processor_session_results.json --markdown-output docs/generated/hf_edgetam_streaming_processor_session_validation.md`
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/verify_hf_edgetam_streaming.py --help`
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/experiments/probe_edgetam_video_trt_compile.py --help`
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/check_harness_catalog.py`
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`
