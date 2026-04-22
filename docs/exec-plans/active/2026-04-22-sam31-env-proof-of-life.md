# 2026-04-22 SAM 3.1 Environment Proof-of-Life

## Goal

Validate that the local `qqtt-ffs-compat` environment can import the upstream `sam3` runtime and resolve or load an external `SAM 3.1` checkpoint for QQTT's existing sidecar mask helper.

## Non-Goals

- no change to canonical recording, alignment, or viewer outputs
- no expansion of segmentation into the repo's formal in-scope product surface
- no change to `scripts/harness/sam31_mask_helper.py` behavior unless validation exposes a concrete compatibility bug
- no attempt to vendor checkpoints or external runtimes into this repo

## Files To Touch

- new `docs/generated/sam31_env_validation.md`
- `docs/generated/README.md`
- `docs/envs.md`

## Implementation Plan

1. validate the current baseline:
   - confirm the target env, CUDA visibility, and current `sam3` import state
   - confirm the external checkpoint path as seen from WSL
2. install the minimal upstream `sam3` runtime needed by QQTT's helper:
   - prefer installing into `qqtt-ffs-compat`
   - keep the checkpoint external and referenced by path
3. run a bounded proof-of-life:
   - verify `sam3` import success
   - run a minimal checkpoint resolution / predictor construction probe against the external `SAM 3.1` checkpoint
   - capture any compatibility caveats
4. record exact commands, versions, and outcomes under `docs/generated/`, then refresh the validated environment doc

## Validation Plan

- `conda run -n qqtt-ffs-compat python -c "import sam3, torch; print(torch.cuda.is_available())"`
- `conda run -n qqtt-ffs-compat python - <<'PY'`
- `from scripts.harness.sam31_mask_helper import build_sam31_video_predictor`
- `predictor, checkpoint = build_sam31_video_predictor(checkpoint_path='/mnt/c/Users/zhang/external/sam3_checkpoints/sam3.1_multiplex.pt')`
- `print(type(predictor).__name__, checkpoint)`
- `PY`
