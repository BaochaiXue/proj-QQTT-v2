# 2026-05-10 Demo 2.2 Init Latency Under 10s

## Goal
Reduce Demo 2.2 startup latency toward <10s before the steady 15 FPS profile window.

## Findings
- SAM3.1 live first-frame init currently rebuilds/releases the SAM3.1 image model per camera.
- Demo 2.2 single-owner currently uses replicated HF EdgeTAM models, causing three model loads and three lazy compile paths even though the GPU owner processes cameras sequentially.

## Changes
- Cache the SAM3.1 image processor/model during live first-frame init and release it only after all three camera EdgeTAM sessions are initialized.
- Default Demo 2.2 async-filter single-owner to shared EdgeTAM model topology while keeping independent per-camera streaming sessions.
- Keep staged-parallel topology unchanged because it needs separate per-camera model execution.
- Record the init cache policy in the contract/profile and add smoke coverage.
- Add explicit init-stage profiling for camera startup, FFS runner/first run,
  EdgeTAM model load/compile/first forward, SAM3.1 model/segmentation/release,
  EdgeTAM session init/prompt add, and time to first complete/rendered group.
- Move EdgeTAM torch.compile lazy first-forward cost into an explicit init
  prewarm step for Demo 2.2, using a dummy streaming session and dummy mask
  prompt before the live first-frame sessions are created.
- Start camera setup, FFS runner init, EdgeTAM shared-model/session prewarm,
  and SAM3.1 image-processor preload concurrently for Demo 2.2 presets.
- Make SAM3.1 helper CUDA autocast state thread-local enough for background
  preload plus foreground first-frame segmentation.

## Validation
- Demo 2.2 dry-run contract.
- Demo 2.2/Demo 2.1 smoke tests.
- `scripts/harness/check_all.py`.
- Then rerun hardware startup/profile.
- Parallel-init profile:
  `docs/generated/demo2_2_async_filter_parallel_init_20s_warmup_20s_formal_profile.md`.
