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

## Validation
- Demo 2.2 dry-run contract.
- Demo 2.2/Demo 2.1 smoke tests.
- `scripts/harness/check_all.py`.
- Then rerun hardware startup/profile.
