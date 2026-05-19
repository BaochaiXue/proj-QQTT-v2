# Demo 2.3 EdgeTAM bounded live session

## Goal

Prevent Demo 2.3 HF EdgeTAM live tracking from accumulating unbounded video-session tensors during long rendered runs.

## Scope

- Keep the existing HF EdgeTAM model and Demo 2.3 dual-GPU scheduling.
- Add QQTT-side pruning of old live-session frames and non-conditioning outputs.
- Preserve recent memory frames and conditioning outputs required by EdgeTAM tracking.
- Re-run rendered Demo 2.3 profiling and confirm `fatal_error` is clear.

## Validation

- Syntax check the changed runtime.
- Dry-run Demo 2.3 contract.
- Run rendered Demo 2.3 profile with Open3D window and inspect JSON/MD summary.

## Outcome

- Added bounded HF EdgeTAM live-session pruning with a default 64-frame retention window.
- Demo 2.3 rendered profile completed with `fatal_error=None`.
- GPU1 after-warmup memory stayed bounded at about 6.1 GB max instead of growing to OOM.
- Quick deterministic checks passed.
