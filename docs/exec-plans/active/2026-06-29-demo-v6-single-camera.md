# Demo v6 Single-Camera Compatibility Plan

**Goal:** Make `demo_v6` run against the actual camera count present in a case,
including a single-camera case, without reading camera 1/2 when they do not
exist.

**Scope:** Keep the change inside the demo-only `demo_v6/` flow. Do not change
formal recording or aligned-case outputs.

## Steps

- [x] Replace hard-coded camera count in segmentation and dense tracking with
  camera indices discovered from `depth/<camera_idx>/`.
- [x] Replace hard-coded three-camera observation gathering in shape alignment
  with the camera count stored in `pcd/0.npz`.
- [x] Point demo segmentation helpers at `demo_v6/groundedSAM_checkpoints`
  instead of `./data_process/groundedSAM_checkpoints`.
- [x] Validate syntax and static hard-code removal.

## Validation Results

- `rg -n "num_cam = 3|camera_num = 3|range\\(3\\)|\\.\\/data_process|data_process/" demo_v6 -S`
  only reports `models/superglue.py`, where `range(3)` is a neural network
  projection-layer count, not a camera count.
- `python -m py_compile demo_v6/dense_track.py demo_v6/segment.py
  demo_v6/segment_util_video.py demo_v6/segment_util_image.py demo_v6/align.py`
  passed.
- `conda run -n demo_2_max --no-capture-output python
  scripts/harness/validation/run.py --profile smoke` passed.

## Notes

- `git pull --ff-only origin main` was attempted before edits on
  `single-camera`, but Git refused because the branch cannot fast-forward from
  `origin/main`.
