# External Dependencies

External repos, checkpoints, TensorRT engines, SAM assets, and generated proof
artifacts stay outside this repo. This file records source-of-truth locations
for the single-camera branch.

## Fast-FoundationStereo

- Default external repo: `../Fast-FoundationStereo`
- Typical checkpoint family: `20-30-48`
- Example checkpoint path:
  - `../Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth`
- Runtime roles:
  - `cameras_viewer_FFS.py`
  - `data_process/record_data_align.py --depth_backend ffs`
  - `scripts/harness/realtime_single_camera_pointcloud.py --depth-source ffs`
  - native-vs-FFS visualization harnesses

## SAM 3.1

- Runtime role:
  - single-camera masked PCD demo initialization
  - mask helper / visualization diagnostics
- Keep checkpoints outside this repo. Prefer environment variables or explicit
  CLI arguments when selecting local checkpoints.

## EdgeTAM

- Runtime role:
  - optional single-camera mask propagation in demo workflows
- Keep the external checkout outside this repo and document the path in the
  relevant validation plan or local environment notes.

## TensorRT Engines

- Runtime role:
  - optional FFS acceleration for viewer, demo, and visualization experiments
- Engines are generated artifacts and must not be committed.
- Store local engine paths under `data/experiments/`, `data/ffs_proof_of_life/`,
  or another ignored/generated location documented by the active plan.
