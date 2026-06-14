# Single Demo 3.1 TAPNext++ Overlay Plan

## Goal

Bring back a single-camera version of the original Demo 3.2 TAPNext++ point-tracker overlay while preserving the single-camera replay/runtime scope. The recording replay must still render masked point clouds; TAPNext++ adds visible 3D query/track markers on top of that point-cloud path.

## Constraints

- Keep the formal Demo 3/3.1 recording replay on `--render-mode pointcloud`.
- Keep RealSense RGB-D replay single-camera only; do not restore three-camera fusion, batch3 FFS, shape prior, or robot/physics paths.
- Use TAPNext++ as the tracker backend, matching original Demo 3.2 semantics where practical.
- Use GPU0 for SAM3.1/EdgeTAM/PCD and GPU1 for the tracker when available.
- Fail clearly when external TAPNext++ repo/checkpoint paths are not configured.

## Planned Implementation

1. Add a minimal tracking package with PhysTwin-style query sampling and a TAPNext++ adapter copied down from the original repo only where needed.
2. Add a single-camera tracker worker in `qqtt.demo.realtime_masked_edgetam_pcd`:
   - initialize query points after both object and controller masks are non-empty
   - default to union/all-tracks display scope, capped with visible-spread selection
   - update TAPNext++ serially from RGB frames
   - lift visible tracks into camera-color-frame 3D using the same RGB-D depth/intrinsics path as the PCD
3. Add a third Open3D geometry layer for tracker markers.
4. Add CLI/contract options through `single_demo_v3_runtime` and the masked PCD delegate:
   - `--tracker-backend tapnextpp|none`
   - `--tracker-device`
   - `--tapnet-repo-dir`
   - `--tapnextpp-checkpoint`
   - query/overlay marker controls
5. Add deterministic tests for contract defaults, sampling/lift helpers, and a fake-backend tracker worker path.

## Validation

- Targeted unit tests for new tracker helpers/runtime contract.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- Manual hardware/replay smoke only after deterministic checks pass.

## Outcome

- Added a single-camera TAPNext++ serial backend and 3D marker overlay worker.
- Demo 3/3.1 recording replay now requires `--tracker-backend tapnextpp` and the point-cloud render path.
- Default public tracker convention is q4096, GPU1, union/all-tracks display, visible-spread cap 512.
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_tapnextpp_overlay tests.test_single_demo_v3_runtime`
- PASS: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
