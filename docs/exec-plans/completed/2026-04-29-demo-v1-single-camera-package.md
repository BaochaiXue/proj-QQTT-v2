# Demo V1 Single-Camera Package

## Goal

Create `demo_v1/` as the operator-facing home for the single-D455 realtime demo
so it can be run without browsing through harness engineering utilities.

## Scope

- Add `demo_v1/realtime_single_camera_pointcloud.py` from the current
  single-camera harness implementation.
- Add a `demo_v1/README.md` with native RealSense and FFS commands.
- Add a demo-local WSLg/Open3D wrapper.
- Keep the existing harness entrypoint working for compatibility.
- Update docs/tests to reference the new package path.

## Non-Goals

- No change to 3-camera preview, calibration, recording, or alignment behavior.
- No removal of existing harness commands.
- No hardware-dependent automated test.
- No vendoring of external FFS weights or TensorRT engines.

## Validation

- `python demo_v1/realtime_single_camera_pointcloud.py --help`
- `python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke`
- `python scripts/harness/check_experiment_boundaries.py`

