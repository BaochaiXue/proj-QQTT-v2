# Demo V2 Single-Camera Clone

## Goal

Clone `demo_v1/` into `demo_v2/` so current-week single-camera demo changes can
land in v2 while v1 remains a stable baseline.

## Scope

- Copy the current single-D455 realtime demo package from `demo_v1/` to
  `demo_v2/`.
- Update demo-facing docs so v2 is the recommended active path.
- Keep v1 available as a baseline.
- Route the harness compatibility entrypoint through v2 so harness validation
  manages the current demo.
- Extend tests/check lists to cover v2.

## Non-Goals

- No change to camera capture behavior.
- No removal of `demo_v1/`.
- No hardware-dependent automated test.
- No vendoring of external FFS repos, weights, or TensorRT engines.

## Validation

- `python demo_v2/realtime_single_camera_pointcloud.py --help`
- `python scripts/harness/realtime_single_camera_pointcloud.py --help`
- `conda run -n FFS-SAM-RS python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke tests.test_check_all_smoke`
- `conda run -n FFS-SAM-RS python scripts/harness/check_all.py`

