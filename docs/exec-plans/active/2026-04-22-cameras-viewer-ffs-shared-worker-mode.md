## Goal

Add an optional shared-worker mode to `cameras_viewer_FFS.py` so multiple active cameras can feed one Fast-FoundationStereo worker process instead of the current one-worker-per-camera topology.

## Scope

- `cameras_viewer_FFS.py`
- `tests/test_cameras_viewer_ffs_smoke.py`
- `README.md`
- `docs/WORKFLOWS.md`
- `docs/ARCHITECTURE.md`
- `docs/HARDWARE_VALIDATION.md`

## Design

1. keep the existing `per_camera` worker topology as the default behavior
2. add `--ffs_worker_mode {per_camera,shared}`
3. in `shared` mode:
   - keep per-camera latest-only request/result queues
   - start one FFS worker process for all active cameras
   - poll per-camera request queues in round-robin order inside the shared worker
   - reuse the same FFS runner instance for all cameras
   - keep per-camera result delivery, display, and stats paths intact
4. do not add batched FFS inference in this change; shared mode remains sequential inside one worker

## Validation

- `python cameras_viewer_FFS.py --help`
- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `python scripts/harness/check_all.py`
