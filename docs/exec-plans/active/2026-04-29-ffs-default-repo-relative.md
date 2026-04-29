# FFS Default Repo Path Repo-Relative

## Goal

Change Linux/WSL Fast-FoundationStereo defaults from user-specific absolute
paths to the QQTT repo sibling path `../Fast-FoundationStereo`, resolved from
the repository root rather than the process current working directory.

## Scope

- Update the central FFS defaults used by live viewers and harness scripts.
- Keep explicit CLI overrides unchanged.
- Keep Windows-only validation defaults and historical generated validation logs
  unchanged.
- Update operator-facing docs and smoke coverage for the new default.

## Validation

Run:

```bash
python -m unittest -v tests.test_cameras_viewer_ffs_smoke
python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke
python scripts/harness/check_all.py
```
