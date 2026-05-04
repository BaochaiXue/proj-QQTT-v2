# Demo V2 Realtime Masked EdgeTAM PCD

## Goal

Create a new single-camera realtime demo entrypoint that keeps the existing
RealSense/Open3D/latest-wins scaffolding but changes the processing path to:

```text
RealSense color + aligned depth -> HF EdgeTAM streaming masks -> masked PCD only
```

The demo tracks two objects in one HF EdgeTAM session:

- `obj_id=1`: controller
- `obj_id=2`: object

## Scope

- Add `demo_v2/realtime_masked_edgetam_pcd.py`.
- Support `saved-masks` initialization for frame 0 and expose the
  `sam31-first-frame` mode behind lazy runtime dependencies.
- Keep heavy dependencies lazy so import and `--help` do not require
  RealSense, Open3D, Transformers, or CUDA.
- Add focused smoke coverage for CLI shape, saved-mask validation, object-id
  mapping, and masked-only backprojection.
- Update demo V2 documentation.

## Validation

- `python demo_v2/realtime_masked_edgetam_pcd.py --help`
- `python -m py_compile demo_v2/realtime_masked_edgetam_pcd.py`
- focused realtime smoke tests
- `python scripts/harness/check_all.py`

