## Goal

Extend `cameras_viewer_FFS.py` to support three explicit live FFS modes:

- current PyTorch path (`original`)
- current two-stage TensorRT path
- new single-engine TensorRT path

This change only affects the live viewer path. It does not modify `record_data_align.py` or batch scheduling.

## Scope

- `cameras_viewer_FFS.py`
- `data_process/depth_backends/fast_foundation_stereo.py`
- `data_process/depth_backends/__init__.py`
- `tests/test_cameras_viewer_ffs_smoke.py`
- new TensorRT single-engine backend smoke tests
- `README.md`
- `docs/WORKFLOWS.md`
- `docs/ARCHITECTURE.md`
- `docs/HARDWARE_VALIDATION.md`

## Design

1. keep `--ffs_backend {pytorch,tensorrt}` unchanged for compatibility
2. add `--ffs_trt_mode {two_stage,single_engine}` with default `two_stage`
3. keep `--ffs_backend tensorrt` defaulting to the current two-stage engine directory
4. reuse `--ffs_trt_model_dir` for both TRT modes
5. add a new single-engine TensorRT runner in `fast_foundation_stereo.py`
6. generalize TensorRT config discovery so both TRT modes share the same loader
7. keep `worker_mode={per_camera,shared}` orthogonal to the new TRT mode

## Validation

- `python cameras_viewer_FFS.py --help`
- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `python -m unittest -v tests.test_ffs_tensorrt_single_engine_smoke`
- `python scripts/harness/check_all.py`
