# Rendered Profile Open3D Shutdown Sync

## Goal

Apply the current rendered profiling lessons to Demo 2.2, Demo 2.3, Demo 3.0, and Demo 3.1 without running those demos: rendered FPS must come from `pointcloud` rendering, and finite Open3D runs must preserve profile artifacts even if the GUI/Filament teardown path hangs or crashes.

## Plan

- Update the shared three-view runtime so Open3D duration/window shutdown stops workers and writes summary/profile before requesting Open3D window/app teardown.
- Keep the existing `QQTT_WSLG_OPEN3D_FAST_EXIT=1` escape hatch for workstations where native teardown is still unsafe.
- Document the rendered profiling commands and caveats for Demo 2.2, Demo 2.3, Demo 3.0, and Demo 3.1.
- Add deterministic smoke coverage for the profile-first Open3D shutdown path.

## Validation

- `python -m py_compile qqtt/demo/three_view_masked_fused_pcd_runtime.py tests/test_demo_v2_3_dual_gpu_smoke.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo_v2_3_dual_gpu_smoke`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
