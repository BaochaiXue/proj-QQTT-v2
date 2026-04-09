# Architecture

## Kept Runtime Surface

The repo is intentionally small.

### Entry Points

- `cameras_viewer.py`
- `cameras_calibrate.py`
- `record_data.py`
- `data_process/record_data_align.py`

### Shared Camera Package

- `qqtt/__init__.py`
- `qqtt/env/__init__.py`
- `qqtt/env/camera/defaults.py`
- `qqtt/env/camera/camera_system.py`
- `qqtt/env/camera/realsense/**`
- `qqtt/env/camera/recording_metadata.py`

### Optional FFS Depth Backend

- `data_process/depth_backends/__init__.py`
- `data_process/depth_backends/geometry.py`
- `data_process/depth_backends/fast_foundation_stereo.py`

### Comparison Visualization

- `data_process/visualization/__init__.py`
- `data_process/visualization/calibration_io.py`
- `data_process/visualization/camera_frusta.py`
- `data_process/visualization/depth_diagnostics.py`
- `data_process/visualization/object_compare.py`
- `data_process/visualization/object_roi.py`
- `data_process/visualization/panel_compare.py`
- `data_process/visualization/pointcloud_compare.py`
- `data_process/visualization/reprojection_compare.py`
- `data_process/visualization/source_compare.py`
- `data_process/visualization/support_compare.py`
- `data_process/visualization/turntable_compare.py`
- `scripts/harness/visual_compare_depth_panels.py`
- `scripts/harness/visual_compare_reprojection.py`
- `scripts/harness/visual_compare_depth_video.py`
- `scripts/harness/visual_compare_turntable.py`

### Tooling / Harness

- `env_install/env_install.sh`
- `scripts/harness/check_scope.py`
- `scripts/harness/check_all.py`
- `tests/test_record_data_align_smoke.py`
- `docs/*`

## Dependency Flow

`cameras_calibrate.py` and `record_data.py` import `CameraSystem`.

`CameraSystem` depends on:

- `qqtt/env/camera/realsense/multi_realsense.py`
- `qqtt/env/camera/realsense/single_realsense.py`
- shared-memory helpers under `qqtt/env/camera/realsense/shared_memory/`

`data_process/record_data_align.py` remains the terminal product stage. It:

- stays stdlib-only at import time so `--help` remains cheap
- lazily imports `data_process/depth_backends/*` only when `--depth_backend ffs|both` is requested
- keeps `realsense` as the default backend

Harness scripts for FFS proof-of-life now reuse `data_process/depth_backends/*` instead of maintaining a second geometry implementation.

The visualization layer intentionally uses three different diagnostics built on aligned cases:

- per-camera panels for local depth quality
- reprojection / warp comparison for multi-view consistency
- fused point-cloud rendering for global geometry shape

The fused point-cloud visualization is now split into two user-facing workflows:

- `visual_compare_turntable.py`
  - primary single-frame object-centric coverage-aware compare
  - explicit camera-frusta visualization from real `c2w`
  - large side-by-side Native vs FFS panels
  - automatic geometry + RGB + support videos and keyframe sheets
  - old 2x3 near-camera board retained only as a secondary mode
- `visual_compare_depth_video.py`
  - older temporal fused compare over a frame range
  - still useful as a secondary motion/consistency diagnostic

The fused point-cloud renderer now supports two view-selection modes:

- `fixed`: synthetic deterministic views such as `oblique`, `top`, and `side`
- `camera_poses_table_focus`: the 3 real calibrated camera poses, all refocused toward a shared tabletop center

The fused renderer also supports two layout modes:

- `pair`: one native-vs-ffs panel per view
- `grid_2x3`: a single 2x3 panel where the top row is Native, the bottom row is FFS, and the 3 columns use the selected camera-pose views

`turntable_compare.py` reuses the same aligned-case loading and fallback rendering primitives, then adds:

- single-frame case selection
- world-space ROI cropping before orbit computation
- optional per-camera image-ROI filtering before fusion for object-only review
- object-first dense/context splitting through `object_compare.py`
- seeded object-union bbox crop when manual image ROI masks are available
- default automatic pass1 -> pass2 object ROI refinement when manual image ROI is absent
- camera-frustum geometry extraction
- object-centric ROI extraction from the tabletop scene
- coverage-aware orbit planning informed by the real camera layout
- large side-by-side compare composition
- automatic geom/rgb/support output planning
- automatic mp4/gif animation export from the shared per-frame PNG sequence
- larger orthographic top-view position-map rendering with orbit path, supported arc, and crop visualization
- debug artifact export for per-camera masks, per-camera object clouds, fused object-only clouds, and compare metrics
- source-attribution overlay rendering and mismatch residual rendering through `source_compare.py`

The object-ROI stack now has two distinct roles:

- `object_roi.py`
  - table-plane estimation
  - component scoring
  - `graph_union` component closure for torso + protrusion recovery
- `object_compare.py`
  - projected coarse bbox generation from world ROI
  - automatic per-camera foreground-mask refinement
  - pixel-mask filtering back into world-space seed clouds
- `source_compare.py`
  - source-camera color mapping and legend generation
  - semi-transparent per-camera provenance overlay
  - split per-camera source contribution renders
  - overlap mismatch residual computation and rendering

This means the professor-facing compare no longer treats the initial fused world ROI as the sole authority. Pixel-derived object evidence is allowed to expand and refine the world ROI before the final compare is rendered.

It also means fused object clouds no longer drop source provenance as soon as they are concatenated: `source_camera_idx` now survives object/context/fused construction so the final compare can render provenance and mismatch as first-class diagnostics.

The shared fallback projection convention in `pointcloud_compare.py` maps positive view-space `y` upward on screen, so larger view-space height becomes a smaller image-row index without requiring any late image flip.

`calibrate.pkl` support is intentionally narrow and matches the current producer:

- object type: `list` / `tuple` or `numpy.ndarray`
- shape: `(N, 4, 4)`
- convention: each transform is `camera -> world` (`c2w`)
- ordering: calibration-time camera order

Subset capture cases rely on `metadata["calibration_reference_serials"]` to map case serials back to the full calibration order.

## Architectural Invariants

- No dependency from kept code into deleted downstream packages.
- No physics / rendering exports at the `qqtt` top level.
- Alignment remains the canonical data product of this repo; comparison visualization is an ancillary utility built on aligned cases.
- `depth/` remains the canonical compatibility output in aligned cases.
- Comparison visualization is diagnostic-only. It reads aligned cases and does not create new training or downstream simulation artifacts.
