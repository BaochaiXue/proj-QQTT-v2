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
- `qqtt/env/camera/preflight.py`
- `qqtt/env/camera/realsense/**`
- `qqtt/env/camera/realsense/depth_postprocess.py`
- `qqtt/env/camera/recording_metadata.py`

### Optional FFS Depth Backend

- `data_process/depth_backends/__init__.py`
- `data_process/depth_backends/benchmarking.py`
- `data_process/depth_backends/geometry.py`
- `data_process/depth_backends/fast_foundation_stereo.py`

### Comparison Visualization

- `data_process/visualization/__init__.py`
- `data_process/visualization/calibration_frame.py`
- `data_process/visualization/calibration_io.py`
- `data_process/visualization/camera_frusta.py`
- `data_process/visualization/depth_diagnostics.py`
- `data_process/visualization/floating_point_diagnostics.py`
- `data_process/visualization/compare_scene.py`
- `data_process/visualization/io_artifacts.py`
- `data_process/visualization/io_case.py`
- `data_process/visualization/layouts.py`
- `data_process/visualization/object_compare.py`
- `data_process/visualization/object_roi.py`
- `data_process/visualization/panel_compare.py`
- `data_process/visualization/pointcloud_compare.py`
- `data_process/visualization/reprojection_compare.py`
- `data_process/visualization/renderers/**`
- `data_process/visualization/rerun_compare.py`
- `data_process/visualization/roi.py`
- `data_process/visualization/selection_contracts.py`
- `data_process/visualization/semantic_world.py`
- `data_process/visualization/source_compare.py`
- `data_process/visualization/stereo_audit.py`
- `data_process/visualization/support_compare.py`
- `data_process/visualization/turntable_compare.py`
- `data_process/visualization/triplet_ply_compare.py`
- `data_process/visualization/triplet_video_compare.py`
- `data_process/visualization/types.py`
- `data_process/visualization/views.py`
- `data_process/visualization/workflows/**`
- `scripts/harness/visual_compare_masked_pointcloud.py`
- `scripts/harness/visual_compare_masked_camera_views.py`
- `scripts/harness/visual_compare_depth_triplet_ply.py`
- `scripts/harness/visual_compare_depth_triplet_video.py`
- `scripts/harness/visual_compare_depth_panels.py`
- `scripts/harness/diagnose_floating_point_sources.py`
- `scripts/harness/visual_compare_reprojection.py`
- `scripts/harness/visual_compare_depth_video.py`
- `scripts/harness/visual_compare_rerun.py`
- `scripts/harness/visual_compare_stereo_order_pcd.py`
- `scripts/harness/visual_compare_turntable.py`

### Tooling / Harness

- `env_install/env_install.sh`
- `scripts/harness/check_scope.py`
- `scripts/harness/check_all.py`
- `scripts/harness/benchmark_ffs_configs.py`
- `scripts/harness/cleanup_different_types_cases.py`
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
- can optionally write auxiliary `depth_ffs_native_like_postprocess*` streams without changing canonical aligned depth outputs
- now writes aligned metadata in two files:
  - `metadata.json` for legacy `proj-QQTT` compatibility
  - `metadata_ext.json` for QQTT-only aligned metadata extensions

Downstream-facing formal exports under `data/different_types/` may be narrowed further with the cleanup harness:

- `cleanup_different_types_cases.py`
  - default dry-run
  - in-place removal of IR streams, FFS auxiliary depth streams, and `metadata_ext.json`
  - preserves the minimal downstream structure expected by external consumers, plus optional `color/<camera>.mp4` RGB sidecars
  - execute mode backfills missing color mp4 sidecars from `color/<camera>/*.png` before cleanup

Aligned exports written directly under `data/different_types/<case_name>/` auto-generate `color/0.mp4`, `1.mp4`, and `2.mp4` sidecars because downstream formal pipelines consume them.
Those formal exports also rewrite `calibrate.pkl` into case camera order so old downstream code that indexes `c2ws[cam_idx]` remains compatible.

Harness scripts for FFS proof-of-life now reuse `data_process/depth_backends/*` instead of maintaining a second geometry implementation.

The FFS benchmark helper stack is intentionally split like this:

- `data_process/depth_backends/benchmarking.py`
  - pure config-grid expansion
  - latency summary stats
  - reference-depth agreement metrics
  - target-FPS tradeoff selection
- `scripts/harness/benchmark_ffs_configs.py`
  - aligned-case stereo pair loading
  - repeated CUDA benchmark execution
  - JSON / markdown benchmark report writing

This keeps the deterministic summary logic testable without requiring CUDA while leaving the actual model execution in the thin harness CLI.

The native RealSense depth filter contract is now centralized in:

- `qqtt/env/camera/realsense/depth_postprocess.py`
  - source-of-truth chain and parameters for native depth postprocessing
  - live-frame application for `SingleRealsense.depth_process`
  - software-frame application for optional FFS native-like depth postprocessing during alignment and comparison

Record-time preflight policy is now explicit instead of being partially inlined inside `record_data.py`:

- `qqtt/env/camera/preflight.py`
  - maps capture mode to the relevant D455 stream-probe stream set
  - distinguishes:
    - supported
    - blocked
    - experimental warning
    - probe unknown
    - pending serial resolution
  - produces the operator-facing summary that `record_data.py` prints before recording continues

The visualization layer intentionally uses three different diagnostics built on aligned cases:

- per-camera panels for local depth quality
- reprojection / warp comparison for multi-view consistency
- fused point-cloud rendering for global geometry shape

The visualization package is now split by responsibility instead of concentrating everything inside the old compare modules:

- `io_case.py`
  - aligned-case metadata loading
  - merged loading of `metadata.json` + optional `metadata_ext.json`
  - grouped aligned-case resolution under `data/<type>/<case_name>/`
  - aligned depth-frame loading and FFS native-like postprocess selection/fallback
  - depth decoding
  - per-camera point-cloud generation
  - fused-cloud loading helpers
- `io_artifacts.py`
  - json / png / ply / mp4 / gif writing
- `roi.py`
  - focus estimation
  - table/object crop bounds
  - world-space crop filtering
- `views.py`
  - fixed view configs
  - camera-pose-derived views
  - orbit planning and supported-coverage math
- `layouts.py`
  - 2x3 grids
  - side-by-side professor-facing boards
  - keyframe sheets
  - shared text/label composition
- `renderers/fallback.py`
  - projection math
  - rasterization
  - fallback surfel/point rendering
- `types.py`
  - typed internal contracts for case selection, render specs, crops, and view config payloads
- `selection_contracts.py`
  - shared angle-selection ranking and summary contracts
- `compare_scene.py`
  - shared single-frame turntable scene assembly
  - shared orbit-state construction
  - shared object-view metric computation used by multiple workflows
- `calibration_frame.py`
  - explicit distinction between raw calibration-world semantics and any future semantic-world layer
- `semantic_world.py`
  - visualization-only semantic display transform inference
  - shared calibration-world vs semantic-world display selection
- `workflows/merge_diagnostics.py`
  - typed render-output planning for `geom / rgb / support / source / mismatch`
- `rerun_compare.py`
  - multi-frame native-vs-FFS raw point-cloud export to Rerun
  - shared full-scene fused PLY writing for `native / ffs_remove_1 / ffs_remove_0`
- `triplet_ply_compare.py`
  - single-frame fused PLY compare for `native / ffs_raw / ffs_postprocess`
  - aligned auxiliary postprocess preference plus on-the-fly fallback
- `triplet_video_compare.py`
  - multi-frame point-cloud video compare for `native / ffs_raw / ffs_postprocess`
  - fixed RGB coloring from aligned `color/`
  - local vertical image flip for the Open3D hidden-window capture path
- legacy compatibility modules:
  - `pointcloud_compare.py`
  - `turntable_compare.py`
  - still provide the old import paths, but now delegate to the shared lower-level modules

The fused point-cloud visualization is now split into four user-facing workflows:

- `visual_compare_depth_triplet_ply.py`
  - single-frame fused PLY-only compare for `Native`, `FFS raw`, and `FFS postprocess`
  - keeps raw calibration-world coordinates
  - prefers aligned postprocessed FFS depth when present and otherwise falls back to on-the-fly postprocessing
  - writes exactly 3 fused PLYs plus one compact summary
- `visual_compare_depth_triplet_video.py`
  - multi-frame RGB-colored point-cloud videos for `Native`, `FFS raw`, and `FFS postprocess`
  - shares one crop/view contract across all 3 outputs
  - uses the Open3D hidden-window path with an explicit vertical flip before writing frames
- `visual_compare_masked_pointcloud.py`
  - single-frame `Native` / `FFS` before-vs-after SAM 3.1 mask compare
  - keeps one shared crop and one shared oblique Open3D view across all 4 panels
  - reads or generates QQTT-local `sam31_masks` sidecars without introducing a PhysTwin runtime dependency
- `visual_compare_masked_camera_views.py`
  - single-frame masked `Native` vs masked `FFS` compare under the 3 original calibrated camera views
  - uses exact camera `c2w` position/forward/up per column rather than a shared oblique view
  - writes one fixed-view `2x3` Open3D board plus per-view debug renders
- `visual_compare_turntable.py`
  - primary single-frame object-centric coverage-aware compare
  - explicit camera-frusta visualization from real `c2w`
  - large side-by-side Native vs FFS panels
  - automatic geometry + RGB + support videos and keyframe sheets
  - old 2x3 near-camera board retained only as a secondary mode

- `visual_compare_depth_video.py`
  - older temporal fused compare over a frame range
  - still useful as a secondary motion/consistency diagnostic
- `visual_compare_rerun.py`
  - raw multi-frame remove-invisible diagnostic for aligned `native` and aligned `ffs`
  - reruns FFS from aligned `ir_left` / `ir_right`, then derives both `remove_1` and `remove_0` from the same disparity
  - writes fused full-scene PLYs and streams the same variants into a Rerun timeline
  - lazily imports `rerun-sdk` so `--help` remains cheap when the optional dependency is absent

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

The turntable workflow still keeps the old public entrypoint, but the internal split is now clearer:

- lower-level crop / view / layout / artifact helpers live outside the workflow module
- `turntable_compare.py` is primarily orchestration plus a small amount of turntable-specific overview/debug logic
- `scripts/harness/visual_compare_turntable.py` remains a thin CLI wrapper

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
- `io_artifacts.py`
  - shared product/debug artifact-set helpers so workflows stop hand-rolling output contracts

This means the professor-facing compare no longer treats the initial fused world ROI as the sole authority. Pixel-derived object evidence is allowed to expand and refine the world ROI before the final compare is rendered.

It also means fused object clouds no longer drop source provenance as soon as they are concatenated: `source_camera_idx` now survives object/context/fused construction so the final compare can render provenance and mismatch as first-class diagnostics.

The shared fallback projection convention in `pointcloud_compare.py` maps positive view-space `y` upward on screen, so larger view-space height becomes a smaller image-row index without requiring any late image flip.

## Visualization Layering

The intended import layering is now:

1. `scripts/harness/*.py`
   - parse args
   - call workflow entrypoints
2. workflow-facing modules
   - `panel_compare.py`
   - `reprojection_compare.py`
   - `turntable_compare.py`
   - `workflows/*`
3. shared visualization helpers
   - `io_case.py`
   - `io_artifacts.py`
   - `roi.py`
   - `views.py`
   - `layouts.py`
   - `types.py`
4. specialized diagnostics
   - `object_compare.py`
   - `object_roi.py`
   - `source_compare.py`
   - `support_compare.py`
   - `renderers/*`

`scripts/harness/check_visual_architecture.py` now enforces basic layering and file-size guardrails so the visualization stack does not regress back into a small number of giant mixed-responsibility files.

`calibrate.pkl` support is intentionally narrow and matches the current producer:

- object type: `list` / `tuple` or `numpy.ndarray`
- shape: `(N, 4, 4)`
- convention: each transform is `camera -> world` (`c2w`)
- ordering: calibration-time camera order

Important frame-semantics distinction:

- `calibrate.pkl` remains raw ChArUco-board calibration-world `c2w`
- professor-facing turntable and stereo-order workflows now default to a visualization-only `semantic_world` display frame
- that display frame is inferred from:
  - the tabletop plane
  - current camera centers
- it is applied only in memory for visualization
- raw calibration display remains available explicitly through `display_frame=calibration_world`
- both workflows now keep the distinction explicit in metadata/debug outputs instead of burying it inside the renderer

Important selection/artifact-contract distinction:

- angle-selection ranking and summary contracts are now shared through `selection_contracts.py`
- product-vs-debug output sets are now built through `io_artifacts.py` helpers and typed artifact contracts
- this does not force every workflow to emit the same files, but it does stop each workflow from inventing its own implicit output schema from scratch

Aligned cases rely on `metadata["calibration_reference_serials"]` whenever the case serial order differs from the calibration order, including same-length captures whose camera order changed and true subset captures.

## Architectural Invariants

- No dependency from kept code into deleted downstream packages.
- No physics / rendering exports at the `qqtt` top level.
- Alignment remains the canonical data product of this repo; comparison visualization is an ancillary utility built on aligned cases.
- `depth/` remains the canonical compatibility output in aligned cases.
- Comparison visualization is diagnostic-only. It reads aligned cases and does not create new training or downstream simulation artifacts.
