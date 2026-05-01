# proj-QQTT-v2

This repository handles 3-camera RealSense preview, calibration, synchronized raw capture, aligned case generation, and native-vs-FFS comparison visualization for aligned cases.

## Scope

This repo is intentionally narrow. It supports only:

1. RealSense camera preview / debugging
2. multi-camera calibration
3. synchronized recording with:
   - default RealSense RGB-D
   - optional raw D455 IR stereo capture
4. raw recording alignment and trimming
5. optional Fast-FoundationStereo depth generation during alignment
6. native-vs-FFS aligned depth comparison visualization

This repo does **not** include:

- segmentation
- dense tracking
- shape-prior generation
- downstream point-cloud processing beyond alignment packaging
- inverse physics
- Warp training / inference
- Gaussian Splatting
- rendering evaluation
- teleoperation or interaction demos

See [docs/SCOPE.md](/c:/Users/zhang/proj-QQTT/docs/SCOPE.md) for the exact boundary.

## Hardware Assumptions

- 3 Intel RealSense D400-series cameras
- a ChArUco calibration board
- Windows or Linux with librealsense-compatible device access
- optional footswitch or keyboard input for recording
- optional `ffmpeg` if you want aligned mp4 files

## Installation

Create and activate a Python 3.10 conda environment, then run:

```bash
bash ./env_install/env_install.sh
```

The install script is camera-only. It installs only the dependencies needed for:

- preview
- calibration
- recording
- alignment

## Preview

Live preview / debugging:

```bash
python cameras_viewer.py --help
python cameras_viewer.py
```

Default preview settings come from the shared camera defaults in [defaults.py](/c:/Users/zhang/proj-QQTT/qqtt/env/camera/defaults.py).

The preview depth panel now uses the same `TURBO` metric-depth colorization path as the repo's aligned-case depth diagnostics. You can keep the display range explicit:

```bash
python cameras_viewer.py --depth-vis-min-m 0.1 --depth-vis-max-m 3.0
```

Live RGB + Fast-FoundationStereo preview:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py
```

Single-D455 realtime demo:

```bash
conda run -n FFS-SAM-RS python demo_v2/realtime_single_camera_pointcloud.py --profile 848x480 --fps 60
```

See [demo_v2/README.md](demo_v2/README.md) for native RealSense, FFS depth,
and WSLg/Open3D commands. `demo_v1/` is kept as the baseline copy.

True no-render FFS throughput probe:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --render-mode none
```

In `--render-mode none`, the viewer skips panel assembly / `imshow()`, does not
reproject FFS depth into the color frame, and prints periodic console FPS stats
instead.

The default TensorRT FFS path for viewer, proxy, and visualization runs is:

- environment: `FFS-SAM-RS`
- external repo default: sibling `../Fast-FoundationStereo`
- checkpoint: `20-30-48`
- `--ffs_valid_iters 4`
- `--ffs_max_disp 192`
- backend: `--ffs_backend tensorrt --ffs_trt_mode two_stage`
- TensorRT artifact: `result/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/engines/model_20-30-48_iters_4_res_480x864/`
- TensorRT build level: `builder_optimization_level=5`

For actual QQTT performance, use the recorded `848x480` RealSense image size. If the TRT shape needs a multiple of `32`, pad `848x480` to `864x480` and unpad the output; do not resize down to `640x480` and report that as QQTT runtime.

Performance boundary:

- Live PyTorch 3-camera FFS is not realtime on the RTX 5090 laptop. The best measured `20-30-48 / valid_iters=4 / scale=0.5` live PyTorch run reached only about `22.6` aggregate FFS FPS, or about `7.5` FPS per camera.
- The `20-30-48 / valid_iters=4 / 848x480 -> 864x480 / builderOptimizationLevel=5` result is the current TensorRT static replay / proxy target and has basically reached that proxy goal.
- Do not merge static replay or TensorRT proxy numbers into a "live PyTorch 3-camera realtime" claim.

Available live FFS modes:

- `--ffs_backend tensorrt --ffs_trt_mode two_stage`
  - default
- `--ffs_backend pytorch`
  - current original PyTorch path
- `--ffs_backend tensorrt --ffs_trt_mode single_engine`
  - new single-engine TensorRT path

The default FFS viewer topology is still `--ffs_worker_mode per_camera`, which means one FFS worker process per active camera. You can also force a single shared FFS worker process for all active cameras:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --ffs_worker_mode shared
```

Strict 3-camera batch mode is now also available:

- `--ffs_batch_mode strict3`
  - requires `--ffs_worker_mode shared`
  - requires exactly 3 active cameras
  - runs one shared FFS inference over the 3 latest camera IR pairs at once
- TensorRT batch mode requires batch-3 TRT artifact directories
  - intended two-stage directory: `data/ffs_proof_of_life/trt_two_stage_batch3_864x480_wsl`
  - intended single-engine directory: `data/ffs_proof_of_life/trt_single_engine_batch3_864x480_wsl_fp32`
  - current machine-side batch-3 TRT build status is tracked in `docs/generated/ffs_batch3_viewer_validation.md`

Strict 3-camera batch preview examples:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --ffs_backend pytorch --ffs_repo ../Fast-FoundationStereo --ffs_model_path ../Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth --ffs_valid_iters 4 --ffs_worker_mode shared --ffs_batch_mode strict3
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --ffs_backend tensorrt --ffs_trt_mode two_stage --ffs_trt_model_dir data/ffs_proof_of_life/trt_two_stage_batch3_864x480_wsl --ffs_worker_mode shared --ffs_batch_mode strict3
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --ffs_backend tensorrt --ffs_trt_mode single_engine --ffs_trt_model_dir data/ffs_proof_of_life/trt_single_engine_batch3_864x480_wsl_fp32 --ffs_worker_mode shared --ffs_batch_mode strict3
```

Explicit single-engine TensorRT preview:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --ffs_backend tensorrt --ffs_trt_mode single_engine --ffs_trt_model_dir /path/to/single_engine_trt_dir
```

Static-round TRT matrix replay + PPTX export:

```bash
conda run -n FFS-SAM-RS python scripts/harness/run_ffs_static_replay_matrix.py --output_root ./result/ffs_static_replay_matrix_my_run --artifact_root ./result/_archived_obsolete/ffs_static_replay_matrix_20260422_sequential_obsolete_fullrun/artifacts --reuse_artifacts
```

This workflow is the current offline static replay / TensorRT proxy harness for the three static FFS rounds. It:

- replays `static/ffs_30_static_round1_20260410_235202`, `round2`, and `round3`
- searches the fixed `3 models × 3 scales × 3 valid_iters × 2 TRT engines = 54` matrix
- benchmarks `3` cameras simultaneously within each round using `3` worker processes and `batch=1`
- measures `30 / elapsed_seconds` FPS after a `10`-frame warmup for every `Round X cam Y`
- writes one shared masked RGB `3x3` board artifact, one masked FFS-only PCD `3x3` board per experiment, `results.csv`, `manifest.json`, and a ranked PPTX
- the PPT itself now contains only `1` slide per experiment:
  - top-of-slide summary text + frame-10 masked PCD panel

Important:

- run new static replay / TensorRT proxy visualization work from `FFS-SAM-RS`
- `python-pptx` and `onnx` must be available in that environment
- `--artifact_root` can point at an older artifact tree so TRT engines are reused without reusing old benchmark results
- the harness uses `stuffed animal` masks at `frame_idx=10`
- if the local SAM 3.1 checkpoint is unavailable, the harness falls back to the existing static frame-0 stuffed-animal masks and copies them to frame 10 for this static-only workflow

## Calibration

Calibrate the 3-camera setup:

```bash
python cameras_calibrate.py --help
python cameras_calibrate.py
```

Successful calibration writes `calibrate.pkl` in the repo root by default.

Current calibration defaults are optimized for board detection rather than live throughput:

```bash
python cameras_calibrate.py --width 1280 --height 720 --fps 5
```

## Recording

Record a raw case.

Default path:

```bash
python record_data.py --help
python record_data.py --case_name my_case --capture_mode rgbd
```

If `--case_name` is omitted, a timestamp-based folder name is used.

Raw cases are written under `data_collect/<case_name>/`.

If `calibrate.pkl` exists, `record_data.py` copies it into the recorded case folder.

Optional FFS raw capture path:

```bash
python record_data.py --case_name my_case --capture_mode stereo_ir --emitter on
```

Optional experimental comparison path:

```bash
python record_data.py --case_name my_case --capture_mode both_eval --emitter on
```

`both_eval` is intentionally gated. On the current machine it is now warning-allowed experimentally even though the latest long-duration D455 stream capability probe still marks the profile unstable.

Current recording preflight policy:

- `rgbd`: supported directly
- `stereo_ir`: probe-aware, warning-only when unsupported
- `both_eval`: probe-aware, warning-only when unsupported

`record_data.py` now prints an explicit preflight summary before recording continues.

## Alignment

Align and trim a raw case:

```bash
python data_process/record_data_align.py --help
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend realsense
```

Defaults:

- `--base_path ./data_collect`
- `--output_path ./data`
- `--depth_backend realsense`
- output fps comes from raw recording metadata unless `--fps` is provided
- mp4 generation is off unless `--write_mp4` is passed

Aligned cases are written to `data/<case_name>/`.

Optional FFS alignment backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --write_ffs_float_m
```

Optional FFS native-like postprocess during alignment:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --ffs_native_like_postprocess
```

This keeps canonical `depth/` unchanged and additionally writes:

- `depth_ffs_native_like_postprocess/`
- `depth_ffs_native_like_postprocess_float_m/`

Optional PyTorch logits confidence filtering during FFS alignment:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --ffs_confidence_mode max_softmax --ffs_confidence_threshold 0.6 --ffs_confidence_depth_min_m 0.2 --ffs_confidence_depth_max_m 1.5
```

Confidence-filtered depth remains uint16 depth with invalid / rejected pixels
encoded as zero. See `docs/ffs_confidence_filtering.md` for sweep usage and
threshold interpretation.

Optional Open3D radius-outlier filtering during alignment:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --ffs_radius_outlier_filter --ffs_radius_outlier_radius_m 0.01 --ffs_radius_outlier_nb_points 40 --write_ffs_float_m
```

This rewrites the main aligned FFS depth to the filtered result and archives the unfiltered raw FFS depth:

- for `--depth_backend ffs`:
  - main filtered depth: `depth/`
  - raw archive: `depth_original/`
  - optional filtered float: `depth_ffs_float_m/`
  - optional raw float archive: `depth_ffs_float_m_original/`
- for `--depth_backend both`:
  - native `depth/` stays unchanged
  - main filtered FFS depth: `depth_ffs/`
  - raw archive: `depth_ffs_original/`
  - optional filtered float: `depth_ffs_float_m/`
  - optional raw float archive: `depth_ffs_float_m_original/`

Optional comparison backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend both --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth
```

Important:

- raw Fast-FoundationStereo output is not color-aligned by itself
- this repo explicitly reprojects FFS depth from IR-left coordinates into color coordinates during alignment
- canonical aligned `depth/` remains compatibility-oriented

Realtime native RGB-D formal export:

```bash
python record_data_realtime_align.py --case_name native_rt_baseline
```

This writes one growing case under `data/different_types_real_time/<case_name>/`
with only `color/`, `depth/`, `calibrate.pkl`, and legacy `metadata.json`.
It does not generate mp4 sidecars. FPS stats are written under
`data/different_types_real_time/_logs/` and report complete 3-camera aligned
frame sets per second.

## Compare Native vs FFS

The repo now provides three complementary comparison views. Use them together:

1. Per-camera diagnostic panels:

```bash
python scripts/harness/visual_compare_depth_panels.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --write_mp4 --use_float_ffs_depth_when_available
```

Optional FFS native-like postprocess in the panel workflow:

```bash
python scripts/harness/visual_compare_depth_panels.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --use_float_ffs_depth_when_available --ffs_native_like_postprocess
```

When enabled, the panel workflow prefers aligned `depth_ffs_native_like_postprocess*` streams when they exist and otherwise computes the same FFS native-like postprocess on the fly.
If an aligned case was built with `--ffs_radius_outlier_filter`, the default `FFS` path already uses the filtered main FFS depth.

This is the primary first-pass diagnostic. It shows:

- native RGB and FFS RGB
- native depth and FFS depth with the same scale
- absolute depth difference heatmap
- valid-mask comparison
- surface-shaded depth
- deterministic ROI crops

2. Cross-view reprojection / warp comparison:

```bash
python scripts/harness/visual_compare_reprojection.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --camera_pair 0,1 --camera_pair 0,2 --write_mp4 --use_float_ffs_depth_when_available
```

This is the main multi-view consistency diagnostic. It warps source RGB into the target view using native depth and FFS depth separately, then compares each warp against the target RGB with residual heatmaps and summary metrics.

3. Professor-facing three-figure pack:

Use this when you need one slide-ready conclusion pack instead of dozens of debug artifacts:

```bash
python scripts/harness/visual_make_professor_triptych.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0
```

By default it writes only:

- `01_hero_compare.png`
- `02_merge_evidence.png`
- `03_truth_board.png`
- `summary.json`

The three figures answer three different questions:

- `01_hero_compare.png`
  - overall, which fused result looks better?
- `02_merge_evidence.png`
  - do source attribution, support count, and mismatch residual support that conclusion?
- `03_truth_board.png`
  - is the apparent winner actually more multi-view truthful, or just more filled in?

Default clutter is off:

- no debug dump
- no mp4 / gif
- no orbit keyframe sheets

Optional raw Rerun diagnostic:

```bash
python -m pip install rerun-sdk
python scripts/harness/visual_compare_rerun.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --rerun_output viewer_and_rrd --viewer_layout horizontal_triple
```

This raw workflow keeps calibration-world coordinates, logs multi-frame `native / ffs_remove_1 / ffs_remove_0` fused point clouds to a Rerun timeline, and writes fused full-scene PLYs for each frame.

Generic fused point-cloud workflows now default to a tabletop-oriented depth filter: valid depth must stay `> 0m` and points farther than `1.5m` are dropped unless you override `--depth_min_m` / `--depth_max_m`.

Enable those only when needed:

```bash
python scripts/harness/visual_make_professor_triptych.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --write_debug --write_video --write_keyframes
```

4. Single-frame object-centric coverage-aware orbit compare:

Same-case comparison, when an aligned case contains both native depth and FFS depth:

```bash
python scripts/harness/visual_compare_turntable.py --case_name my_case --aligned_root ./data --frame_idx 0
```

Fallback two-case comparison:

```bash
python scripts/harness/visual_compare_turntable.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --renderer fallback --scene_crop_mode auto_object_bbox
```

If the automatic object crop still keeps too much tabletop, constrain the fused cloud at the source-image level with per-camera RGB boxes:

```bash
python scripts/harness/visual_compare_turntable.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --renderer fallback --scene_crop_mode auto_object_bbox --manual_image_roi_json docs/generated/object_only_manual_image_roi_native_30_static_frame_0000.json
```

This is now the main fused-cloud diagnostic engine behind the professor-facing three-figure pack. It:

- loads one selected aligned frame instead of a temporal frame range
- fuses one native point cloud and one FFS point cloud
- visualizes the 3 real camera poses from `calibrate.pkl`
- computes an object ROI rather than only a tabletop crop
- defaults to `observed_hemisphere` instead of naive full-360
- limits the default orbit to the supported viewing arc inferred from the real camera layout
- defaults to `object_component_mode=graph_union` so sparse protrusions are recovered through a transitive component graph instead of only a top-component anchor
- renders one large left-right comparison:
  - left = Native
  - right = FFS
- uses the exact same orbit path for both renders
- automatically writes all three:
  - geometry diagnostic outputs
  - RGB-colored reference outputs
  - support-count outputs
- also writes source-provenance merge diagnostics:
  - source-attribution overlay outputs
  - source-split keyframe sheets
  - mismatch residual outputs
- can optionally apply `--manual_image_roi_json` to keep only object pixels from each real camera before fusion when the professor-facing view should exclude the tabletop as much as possible
- when `--manual_image_roi_json` is provided, the compare now runs an object-first path:
  - load the full dense per-camera cloud first
  - derive per-camera object masks before context subsampling
  - use the seeded object union bbox to drive crop / focus / orbit
  - keep object points dense while treating `--max_points_per_camera` as the context-layer cap
- when `--manual_image_roi_json` is not provided, the default auto-object flow now does:
  - pass1 coarse world ROI from fused points
  - project that ROI into each real camera to get coarse 2D bboxes
  - refine per-camera foreground masks automatically
  - rebuild pass2 world ROI from the pixel-derived object evidence
  - use pass2 ROI plus pass2 masks for the final compare
- writes debug artifacts that show where object detail is lost or preserved:
  - `object_roi_pass1_world.json`
  - `object_roi_pass2_world.json`
  - `per_camera_auto_bbox/cam*.json`
  - `debug/per_camera_object_mask_overlay/*.png`
  - `debug/per_camera_object_cloud/*.png`
  - `debug/fused_object_only/*`
  - `debug/fused_object_context/*`
  - `debug/compare_debug_metrics.json`
- preserves `source_camera_idx` through the object/context/fused pipeline so the final compare also writes:
  - `orbit_compare_source.mp4`
  - `orbit_compare_source.gif`
  - `turntable_keyframes_source.png`
  - `turntable_keyframes_source_split.png`
  - `orbit_compare_mismatch.mp4`
  - `orbit_compare_mismatch.gif`
  - `turntable_keyframes_mismatch.png`
  - `source_attribution_legend.png`
  - `source_metrics.json`
  - `mismatch_metrics.json`
- shows a larger orthographic top-view position map so the real camera locations stay readable without stretching the inset
- writes:
  - `scene_overview_with_cameras.png`
  - `frames_geom/*.png`
  - `frames_rgb/*.png`
  - `frames_support/*.png`
  - `orbit_compare_geom.mp4`
  - `orbit_compare_geom.gif`
  - `orbit_compare_rgb.mp4`
  - `orbit_compare_rgb.gif`
  - `orbit_compare_support.mp4`
  - `orbit_compare_support.gif`
  - `turntable_keyframes_geom.png`
  - `turntable_keyframes_rgb.png`
  - `turntable_keyframes_support.png`
  - `support_metrics.json`

`full_360` remains available as a presentation mode, but unsupported backside views are labeled instead of being treated as equally trustworthy:

```bash
python scripts/harness/visual_compare_turntable.py --case_name my_case --orbit_mode full_360 --show_unsupported_warning
```

The older 2x3 near-camera turntable board remains available only as a secondary advanced mode via:

```bash
python scripts/harness/visual_compare_turntable.py --case_name my_case --layout_mode camera_neighborhood_grid --orbit_mode camera_neighborhood --num_orbit_steps 6 --orbit_degrees 30
```

For teddy-bear-like cases, `auto_object_bbox` can still under-segment sparse protrusions or keep too much tabletop. Prefer a tight `--manual_image_roi_json` and inspect `debug/compare_debug_metrics.json` plus the per-camera mask overlays before treating the fused professor-facing compare as final.

## Visualization Architecture

The user-facing compare commands are unchanged, but the visualization implementation is now split into clearer internal layers:

- `data_process/visualization/io_case.py`
  - aligned-case metadata, depth decoding, and per-camera/fused cloud loading
- `data_process/visualization/io_artifacts.py`
  - json / png / ply / mp4 / gif writing
- `data_process/visualization/roi.py`
  - focus estimation and world-space crop bounds
- `data_process/visualization/views.py`
  - fixed views, camera-pose views, orbit planning, and supported-coverage math
- `data_process/visualization/layouts.py`
  - shared board composition, labels, and keyframe sheets
- `data_process/visualization/renderers/`
  - rendering-only projection/rasterization backends
- `data_process/visualization/workflows/`
  - thin workflow-facing wrappers and render-output planning
- compatibility entrypoints kept in:
  - `data_process/visualization/pointcloud_compare.py`
  - `data_process/visualization/turntable_compare.py`

This keeps the existing commands stable while making it easier to test and evolve the visualization stack without re-growing a small number of mixed-responsibility modules.

Current compare frame semantics are also explicit now:

- aligned-case comparison paths use the raw calibration-board `c2w` world frame
- no semantic-world transform is silently applied by default
- professor-facing turntable outputs now also write `scene_overview_calibration_frame.png` and record the frame contract in `turntable_metadata.json`

Additional internal documentation:

- `scripts/harness/README.md`
- `docs/generated/README.md`
- `docs/generated/visual_stack_cleanup_inventory.md`
- `docs/generated/visual_stack_cleanup_validation.md`

Why the new source diagnostics matter:

- `geom` shows surface readability but hides which camera contributed which points.
- `rgb` can hide misalignment behind texture.
- `support` shows how many cameras agree, but not which cameras created a double surface or fringe.
- `source` colors points by camera provenance with semi-transparent overlay:
  - `Cam0` = red
  - `Cam1` = green
  - `Cam2` = blue
- `source_split` shows each camera contribution separately on the same orbit and crop.
- `mismatch` colors overlap residual magnitude so merge disagreement is visible even when RGB still looks plausible.

5. Temporal fused point-cloud comparison video:

Same-case comparison, when an aligned case contains both native depth and FFS depth:

```bash
python scripts/harness/visual_compare_depth_video.py --case_name my_case --aligned_root ./data --preset tabletop_compare_2x3 --write_mp4
```

Fallback two-case comparison, when `both_eval` is not supported on the current machine:

```bash
python scripts/harness/visual_compare_depth_video.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --renderer fallback --preset tabletop_compare_2x3 --write_mp4 --use_float_ffs_depth_when_available
```

The temporal fused-cloud utility remains available as a secondary diagnostic. It:

- decodes compatible depth to meters
- deprojects with color intrinsics
- uses `calibrate.pkl` camera-to-world transforms
- fuses per-camera point clouds into a common frame
- can render from fixed synthetic views or from the 3 real calibrated camera poses
- supports a world-space tabletop crop before view bounds are computed
- supports geometry-aware camera distance control and perspective / orthographic projection
- supports a table-focus mode that keeps the view centered on the tabletop
- uses denser splat-like fallback rendering instead of isolated 1-pixel dots
- supports:
  - classic pair output per view
  - a 2x3 summary grid with top row = Native and bottom row = FFS
- the `tabletop_compare_2x3` preset now uses `color_by_height` plus orthographic tabletop framing for readability
- keeps `color_by_rgb` as a secondary reference mode
- writes per-view frame sequences and optional videos, plus `grid_2x3_frames/` and `videos/grid_2x3.mp4` when requested

By default, this generic fused point-cloud path now clips farther-background geometry at `1.5m`; pass explicit depth limits if you need a wider full-scene export.

6. Raw multi-frame Rerun remove-invisible compare:

```bash
python -m pip install rerun-sdk
python scripts/harness/visual_compare_rerun.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --rerun_output viewer_and_rrd --viewer_layout horizontal_triple
```

This workflow:

- uses the full shared aligned frame range by default
- can pin the viewer to a `1x3` layout through `--viewer_layout horizontal_triple`
- keeps raw calibration-world coordinates
- fuses 3 cameras into one full-scene point cloud for each variant
- re-runs FFS from aligned `ir_left` / `ir_right`
- derives both `ffs_remove_1` and `ffs_remove_0` from the same disparity
- writes:
  - `pointcloud_compare.rrd`
  - `ply_fullscene/native_frame_<idx>_fused_fullscene.ply`
  - `ply_fullscene/ffs_remove_1_frame_<idx>_fused_fullscene.ply`
  - `ply_fullscene/ffs_remove_0_frame_<idx>_fused_fullscene.ply`
  - `summary.json`

## Output Layout

### Raw case layout

```text
data_collect/<case_name>/
  calibrate.pkl
  metadata.json
  color/
    0/<step>.png
    1/<step>.png
    2/<step>.png
  depth/                # for rgbd or both_eval
    0/<step>.npy
    1/<step>.npy
    2/<step>.npy
  ir_left/              # for stereo_ir or both_eval
    0/<step>.png
    1/<step>.png
    2/<step>.png
  ir_right/             # for stereo_ir or both_eval
    0/<step>.png
    1/<step>.png
    2/<step>.png
```

### Aligned case layout

```text
data/<case_name>/
  calibrate.pkl
  metadata.json
  color/
    0/<frame>.png
    1/<frame>.png
    2/<frame>.png
    0.mp4          # only if --write_mp4
    1.mp4          # only if --write_mp4
    2.mp4          # only if --write_mp4
  depth/
    0/<frame>.npy
    1/<frame>.npy
    2/<frame>.npy
  ir_left/              # copied through when present
    0/<frame>.png
    1/<frame>.png
    2/<frame>.png
  ir_right/             # copied through when present
    0/<frame>.png
    1/<frame>.png
    2/<frame>.png
  depth_ffs/            # only for --depth_backend both
    0/<frame>.npy
    1/<frame>.npy
    2/<frame>.npy
  depth_ffs_float_m/    # optional
    0/<frame>.npy
    1/<frame>.npy
    2/<frame>.npy
  depth_original/       # only for --depth_backend ffs with --ffs_radius_outlier_filter
    0/<frame>.npy
    1/<frame>.npy
    2/<frame>.npy
  depth_ffs_original/   # only for --depth_backend both with --ffs_radius_outlier_filter
    0/<frame>.npy
    1/<frame>.npy
    2/<frame>.npy
  depth_ffs_float_m_original/ # optional raw float archive when --ffs_radius_outlier_filter is enabled
    0/<frame>.npy
    1/<frame>.npy
    2/<frame>.npy
  comparison/           # optional native-vs-FFS visualization output
    native_frames/
    ffs_frames/
    side_by_side_frames/
    grid_2x3_frames/    # optional 2x3 comparison grid
    videos/
      grid_2x3.mp4
    metrics.json
    comparison_metadata.json
  turntable_frame_<idx>/ # optional single-frame object-centric coverage-aware compare
    scene_overview_with_cameras.png
    orbit_compare_geom.mp4
    orbit_compare_rgb.mp4
    orbit_compare_support.mp4
    turntable_keyframes_geom.png
    turntable_keyframes_rgb.png
    turntable_keyframes_support.png
    support_metrics.json
    turntable_metadata.json
    frames_geom/
      000_angle_*.png
    frames_rgb/
      000_angle_*.png
    frames_support/
      000_angle_*.png
  depth_panels/         # optional per-camera diagnostic panels
    camera_0/frames/
    camera_1/frames/
    camera_2/frames/
    summary.json
  reprojection_compare/ # optional cross-view warp diagnostics
    pair_0_to_1/frames/
    pair_0_to_2/frames/
    summary_metrics.json
  professor_triptych_frame_<idx>/ # optional professor-facing three-figure pack
    01_hero_compare.png
    02_merge_evidence.png
    03_truth_board.png
    summary.json
```

## Validation

Deterministic checks:

```bash
python scripts/harness/check_all.py
```

This default path is the fast deterministic profile and is intended to finish in
under one minute on a healthy repo environment.

When you need the broader legacy validation surface, run:

```bash
python scripts/harness/check_all.py --full
```

Manual hardware validation checklist:

- [docs/HARDWARE_VALIDATION.md](/c:/Users/zhang/proj-QQTT/docs/HARDWARE_VALIDATION.md)
- [docs/generated/ffs_depth_backend_integration_validation.md](/c:/Users/zhang/proj-QQTT/docs/generated/ffs_depth_backend_integration_validation.md)
- [docs/generated/ffs_comparison_workflow_validation.md](/c:/Users/zhang/proj-QQTT/docs/generated/ffs_comparison_workflow_validation.md)

## Future Changes

This repo uses lightweight harness engineering:

- short map in [AGENTS.md](/c:/Users/zhang/proj-QQTT/AGENTS.md)
- versioned plans under [docs/exec-plans](/c:/Users/zhang/proj-QQTT/docs/exec-plans)
- deterministic scope guard in [check_scope.py](/c:/Users/zhang/proj-QQTT/scripts/harness/check_scope.py)

Any future change must preserve the camera-only charter.
