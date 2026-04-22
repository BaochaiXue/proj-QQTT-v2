# Workflows

## 1. Preview

```bash
python cameras_viewer.py
```

Use this to verify that all 3 cameras enumerate and stream correctly before calibration or recording.

Each panel now shows both the negotiated stream `configured fps` and a per-camera `measured fps` computed from the recent valid color+depth delivery rate, so fallback startup profiles and live stalls are easier to see during preview.

The viewer uses the same `TURBO` metric-depth colormap as the aligned-case depth diagnostics. Keep the display range explicit when you want preview colors to match later panels:

```bash
python cameras_viewer.py --depth-vis-min-m 0.1 --depth-vis-max-m 3.0
```

FFS preview for live RGB plus color-aligned FFS depth now defaults to the repo-local TensorRT path:

```bash
python cameras_viewer_FFS.py --ffs_repo /home/zhangxinjie/Fast-FoundationStereo
```

Use this as a debug viewer only:

- top = live RGB
- bottom = latest available color-aligned FFS depth
- overlay = negotiated stream profile plus live `capture` and `ffs` fps
- preview favors freshness over completeness and may drop stale stereo work while FFS catches up

If you explicitly want the older PyTorch viewer path instead of the default TensorRT engines:

```bash
python cameras_viewer_FFS.py --ffs_backend pytorch --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --ffs_model_path /home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth
```

Saved-pair FFS speed / tradeoff benchmark:

```bash
python scripts/harness/benchmark_ffs_configs.py --aligned_root ./data --case_ref static/ffs_30_static_round3_20260414 --camera_idx 0 --frame_idx 0 1 2 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-26-39\model_best_bp2_serialize.pth --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\20-30-48\model_best_bp2_serialize.pth --scale 1.0 0.75 0.5 --valid_iters 8 4 --max_disp 192 --warmup_runs 2 --repeats 4 --target_fps 15 25 30 --out_dir ./data/ffs_benchmarks/my_tradeoff_run
```

This benchmark-only workflow:

- loads aligned `ir_left` / `ir_right` plus FFS geometry from one aligned case
- sweeps checkpoint / `scale` / `valid_iters` / `max_disp`
- reports warmup-adjusted latency, FPS, and peak GPU memory
- compares each config against the chosen reference config after nearest-neighbor resize back to the reference depth shape
- writes:
  - `summary.json`
  - `report.md`

Use this first when the main question is:

- can current PyTorch FFS reach online-setting FPS on our machine?
- how much reference-depth drift appears when we lower `scale` or `valid_iters`?
- which config is the best compromise for a target FPS threshold?

Realistic live 3-camera FFS benchmark:

```bash
python cameras_viewer_FFS.py --ffs_backend pytorch --duration-s 20 --stats-log-interval-s 5 --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --ffs_model_path /home/zhangxinjie/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth --ffs_scale 0.75 --ffs_valid_iters 4 --ffs_max_disp 192
```

Use this when the main question is the real online path:

- 3 cameras streaming at once
- 3 FFS workers sharing one GPU
- latest-only queue pressure
- actual viewer-side `capture` vs `ffs` throughput

The live viewer now supports:

- `--duration-s`
  - auto-stops after the requested benchmark window
- `--stats-log-interval-s`
  - prints aggregate and per-camera runtime stats to stdout

The logged stats include:

- aggregate capture fps across all cameras
- aggregate FFS result fps across all cameras
- per-camera capture fps
- per-camera FFS fps
- latest per-camera inference ms
- per-camera `seq_gap` between the latest captured frame id and the latest completed FFS result id

Treat this live 3-camera viewer benchmark as the authoritative online-setting measurement. The saved-pair benchmark above is still useful for offline checkpoint/parameter screening, but it is not a substitute for simultaneous 3-camera runtime behavior.

## 2. Calibrate

```bash
python cameras_calibrate.py
```

This writes `calibrate.pkl` in the repo root by default.

Useful options:

```bash
python cameras_calibrate.py --width 1280 --height 720 --fps 5 --num-cam 3
```

## 3. Record

```bash
python record_data.py --case_name my_case --capture_mode rgbd
```

If `--case_name` is omitted, a timestamp is used.

The recorder writes raw data to `data_collect/<case_name>/`.

Default RealSense path:

```bash
python record_data.py --case_name my_case --capture_mode rgbd
```

Optional FFS raw capture path:

```bash
python record_data.py --case_name my_case --capture_mode stereo_ir --emitter on
```

Current preflight policy:

- `rgbd`
  - supported directly
  - no D455 IR-pair probe gate
- `stereo_ir`
  - probe-aware
  - unsupported profile remains allowed experimentally with a warning
- `both_eval`
  - probe-aware
  - unsupported profile remains allowed experimentally with a warning

`record_data.py` now prints a preflight summary before recording, including:

- selected or pending serials
- requested profile
- probe support result
- current repo policy
- whether recording is allowed

If `--serials` is omitted, the first summary is intentionally provisional:

- stage = `before camera discovery`
- serials = `<pending>`

After `CameraSystem` resolves the actual camera serials, `record_data.py` prints a second summary:

- stage = `after camera discovery`
- final support / blocked / experimental / unknown status for the discovered serial set

When `--max_frames` is used, recording now fails fast if one or more cameras stop making progress while others continue, instead of waiting indefinitely for the slowest camera to catch up.

If a camera drops during or immediately after stop, the worker now exits cleanly instead of emitting a secondary inactive-pipeline traceback during reconnect handling.

Optional non-interactive short capture:

```bash
python record_data.py --case_name smoke_case --capture_mode rgbd --max_frames 5 --disable-keyboard-listener
```

Optional single-camera selection for validation:

```bash
python record_data.py --case_name smoke_case --capture_mode stereo_ir --serials 239222300781 --max_frames 5 --disable-keyboard-listener
```

## 4. Align

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend realsense
```

Optional mp4 generation:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --write_mp4
```

Optional output location override:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --output_path ./data
```

The grouped `data/<type>/<case_name>/` layout applies to aligned cases under `data/`, not to raw recordings under `data_collect/`.

Grouped aligned layouts are supported by choosing a grouped output root. For example:

```bash
python record_data.py --case_name my_case --capture_mode rgbd
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend realsense --output_path ./data/static
```

This keeps the aligned case under `data/static/my_case/` while the raw recording remains under `data_collect/my_case/`.

Future aligned cases now write:

- `metadata.json`
  - old `proj-QQTT` compatible aligned fields only
- `metadata_ext.json`
  - QQTT extension fields such as depth backend, stream layout, calibration-reference serials, and FFS geometry/config metadata

Visualization case resolution now accepts either:

- a relative grouped ref such as `static/my_case`
- a unique bare case name such as `my_case` when it appears only once under `aligned_root`

Optional FFS backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --write_ffs_float_m
```

Optional FFS native-like postprocess during alignment:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --ffs_native_like_postprocess
```

This keeps canonical FFS compatibility depth unchanged and additionally writes:

- `depth_ffs_native_like_postprocess/`
- `depth_ffs_native_like_postprocess_float_m/`

Optional Open3D radius-outlier filtering during alignment:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --ffs_radius_outlier_filter --ffs_radius_outlier_radius_m 0.01 --ffs_radius_outlier_nb_points 40 --write_ffs_float_m
```

This PhysTwin-style filtering:

- runs on each per-camera color-aligned FFS depth frame
- applies Open3D `remove_radius_outlier(nb_points=40, radius=0.01)` by default
- writes the filtered result as the main aligned FFS depth
- archives the unfiltered raw FFS depth beside it
- does not change native `depth/` when `--depth_backend both`

Experimental comparison backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend both --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth
```

Important:

- `realsense` remains the default backend.
- `ffs` requires raw `ir_left` / `ir_right` plus runtime geometry metadata.
- `both` is experimental and should only be used when the hardware probe says the same-take stream set is supported.
- current repo visualization loaders merge `metadata.json` and `metadata_ext.json` automatically when both are present
- `ffs_raw` triplet workflows now prefer archived raw FFS depth when present and otherwise fall back to legacy pre-archive `depth_ffs*`

For downstream-facing formal exports under `data/different_types/`, use the cleanup script after alignment:

```bash
python scripts/harness/cleanup_different_types_cases.py --root ./data/different_types --case_name sloth_base_motion_native --case_name sloth_base_motion_ffs
```

Default behavior is `dry-run`. Add `--execute` to apply the cleanup in place.

The cleanup keeps the formal downstream layout minimal, but it may also preserve optional `color/0.mp4`, `color/1.mp4`, and `color/2.mp4` sidecars for consumers that require per-camera RGB videos.
If those RGB sidecars are missing, execute mode auto-generates them from `color/<camera>/*.png` before removing non-formal extras.

When `record_data_align.py` writes directly under `data/different_types/<case_name>/`, it now auto-generates those `color/<camera>.mp4` sidecars even if `--write_mp4` was not passed, because downstream formal consumers depend on them.
For the same old-downstream compatibility reason, those formal exports also normalize `calibrate.pkl` into the case camera order (`color/0`, `1`, `2`) instead of preserving a separate calibration-reference order.

After cleanup, each case keeps only:

- `color/0|1|2`
- `depth/0|1|2`
- `calibrate.pkl`
- `metadata.json`

This formal downstream export is intentionally narrower than the repo's internal aligned-case comparison contract and deletes `metadata_ext.json`, IR streams, FFS raw archives such as `*_original*`, and FFS auxiliary depth directories.

## 5. Compare

Start with single-camera panels:

```bash
python scripts/harness/visual_compare_depth_panels.py --aligned_root ./data --realsense_case static/native_case --ffs_case static/ffs_case --write_mp4 --use_float_ffs_depth_when_available
```

Optional FFS native-like postprocess in the panel workflow:

```bash
python scripts/harness/visual_compare_depth_panels.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --use_float_ffs_depth_when_available --ffs_native_like_postprocess
```

Use this to judge:

- per-camera holes / invalid regions
- depth edge quality
- local surface smoothness
- ROI crops on the same spatial region

When `--ffs_native_like_postprocess` is enabled:

- the workflow prefers aligned `depth_ffs_native_like_postprocess*` streams when present
- otherwise it computes the same FFS native-like postprocess on the fly before rendering the panels
- `summary.json` records whether FFS native-like postprocessing was enabled and which FFS depth source was used per frame

For professor-/review-quality static boards, use the publication-style preset:

```bash
python scripts/harness/visual_compare_depth_panels.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --preset review_quality --camera_ids 0 1 2 --use_float_ffs_depth_when_available
```

The upgraded panel workflow now:

- uses a larger title strip with case id, camera id, and frame id
- keeps native and FFS depth panels on the exact same depth range
- overlays ROI boxes on both RGB and depth panels
- accepts named ROIs via `--roi name:x0,y0,x1,y1`
- enlarges ROI detail panels
- can add `RGB vs Depth Edges` comparison panels through the `review_quality` preset or `--show_edge_overlay`
- writes per-frame summary metrics into `summary.json`, including:
  - valid pixel ratios
  - median / p90 absolute depth difference
  - ROI-specific median absolute depth difference

Then run cross-view reprojection diagnostics:

```bash
python scripts/harness/visual_compare_reprojection.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --camera_pair 0,1 --camera_pair 0,2 --write_mp4 --use_float_ffs_depth_when_available
```

Use this to judge:
- which depth is more multi-view consistent
- which source depth produces lower reprojection residuals in the target camera
- whether failures are localized to one camera pair or happen everywhere

To diagnose where floating-point outliers come from without changing any aligned outputs:

```bash
python scripts/harness/diagnose_floating_point_sources.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --use_float_ffs_depth_when_available
```

This diagnostic-only workflow:

- loads aligned `color + depth` data only; it does not rewrite `depth/`, metadata, or calibration artifacts
- applies a PhysTwin-style full-scene radius outlier rule independently to `Native` and `FFS`
- projects each outlier back to its source color image and classifies it as `occlusion`, `edge`, `dark`, or `other`
- writes:
  - `native/frames/*.png`
  - `ffs/frames/*.png`
  - `native/per_frame_metrics.json`
  - `ffs/per_frame_metrics.json`
  - `summary.json`
  - optional `comparison.mp4` when `--write_mp4` is enabled

Use this to judge:

- whether most outliers cluster around image/depth edges
- whether they are concentrated in dark image regions
- whether they disappear under cross-view support and instead look like occlusion failures
- which camera contributes most of the outliers on each source path

For triplet time-axis point-cloud videos across `Native`, `FFS raw`, and `FFS postprocess`:

```bash
python scripts/harness/visual_compare_depth_triplet_video.py --aligned_root ./data --realsense_case dynamics/native_case --ffs_case dynamics/ffs_case
```

This workflow:

- writes `native_open3d.mp4`, `ffs_raw_open3d.mp4`, and `ffs_postprocess_open3d.mp4`
- uses aligned RGB colors for all 3 videos
- keeps one shared `auto_table_bbox` crop and one shared `oblique` view across all variants
- applies a vertical image flip to correct the Open3D hidden-window capture orientation

For single-frame point-cloud quality diagnosis before and after SAM 3.1 masking:

```bash
python scripts/harness/visual_compare_masked_pointcloud.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --text_prompt sloth
```

This workflow:

- keeps `PhysTwin` only as a reference design; it does not import or depend on that repo
- prefers existing `sam31_masks/` sidecars when present
- otherwise can generate workflow-local SAM 3.1 sidecars through QQTT's own helper path
- fuses 4 fixed-view variants under one shared crop and one shared oblique Open3D camera:
  - `Native Unmasked`
  - `Native Masked`
  - `FFS Unmasked`
  - `FFS Masked`
- writes:
  - `01_masked_pointcloud_board.png`
  - `summary.json`
  - `debug/native_mask_overlay_cam*.png`
  - `debug/ffs_mask_overlay_cam*.png`
  - `debug/native_unmasked_fused.ply`
  - `debug/native_masked_fused.ply`
  - `debug/ffs_unmasked_fused.ply`
  - `debug/ffs_masked_fused.ply`

Use this when the question is specifically:

- does background suppression change the apparent Native-vs-FFS point-cloud quality?
- do we get a cleaner object-focused compare after masking?
- how much point count is removed by masking per camera and per source?

For single-frame masked point-cloud diagnosis under the 3 original calibrated camera views:

```bash
python scripts/harness/visual_compare_masked_camera_views.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --text_prompt sloth
```

To compare after applying the same PhysTwin-like depth postprocess to both `Native` and `FFS` before fusion/rendering:

```bash
python scripts/harness/visual_compare_masked_camera_views.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --text_prompt sloth --native_depth_postprocess --ffs_native_like_postprocess
```

This workflow:

- reuses the same QQTT-local `sam31_masks` resolution / generation policy as `visual_compare_masked_pointcloud.py`
- uses the 3 real calibrated camera extrinsics from `calibrate.pkl`
- uses the original per-camera `K_color` pinhole projection when rendering point clouds so each Open3D panel matches the corresponding RGB view scale more closely
- can optionally apply the same software postprocess chain to `Native` depth on the fly
- can optionally prefer aligned `depth_ffs_native_like_postprocess*` for `FFS` and otherwise run the same postprocess on the fly
- fixes one exact original camera view per column:
  - `Cam0`
  - `Cam1`
  - `Cam2`
- writes one `1x3` masked RGB reference board with the background zeroed outside the resolved mask
- keeps one shared masked-object crop across the `2x3` point-cloud board
- renders one `2x3` Open3D board:
  - top row = masked `Native`
  - bottom row = masked `FFS`
- writes:
  - `00_masked_rgb_board.png`
  - `01_masked_camera_view_board.png`
  - `summary.json`
  - `debug/masked_rgb_cam*.png`
  - `debug/native_cam*.png`
  - `debug/ffs_cam*.png`
  - `debug/native_mask_overlay_cam*.png`
  - `debug/ffs_mask_overlay_cam*.png`
  - `debug/native_masked_fused.ply`
  - `debug/ffs_masked_fused.ply`

Use this when the question is specifically:

- how do `Native` and `FFS` look from the exact 3 original camera viewpoints?
- if we fix the camera extrinsics, where does FFS geometry break relative to native depth?
- does masking still leave visible floating fragments when judged from the original camera views?

For professor-facing 3-view point-cloud match diagnosis, start with the single match board:

```bash
python scripts/harness/visual_make_match_board.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0
```

Default top-level outputs are only:

- `01_pointcloud_match_board.png`
- `match_board_summary.json`

This board is intentionally narrow:

- rows:
  - `Native`
  - `FFS`
- columns:
  - `Source attribution`
  - `Support count`
  - `Mismatch residual`

The match angle is object-aware and match-oriented:

- supported-hemisphere only
- prefers higher object-only multi-camera support
- prefers lower object-only mismatch residual
- still requires enough projected object area to stay readable
- penalizes table/context dominance
- penalizes thin edge-on silhouettes

All clutter is gated off by default:

- no debug directory
- no videos
- no orbit keyframe sheets
- no extra top-level figures

Enable those only when needed:

```bash
python scripts/harness/visual_make_match_board.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --write_debug
```

When `--write_debug` is enabled, selection-specific artifacts stay under `debug/`, for example:

- `debug/match_angle_candidates.json`

For a three-figure slide pack that reuses the same object-first compare stack but answers a broader presentation question, use:

```bash
python scripts/harness/visual_make_professor_triptych.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0
```

Default top-level outputs are only:

- `01_hero_compare.png`
- `02_merge_evidence.png`
- `03_truth_board.png`
- `summary.json`

Selection/debug artifacts stay under `debug/` only when requested.

Use the single-frame object-centric coverage-aware orbit compare when you need the richer fused-cloud diagnostics behind that single board:

Same-case comparison when an aligned case contains both native and FFS depth:

```bash
python scripts/harness/visual_compare_turntable.py --case_name my_case --aligned_root ./data --frame_idx 0
```

Fallback two-case comparison when `both_eval` is not supported:

```bash
python scripts/harness/visual_compare_turntable.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --renderer fallback --scene_crop_mode auto_object_bbox
```

When the automatic object crop still includes too much tabletop, provide per-camera RGB boxes so only object pixels are fused:

```bash
python scripts/harness/visual_compare_turntable.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --renderer fallback --scene_crop_mode auto_object_bbox --manual_image_roi_json docs/generated/object_only_manual_image_roi_native_30_static_frame_0000.json
```

The turntable workflow:

- selects one aligned frame
- reuses the fused native and fused FFS cloud loader
- computes an object ROI from the cropped tabletop scene before orbit generation
- visualizes the 3 real camera frusta from `calibrate.pkl`
- keeps `calibrate.pkl` as the raw calibration-board `c2w` frame on disk
- now defaults to a visualization-only `semantic_world` display frame so:
  - tabletop appears horizontal
  - cameras appear above the object/table
  - top/front/side views are human-readable
- still allows raw rendering for debugging:
  - `--display_frame calibration_world`
- defaults to `observed_hemisphere` instead of naive full-360
- defaults to `object_component_mode=graph_union`
- infers the supported viewing arc from the real camera azimuth layout
- renders a large side-by-side compare:
  - left = Native
  - right = FFS
- writes clean slide-ready hero stills first:
  - `hero_compare_geom.png`
  - `hero_compare_rgb.png`
- uses the exact same orbit path for Native and FFS
- automatically writes geometry, RGB, and support-count products in one run
- automatically writes source-attribution and mismatch products in the same run
- optionally applies `--manual_image_roi_json` before fusion to suppress tabletop pixels when the object itself should dominate the professor-facing render
- when `--manual_image_roi_json` is present, runs an object-first selection path:
  - full dense camera clouds are loaded first
  - per-camera object masks are built before context subsampling
  - a seeded object union bbox drives crop / focus / orbit
  - `--max_points_per_camera` becomes the context-layer cap instead of an early object-point cap
- when `--manual_image_roi_json` is absent, the default auto-object path now adds an explicit refinement loop:
  - pass1 coarse world ROI from fused object-above-table points
  - projected per-camera coarse bbox generation
  - automatic per-camera foreground-mask refinement
  - pass2 world ROI rebuilt from those per-camera masked object points
  - final compare uses pass2 ROI plus pass2 masks
- uses a larger orthographic top-view position map so the real calibrated camera positions stay readable without stretching the inset
- produces:
  - `hero_compare_geom.png`
  - `hero_compare_rgb.png`
  - `scene_overview_with_cameras.png`
  - `scene_overview_calibration_frame.png`
  - `orbit_compare_geom.mp4`
  - `orbit_compare_geom.gif`
  - `orbit_compare_rgb.mp4`
  - `orbit_compare_rgb.gif`
  - `orbit_compare_support.mp4`
  - `orbit_compare_support.gif`
  - `orbit_compare_source.mp4`
  - `orbit_compare_source.gif`
  - `orbit_compare_mismatch.mp4`
  - `orbit_compare_mismatch.gif`
  - `turntable_keyframes_geom.png`
  - `turntable_keyframes_rgb.png`
  - `turntable_keyframes_support.png`
  - `turntable_keyframes_source.png`
  - `turntable_keyframes_source_split.png`
  - `turntable_keyframes_mismatch.png`
  - `support_metrics.json`
  - `source_metrics.json`
  - `mismatch_metrics.json`
  - `source_attribution_legend.png`
  - `object_roi_pass1_world.json`
  - `object_roi_pass2_world.json`
  - `per_camera_auto_bbox/cam*.json`
  - `debug/per_camera_object_mask_overlay/*.png`
  - `debug/per_camera_object_cloud/*.png`
  - `debug/fused_object_only/*`
  - `debug/fused_object_context/*`
  - `debug/compare_debug_metrics.json`
  - per-angle `frames_source/*.png`, `frames_source_split/*.png`, and `frames_mismatch/*.png`
  - per-angle `frames_geom/*.png`, `frames_rgb/*.png`, and `frames_support/*.png`

The user-facing compare commands are unchanged, but the implementation stack is now split more cleanly:

- harness CLIs stay thin
- workflow modules coordinate steps
- shared case IO, crop math, view planning, artifact writing, and layouts live in dedicated modules under `data_process/visualization/`

For the current internal map and migration notes, see:

- `docs/generated/README.md`
- `docs/generated/visual_stack_cleanup_inventory.md`
- `docs/generated/visual_stack_cleanup_validation.md`

Use `full_360` only when you explicitly want the unsupported backside visualization to appear, with warnings:

```bash
python scripts/harness/visual_compare_turntable.py --case_name my_case --orbit_mode full_360 --show_unsupported_warning
```

The old 2x3 near-camera board remains available only as a secondary mode:

```bash
python scripts/harness/visual_compare_turntable.py --case_name my_case --layout_mode camera_neighborhood_grid --orbit_mode camera_neighborhood --num_orbit_steps 6 --orbit_degrees 30
```

When teddy/head/ear regions are still incomplete, inspect the debug artifacts in this order:

- `debug/per_camera_object_mask_overlay/*.png`
- `debug/per_camera_object_cloud/*.png`
- `debug/fused_object_only/*.png`
- `debug/compare_debug_metrics.json`

If the head is missing already in the per-camera overlays, tighten `--manual_image_roi_json` before rerunning. If the per-camera overlays look correct but the fused object remains weak, use the support render to confirm whether the missing region is mostly only 1-camera supported.

Single-frame triplet fused PLY compare:

Use this when the question is specifically “for one aligned frame, how do `Native`, `FFS raw`, and `FFS postprocess` differ after fusing all 3 cameras into calibration-world point clouds?”, and you only need `.ply` outputs plus a compact summary:

```bash
python scripts/harness/visual_compare_depth_triplet_ply.py --aligned_root ./data --realsense_case native_30_static_20260410_235202 --ffs_case ffs_30_static_20260410_235202 --frame_idx 0
```

This workflow:

- selects one aligned frame only
- keeps raw `calibration_world` coordinates
- fuses all 3 cameras for exactly 3 variants:
- `native`
- `ffs_raw`
- `ffs_postprocess`
- reuses aligned `depth/` for Native
- prefers archived raw FFS depth for `ffs_raw`:
  - `depth_ffs_float_m_original/`
  - `depth_ffs_original/`
  - `depth_original/` when the aligned case itself is FFS-backed
- otherwise falls back to legacy pre-archive `depth_ffs*`
- prefers aligned `depth_ffs_native_like_postprocess*` for `ffs_postprocess`
- otherwise applies the same native-like depth postprocess on the fly before fusion
- writes:
  - `ply_fullscene/native_frame_<idx>_fused_fullscene.ply`
  - `ply_fullscene/ffs_raw_frame_<idx>_fused_fullscene.ply`
  - `ply_fullscene/ffs_postprocess_frame_<idx>_fused_fullscene.ply`
  - `summary.json`

Default point-cloud filtering in this generic triplet workflow now keeps only valid depth `> 0m` and clips points beyond `1.5m`. Override `--depth_min_m` / `--depth_max_m` if you need a wider export.

Raw multi-frame Rerun remove-invisible compare:

Use this when the main question is not slide composition, but “how do the fused full-scene point clouds evolve over time, and what exactly changes when `remove_invisible` is on vs off?”:

```bash
python scripts/harness/visual_compare_rerun.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --rerun_output viewer_and_rrd --viewer_layout horizontal_triple
```

This workflow:

- uses the full shared aligned frame range by default
- can force a `1x3` horizontal viewer layout so `native`, `ffs_remove_1`, and `ffs_remove_0` stay side by side while you scrub time
- keeps raw calibration-world coordinates
- fuses all 3 cameras into one full-scene cloud for each variant
- re-runs FFS from aligned `ir_left` / `ir_right` instead of reusing the aligned `depth/` output
- derives both `ffs_remove_1` and `ffs_remove_0` from the same disparity so the only intended delta is the overlap invalidation
- logs only 3 entity paths to Rerun:
  - `native`
  - `ffs_remove_1`
  - `ffs_remove_0`
- writes:
  - `pointcloud_compare.rrd`
  - `ply_fullscene/native_frame_<idx>_fused_fullscene.ply`
  - `ply_fullscene/ffs_remove_1_frame_<idx>_fused_fullscene.ply`
  - `ply_fullscene/ffs_remove_0_frame_<idx>_fused_fullscene.ply`
  - `summary.json`

Default point-cloud filtering in this generic Rerun workflow now keeps only valid depth `> 0m` and clips points beyond `1.5m`. Override `--depth_min_m` / `--depth_max_m` if you need a wider export.

Use `--rerun_output rrd_only` when you want a non-interactive run that still saves the timeline for later replay.

Focused stereo-depth audits:

Use the point-cloud-only stereo-order registration board when the main question is not “which one looks prettier,” but “does current left/right or swapped left/right produce tighter 3-view 3D alignment?”:

```bash
python scripts/harness/visual_compare_stereo_order_pcd.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth
```

Default top-level outputs are only:

- `01_stereo_order_registration_board.png`
- `match_board_summary.json`

This board is intentionally narrow:

- rows:
  - `Native`
  - `FFS-current`
  - `FFS-swapped`
- columns:
  - `Oblique`
  - `Top`
  - `Front`
  - `Side`

Interpret it like this:

- thinner, tighter, less color-separated surfaces = better 3-camera registration
- thicker shells and stronger red/green/blue fringing = worse registration
- if `FFS-swapped` collapses more tightly than `FFS-current`, current left/right ordering stays suspicious

All panels are:

- point-cloud-only
- colored only by source camera
- rendered in the same display frame
- rendered with the same frame / ROI / crop / view scaling semantics

By default this board also uses `semantic_world` display coordinates for readability:

- table approximately horizontal
- cameras above the table/object
- top/front/side columns consistent with human intuition

Switch back to raw calibration display only when you explicitly want to inspect the original ChArUco world:

```bash
python scripts/harness/visual_compare_stereo_order_pcd.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --display_frame calibration_world
```

Optional closeup/debug outputs stay gated:

```bash
python scripts/harness/visual_compare_stereo_order_pcd.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --write_closeup --write_debug
```

Across the current professor-facing workflows, the output rule is now explicit:

- top-level directory contains only the intended product artifacts for that workflow
- optional diagnostics go under `debug/`
- selection summaries use shared typed contracts for display-frame, angle-selection, and product/debug artifact sets

Use a left/right audit on one aligned FFS case, one camera, and one frame when you want to verify that the repo is really feeding Fast-FoundationStereo the correct IR ordering:

```bash
python scripts/harness/audit_ffs_left_right.py --aligned_root ./data --ffs_case ffs_case --frame_idx 0 --camera_idx 0 --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --face_patches_json docs/generated/box_face_patches_static_frame_0000.json
```

This writes:

- `left_right_audit.json`
- `left_right_audit_board.png`

Use fixed face patches when you want to compare smoothness / noise on planar surfaces rather than on the whole scene:

```bash
python scripts/harness/compare_face_smoothness.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0 --face_patches_json docs/generated/box_face_patches_static_frame_0000.json --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth
```

This writes:

- `face_quality_board.png`

Interpret these face-patch metrics as:

- better = higher valid depth ratio
- better = lower plane-fit RMSE
- better = lower MAD
- better = lower p90 point-to-plane residual

Frame semantics note:

- `calibrate.pkl` remains raw calibration-board `c2w`
- professor-facing turntable and stereo-order point-cloud workflows now default to a visualization-only `semantic_world` display frame inferred from:
  - the fitted tabletop plane
  - the current camera centers
- this transform is applied only in memory for visualization
- `scene_overview_calibration_frame.png` and `scene_overview_semantic_frame.png` make the distinction explicit
- raw calibration-world display is still available through `--display_frame calibration_world`

Use the new merge-diagnostic outputs like this:

- `source`:
  - semi-transparent overlay by source camera
  - red = `Cam0`
  - green = `Cam1`
  - blue = `Cam2`
  - use this to spot fringing and double surfaces
- `source_split`:
  - inspect which camera is actually providing or missing the teddy head
- `mismatch`:
  - high residual = poor 3-view merge agreement
  - low residual = stable overlap
- `support`:
  - still useful, but it only answers “how many cameras agree,” not “which cameras disagree”

Keep the older fused-cloud temporal video workflow only as a secondary diagnostic:

Same-case comparison when an aligned case contains both native and FFS depth:

```bash
python scripts/harness/visual_compare_depth_video.py --case_name my_case --aligned_root ./data --preset tabletop_compare_2x3 --write_mp4
```

Fallback two-case comparison when `both_eval` is not supported:

```bash
python scripts/harness/visual_compare_depth_video.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --renderer fallback --preset tabletop_compare_2x3 --write_mp4 --use_float_ffs_depth_when_available
```

The temporal comparison workflow:

- decodes compatible depth to meters
- deprojects with `K_color`
- transforms to world using `calibrate.pkl`
- fuses the aligned camera clouds
- can render from the 3 real calibrated camera poses
- applies a world-space tabletop crop before framing
- can move the effective viewpoint closer to the tabletop via `--view_distance_scale`
- supports `perspective` and `orthographic` projection in the fallback renderer
- uses denser splat-like fallback rendering for tabletop inspection
- can compose a single `2x3` output:

This generic fused point-cloud compare path now defaults to valid depth `> 0m` and a `1.5m` far clip so tabletop scenes shed distant background geometry by default. Override `--depth_min_m` / `--depth_max_m` when you need a wider range.
  - row 1 = Native
  - row 2 = FFS
  - columns = camera 0 / 1 / 2 viewpoints
- recommends geometry-first rendering (`neutral_gray_shaded`) for judging shape
- keeps `color_by_rgb` available as a secondary reference view
- the `tabletop_compare_2x3` preset currently uses `color_by_height` plus orthographic tabletop framing as the most robust readable default
