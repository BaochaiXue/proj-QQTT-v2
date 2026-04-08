# Workflows

## 1. Preview

```bash
python cameras_viewer.py
```

Use this to verify that all 3 cameras enumerate and stream correctly before calibration or recording.

The viewer uses the same `TURBO` metric-depth colormap as the aligned-case depth diagnostics. Keep the display range explicit when you want preview colors to match later panels:

```bash
python cameras_viewer.py --depth-vis-min-m 0.1 --depth-vis-max-m 3.0
```

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

Optional FFS backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend ffs --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --write_ffs_float_m
```

Experimental comparison backend:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend both --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --ffs_model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth
```

Important:

- `realsense` remains the default backend.
- `ffs` requires raw `ir_left` / `ir_right` plus runtime geometry metadata.
- `both` is experimental and should only be used when the hardware probe says the same-take stream set is supported.

## 5. Compare

Start with single-camera panels:

```bash
python scripts/harness/visual_compare_depth_panels.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --write_mp4 --use_float_ffs_depth_when_available
```

Use this to judge:

- per-camera holes / invalid regions
- depth edge quality
- local surface smoothness
- ROI crops on the same spatial region

Then run cross-view reprojection diagnostics:

```bash
python scripts/harness/visual_compare_reprojection.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --camera_pair 0,1 --camera_pair 0,2 --write_mp4 --use_float_ffs_depth_when_available
```

Use this to judge:

- which depth is more multi-view consistent
- which source depth produces lower reprojection residuals in the target camera
- whether failures are localized to one camera pair or happen everywhere

Finally use the single-frame object-centric coverage-aware orbit compare for professor-facing fused geometry review:

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
- defaults to `observed_hemisphere` instead of naive full-360
- infers the supported viewing arc from the real camera azimuth layout
- renders a large side-by-side compare:
  - left = Native
  - right = FFS
- uses the exact same orbit path for Native and FFS
- automatically writes geometry, RGB, and support-count products in one run
- optionally applies `--manual_image_roi_json` before fusion to suppress tabletop pixels when the object itself should dominate the professor-facing render
- produces:
  - `scene_overview_with_cameras.png`
  - `orbit_compare_geom.mp4`
  - `orbit_compare_rgb.mp4`
  - `orbit_compare_support.mp4`
  - `turntable_keyframes_geom.png`
  - `turntable_keyframes_rgb.png`
  - `turntable_keyframes_support.png`
  - `support_metrics.json`
  - per-angle `frames_geom/*.png`, `frames_rgb/*.png`, and `frames_support/*.png`

Use `full_360` only when you explicitly want the unsupported backside visualization to appear, with warnings:

```bash
python scripts/harness/visual_compare_turntable.py --case_name my_case --orbit_mode full_360 --show_unsupported_warning
```

The old 2x3 near-camera board remains available only as a secondary mode:

```bash
python scripts/harness/visual_compare_turntable.py --case_name my_case --layout_mode camera_neighborhood_grid --orbit_mode camera_neighborhood --num_orbit_steps 6 --orbit_degrees 30
```

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
  - row 1 = Native
  - row 2 = FFS
  - columns = camera 0 / 1 / 2 viewpoints
- recommends geometry-first rendering (`neutral_gray_shaded`) for judging shape
- keeps `color_by_rgb` available as a secondary reference view
- the `tabletop_compare_2x3` preset currently uses `color_by_height` plus orthographic tabletop framing as the most robust readable default
