# Hardware Validation

Hardware checks are manual. CI does not attempt to validate RealSense behavior.

## Active Hardware Inventory

Current connected camera:

- `239222300781` - Intel RealSense D455

Design assumption for the `single-camera` branch: treat the active setup as one
D455 on the shared D400-family code path. The `main` branch remains the
protected 3-camera baseline.

## Viewer Checklist

- 1 D400 camera is connected.
- `python cameras_viewer.py` launches successfully.
- the camera shows live color and depth.
- the panel reports negotiated configured FPS plus live measured FPS.
- `conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --ffs_repo <repo>` launches successfully when FFS assets are available.
- optional PyTorch FFS mode launches successfully when requested explicitly:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --ffs_backend pytorch --ffs_repo <repo> --ffs_model_path <weights>
```

- optional TensorRT FFS mode launches successfully with prebuilt engines:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --ffs_backend tensorrt --ffs_trt_mode two_stage --ffs_repo <repo> --ffs_trt_model_dir <engine_dir> --ffs_trt_root <tensorrt_root>
```

## Single-Camera Demo Checklist

- branch-default realtime demo opens exactly one RealSense camera:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py --profile 848x480 --fps 30 --debug
```

- single demo dry-runs report `camera_count = 1`:

```bash
python demo_v3/realtime_single_camera_realsense_masked_pcd.py --dry-run
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

## Demo 3.2 Shape Prior Warmup Checklist

Demo 3.2 enables SAM3D shape-prior warmup by default at the wrapper layer. This
is a diagnostic gray reference layer only: it must not change EdgeTAM masks,
TAPNext++ queries/tracks, current observed PCD, table-world filtering, or the
strict tracking product.

- Start a resident worker on the SAM3D workstation:

```bash
conda run -n <sam3d-env> --no-capture-output \
  python services/shape_prior_remote/server.py \
  --bind tcp://0.0.0.0:7100 \
  --sam3d-root /home/xinjie/external/sam-3d-objects \
  --futurephystwin-root /home/xinjie/FuturePhysTwin
```

- For protocol-only validation without SAM3D weights, start:

```bash
conda run -n demo_2_max --no-capture-output \
  python services/shape_prior_remote/server.py \
  --bind tcp://0.0.0.0:7100 \
  --echo-observation
```

- Run fake-live with the default native RealSense backend and default async
  policy:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --replay-fps 5 \
  --duration-s 60 \
  --shape-prior-profile-json result/bench/fake_native_warm.json
```

- Run fake-live IR-FFS explicitly with the same worker:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --depth-backend ir-ffs \
  --replay-fps 5 \
  --duration-s 60 \
  --shape-prior-profile-json result/bench/fake_ir_ffs_warm.json
```

- For real-live validation, remove `--input-source fake-live` and repeat for
  the native default plus explicit `--depth-backend ir-ffs`.
- For each command, run a baseline with `--no-shape-prior-warmup` and the same
  input/depth settings.
- Repeat the matrix with
  `--shape-prior-start-policy blocking-before-first-output` only for startup
  penalty measurement.
- Verify missing or unreachable worker records `shape_prior_status=failed` in
  the profile JSON/HUD and does not terminate real-live or fake-live.
- Verify async runs publish first track/render before full SAM3D inference
  completes.
- Verify ready runs mount a gray shape reference layer and keep strict product
  `queries_txy`, `tracks_yx`, and visibility unchanged versus the baseline.

## Calibration Checklist

- `python cameras_calibrate.py` uses the current lab Calib.io ChArUco board by default:
  - profile: `calibio-12x9-30mm`
  - grid: `12x9`
  - checker size: 30 mm
  - marker size: 22 mm
  - dictionary: `DICT_5X5_250`
- calibration opens color streams only.
- calibration completes without reprojection failure.
- `calibrate.pkl` is written in the repo root.
- `calibrate_metadata.json` records serial order, board profile, world-frame
  convention, color distortion coefficients, and corner counts.
- rerun calibration after any physical camera-position change.

## Table Z0 Calibration Checklist

- exactly one D455 is connected, or `--serial` selects the intended camera
- the ChArUco board is flat on the tabletop that should define `Z=0`
- `conda run -n demo_2_max --no-capture-output python cameras_calibrate_table.py` exits 0
- `table_calibrate.pkl` exists
- `table_calibrate_metadata.json` exists and reports `world_frame_kind = table_world_z0`
- `table_calibrate_diagnostic.png` shows the board corners and pose axes on the board
- Demo 3.1/3.2/3.3 dry-runs report `pcd_coordinate_frame = table_world_z0`
  and `table_z_above_direction = negative` for the current repo calibration
- if the strict corner count or reprojection check fails, adjust lighting/board visibility and rerun

## Recording Checklist

- `python record_data.py --case_name smoke_case --capture_mode rgbd` creates `data_collect/smoke_case/`.
- `color/0/<step>.png` files are written.
- `depth/0/<step>.npy` files are written for `rgbd`.
- `ir_left/0/<step>.png` and `ir_right/0/<step>.png` are written for `stereo_ir`.
- `metadata.json` exists.
- `calibrate.pkl` is copied into the case if available.
- `calibrate_metadata.json` is copied into the case when available.
- short non-interactive `--max_frames ... --disable-keyboard-listener` runs fail quickly instead of hanging forever when the camera stalls.

## Alignment Checklist

- `python data_process/record_data_align.py --case_name smoke_case --start <start> --end <end> --depth_backend realsense` completes.
- aligned case exists under `data/smoke_case/`.
- aligned `metadata.json` exists.
- aligned `metadata_ext.json` exists for QQTT extension fields.
- aligned `color/0/` exists.
- aligned `depth/0/` exists for `realsense` and `ffs`.
- aligned `calibrate.pkl` is normalized to aligned case camera order.
- optional `--write_mp4` produces per-camera mp4 files if ffmpeg is installed.

## Current D455 Notes

- integrated `stereo_ir -> ffs` path has been validated on serial `239222300781`.
- fallback two-case comparison video workflow has been validated on serial `239222300781`.
- `both_eval` remains experimental and should not be promised as the default workflow.
