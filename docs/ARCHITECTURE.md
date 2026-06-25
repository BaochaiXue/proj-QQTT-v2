# Architecture

## Kept Runtime Surface

The core recording/alignment surface is intentionally small. Sanctioned
single-camera demo, remote FFS proxy, and visualization diagnostics are kept as
explicit boundaries around that core instead of becoming part of the formal
aligned-case data product.

## Entry Points

- `cameras_viewer.py`
- `cameras_viewer_FFS.py`
- `cameras_calibrate.py`
- `record_data.py`
- `record_data_realtime_align.py`
- `data_process/record_data_align.py`
- `scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py`
- `demo_v3/realtime_single_camera_realsense_masked_pcd.py`
- `demo_v3_1/realtime_single_camera_realsense_masked_pcd.py`
- `demo_v3_2/realtime_single_camera_ffs_masked_pcd.py`
- `demo_v3_3/realtime_single_camera_ffs_masked_pcd.py`
- `demo_v4/realtime_futurephystwin_chunks.py`
- `demo_v5/realtime_futurephystwin_chunks.py`

## Shared Packages

- `qqtt/env/camera/`: RealSense discovery, capture, defaults, preflight policy,
  recording metadata, and depth postprocess helpers.
- `data_process/depth_backends/`: FFS geometry, runners, benchmarking helpers,
  and optional postprocess filters.
- `data_process/visualization/`: aligned-case native-vs-FFS comparison and
  render/output helpers.
- `qqtt/demo/realtime_single_camera_pointcloud.py`: branch-default live
  single-camera point-cloud demo implementation.
- `qqtt/demo/single_demo_v3_runtime.py`: shared launcher for the
  `demo_v3*` entrypoints.
- `qqtt/demo/demo32_side_by_side_panel.py`: pure Demo 3.2 side-by-side panel
  renderer shared by offline headless rendering and runtime fake-live panel
  display. It composes latest RGB, projected filtered PCD, tracking overlay,
  and timing HUD frames; it does not own camera capture, FFS, EdgeTAM, or
  runtime queues.
- `qqtt/demo/realtime_masked_edgetam_pcd.py`: shared masked PCD runtime. Demo
  3.2/3.3 tracking visualizations and PCD-only inspection default
  object/controller filters to `none` while still running the full tracker
  pipeline for honest FPS. The high-level
  `--pcd-filter-preset {original,pt,enhanced-pt}` option overrides those
  defaults for both object/controller layers and also selects the residual PCD
  pixels used for TAPNext++ query initialization. When explicitly selected,
  object enhanced-pt filtering keeps one component and controller enhanced-pt
  filtering keeps two components, matching single-object plus two-hand demo
  scenes. It starts
  Open3D in a third-person orbit view by default, falls back to capped
  current-frame PCD when filtering
  would empty a nonempty layer, falls back for controller output that retains
  less than half of its capped points, falls back to raw current-frame
  controller PCD when voxel capping makes the controller output less than half
  of the raw semantic controller points, and rejects async filtered outputs
  older than three frames for rendering. Render-point caps use deterministic
  coarse spatial bucket balancing so sparse separated controller regions are not
  hidden by denser regions when the display layer is capped. In demo-mode hand
  tracking it propagates `hand_a`, `object`, and `hand_b` as separate EdgeTAM
  identities while preserving the legacy controller PCD mask as the union of
  `hand_a` and `hand_b`. Its EdgeTAM path is a frame-by-frame live session, not
  an offline batch video path; runtime state is bounded to a recent 64-frame
  window by default for fake-live and live stability. When TAPNext++ is enabled,
  the runtime uses ordered lossless queues so the 5 FPS task stream is processed
  without latest-wins drops; independent PCD and tracker workers publish only
  same-sequence pairs, and bounded backlog overflow is a fatal pipeline error.
  Local FFS depth is serialized and cached by sequence inside the runtime so PCD
  and tracker threads share color-aligned depth without entering one TensorRT
  runner concurrently. Demo 3.1/3.2/3.3 table-calibrated output uses
  `table_world_z0`; after the current PCD preset output is transformed to table world,
  the runtime records per-class world-Z quantiles and table-band candidate
  counts, including hand_a/hand_b when those masks are available. The tabletop
  is `table_z_m = 0.0`; the current single-camera table calibration marks the
  workspace above the table as negative Z (`table_z_above_direction =
  negative`). Demo 3.1/3.2/3.3 PCD and tracking visual modes, plus Demo 3.2/3.3
  headless captures, enable table-Z removal by default at 0 mm signed clearance;
  `--disable-table-z-filter` keeps an unfiltered ablation path. Headless captures
  store `camera_to_world_c2w`, `table_z_above_direction`, plus
  `world_z_stats.jsonl` for offline RGB overlay sweeps.
- `services/ffs_remote/`: single-camera remote FFS depth request/response
  protocol and server/client utilities.
- `demo_v4/`: isolated FuturePhysTwin-compatible chunk writer for Demo 3.2
  realtime/headless artifacts.
- `demo_v5/`: isolated continuous online optimization orchestration that
  reuses the Demo v4-compatible chunk contract and starts repo-local
  `realtime_phystwin` as a single continuous online consumer.

## Dependency Flow

`cameras_calibrate.py`, `record_data.py`, and `record_data_realtime_align.py`
import `CameraSystem`. `CameraSystem` owns RealSense startup, serial ordering,
stream configuration, and shared RGB color controls.

`data_process/record_data_align.py` remains the terminal product stage. It:

- stays cheap to import so `--help` works without CUDA or model imports
- keeps `realsense` as the default backend
- lazily imports `data_process/depth_backends/*` only when FFS output is requested
- writes `metadata.json` for compatibility and `metadata_ext.json` for QQTT-only extensions
- keeps canonical `depth/` as the compatibility depth output

Demo and proxy entrypoints depend on the core camera/runtime pieces in one
direction only. They may consume `CameraSystem`, calibration loaders, FFS
geometry/runners, and aligned-case visualization helpers. Core recording and
alignment entrypoints must not import demo, proxy, or experiment packages.

## Camera Identity Contract

Aligned cases use logical camera indices only as positions in
`metadata["serial_numbers"]`. `color/<camera_idx>/`, `depth/<camera_idx>/`,
intrinsics, depth scales, and other per-camera lists must all have the same
length and order as that serial list.

New calibrations write `calibrate_metadata.json` next to `calibrate.pkl`. The
sidecar records the calibration transform serial order, ChArUco board profile,
world-frame convention, RealSense color distortion coefficients, and
reprojection diagnostics. Recording entry points prefer that sidecar over
inferred connected-device order.

`table_calibrate.pkl` is a separate single-camera table-world artifact. It uses
the same list-of-4x4 `camera_to_world_c2w` physical shape as `calibrate.pkl`,
but its metadata declares `world_frame_kind = table_world_z0` and compatibility
contract `qqtt_table_calibrate_c2w_v1`. Demo 3.1, Demo 3.2, and Demo 3.3
default to the repo-root `table_calibrate.pkl` and fail fast if it is missing or
invalid; operators may override it with `--table-calibrate`. Demo table-world
filtering uses signed clearance from the table plane because the calibrated
"above table" direction can be either positive or negative Z. Recording,
alignment, and other commands still receive table calibration explicitly via
`--table-calibrate`.

Swapping USB ports is safe because capture uses device serial numbers rather
than port order. Physically moving the camera relative to the scene requires a
new calibration.

## Visualization Boundary

Comparison visualization is an in-scope diagnostic utility built on aligned
cases. It may read native and FFS depth, calibration files, masks, and generated
diagnostic artifacts. It must not become a formal recording/alignment runtime
dependency.

Experiment-only workflow implementations live under
`data_process/visualization/experiments/` and CLIs under
`scripts/harness/experiments/`. `scripts/harness/guards/check_experiment_boundaries.py`
guards that formal runtime code does not import those experiment packages.

## Removed Historical Surface

This branch no longer carries old three-camera demo folders, historical
dual-GPU demo runtimes, tracker backend registries, staged multi-view demo
protocols, or batch-size-specific demo helper scripts. The Demo v5 warmup/online
optimization GPU split is an explicit isolated single-camera diagnostic
carveout. Use `main` for the protected multi-camera baseline.
