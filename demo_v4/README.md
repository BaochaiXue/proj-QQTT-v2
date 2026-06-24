# Demo v4 FuturePhysTwin Chunks

Demo v4 is an isolated single-camera realtime preprocessing path for
FuturePhysTwin acceptance testing. It launches the sanctioned Demo 3.2
headless realtime runtime by default, then streams complete FuturePhysTwin case
roots one chunk at a time.

This is a diagnostic carveout, not the formal aligned-case data product. The
canonical recording/alignment product still stops at `data_process/`.

## Default Realtime Contract

The default path is full fake realtime camera input:

```text
Demo 3.2 fake-live camera
  -> native RealSense color-aligned depth
  -> EdgeTAM masks
  -> TAPNext++ strict same-seq tracks
  -> async SAM3D shape-prior warmup
  -> Demo v4 5 FPS / 25-frame FuturePhysTwin chunks
```

Defaults:

```text
input_source=fake-live
depth_backend=native-realsense
replay_fps=5
chunk_seconds=5
chunk_frame_count=25
max_chunks=7
demo32_cuda_visible_devices=0
demo32_device=cuda
demo32_tracker_device=cuda
shape_prior_warmup=true
shape_prior_execution=remote-worker
shape_prior_start_policy=async-after-first-mask-depth-pair
```

Demo v4 writes each complete case under `--futurephystwin-base-path`:

```text
<base>/<case>/
  final_data.pkl
  track_process_data.pkl
  calibrate.pkl
  metadata.json
  split.json
  color/0/<frame>.png
  mask/processed_masks.pkl
  pcd/<frame>.npz
  tracking/0.npz
  cotracker/0.npz
  manifest.json
```

The `final_data.pkl` schema follows
`/home/xinjie/FuturePhysTwin/qqtt/data/real_data.py`:

```text
object_points
object_colors
object_visibilities
object_motions_valid
controller_points
controller_mask
surface_points
interior_points
```

FuturePhysTwin loads structure points as:

```text
object_points[0] + surface_points + interior_points
```

## Data Process SAM3D Compatibility

Demo v4 intentionally mirrors the parts of
`/home/xinjie/FuturePhysTwin/data_process_sam3d` that matter for optimization:

- first-frame object/controller semantic labels define the track classes
- per-frame masks gate semantic visibility
- masks are intersected with valid depth before track processing
- masks then run the same 3D radius-outlier refinement used by
  `data_process_sam3d/data_process_mask.py` by default: 1 cm radius and 40
  neighbors
- object/controller overlap is resolved with controller priority
- motion filtering uses the same 1 cm neighborhood, 5-neighbor minimum, and 5
  mm motion-similarity threshold
- controller points are selected by FPS down to 30 points
- object points are sampled on a 5 mm grid with observed object points taking
  priority over shape-prior points
- shape-prior surface/interior samples are kept as separate final-data fields
- z-down/table frame is preserved; the ground policy is not clamped by default

The important realtime-specific difference is scheduling: `data_process_sam3d`
is offline and can block between stages, while Demo v4 streams chunks from the
fake-live camera timeline. SAM3D shape prior is asynchronous and does not change
EdgeTAM masks, TAPNext++ queries/tracks, current observation PCD, or strict
tracking identities.

## Shape Prior Worker

Start the SAM3D worker separately in the FuturePhysTwin/SAM3D environment:

```bash
conda run -n phystwin-max --no-capture-output \
  python services/shape_prior_remote/server.py \
  --bind tcp://127.0.0.1:7100 \
  --sam3d-root /home/xinjie/external/sam-3d-objects \
  --device cuda:0 \
  --preload-models \
  --warmup-models \
  --debug
```

Then run Demo v4:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --futurephystwin-base-path result/demo_v4/full_fake_realtime_native_full_sam3d_20260624/cases \
  --case-prefix demo_v4_native_full_sam3d \
  --max-chunks 7 \
  --capture-extra-seconds 220 \
  --shape-prior-chunk-wait-timeout-s 420
```

The process terminates the Demo 3.2 subprocess after `max_chunks` are written;
`demo32_return_code=-15` with `demo32_stop_reason=max_chunks_reached` is the
expected controlled stop.

## Chunk Cadence Telemetry

The 25-frame chunk setting defines the source window, not by itself the
wall-clock publish cadence. Every chunk manifest records:

```text
source_window_start_s
source_window_end_s
materialize_start_wall_s
materialize_end_wall_s
publish_wall_s
materialize_latency_ms
publish_lag_ms
backlog_chunks
```

`source_window_*` is nominal source time from row offsets and FPS. The
`*_wall_s` fields are relative to Demo v4 chunk streaming startup. Realtime
cadence is acceptable only when steady-state `publish_wall_s` intervals are no
larger than the chunk source window and `backlog_chunks` does not grow after
startup.

## Validation Chunks

The validation selector uses the second-last and fifth-last chunks. With the
default seven chunks this means:

```text
demo_v4_native_full_sam3d_chunk_0006
demo_v4_native_full_sam3d_chunk_0003
```

This avoids proving the path only on an early chunk where the controller may not
have moved enough.

Demo v4 now defaults to one visible GPU for the native RealSense realtime path:
`CUDA_VISIBLE_DEVICES=0`, logical `--device cuda`, and logical
`--tracker-device cuda`. This keeps EdgeTAM, TAPNext++, and the native depth
path in one CUDA namespace. A dual-GPU isolation run is still possible by
explicitly passing `--demo32-cuda-visible-devices 1` and running the SAM3D
worker elsewhere, but that is no longer the default validation contract.

## Verified 2026-06-24 Run

The latest full fake realtime native-RealSense stream produced seven chunks at
25 frames each:

```text
result/demo_v4/full_fake_realtime_native_full_sam3d_20260624/cases
```

The Demo 3.2 capture metadata recorded:

```text
input_source=fake-live
depth_backend=native-realsense
depth_source_internal=realsense
replay_fps=5.0
tracking_product_backend=phystwin-strict-tracking
tracker_query_count=5000
shape_prior_status=ready
table_z_above_direction=negative
table_z_filter_threshold_m=0.0
```

The shape-prior profile recorded image upscale, SAM3D inference, single-view
alignment, and sampling. New runs should also inspect
`worker_preload_upscaler_ms`, `worker_preload_sam3d_ms`,
`worker_dummy_warmup_ms`, and `worker_ready_ms` so cold worker startup is not
mixed with warm request latency:

```text
image_upscale_ms=20508.7
sam3d_inference_ms=10881.6
single_view_alignment_ms=2.4
sampling_ms=29954.4
shape_prior_total_ms=78569.7
time_to_shape_prior_ready_ms=96647.7
```

Chunk geometry audits:

```text
chunk_0006: object=(25,2136,3), controller=(25,30,3),
            surface=(700,3), interior=(1000,3),
            finite object/controller/shape-prior points=True,
            first-frame zero object/controller points=0/0

chunk_0003: object=(25,2152,3), controller=(25,30,3),
            surface=(700,3), interior=(1000,3),
            finite object/controller/shape-prior points=True,
            first-frame zero object/controller points=0/0
```

This run includes the `data_process_sam3d/data_process_mask.py` radius-outlier
mask refinement before chunk finalization. Tiny synthetic tests can disable it
with `--no-mask-radius-outlier-filter`; product runs keep it enabled.
Shape-prior sampling records `single_view_shape_prior_sampling_backend=
sam3d-single-view`, source `data_process_sam3d/data_process_sample.py`, and the
same full target counts used by the offline SAM3D route: 700 surface points and
1000 interior points.

Both validation chunks loaded in FuturePhysTwin and completed 0-order CMA plus
1-order `train_warp.py`. Exact commands and outcomes are recorded in
`docs/generated/demo_v4_futurephystwin_validation_20260624.md`.

## Known External Environment Notes

- SAM3D/FuturePhysTwin weights and repos remain external.
- The worker can finish SAM3D even if optional gsplat post-optimization fails
  because the local `phystwin-max` environment has no `nvcc`.
- A FuturePhysTwin visualization-only W&B logging compatibility patch was made
  outside this repo in `/home/xinjie/FuturePhysTwin/qqtt/engine/trainer_warp.py`
  so missing H.264 videos do not abort `train_warp.py`. This does not change
  optimization math or final-data loading.
