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
max_chunks=None
gpu_mode=single
realtime_gpu_mode=single
warmup_gpu_mode=single
demo32_cuda_visible_devices=0
shape_prior_device=cuda:0
demo32_device=cuda
demo32_tracker_device=cuda
shape_prior_warmup=true
shape_prior_execution=remote-worker
shape_prior_start_policy=async-after-first-mask-depth-pair
```

Chunk length is time-first. Operators should normally change
`--chunk-seconds`; Demo v4 derives `chunk_frame_count` as
`round(replay_fps * chunk_seconds)`. The default is therefore 5 seconds at
5 FPS, or 25 frames. `--chunk-frame-count` remains available as an explicit
frame-count override for tests and advanced debugging, but `--chunk-seconds`
and `--replay-fps` must still be positive so manifests keep a meaningful source
time window.

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
  READY
```

Consumers must treat `READY` as the publish marker and ignore directories
without it. Demo v4 writes and validates each chunk under `<base>/.publishing/`,
creates `READY` last, and then atomically renames the staged directory to
`<base>/<case>/`.

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
  --capture-extra-seconds 220 \
  --shape-prior-chunk-wait-timeout-s 420
```

For a short debug run, add `--max-chunks <N>`. Omit `--max-chunks` for the
default full fake-live recording pass.

GPU routing is explicit and split by role:

```bash
# Default: Demo 3.2 realtime and SAM3D warmup both resolve to GPU0.
python demo_v4/realtime_futurephystwin_chunks.py \
  --realtime-gpu-mode single \
  --warmup-gpu-mode single

# Dual warmup + single realtime: run Demo 3.2 with CUDA_VISIBLE_DEVICES=0
# and resolve shape-prior device to cuda:1.
python demo_v4/realtime_futurephystwin_chunks.py \
  --realtime-gpu-mode single \
  --warmup-gpu-mode dual

# Realtime isolation: run Demo 3.2 inside CUDA_VISIBLE_DEVICES=1.
python demo_v4/realtime_futurephystwin_chunks.py \
  --realtime-gpu-mode dual
```

`--gpu-mode` remains a backward-compatible alias for realtime routing.
`--demo32-cuda-visible-devices` and `--shape-prior-device` remain explicit
debug overrides.

When `--max-chunks` is supplied, the process terminates the Demo 3.2 subprocess
after that many chunks are written; `demo32_return_code=-15` with
`demo32_stop_reason=max_chunks_reached` is the expected controlled stop. The
default unlimited fake-live path runs until the recording source finishes.

## Chunk Cadence Telemetry

The 25-frame chunk setting defines the source window, not by itself the
wall-clock publish cadence. Every chunk manifest records:

```text
source_window_start_s
source_window_end_s
window_closed_wall_s
track_finalize_done_wall_s
final_data_written_wall_s
validation_done_wall_s
atomic_rename_done_wall_s
materialize_start_wall_s
materialize_end_wall_s
publish_wall_s
materialize_latency_ms
publish_latency_ms
publish_lag_ms
backlog_chunks
```

`source_window_*` is nominal source time from row offsets and FPS. The
`*_wall_s` fields are relative to Demo v4 chunk streaming startup.
`publish_wall_s` is an alias of `atomic_rename_done_wall_s`, so
`publish_latency_ms` measures the real consumer-visible path from
`window_closed_wall_s` until the validated READY case has been atomically
published. `materialize_latency_ms` remains useful for internal finalizer work
but is not the READY-visible publish metric. Realtime cadence is acceptable only
when steady-state `publish_wall_s` intervals are no larger than the chunk source
window and `backlog_chunks` does not grow after startup.

## Validation Chunks

The validation selector uses the second-last and fifth-last chunks. A short
seven-chunk debug run selects:

```text
demo_v4_native_full_sam3d_chunk_0006
demo_v4_native_full_sam3d_chunk_0003
```

This avoids proving the path only on an early chunk where the controller may not
have moved enough.

Demo v4 supports independent warmup and realtime GPU routing:

- `--realtime-gpu-mode single` is the default. Demo 3.2 receives
  `CUDA_VISIBLE_DEVICES=0`, logical `--device cuda`, and logical
  `--tracker-device cuda`.
- `--realtime-gpu-mode dual` runs Demo 3.2 with `CUDA_VISIBLE_DEVICES=1`.
- `--warmup-gpu-mode single` resolves `--shape-prior-device cuda:0`.
- `--warmup-gpu-mode dual` resolves `--shape-prior-device cuda:1`, supporting
  dual-GPU warmup plus single-GPU realtime camera/fake-camera finalization.
- `--gpu-mode` remains a compatibility alias for realtime routing.

## Verified 2026-06-24 Run

The latest single-GPU full fake realtime native-RealSense stream produced seven
chunks at 25 frames each:

```text
result/demo_v4/full_fake_realtime_native_single_gpu_fast_20260624/cases
```

The Demo v4 summary and Demo 3.2 capture metadata recorded:

```text
input_source=fake-live
gpu_mode=single
demo32_cuda_visible_devices=0
depth_backend=native-realsense
depth_source_internal=realsense
replay_fps=5.0
tracking_product_backend=phystwin-strict-tracking
tracker_query_count=5000
external_shape_prior_points=true
table_z_above_direction=negative
table_z_filter_threshold_m=0.0
```

The source SAM3D snapshot used for this single-GPU run is:

```text
result/demo_v4/single_gpu_shape_bootstrap_20260624

status=ready
alignment_valid=true
ground_z_fraction=0.2759
image_upscale_ms=15779.0
sam3d_model_load_ms=18714.5
sam3d_inference_ms=11040.0
single_view_alignment_ms=2.5
sampling_ms=28675.6
shape_prior_total_ms=79241.4
```

Chunk geometry audits:

```text
steady publish intervals after startup:
  [4.722, 4.969, 4.904, 5.135, 4.874, 5.049] seconds
backlog_chunks:
  [0, 0, 0, 0, 0, 0, 0]
materialize_latency_s:
  [4.298, 4.046, 4.022, 3.888, 4.049, 3.934, 3.990]

chunk_0006: object=(25,2122,3), controller=(25,30,3),
            surface=(700,3), interior=(1000,3),
            finite object/controller/shape-prior points=True,
            first-frame zero object/controller points=0/0

chunk_0003: object=(25,2145,3), controller=(25,30,3),
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
