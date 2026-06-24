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
demo32_headless_prepared_only=true
write_final_pcd=false
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
  pcd/<frame>.npz        # only when --write-final-pcd is enabled
  tracking/0.npz
  cotracker/0.npz
  manifest.json
  READY
```

Consumers must treat `READY` as the publish marker and ignore directories
without it. Demo v4 writes and validates each chunk under `<base>/.publishing/`,
creates `READY` last, and then atomically renames the staged directory to
`<base>/<case>/`.

The default realtime path is optimized for FuturePhysTwin `final_data.pkl`
cadence and skips dense per-frame `pcd/` files. The final-data, mask, RGB,
tracking/cotracker, calibration, metadata, split, manifest, and READY contract
remain complete. Use `--write-final-pcd` when a diagnostic/export consumer
needs dense per-frame point-cloud files in each published chunk.

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
CUDA_VISIBLE_DEVICES=1 \
conda run -n phystwin-max --no-capture-output \
  python services/shape_prior_remote/server.py \
  --bind tcp://127.0.0.1:7103 \
  --sam3d-root /home/xinjie/external/sam-3d-objects \
  --device cuda:0 \
  --preload-models \
  --debug
```

`--preload-models` keeps the x4 upscaler and SAM3D model resident before Demo
v4 starts. `--warmup-models` runs an additional dummy SAM3D pass, but it needs
more VRAM; on the 2026-06-24 GPU1 test it OOMed during `decode_slat`, while
preload-only was stable.

Then run Demo v4:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --realtime-gpu-mode single \
  --warmup-gpu-mode dual \
  --demo32-cuda-visible-devices 0 \
  --shape-prior-endpoint tcp://127.0.0.1:7103 \
  --demo32-source-replay-fps 5.2 \
  --futurephystwin-base-path result/demo_v4/warmup_fast_sampling_dual_rt_single_lossless52_20260624/cases \
  --case-prefix demo_v4_warmup_fast_sampling_dual_rt_single_lossless52 \
  --capture-extra-seconds 140 \
  --shape-prior-timeout-ms 240000 \
  --shape-prior-chunk-wait-timeout-s 240 \
  --demo32-lossless-max-backlog-seconds 45
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

`--replay-fps` remains the PhysTwin logical FPS used for chunk window math and
published `metadata.json`. `--demo32-source-replay-fps` is separate: it controls
Demo 3.2 fake-live/lossless wall-clock pacing and, when non-default, Demo v4
forwards the same value as Demo 3.2 `--lossless-input-fps`. The 2026-06-24
cadence proof used `--replay-fps 5.0`, 25-frame chunks, and
`--demo32-source-replay-fps 5.2` to give wall-clock headroom while preserving
published `fps=5`.

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

## Verified 2026-06-24 Optimization Run

The current passing warmup run is:

```text
result/demo_v4/warmup_fast_sampling_dual_rt_single_lossless52_20260624/cases

realtime_gpu_mode=single
warmup_gpu_mode=dual
demo32_cuda_visible_devices=0
shape_prior_worker=GPU1 preload-only remote worker
demo32_source_replay_fps=5.2
demo32_lossless_input_fps=5.2
write_final_pcd=false
chunk_count=7
first_shape_prior_ready_chunk_wall_s=43.942
shape_prior_total_ms=27154.2
sampling_ms=64.1
steady_state_publish_interval_max_s=1.747
max_backlog_chunks=4, drained to 0 by chunk_0007
surface/interior target counts=700/1000
```

The steady-state no-warmup cadence proof is:

```text
result/demo_v4/realtime_final_data_only_lossless52_20260624/cases

external_shape_prior_points=true
chunk_count=7
steady_publish_intervals_s=[4.706, 4.779, 4.820, 4.766, 4.760, 4.853]
steady_state_publish_interval_max_s=4.853
max_backlog_chunks=0
materialize_latency_s=[1.714, 1.603, 1.582, 1.607, 1.557, 1.516, 1.560]
```

Both validation chunks in both runs passed `validate_futurephystwin_case` with
`require_ready=True`, 25 frames, 30 controller points, finite object/controller
points, and 700/1000 shape-prior target counts. The warmup run still records
`single_view_shape_prior_sampling_backend=sam3d-single-view`, source
`data_process_sam3d/data_process_sample.py`, and keeps the same final-data
fields consumed by FuturePhysTwin.

Single-GPU cold same-card SAM3D remains above target and can break realtime
backlog: the 2026-06-24 cold run produced no chunks and hit lossless backlog
while the worker reported about 78.8 seconds for shape prior. The supported
sub-60s path is therefore dual warmup plus single realtime, with the long-lived
remote worker preloaded before the Demo v4 run.

## Known External Environment Notes

- SAM3D/FuturePhysTwin weights and repos remain external.
- The worker can finish SAM3D even if optional gsplat post-optimization fails
  because the local `phystwin-max` environment has no `nvcc`.
- A FuturePhysTwin visualization-only W&B logging compatibility patch was made
  outside this repo in `/home/xinjie/FuturePhysTwin/qqtt/engine/trainer_warp.py`
  so missing H.264 videos do not abort `train_warp.py`. This does not change
  optimization math or final-data loading.
