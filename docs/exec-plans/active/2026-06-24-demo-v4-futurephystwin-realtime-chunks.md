# Demo v4 FuturePhysTwin Realtime Chunks Execution Record

## Goal

Build and verify a single-camera realtime preprocessing path that turns Demo
3.2 fake-live/live camera output into FuturePhysTwin-consumable chunk case
roots. The default test path is fake realtime camera input at 5 FPS with native
RealSense depth, async SAM3D shape-prior warmup, and one 25-frame case per
chunk.

The acceptance target is not only schema generation. The generated chunks must
load in `/home/xinjie/FuturePhysTwin/qqtt/data/real_data.py` and run through
FuturePhysTwin 0-order CMA and 1-order `train_warp.py`.

## Implemented Architecture

Demo v4 reuses Demo 3.2 headless strict artifacts instead of duplicating the
realtime stack:

```text
Demo 3.2 fake-live/live
  -> color-aligned depth
  -> EdgeTAM processed masks
  -> TAPNext++ strict trajectories
  -> async SAM3D shape-prior snapshot/worker
  -> headless capture rows
  -> Demo v4 streaming chunk finalizer
  -> FuturePhysTwin case root per chunk
```

Each chunk case contains:

```text
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

The implementation is intentionally single-camera. It does not fake three
cameras, does not call the old `data_process_sam3d/align.py`, and keeps SAM3D
shape prior isolated from tracking identities and live observation PCD.

## Compatibility Decisions

- `demo_v4/realtime_futurephystwin_chunks.py` defaults to fake-live, 5 FPS,
  native RealSense, 5-second chunks, 7 chunks, and shape-prior warmup enabled.
  It starts Demo 3.2 with `CUDA_VISIBLE_DEVICES=1` by default and passes
  logical `--device cuda` / `--tracker-device cuda`, so local SAM3D workers can
  occupy physical GPU0 without starving EdgeTAM or exposing physical GPU
  indices to SAM3.1/EdgeTAM internals.
- The validation selector uses the second-last and fifth-last chunk. With seven
  chunks this validates chunk `0006` and chunk `0003`.
- `demo_v4/headless_chunk_bridge.py` tails Demo 3.2 headless rows while the
  subprocess is running, so chunk output is streamed from the fake realtime
  camera timeline. Before track processing it intersects masks with valid depth
  and applies the same 1 cm / 40-neighbor 3D radius-outlier refinement used by
  `data_process_sam3d/data_process_mask.py`.
- `demo_v4/futurephystwin_chunk_writer.py` writes the exact final-data and case
  root files expected by FuturePhysTwin.
- `qqtt/demo/phystwin_strict_product.py` keeps first-frame semantic identity,
  applies per-frame semantic masks, performs the same neighbor motion filtering
  constants, and selects 30 controller points by FPS.
- `qqtt/demo/single_view_shape_prior_sampling.py` ports the relevant legacy
  `data_process_sam3d/data_process_sample.py` sampling semantics for single
  view: surface/interior sampling, NN distance filtering, observed-object
  priority, and 5 mm voxel dedupe.

## Illegal Simplification Found And Removed

During optimization validation, the original Demo v4 bridge allowed semantic
object/controller masks to include pixels with invalid or zero depth. Those
pixels became `(0,0,0)` placeholders in PCD sampling and could enter
`final_data.pkl`. A final audit also found that the bridge had not yet mirrored
`data_process_sam3d/data_process_mask.py` radius-outlier refinement. Both were
illegal simplifications relative to the offline SAM3D data-processing route.

Fixes:

- `headless_chunk_bridge.py` now intersects every processed mask with
  `finite(depth) & depth > 0`.
- `headless_chunk_bridge.py` now applies the 3D radius-outlier mask refinement
  before track processing; Demo v4 CLI keeps this enabled by default.
- `phystwin_strict_product.py` rejects sampled track points that are nonfinite
  or zero-norm.
- `normalize_processed_mask_frame` resolves object/controller overlap with
  controller priority.
- `futurephystwin_chunk_writer.py` validates that first-frame object and
  controller final-data points contain no zero-depth placeholders and no exact
  object/controller overlap.

After this fix, FuturePhysTwin CMA and train no longer produce the earlier
all-NaN failure.

## Verified Full Fake Realtime Run

Command:

```bash
conda run -n phystwin-max --no-capture-output \
  python services/shape_prior_remote/server.py \
  --bind tcp://127.0.0.1:7100 \
  --sam3d-root /home/xinjie/external/sam-3d-objects \
  --device cuda:0 \
  --debug
```

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --futurephystwin-base-path result/demo_v4/full_fake_realtime_native_radius_20260624/cases \
  --case-prefix demo_v4_native_radius \
  --max-chunks 7 \
  --capture-extra-seconds 220 \
  --shape-prior-chunk-wait-timeout-s 420
```

Result:

```text
mode=full-fake-realtime-camera
demo32_cuda_visible_devices=1
demo32_return_code=-15
demo32_stop_reason=max_chunks_reached
chunk_frame_count=25
chunk_count=7
validation_chunk_cases=[
  demo_v4_native_radius_chunk_0006,
  demo_v4_native_radius_chunk_0003
]
```

Demo 3.2 capture metadata:

```text
input_source=fake-live
depth_backend=native-realsense
depth_source_internal=realsense
replay_fps=5.0
tracking_product_backend=phystwin-strict-tracking
tracker_query_count=5000
shape_prior_enabled=true
shape_prior_status=ready
table_z_above_direction=negative
table_z_filter_threshold_m=0.0
```

Shape-prior profile:

```text
image_upscale_ms=15533.8
sam3d_inference_ms=8485.0
single_view_alignment_ms=2.4
sampling_ms=5040.7
shape_prior_total_ms=29063.3
time_to_shape_prior_ready_ms=46986.5
shape_prior_ready_seq=0
```

## FuturePhysTwin Acceptance Results

The validated chunks were generated from the full fake-realtime capture after
the final radius-outlier/default GPU-isolation audit:

```text
result/demo_v4/full_fake_realtime_native_radius_20260624/cases/demo_v4_native_radius_chunk_0006
result/demo_v4/full_fake_realtime_native_radius_20260624/cases/demo_v4_native_radius_chunk_0003
```

FuturePhysTwin loader accepted both chunks:

```text
chunk_0006: frames=25, object=(25,2135,3), controller=(25,30,3),
            surface=(465,3), interior=(610,3),
            first-frame zero object/controller=0/0
chunk_0003: frames=25, object=(25,2149,3), controller=(25,30,3),
            surface=(465,3), interior=(610,3),
            first-frame zero object/controller=0/0
```

Optimization results:

```text
chunk_0006 optimize_cma:
  optimal_params.pkl written
  Optimal error: 8.20782170194434e-05

chunk_0006 train_warp:
  completed iteration 199/199
  iteration 199 loss: 2.656478261542361e-05
  best_160.pth and iter_199.pth written

chunk_0003 optimize_cma:
  optimal_params.pkl written
  Optimal error: 0.00010896797653003887

chunk_0003 train_warp:
  completed iteration 199/199
  iteration 199 loss: 2.4058800818238524e-05
  best_199.pth and iter_199.pth written
```

The exact commands and external log paths are recorded in
`docs/generated/demo_v4_futurephystwin_validation_20260624.md`.

## Remaining Boundaries

- Real camera hardware validation still needs to be run manually on the D455.
- SAM3D/FuturePhysTwin dependencies and weights remain external.
- Optional gsplat post-optimization inside the SAM3D worker failed in the local
  environment because `phystwin-max` has no `nvcc`; the worker still returned
  shape-prior points and Demo v4 chunks used them.
- A FuturePhysTwin visualization/logging compatibility patch was made outside
  this repo in `/home/xinjie/FuturePhysTwin/qqtt/engine/trainer_warp.py` so
  missing H.264 videos do not abort `train_warp.py`.
- Local full-pipeline runs require GPU isolation. If a SAM3D worker is resident
  on GPU0, Demo 3.2 and FuturePhysTwin should run with
  `CUDA_VISIBLE_DEVICES=1` or an equivalent separate device.

## Validation Commands

Focused repo validation:

```bash
conda run -n demo_2_max --no-capture-output \
  python -m unittest tests.test_demo_v4_futurephystwin_chunks
```

```bash
conda run -n demo_2_max --no-capture-output \
  python -m unittest tests.test_demo32_shape_prior_warmup
```

Integrated validation:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/guards/check_scope.py
```

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
```
