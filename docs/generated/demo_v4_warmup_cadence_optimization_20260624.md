# Demo v4 Warmup And Cadence Optimization - 2026-06-24

## Goal

Keep FuturePhysTwin-quality Demo v4 products complete while reducing:

- warmup to first shape-prior READY chunk below 60 seconds;
- fake/live camera to `final_data.pkl` cadence to at least 5 FPS.

## Code Changes Under Test

- Demo v4 can route realtime and warmup separately:
  `--realtime-gpu-mode` and `--warmup-gpu-mode`.
- Demo v4 can pass fake-live wall-clock headroom with
  `--demo32-source-replay-fps`; output `metadata.json` still records
  PhysTwin logical `fps=5`.
- Demo 3.2 lossless cadence is configurable with `--lossless-input-fps`.
- Demo 3.2 prepared-only headless mode writes only the prepared PhysTwin frames
  needed by Demo v4.
- Demo v4 defaults to final-data cadence mode and skips dense per-frame `pcd/`;
  use `--write-final-pcd` for diagnostic/export PCD files.
- Single-view SAM3D shape-prior interior sampling now uses deterministic
  voxel/raycast candidates before falling back to random volume sampling.

## Failed / Boundary Experiments

### Single-GPU Cold Same-Card SAM3D

GPU assignment:

```text
physical GPU1: Demo 3.2 + remote SAM3D worker
```

Outcome:

```text
chunk_count=0
failure=LosslessPipelineError
stage=pair-output
queue_len=16
max=15
worker_shape_prior_total_ms=78783.6
```

Conclusion: cold single-GPU same-card warmup cannot honestly be claimed as
sub-60s, and it can stall the strict realtime pipeline before the first chunk.

### Preload + Dummy Worker Warmup On GPU1

Command included `--preload-models --warmup-models`.

Outcome:

```text
failure=CUDA out of memory
phase=SAM3D dummy warmup decode_slat
```

Conclusion: GPU1 did not have enough VRAM margin for the optional dummy
worker warmup. Preload-only is the stable worker mode for this machine.

## Passing Realtime Cadence Proof

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --realtime-gpu-mode single \
  --demo32-cuda-visible-devices 1 \
  --demo32-source-replay-fps 5.2 \
  --no-shape-prior-warmup \
  --surface-points-npy result/demo_v4/single_gpu_shape_bootstrap_20260624/surface_points.npy \
  --interior-points-npy result/demo_v4/single_gpu_shape_bootstrap_20260624/interior_points.npy \
  --futurephystwin-base-path result/demo_v4/realtime_final_data_only_lossless52_20260624/cases \
  --case-prefix demo_v4_realtime_final_data_only_lossless52 \
  --max-chunks 7 \
  --capture-extra-seconds 60 \
  --shape-prior-chunk-wait-timeout-s 120 \
  --demo32-lossless-max-backlog-seconds 30
```

Result:

```text
chunk_count=7
write_final_pcd=false
demo32_source_replay_fps=5.2
demo32_lossless_input_fps=5.2
steady_publish_intervals_s=[4.706, 4.779, 4.820, 4.766, 4.760, 4.853]
steady_state_publish_interval_max_s=4.853
max_backlog_chunks=0
materialize_latency_s=[1.714, 1.603, 1.582, 1.607, 1.557, 1.516, 1.560]
```

Validation chunks:

```text
demo_v4_realtime_final_data_only_lossless52_chunk_0006: valid, 25 frames, 700/1000 shape prior
demo_v4_realtime_final_data_only_lossless52_chunk_0003: valid, 25 frames, 700/1000 shape prior
```

Conclusion: with shape points already available, camera/fake-camera to
`final_data.pkl` maintains 5 FPS wall-clock publish cadence.

## Passing Warmup Proof

Worker command:

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

Worker startup:

```text
worker_ready_ms=21337.1
worker_preloaded_models=true
worker_warmed_models=false
```

Demo v4 command:

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
  --max-chunks 7 \
  --capture-extra-seconds 140 \
  --shape-prior-timeout-ms 240000 \
  --shape-prior-chunk-wait-timeout-s 240 \
  --demo32-lossless-max-backlog-seconds 45
```

Result:

```text
chunk_count=7
realtime_gpu_mode=single
warmup_gpu_mode=dual
demo32_cuda_visible_devices=0
shape_prior_device=cuda:1
demo32_source_replay_fps=5.2
demo32_lossless_input_fps=5.2
first_shape_prior_ready_chunk_wall_s=43.942
shape_prior_total_ms=27154.2
time_to_shape_prior_ready_ms=42462.5
image_upscale_ms=15802.7
sam3d_inference_ms=11284.0
sampling_ms=64.1
shape_prior_surface_candidates=2002
shape_prior_interior_candidates=8772
steady_state_publish_interval_max_s=1.747
max_backlog_chunks=4, drained to 0 by chunk_0007
```

Validation chunks:

```text
demo_v4_warmup_fast_sampling_dual_rt_single_lossless52_chunk_0006: valid, 25 frames, 700/1000 shape prior
demo_v4_warmup_fast_sampling_dual_rt_single_lossless52_chunk_0003: valid, 25 frames, 700/1000 shape prior
```

Conclusion: dual warmup plus single realtime meets the sub-60s first READY
target while preserving FuturePhysTwin final-data contents and shape-prior
target counts.

