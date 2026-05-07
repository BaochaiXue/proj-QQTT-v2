# Demo 2.1 Three EdgeTAM GPU Probe

Question: how much GPU does three compiled HF EdgeTAM workers use when tracking both controller and object?

This probe used the Demo 2.1 live path with:

- Three D455 cameras in WSL
- `track_mode=controller-object`
- `controller_prompt=towel`
- `object_prompt=stuffed animal`
- live `sam31-first-frame` initialization
- HF EdgeTAMVideo streaming
- `compile_mode=vision-reduce-overhead`
- `dtype=bfloat16`
- `depth_source=none`
- `render_mode=none`
- `fusion_target_fps=15`

The goal was to isolate three EdgeTAM workers, not FFS or Open3D.

## Commands

Original true-concurrent attempt:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --depth-source none \
  --track-mode controller-object \
  --controller-prompt "towel" \
  --object-prompt "stuffed animal" \
  --render-mode none \
  --gpu-pipeline-mode separate-workers \
  --gpu-gate-mode off \
  --fusion-target-fps 15 \
  --duration-s 90 \
  --debug \
  --profile-pipeline \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 30 \
  --profile-json-output docs/generated/demo2_1_three_edgetam_gpu_probe/three_edgetam_controller_object_towel_stuffed_gateoff_target15_90s.json
```

Safe serialized baseline:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --depth-source none \
  --track-mode controller-object \
  --controller-prompt "towel" \
  --object-prompt "stuffed animal" \
  --render-mode none \
  --gpu-pipeline-mode separate-workers \
  --gpu-gate-mode limited \
  --gpu-gate-max-concurrent 1 \
  --fusion-target-fps 15 \
  --duration-s 90 \
  --debug \
  --profile-pipeline \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 30 \
  --profile-json-output docs/generated/demo2_1_three_edgetam_gpu_probe/three_edgetam_controller_object_towel_stuffed_gate1_target15_90s.json
```

Gate-off true parallel after adding a compiled vision-encoder output clone
wrapper:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --depth-source none \
  --track-mode controller-object \
  --controller-prompt "towel" \
  --object-prompt "stuffed animal" \
  --render-mode none \
  --gpu-pipeline-mode separate-workers \
  --gpu-gate-mode off \
  --fusion-target-fps 15 \
  --duration-s 90 \
  --debug \
  --profile-pipeline \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 30 \
  --profile-json-output docs/generated/demo2_1_three_edgetam_gpu_probe/three_edgetam_controller_object_towel_stuffed_gateoff_clonewrap_target15_90s.json
```

GPU was sampled with:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1
```

## Results

| Mode | Result | GPU util median / p90 / p95 / max | EdgeTAM model ms median cam0 / cam1 / cam2 | EdgeTAM gate wait ms median cam0 / cam1 / cam2 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `gpu_gate=off` | failed | not a valid steady run | `101.5 / 0.0 / 95.2` | `0 / 0 / 0` | crashed before all three cameras reached steady state |
| `gpu_gate=max_concurrent=2` | failed | not a valid steady run | `0.0 / 0.0 / 50.8` | `0 / 0 / 0` | same CUDAGraph overwrite failure |
| `gpu_gate=max_concurrent=1` | completed 90s | `28 / 37 / 39 / 47 %` | `41.2 / 40.0 / 40.1 ms` | `69.3 / 73.7 / 69.4 ms` | stable serialized baseline |
| `gpu_gate=off + clone_wrap` | completed 90s | `24 / 26 / 26 / 28 %` | `159.6 / 162.8 / 161.7 ms` | `0 / 0 / 0` | true three-worker parallel run; no gate wait, but each EdgeTAM forward is much slower |

The original true-concurrent runs failed with:

```text
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
...
To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
```

## Interpretation

Three compiled HF EdgeTAM workers tracking both controller and object cannot
be run fully concurrently by simply turning the GPU gate off. The original
compiled/CUDAGraph path crashes before producing a valid steady-state GPU
utilization measurement.

Adding a small output-clone wrapper around the compiled `vision_encoder`
prevents the CUDAGraph output overwrite and allows the gate-off run to finish.
That makes the comparison measurable:

```text
gate=1 serialized:
  per-camera model median: ~40 ms
  gate wait median:        ~70 ms per camera
  GPU util median/p95:     28% / 39%

gate=off true parallel + clone_wrap:
  per-camera model median: ~160 ms
  gate wait median:        0 ms
  GPU util median/p95:     24% / 26%
```

So gate-off removes scheduler wait but increases each EdgeTAM forward by about
4x. For this controller-object/no-FFS isolation test, true parallel EdgeTAM
does not eat the GPU efficiently and does not improve per-camera inference
latency.

The stable serialized baseline shows:

```text
three controller-object EdgeTAM workers, serialized:
  GPU util median: 28%
  GPU util p90:    37%
  GPU util p95:    39%
  GPU util max:    47%
  per-camera model median: ~40 ms
  per-camera gate wait median: ~70 ms
```

The low utilization percentage does not mean the path is fast. It means the GPU is being used in bursts while the three workers queue behind the gate and while CPU/SAM/capture scheduling runs around it. The practical bottleneck is latency and scheduling, not sustained 100% GPU occupancy.

For Demo 2.1, this confirms:

- `gpu_gate=off` is now the default contract, but true three-worker compiled
  EdgeTAM needs the CUDAGraph output clone wrapper to avoid overwrite errors.
- With that wrapper, gate-off is stable but slower per camera than serialized
  EdgeTAM in this isolation test.
- `gpu_gate=2` without the clone wrapper was not safe in the tested
  controller-object/no-FFS stress configuration.
- `gpu_gate=1` is stable but serializes the three workers and yields roughly 7-9 EdgeTAM FPS per camera after initialization.
- The single-owner pipeline remains the safer architecture because it avoids partial-group join failures and CUDAGraph overlap hazards.
