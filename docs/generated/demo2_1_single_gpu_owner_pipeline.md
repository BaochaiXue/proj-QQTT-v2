# Demo 2.1 Single GPU-Owner Pipeline

Status: implementation ready; live performance matrix pending.

Purpose:

```text
Reduce FFS/EdgeTAM GPU worker contention and same-group partial joins by
making one worker own all heavy GPU inference for each temporal-coherent group.
```

Pipeline:

```text
CaptureGroup(group_id)
  -> GPUOwnerPipelineWorker
       FFS cam0
       FFS cam1
       FFS cam2
       EdgeTAM cam0
       EdgeTAM cam1
       EdgeTAM cam2
       publish CompleteInferenceGroup(depths + masks)
  -> Fusion / filter / render
```

Quality contract unchanged:

```text
live SAM3.1 first-frame init
no saved-mask fallback
timestamp-nearest temporal grouping
FFS 20-30-48 / valid_iters=4 / 480x864 / builderOpt5
HF EdgeTAM vision-reduce-overhead
object enhanced-PT
controller pt-filter
object/controller separated fusion
```

New CLI:

```text
--gpu-pipeline-mode separate-workers|single-owner
--single-owner-order ffs-then-edgetam|edgetam-then-ffs
--static-device-buffers
--preallocate-pcd-buffers
```

`interleaved` is reserved and fails fast in the first implementation slice; use
`ffs-then-edgetam` or `edgetam-then-ffs` for current profiling.

Default behavior remains:

```text
gpu_pipeline_mode=separate-workers
```

New preset:

```text
visual-5fps-single-owner:
  profile=848x480
  fps=30
  fusion_target_fps=5
  render_mode=pointcloud
  depth_source=ffs
  init_mode=sam31-first-frame
  gpu_pipeline_mode=single-owner
  single_owner_order=ffs-then-edgetam
```

Recommended first live run:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps-single-owner \
  --track-mode object-only \
  --init-mode sam31-first-frame \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-h2d \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_1_visual5fps_single_owner_no_pin_object_only_120s.json
```

Compare against:

```text
separate-workers gate=2
separate-workers no-gate
single-owner no-pin
single-owner pin-all
single-owner pin-all + static-device-buffers flags
```

Interpretation:

```text
If complete group ratio improves and fusion timeouts drop, separate workers
were a major source of partial group loss.

If single-owner is slower but p95 is stable, it may still be useful for
professor-safe mode.

If H2D metrics do not improve with pin-memory modes, the bottleneck is compute
or scheduling rather than transfer.
```

## Towel-controller single-owner experiment

Status on 2026-05-06: `cloth` was not segmentable in cam0, but the same two
physical cloth objects were correctly segmented as `towel`. `towel` is a
temporary experiment prompt only. The default controller prompt remains `hand`.

Dry-run contract passed for the temporary controller experiment:

```text
track_mode=controller-object
controller_prompt=towel
object_prompt=stuffed animal
controller obj_id=1 -> pt-filter
object obj_id=2 -> enhanced-pt
depth_source=ffs
gpu_pipeline_mode=single-owner
single_owner_order=ffs-then-edgetam
init_mode=sam31-first-frame
fallback_allowed=false
```

Initial camera attach issue:

```text
AssertionError: Only 0 cameras are connected.
```

Windows `usbipd list` showed three D455 devices as shared (`1-3`, `1-4`,
`2-19`), but the `usbipd` service was not running, so WSL could not attach or
enumerate them. After starting `usbipd` and attaching all three devices, WSL
enumerated the three D455 cameras.

`cloth` live sanity result:

```text
capture groups emitted: 124
capture skew median/p95/max: 8.87 / 26.81 / 32.68 ms
SAM3.1 cam2: object_px=11241, controller_px=16956
SAM3.1 cam0: failed to produce a mask for label "cloth"
complete fused groups: 0
drop reason: missing_mask_cam0
```

This is a valid no-fallback failure for the explicit cloth-controller
experiment.

`towel` live sanity result:

```text
cam0 object_px=19074, controller_px=26153
cam1 object_px=18203, controller_px=17556
cam2 object_px=11255, controller_px=16938
complete fused groups: 12 in the 60s sanity run
```

After-warmup 120s benchmark results:

| Mode | Render FPS | Fusion FPS | Complete / Total | Timeout | Controller pts | Object pts | FFS cycle p95 | EdgeTAM p95 | Filter p95 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| separate-workers gate2 towel-controller | 0.51 | 0.51 | 38 / 367 | 170 | 19486 / 19531.0 | 11488 / 11590.5 | 561.8 | ~178 | 32.9 object / 19.1 controller | too many partial-group timeouts |
| single-owner no-pin towel-controller | 3.85 | 3.85 | 315 / 367 | 0 | 19482 / 19535.0 | 11332 / 11593.0 | 106.7 | ~70 | 31.2 object / 18.3 controller | best current candidate |
| single-owner pin-ffs towel-controller | 3.59 | 3.59 | 299 / 383 | 2 | 19487 / 19538.0 | 11415 / 11607.0 | 114.4 | ~74 | 34.1 object / 18.6 controller | pinned FFS staging did not help |
| single-owner edge-first towel-controller | 3.74 | 3.74 | 313 / 360 | 1 | 19483 / 19535.0 | 11320 / 11597.0 | 74.2 | cam0 ~102 / cam1-2 ~65 | 33.5 object / 18.5 controller | stable but slower than ffs-then-edgetam |

Conclusion:

```text
single-owner removes most same-group join loss and is the best current
towel-controller path. Pinned FFS staging should stay an ablation flag, not the
default. The preferred order remains ffs-then-edgetam.
```
