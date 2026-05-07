# 2026-05-06 Demo 2.1 Cloth-Controller Single-Owner Benchmark

Status: blocked at live SAM3.1 initialization for cam0 cloth controller.

## Goal

Run Demo 2.1 controller-object live experiments with:

```text
controller_prompt=cloth
object_prompt=stuffed animal
init_mode=sam31-first-frame
```

`cloth` is an explicit temporary controller experiment only. The default
controller remains `hand`.

## Non-Goals

- do not change default controller prompt
- do not use saved-mask fallback
- do not use native RealSense depth fallback
- do not downgrade to object-only fallback
- do not change FFS checkpoint / TensorRT contract
- do not change object enhanced-PT or controller pt-filter

## Run Plan

1. dry-run contract check for `controller=cloth`
2. 30s visual-5fps sanity run
3. 120s separate-workers baseline
4. 120s single-owner no-pin
5. 120s single-owner pin-ffs
6. optional pin-all / edge-first only if earlier results justify it

## Validation Plan

- parse generated profile JSONs for fusion/render FPS and completeness
- update generated Demo 2.1 validation docs with the benchmark table
- run deterministic checks after report/code changes

## Previous Camera-Attach Result

- Dry-run contract passed for `controller_prompt=cloth`, `object_prompt=stuffed animal`,
  `track_mode=controller-object`, and `gpu_pipeline_mode=single-owner`.
- The 30s live sanity run failed before SAM3.1 / FFS / EdgeTAM startup because
  `CameraSystem` detected zero connected cameras in WSL.
- Windows `usbipd list` shows three D455 devices as shared at bus IDs `1-3`,
  `1-4`, and `2-19`, but `usbipd` service is not running.
- `usbipd attach --wsl --busid ...` fails until the Windows `usbipd` service is
  started from an elevated/admin Windows session or the host is rebooted.

## Current Result

- Windows `usbipd` was restored and all three D455 devices attached into WSL.
- WSL pyrealsense2 enumerated three devices:
  `239222303506`, `239222300412`, `239222300781`.
- After the camera startup stability patch, the 30s sanity run reached live
  SAM3.1 initialization and emitted capture groups.
- SAM3.1 initialized cam2 with nonzero masks:
  `object_px=11241`, `controller_px=16956`.
- SAM3.1 failed on cam0 for the temporary controller prompt:
  `SAM3.1 did not produce a mask for label 'cloth'`.
- This is a correct no-fallback stop condition for the cloth-controller
  experiment, so the 120s A/B benchmarks remain pending until the cloth is
  visible/segmentable in all three camera views.

## Prompt Probe Update

- Captured current RGB frames under `docs/generated/demo2_1_prompt_probe/`.
- Cam0 prompt probe showed `cloth` only selected one cloth (`14772 px`) while
  `towel` selected both visible cloth/towel objects (`26121 px`).
- The current scene is better described by `controller_prompt=towel` than by
  `controller_prompt=cloth`.
- Static one-frame SAM3.1 probe with `text_prompt=stuffed animal,towel`
  returned nonzero object and controller masks on all three captured views:

```text
cam0 stuffed animal=19044 px, towel=26121 px
cam1 stuffed animal=18232 px, towel=17563 px
cam2 stuffed animal=11266 px, towel=16946 px
```

- A short live sanity with `--controller-prompt towel` initialized cam0
  successfully (`object_px=19038`, `controller_px=26111`). The run ended after
  writing summary/profile artifacts before all camera EdgeTAM workers completed
  initialization, so it is not a valid FPS benchmark.

## Towel Benchmark Result

The scene was benchmarked with:

```text
controller_prompt=towel
object_prompt=stuffed animal
init_mode=sam31-first-frame
```

60s sanity initialized all three cameras with nonzero live SAM3.1 masks:

```text
cam0 object_px=19074 controller_px=26153
cam1 object_px=18203 controller_px=17556
cam2 object_px=11255 controller_px=16938
```

After-warmup 120s results:

| Mode | Render FPS | Fusion FPS | Complete / Total | Timeout | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| separate-workers gate2 | 0.51 | 0.51 | 38 / 367 | 170 | too many partial-group timeouts |
| single-owner no-pin | 3.85 | 3.85 | 315 / 367 | 0 | best current candidate |
| single-owner pin-ffs | 3.59 | 3.59 | 299 / 383 | 2 | pinned FFS staging did not help |
| single-owner edge-first | 3.74 | 3.74 | 313 / 360 | 1 | stable but slower than ffs-then-edgetam |

Conclusion: single-owner `ffs-then-edgetam` is the preferred temporary
towel-controller path. Pinned FFS staging remains an ablation flag.
