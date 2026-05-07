# Demo 2.1 Shared FFS + Per-Camera EdgeTAM Validation

Date: 2026-05-06

## Contract

- `frame_by_frame_streaming=true`
- `offline_video_input_used=false`
- official depth source: `ffs`
- FFS checkpoint: `20-30-48`
- FFS valid iters: `4`
- FFS input shape: `480x864`
- FFS capture shape: `480x848`
- FFS padding policy: `pad_width_848_to_864`
- TensorRT builder optimization level: `5`
- EdgeTAM backend: HF `EdgeTAMVideo`
- EdgeTAM compile mode: `vision-reduce-overhead`
- official-lowfps GPU gate default: off
- preset capture: `848x480@15` per camera
- temporal grouping: `timestamp-nearest`, max skew `66.7 ms`
- object filter: `enhanced-pt`
- controller filter: `pt-filter`
- object/controller union before filter: `false`

## Runtime Design

FFS worker:

- one shared worker
- one runner/context owner
- cam0/cam1/cam2 depth generation is sequential
- input must already be a temporal-coherent `CaptureGroup`

EdgeTAM:

- three per-camera workers
- one streaming session per camera
- `obj_id=1 controller`
- `obj_id=2 object`

Fusion:

- strict `group_id` matching
- temporal skew re-check before fusion
- object fused cloud uses `enhanced-pt`
- controller fused cloud uses `pt-filter`
- object and controller are never unioned before filtering

GPU gate:

- one optional shared `GpuInferenceGate`
- Demo 2.1 presets now default to `gpu_gate=off`, `max_concurrent=0`
- serialized/limited gate modes remain explicit profiling overrides
- debug and summary record gate wait for FFS and each EdgeTAM camera worker

Temporal grouping:

- no temporal-coherent `CaptureGroup`, no FFS
- `CaptureGroupBuilder` keeps a small per-camera timestamp buffer
- default policy is `timestamp-nearest`
- default max skew is `66.7 ms`, one frame at `15 FPS`
- skewed groups are dropped before shared FFS
- shared FFS and fusion both re-check the same skew contract
- debug/profile/summary record timestamp source, per-camera offsets, skew, skew drops, and no-candidate drops

## Presets

| Preset | Profile | Target | Render default | Intended use |
| --- | --- | ---: | --- | --- |
| `official-lowfps` | `848x480@15` | 2 FPS | pointcloud | default controller-object FFS-quality fused demo; pass `--track-mode object-only` only when no controller is visible |
| `perf-5fps` | `848x480@15` | 5 FPS | pointcloud | controller-object quality-preserving performance target; current no-hand tests explicitly pass `--track-mode object-only` |
| `perf-5fps-single-owner` | `848x480@15` | 5 FPS | pointcloud | single GPU-owner performance target; publishes complete depth+masks together |
| `perf-5fps-staged` | `848x480@15` | 5 FPS | pointcloud | staged FFS sequential stage followed by parallel EdgeTAM stage |
| `climb-5` | `848x480@15` | 5 FPS | none | headless performance climb |
| `climb-10` | `848x480@15` | 10 FPS | none | diagnostic stress test |
| `diagnostics` | `848x480@15` | 2 FPS | none | capture/EdgeTAM/FFS isolation |

Compatibility aliases are kept for older commands:

```text
professor-safe              -> official-lowfps
visual-5fps                 -> perf-5fps
visual-5fps-no-gate         -> perf-5fps
visual-5fps-single-owner    -> perf-5fps-single-owner
visual-5fps-staged          -> perf-5fps-staged
```

The official-lowfps preset prioritizes semantic correctness and startup stability over frame rate.

## Preset Taxonomy Cleanup

`visual-5fps` was originally a visual performance candidate and then became
overloaded for render, EdgeTAM, FFS, gate, and H2D profiling. New commands
should use canonical preset names:

```text
official-lowfps          formal low-FPS professor-facing quality path
perf-5fps                5 FPS separate-workers performance target
perf-5fps-single-owner   5 FPS single GPU-owner performance target
perf-5fps-staged         5 FPS staged FFS -> parallel EdgeTAM target
diagnostics              capture / EdgeTAM / FFS isolation base
climb-5, climb-10        headless target-rate stress tests
```

Old names remain compatibility aliases only. Historical results below still use
the old filenames and command strings where that is how the experiment was run.

## Validation Commands

```bash
python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py

python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --dry-run \
  --preset professor-safe \
  --track-mode controller-object \
  --ffs-worker-mode shared \
  --edgetam-worker-mode per-camera

python -m unittest -v \
  tests.test_demo_v2_1_three_view_fused_pcd_smoke \
  tests.test_check_all_smoke

conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py
```

## Status

The live runtime code path is implemented with lazy hardware/model imports so deterministic checks can validate the CLI and fusion contract without cameras.

## Hardware Smoke

WSL-5090 hardware smoke was run with three attached D455 cameras, old `calibrate.pkl`, `object-only`, `render-mode=none`, and `profile=848x480`.

The first 60 FPS attempt confirmed the three camera startup path but one D455 detached from WSL under load, so the startup smoke was rerun at 30 FPS while keeping the official FFS shape contract:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --track-mode object-only \
  --depth-source ffs \
  --init-mode sam31-first-frame \
  --object-prompt "stuffed animal" \
  --ffs-worker-mode shared \
  --ffs-schedule strict3-latest \
  --edgetam-worker-mode per-camera \
  --edgetam-model-topology replicated \
  --fusion-target-fps 2 \
  --profile 848x480 \
  --fps 30 \
  --render-mode none \
  --duration-s 90 \
  --debug \
  --profile-cuda-events
```

Summary:

- summary: `result/demo2_1_three_view_fused_pcd/session_20260506_010718_summary.json`
- capture group FPS: `1.997`
- shared FFS cycle FPS: `1.949`
- fusion FPS: `2.019`
- latest group id: `169`
- object points: `10022`
- controller points: `0`
- capture timeout skips: `2`

This validates that the three-view object-only live path can start and produce fused FFS-derived object PCD packets. The old calibration is acceptable only for startup testing; fused geometry correctness still needs a fresh calibration run.

### Serialized GPU Gate Hardware Smoke

After adding the shared GPU gate, a longer professor-safe object-only headless smoke was run:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --track-mode object-only \
  --render-mode none \
  --duration-s 120 \
  --debug \
  --profile-cuda-events
```

Summary:

- summary: `result/demo2_1_three_view_fused_pcd/session_20260506_013134_summary.json`
- capture group FPS: `2.001`
- shared FFS cycle FPS: `2.004`
- fusion FPS: `1.470` over the full run, with steady debug output near the 2 FPS target after warmup
- complete fusion groups: `130`
- fusion timeout groups: `86`
- missing cam0 mask groups: `86`
- capture timeout skips: `1`
- object points: `9987`
- controller points: `0`
- GPU gate wait median:
  - FFS: `0.003 ms`
  - EdgeTAM cam0: `117.632 ms`
  - EdgeTAM cam1: `113.592 ms`
  - EdgeTAM cam2: `115.441 ms`

This validates that the serialized gate path can produce strict group-matched fused object PCD packets for all three cameras. The lower full-run fusion average is dominated by cold-start and missing-mask timeouts; the run should be at least 120 seconds when used as a hardware smoke. Controller-object should only be used when a hand is visible or saved three-camera hand masks are available.

## Current Run Table

| Run | Capture group FPS | FFS cycle FPS | EdgeTAM FPS | Fusion FPS | Render FPS | Notes |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| object-only target2 headless | 1.997 | 1.949 | ~2 FPS/cam after warmup | 2.019 | 0.0 | old calibration, no controller visible |
| object-only professor-safe serialized gate 120s | 2.001 | 2.004 | ~2 FPS/cam after warmup | 1.470 full-run, ~2 steady | 0.0 | 130 complete groups, 86 cam0 mask timeouts, old calibration |
| controller-object professor-safe | TBD | TBD | TBD | TBD | TBD | use only when a hand is visible or saved hand masks are available |
| controller-object climb-5 | TBD | TBD | TBD | TBD | TBD | profiling only |

## Visualization Smoke

A WSLg/Open3D pointcloud render smoke was run with the current no-hand object-only professor-safe path:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=0 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --object-prompt "stuffed animal" \
  --duration-s 90 \
  --debug \
  --profile-cuda-events
```

The Open3D pointcloud window started, rendered fused object PCD packets, and exited with code `0`. During steady runtime, debug output reported `render_fps` around `1.9` to `2.1`, matching the professor-safe 2 FPS target. This first visualization check was log-based; the GUI fast-exit path now writes the normal session summary before exiting.

A second climb run tested whether the same visualization path can reach 5 FPS:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset climb-5 \
  --render-mode pointcloud \
  --object-prompt "stuffed animal" \
  --duration-s 75 \
  --debug \
  --profile-cuda-events \
  > docs/generated/demo2_1_visual_climb5_render_smoke.log 2>&1
```

Parsed steady-state metrics from `docs/generated/demo2_1_visual_climb5_render_smoke.log`:

| Metric | Median | Mean | Notes |
| --- | ---: | ---: | --- |
| capture group FPS | `5.00` | `4.95` | capture/group builder reaches target |
| shared FFS cycle FPS | `5.42` | `5.34` | FFS worker is not the primary 5 FPS blocker |
| EdgeTAM FPS cam0/cam1/cam2 | `4.99 / 3.36 / 4.99` | `4.76 / 3.40 / 4.69` | cam1 often waits behind the serialized GPU gate |
| fusion FPS | `2.48` | `2.52` | below 5 FPS target |
| render FPS | `2.48` | `2.53` | tracks fusion FPS, not an independent Open3D bottleneck |
| FFS cycle ms | `103.90` | `125.90` | occasional gate/wait spikes |
| fusion ms | `44.90` | `51.63` | includes semantic PCD postprocess |
| filter ms | `37.30` | `43.70` | enhanced object filter dominates fusion time; occasional ~230 ms spikes |
| object points | `9981` | `9981` | after object cap/filter |
| EdgeTAM model ms cam0/cam1/cam2 | `32.30 / 33.20 / 31.40` | `34.86 / 32.99 / 31.60` | compiled model compute is not the only limiter |
| GPU gate wait cam0/cam1/cam2 | `66.30 / 345.20 / 228.00` | `116.58 / 260.01 / 204.81` | serialized gate scheduling is the largest climb-5 limiter |

Conclusion:

- Professor-safe visualization succeeds at the intended low-FPS 2 FPS target.
- The current pointcloud renderer can display fused packets at the rate produced by fusion.
- For a 5 FPS visual demo, the current median render rate is short by about `2.5 FPS`, requiring roughly a `2x` throughput improvement.
- The main blockers are serialized GPU gate wait for the three EdgeTAM workers and semantic PCD postprocess cost, not WSLg/Open3D itself.

## 5 FPS Visual Candidate

The quality-preserving hypothesis was tested by keeping the exact same PCD quality path and only allowing two GPU inference workers through the gate at a time:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset climb-5 \
  --render-mode pointcloud \
  --object-prompt "stuffed animal" \
  --gpu-gate-mode limited \
  --gpu-gate-max-concurrent 2 \
  --duration-s 90 \
  --debug \
  --profile-cuda-events \
  > docs/generated/demo2_1_visual_climb5_render_gate2.log 2>&1
```

Parsed steady-state metrics from `docs/generated/demo2_1_visual_climb5_render_gate2.log`:

| Metric | Median | Mean | Notes |
| --- | ---: | ---: | --- |
| capture group FPS | `4.99` | `4.90` | group builder stays near target |
| shared FFS cycle FPS | `4.97` | `4.99` | FFS remains near 5 FPS |
| EdgeTAM FPS cam0/cam1/cam2 | `5.03 / 5.04 / 5.01` | `4.98 / 4.96 / 4.93` | gate2 removes the cam1 starvation seen with serialized gate |
| fusion FPS | `4.86` | `4.69` | near the 5 FPS target |
| render FPS | `4.85` | `4.69` | tracks fusion; p90 render FPS was `5.19` |
| FFS cycle ms | `134.70` | `139.72` | FFS is slower than the serialized run but still supplies target-rate groups |
| fusion ms | `42.80` | `48.38` | includes object enhanced-PT |
| filter ms | `36.20` | `40.94` | unchanged enhanced object filter; occasional spike remains |
| object points | `9982` | `8298` | intermittent point-count drops still need visual review |
| EdgeTAM model ms cam0/cam1/cam2 | `62.80 / 63.00 / 60.50` | `63.89 / 66.15 / 61.12` | model kernels slow under gate2 contention, but total throughput improves |
| GPU gate wait ffs/cam0/cam1/cam2 | `0.00 / 89.30 / 59.50 / 51.70` | `0.06 / 88.95 / 62.98 / 54.12` | much lower than serialized gate waits |

Conclusion:

- The 5 FPS candidate does not change the PCD quality contract: FFS-derived depth, object `enhanced-pt`, controller `pt-filter`, and no native RealSense fallback.
- `gpu_gate_max_concurrent=2` is the first quality-preserving change that brings WSLg/Open3D visualization close to 5 FPS.
- Remaining risk is stability, not renderer capacity: filter spikes and intermittent low object point counts still need visual inspection before replacing `professor-safe`.
- This candidate is now exposed as the `visual-5fps` preset.

## visual-5fps Profiling Schema

`realtime_three_view_masked_fused_pcd.py` now supports explicit profiling flags that are disabled by default:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 20 \
  --profile-json-output docs/generated/demo2_1_visual5fps_profile_object_only.json
```

The JSON profile records one row per `group_id` with:

- capture frame sequence and group build timing
- per-camera EdgeTAM gate wait, model, postprocess, and total timing
- shared FFS gate wait, cycle timing, per-camera FFS and align timing
- fusion wait, raw PCD build timing, object enhanced-PT timing, controller PT-filter timing, point counts
- Open3D geometry update and render callback timing

The companion Markdown report summarizes full-run and warmup-excluded medians, p90, p95, max values, target FPS deficit, bottleneck class, and top slowest object enhanced-PT groups.

A 60s object-only smoke profile was run after adding this schema:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 60 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 20 \
  --profile-json-output docs/generated/demo2_1_visual5fps_profile_object_only.json \
  > docs/generated/demo2_1_visual5fps_profile_object_only.log 2>&1
```

Outputs:

- `docs/generated/demo2_1_visual5fps_profile_object_only.json`
- `docs/generated/demo2_1_visual5fps_profile_object_only.md`
- `docs/generated/demo2_1_visual5fps_profile_object_only.log`

Warmup-excluded summary from this smoke:

| Metric | Value |
| --- | ---: |
| render FPS | `4.56` |
| fusion FPS | `4.56` |
| target deficit | `0.44 FPS` |
| bottleneck class | `upstream_supply` |
| FFS cycle median / p95 | `123.31 / 195.44 ms` |
| EdgeTAM model median cam0/cam1/cam2 | `61.96 / 58.67 / 58.67 ms` |
| EdgeTAM gate wait median cam0/cam1/cam2 | `50.80 / 58.92 / 0.00 ms` |
| object enhanced-PT median / p95 / max | `33.78 / 41.36 / 44.80 ms` |
| Open3D render callback median / p95 | `0.51 / 3.27 ms` |
| Open3D object geometry update median / p95 | `0.03 / 2.24 ms` |

Interpretation:

- Open3D/WSLg is not the current limiter in this profile; the render callback is sub-4ms at p95.
- This run did not reproduce the earlier ~230ms enhanced-PT spike; object enhanced-PT p95 stayed near `41 ms`.
- The remaining deficit is upstream supply: FFS/EdgeTAM scheduling and missing mask groups during startup/initialization.
- The last steady debug window reached about `5.0 FPS`, so a longer 120s profile with a larger warmup exclusion is needed before changing quality settings.

## visual-5fps Follow-up: Initialization-Gated Runs

Two follow-up runs were attempted to separate steady-state throughput from startup effects:

```bash
# 120s gate2, warmup_exclude_s=40
QQTT_WSLG_OPEN3D_FAST_EXIT=1 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_1_visual5fps_profile_object_only_120s.json

# 60s gate2 after narrowing FFS GPU-gate scope
QQTT_WSLG_OPEN3D_FAST_EXIT=1 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 60 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 20 \
  --profile-json-output docs/generated/demo2_1_visual5fps_gate2_narrow_gate_profile_object_only_60s.json
```

Outputs:

- `docs/generated/demo2_1_visual5fps_profile_object_only_120s.json`
- `docs/generated/demo2_1_visual5fps_profile_object_only_120s.md`
- `docs/generated/demo2_1_visual5fps_gate2_narrow_gate_profile_object_only_60s.json`
- `docs/generated/demo2_1_visual5fps_gate2_narrow_gate_profile_object_only_60s.md`

Both runs are intentionally marked invalid for FPS comparison: SAM3.1 first-frame initialization failed on `cam0` with the live scene, so no complete fused groups were rendered. This is an initialization/scene issue, not evidence that Open3D, FFS, or enhanced-PT regressed.

The second run still provides one useful scheduling signal: after narrowing the GPU gate so it only wraps per-camera FFS TensorRT inference and no longer wraps FFS color alignment, the incomplete run reported `FFS cycle median / p95 = 85.31 / 125.32 ms` with near-zero FFS gate wait. Because masks were incomplete, this is not a final throughput number; it is only evidence that the gate scope was previously too broad.

Formal Demo 2.1 must initialize from SAM3.1 on the live first frame. The runtime fails fast by default if SAM3.1 object-only initialization does not register the requested object in a current no-hand run; no saved-mask or native-depth fallback is allowed. `controller-object` remains the default mode, and current no-controller lab runs must explicitly pass `--track-mode object-only`.

A formal pure-object live SAM3.1 profile was run after this contract correction:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 \
  ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --init-mode sam31-first-frame \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_1_visual5fps_live_sam31_profile_object_only_120s.json
```

Outputs:

- `docs/generated/demo2_1_visual5fps_live_sam31_profile_object_only_120s.json`
- `docs/generated/demo2_1_visual5fps_live_sam31_profile_object_only_120s.md`
- `docs/generated/demo2_1_visual5fps_live_sam31_profile_object_only_120s.log`

Result:

- cam1 initialized from live SAM3.1, `object_px=22879`
- cam2 initialized from live SAM3.1, `object_px=14398`
- cam0 failed SAM3.1 object-only initialization: `SAM 3.1 did not register any object for prompt stuffed animal for camera 0`
- complete fused groups: `0`
- rendered groups: `0`

This run is not a valid FPS profile. It is a live initialization failure for cam0. The runtime has been fixed so worker fatal errors are recorded and future runs return nonzero instead of silently writing a zero-FPS profile with process exit code `0`.

Next valid profiling step:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --init-mode sam31-first-frame \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_1_visual5fps_live_sam31_profile_object_only_120s.json
```

`saved-masks` is rejected by the formal Demo 2.1 runtime and is not a professor-facing initialization path.

Recommended current-lab professor-facing command when no hand/controller is visible:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug
```

5 FPS visual candidate command:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-cuda-events
```

Controller-object command for when a hand is visible:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --track-mode controller-object \
  --controller-prompt "hand" \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug
```

## Live SAM3.1 One-Frame Init Correction

The previous live first-frame path reused the generic SAM3.1 video helper even
though it only wrote one frame. Operationally that meant:

```text
live RGB frame -> temporary one-frame case -> SAM3.1 video session
-> add text prompt -> one-frame propagate_in_video -> mask files -> EdgeTAM init
```

The live initialization path now uses the SAM3.1 image API directly:

```text
live RGB frame -> build_sam3_image_model + Sam3Processor
-> set_image(frame) -> set_text_prompt(prompt) -> in-memory masks -> EdgeTAM init
```

The offline/generated-mask helper still uses the existing `run_case_segmentation`
video path. The formal live demo does not.

A proof on the saved cam0 RGB frame that previously failed through the video
helper succeeded with the image one-frame path:

```text
input: docs/generated/demo2_1_cam0_live_rgb_20260506_071723.png
prompt: stuffed animal
object_px: 27274
```

Artifacts:

- `docs/generated/demo2_1_cam0_live_rgb_20260506_071723_sam31_image_one_frame_mask.png`
- `docs/generated/demo2_1_cam0_live_rgb_20260506_071723_sam31_image_one_frame_overlay.png`

The formal failure policy is unchanged: if the live one-frame SAM3.1 image path
does not return the requested object/controller mask, the run fails without
saved-mask fallback.

## Visual-5FPS Profile With Image One-Frame SAM3.1

Command:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 \
timeout 300s ./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --init-mode sam31-first-frame \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-cuda-events \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_1_visual5fps_image_sam31_profile_object_only_120s.json
```

Live SAM3.1 image one-frame initialization succeeded on all three cameras:

```text
cam0 group=31 object_px=27636
cam1 group=31 object_px=22642
cam2 group=31 object_px=13983
```

After excluding the first 40 seconds of warmup, this run did not sustain the
5 FPS target:

```text
render_fps_after_warmup = 3.06
fusion_fps_after_warmup = 3.06
complete_fusion_groups_after_warmup = 233 / 431
fusion_timeout_groups_after_warmup = 177
bottleneck_class = upstream_supply
```

Key warmup-excluded timings:

| metric | median | p95 | max |
| --- | ---: | ---: | ---: |
| FFS cycle | `197.49 ms` | `332.98 ms` | `977.97 ms` |
| FFS gate wait | `100.91 ms` | `226.52 ms` | `803.34 ms` |
| EdgeTAM cam0 model | `96.99 ms` | `141.07 ms` | `443.92 ms` |
| EdgeTAM cam1 model | `97.52 ms` | `143.51 ms` | `497.76 ms` |
| EdgeTAM cam2 model | `91.81 ms` | `136.23 ms` | `348.09 ms` |
| object enhanced-PT | `37.47 ms` | `49.30 ms` | `247.44 ms` |
| render callback | `0.43 ms` | `0.75 ms` | `3.23 ms` |

The current visual-5fps limitation is therefore upstream mask/depth supply and
GPU scheduling, not Open3D rendering. Enhanced-PT still has periodic spikes
around `220-247 ms`, but the primary measured failure mode is incomplete
same-group fusion before the timeout.

Artifacts:

- `docs/generated/demo2_1_visual5fps_image_sam31_profile_object_only_120s.json`
- `docs/generated/demo2_1_visual5fps_image_sam31_profile_object_only_120s.md`

## Temporal-Coherent CaptureGroup Gate

Implementation update:

```text
Capture workers / CameraSystem observation
  -> per-camera ring buffers
  -> timestamp-nearest CaptureGroupBuilder
  -> skew gate
  -> shared FFS / per-camera EdgeTAM / fusion
```

New CLI/contract fields:

```text
--capture-group-policy latest|timestamp-nearest|timestamp-strict
--max-capture-skew-ms 66.7
--max-frame-age-ms 150
--capture-buffer-size 4
--drop-skewed-groups / --no-drop-skewed-groups
```

`professor-safe` and `visual-5fps` default to:

```text
capture_group_policy=timestamp-nearest
max_capture_skew_ms=66.7
max_frame_age_ms=150
capture_buffer_size=4
drop_skewed_groups=true
gpu_gate_mode=off
gpu_gate_max_concurrent=0
```

The shared FFS worker now rejects any group that violates the temporal skew
contract before running TensorRT. The fusion worker re-checks the same contract
before joining masks/depth. This protects dynamic object fused PCD quality at
the cost of dropping incoherent groups instead of producing ghosted geometry.

Deterministic tests cover:

```text
timestamp-nearest min-skew triplet selection
skew threshold drop behavior
per-camera temporal offsets
FFS/fusion shared skew guard
professor-safe and visual-5fps preset temporal defaults
```

The next live profile should report `skew_ms_med/latest`, `skew_drop`,
`no_candidate`, `ffs_skew_drop`, and `fusion_skew_drop` in the debug line and
write the `temporal_grouping` block to the session summary.

## No-GPU-Gate Baseline

Purpose:

```text
Test whether the shared GpuInferenceGate is the main bottleneck after adding
live SAM3.1 one-frame init and temporal-coherent CaptureGroup gating.
```

Quality contract held constant:

```text
live SAM3.1 image one-frame initialization
explicit object-only mode for the current no-controller scene
FFS-derived depth, 20-30-48 / valid_iters=4 / 480x864 / builderOpt5
timestamp-nearest grouping with max_capture_skew_ms=66.7
shared FFS worker remains sequential cam0 -> cam1 -> cam2
three per-camera EdgeTAM streaming sessions
object enhanced-PT
object/controller union before filter = false
```

Implementation note:

```text
Added `visual-5fps-no-gate`.
It disables only the global GpuInferenceGate:
  gpu_gate.mode=off
  gpu_gate.max_concurrent=0
It does not create parallel FFS workers or multiple FFS TensorRT runners.
```

Current default:

```text
Demo 2.1 presets now default to gpu_gate.mode=off and max_concurrent=0.
Serialized and limited gate modes remain explicit profiling overrides.
For separate-workers + gate-off compiled EdgeTAM, the runtime wraps the compiled
vision_encoder output with a clone step to avoid PyTorch CUDAGraph output
overwrite when three camera workers run concurrently.
```

Warmup-excluded live object-only results:

| Mode | Render FPS | Fusion FPS | Complete | Timeouts | FFS cycle median / p95 | FFS gate wait median / p95 | EdgeTAM model median per cam | Enhanced-PT median / p95 | Render p95 | Recommendation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate=2 latest | 2.42 | 2.42 | 186 | 201 | 193.2 / 324.0 ms | 80.6 / 211.9 ms | 88.7 / 88.3 / 90.6 ms | 37.9 / 46.9 ms | 1.75 ms | stable but too conservative |
| gate=3 | 1.37 | 1.37 | 111 | 203 | 226.2 / 397.7 ms | 109.9 / 270.2 ms | 150.3 / 150.9 / 147.2 ms | 28.6 / 39.4 ms | 1.55 ms | reject; FFS wait and timeout worse |
| no gate | 3.74 | 3.74 | 293 | 76 | 136.7 / 169.6 ms | 0.0 / 0.0 ms | 212.7 / 211.6 / 209.6 ms | 31.0 / 39.8 ms | 1.55 ms | useful baseline, not enough for 5 FPS |

Interpretation:

```text
No-gate removes the FFS gate wait and improves throughput from the latest gate=2
run, but it shifts the cost into EdgeTAM contention: per-camera EdgeTAM model
median rises to ~210 ms. It still misses the 5 FPS target by ~1.26 FPS.

gate=3 is worse than both gate=2 and no-gate for this run. It keeps enough gate
contention to slow FFS while also allowing enough concurrency to slow EdgeTAM,
so the complete fusion ratio collapses.

Open3D remains non-bottleneck: render p95 is ~1.5 ms in gate=3 and no-gate.
```

Artifacts:

- `docs/generated/demo2_1_visual5fps_no_gate_profile_20260506_095416.json`
- `docs/generated/demo2_1_visual5fps_no_gate_profile_20260506_095416.md`
- `docs/generated/demo2_1_visual5fps_gate3_profile_20260506_095730.json`
- `docs/generated/demo2_1_visual5fps_gate3_profile_20260506_095730.md`

Next scheduling direction:

```text
Do not use gate=3.
Do not make no-gate the professor default yet.
The data argues for a smarter policy instead:
  prioritize FFS/depth when it is the missing component for fusion
  prevent persistent EdgeTAM stragglers
  keep Open3D and quality filters unchanged
```

## Cloth-Controller No-GPU-Gate Baseline

Purpose:

```text
Repeat the gate baseline with the current experimental controller setup:
  controller = cloth
  object = stuffed animal

This is not the default controller semantics. The default Demo 2.1 controller
label remains hand; cloth is only a temporary live experiment label.
```

Quality contract held constant:

```text
live SAM3.1 image one-frame initialization
no saved-mask fallback
track_mode=controller-object
controller label=cloth, obj_id=1, postprocess=pt-filter
object label=stuffed animal, obj_id=2, postprocess=enhanced-PT
FFS-derived depth, 20-30-48 / valid_iters=4 / 480x864 / builderOpt5
timestamp-nearest grouping with max_capture_skew_ms=66.7
shared FFS worker remains sequential cam0 -> cam1 -> cam2
object/controller union before filter = false
```

Warmup-excluded live cloth-controller results:

| Mode | Render FPS | Fusion FPS | Complete | Timeouts | Capture group FPS | FFS cycle median / p95 | FFS gate wait median / p95 | EdgeTAM model median per cam | Object enhanced-PT median / p95 | Controller PT median / p95 | Render p95 | Recommendation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate=2 | 0.24 | 0.24 | 16 | 183 | 4.32 | 479.2 / 633.7 ms | 384.9 / 526.8 ms | 163.7 / 162.7 / 160.1 ms | 30.3 / 42.2 ms | 18.7 / 24.2 ms | 7.63 ms | valid but far too slow |
| no gate | 1.13 | 1.13 | 82 | 266 | 4.98 | 119.0 / 154.9 ms | 0.0 / 0.0 ms | 257.3 / 259.7 / 269.5 ms | 29.4 / 38.7 ms | 17.7 / 21.8 ms | 2.22 ms | faster than gate=2, still below target |
| gate=3 | invalid | invalid | invalid | invalid | invalid | invalid | invalid | invalid | invalid | invalid | invalid | not measured; camera attach failed |

Interpretation:

```text
For cloth-controller, no-gate is clearly faster than gate=2, but it still only
reaches ~1.13 FPS after warmup. Removing the global gate fixes the FFS wait
problem, but EdgeTAM per-camera model time rises to ~257-270 ms, so the
bottleneck remains upstream GPU contention / scheduling.

The filters are not the primary blocker in this run:
  object enhanced-PT p95 ~= 38.7 ms in no-gate
  controller pt-filter p95 ~= 21.8 ms in no-gate

Open3D is also not the blocker:
  render p95 ~= 2.22 ms in no-gate

gate=3 could not be profiled after no-gate because the Windows usbipd service
stopped and WSL saw 0 RealSense devices. That run is excluded from performance
comparison.
```

Artifacts:

- `docs/generated/demo2_1_visual5fps_controller_cloth_gate2_profile_20260506_103105.json`
- `docs/generated/demo2_1_visual5fps_controller_cloth_gate2_profile_20260506_103105.md`
- `docs/generated/demo2_1_visual5fps_controller_cloth_no_gate_profile_20260506_103346.json`
- `docs/generated/demo2_1_visual5fps_controller_cloth_no_gate_profile_20260506_103346.md`
- invalid gate3 startup log: `docs/generated/demo2_1_visual5fps_controller_cloth_gate3_profile_20260506_103631.log`

Next scheduling direction for cloth-controller:

```text
Do not use gate=2 for the cloth-controller visual path; it starves FFS too much.
No-gate is the better baseline, but still not near 5 FPS.
The next useful experiment is a smarter scheduling gate:
  FFS-priority or deadline-aware gate
  avoid persistent EdgeTAM stragglers
  keep FFS, live SAM3.1, semantic filters, and point quality unchanged
```

## Pinned-Memory H2D Ablation Implementation

Added an explicit transfer ablation without changing the quality chain:

```text
--pin-memory
--pin-memory-mode off|edge|ffs|all
--pinned-ring-size 3
--h2d-stream-mode default|dedicated
--profile-h2d
--ffs-input-staging pinned|pageable
```

Default behavior remains unchanged:

```text
pin_memory=false
pin_memory_mode=off
ffs_input_staging=pinned
```

Important detail:

```text
FFS TensorRT already used pinned input buffers by default.
True no-pin comparison now requires --ffs-input-staging pageable.
```

The EdgeTAM pinned path is opt-in. For `--pin-memory-mode edge|all`, Demo 2.1 runs the HF processor on CPU, stages `pixel_values` through a preallocated pinned ring, then enqueues non-blocking H2D on the selected stream. Pinned slots are not reused until their H2D event has completed.

The FFS runner now exposes `input_staging=pinned|pageable` and records staging/H2D profile fields in every result. The shared FFS worker still owns one TensorRT runner/context and still processes cam0, cam1, cam2 sequentially.

See:

- `docs/generated/demo2_1_pin_memory_ablation.md`
- `docs/generated/demo2_1_pin_memory_ablation.json`

## Cloth-controller live benchmark status

`controller=cloth` is supported as an explicit temporary experiment prompt; the
default controller remains `hand`.

Dry-run passed for:

```text
track_mode=controller-object
controller_prompt=cloth
object_prompt=stuffed animal
controller slot -> pt-filter
object slot -> enhanced-pt
fallback_allowed=false
```

Initial camera attach issue: the first 30s live sanity run did not reach SAM3.1
/ FFS / EdgeTAM. WSL reported zero connected RealSense cameras:

```text
AssertionError: Only 0 cameras are connected.
```

Windows `usbipd list` showed three shared D455 devices at `1-3`, `1-4`, and
`2-19`, but the Windows `usbipd` service was not running, so `usbipd attach
--wsl` failed.

After starting `usbipd` and attaching all three D455 devices, WSL enumerated
three cameras and the live sanity run reached SAM3.1 initialization. It then
failed as a semantic no-fallback condition:

```text
SAM3.1 cam2 initialized: object_px=11241, controller_px=16956
SAM3.1 cam0 failed: no mask for label "cloth"
capture groups emitted=124
complete fused groups=0
drop reason=missing_mask_cam0
```

Prompt probe update: the two physical cloth objects are better segmented as
`towel` in the current scene. Static one-frame SAM3.1 returned nonzero
`stuffed animal` and `towel` masks in all three views, and live sanity with
`--controller-prompt towel` initialized all three cameras.

## Single GPU-Owner Pipeline Experiment

Added an experimental pipeline mode:

```text
--gpu-pipeline-mode single-owner
--single-owner-order ffs-then-edgetam|edgetam-then-ffs
```

`interleaved` is reserved for a later per-camera interleaving implementation and
fails fast in the current runtime so it cannot be mistaken for a profiled mode.

In single-owner mode Demo 2.1 does not start the old `shared-ffs` worker or the
three `edgetam-camN` workers. Instead one worker owns the FFS TensorRT runner
and the EdgeTAM model/session state, processes one temporal-coherent
`CaptureGroup`, and publishes a `CompleteInferenceGroup`.

Default behavior remains unchanged:

```text
gpu_pipeline_mode=separate-workers
```

New preset:

```text
visual-5fps-single-owner:
  profile=848x480
  fps=15
  fusion_target_fps=5
  render_mode=pointcloud
  gpu_pipeline_mode=single-owner
  single_owner_order=ffs-then-edgetam
  depth_source=ffs
  init_mode=sam31-first-frame
```

Quality remains unchanged:

```text
FFS-derived depth
live SAM3.1 first-frame init
timestamp-nearest temporal grouping
object enhanced-PT
controller pt-filter
object/controller union before filter = false
```

See:

- `docs/generated/demo2_1_single_gpu_owner_pipeline.md`
- `docs/generated/demo2_1_single_gpu_owner_pipeline.json`
- `docs/generated/demo2_1_controller_prompt_probe.md`
- `docs/generated/demo2_1_controller_towel_single_owner_benchmark.md`

## Staged FFS -> Parallel EdgeTAM Experiment

Added an experimental staged pipeline mode:

```text
--gpu-pipeline-mode staged
--staged-order ffs-then-parallel-edgetam
--edgetam-stream-mode per-camera
```

Pipeline:

```text
CaptureGroup
  -> FFS stage: cam0 -> cam1 -> cam2, one shared TensorRT runner/context owner
  -> EdgeTAM stage: parallel(cam0, cam1, cam2), replicated HF EdgeTAM sessions
  -> CompleteInferenceGroup(depths + masks)
  -> fusion/filter/render
```

This mode does not start the old `shared-ffs` worker or the three old
`edgetam-camN` workers. FFS and EdgeTAM stages are separated by a stage
barrier, so FFS does not overlap EdgeTAM, while EdgeTAM cam0/cam1/cam2 can
run in parallel inside the EdgeTAM stage.

Profile fields:

```text
edgetam_stage_wall_ms
edgetam_stage_sum_model_ms
edgetam_parallel_efficiency = sum_model_ms / wall_ms
stage_barrier_ms
```

Quality remains unchanged:

```text
FFS-derived depth
live SAM3.1 first-frame init
timestamp-nearest temporal grouping
object enhanced-PT
controller pt-filter
object/controller union before filter = false
```

## Towel-Controller A/B Result

This is an explicit temporary controller experiment:

```text
controller_prompt=towel
object_prompt=stuffed animal
default controller remains hand
```

After-warmup 120s results:

| Mode | Render FPS | Fusion FPS | Complete / Total | Timeout | FFS p95 ms | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| separate-workers gate2 | 0.51 | 0.51 | 38 / 367 | 170 | 561.8 | too many partial-group timeouts |
| single-owner no-pin | 3.85 | 3.85 | 315 / 367 | 0 | 106.7 | best current candidate |
| single-owner pin-ffs | 3.59 | 3.59 | 299 / 383 | 2 | 114.4 | pinned FFS staging did not help |
| single-owner edge-first | 3.74 | 3.74 | 313 / 360 | 1 | 74.2 | stable, but lower FPS than ffs-then-edgetam |

The single-owner pipeline is the main improvement. It reduces partial group
timeouts by publishing depth and masks together as a complete inference group.
