# Demo 2.2 Object + Controller Towel Batch-Vision Retry

Status: live no-render profile completed.

This run retries Demo 2.2 with:

```text
backend: hf_batch_vision_seq_session
track_mode: controller-object
object_prompt: stuffed animal
controller_prompt: towel
compile_mode: vision-reduce-overhead
mask_postprocess: cuda-inline
depth_source: local FFS TensorRT batch=3
render_mode: none
parallel_init: disabled
```

Artifacts:

```text
profile_json: docs/generated/demo22_object_controller_towel_batchvision_no_render_profile.json
profile_md:   docs/generated/demo22_object_controller_towel_batchvision_no_render_profile.md
```

## SAM3.1 Init

SAM3.1 frame-0 initialization succeeded for both object and controller on all
three cameras:

| Camera | stuffed animal pixels | towel pixels |
| --- | ---: | ---: |
| cam0 | `50458` | `37731` |
| cam1 | `14982` | `18793` |
| cam2 | `11328` | `20026` |

This means the current live setup can produce non-empty SAM3.1 masks for the
towel controller. This run is a live pipeline sanity/profile, not a SAM3.1
video replay IoU correctness run.

## Warmup-Excluded Profile

| Metric | Value |
| --- | ---: |
| capture group FPS | `13.01` |
| raw fusion FPS | `6.37` |
| filter output FPS | `6.37` |
| complete fused groups | `264` |
| complete group ratio | `0.455` |
| render FPS | `0.00` |
| Demo 2.2 14.4 FPS pass gate | `FAIL` |
| bottleneck class | `upstream_supply` |

Selected timing:

| Metric | p50 ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| EdgeTAM batch-vision total | `22.78` | `25.54` | `27.04` | `34.33` |
| EdgeTAM batch-vision model | `18.44` | `21.02` | `22.39` | `29.80` |
| EdgeTAM per-camera model | `22.26` | `29.53` | `32.80` | `51.15` |
| GPU owner total | `147.11` | `172.51` | `188.13` | `369.39` |
| FFS cycle | `62.14` | `67.75` | `70.73` | `252.00` |
| GPU owner EdgeTAM cycle | `84.87` | `104.11` | `108.97` | `137.61` |
| raw fusion | `9.87` | `11.50` | `12.05` | `14.23` |
| async filter | `34.33` | `37.88` | `39.41` | `227.52` |
| object enhanced filter | `22.18` | `25.35` | `26.13` | `215.15` |
| controller PT filter | `11.97` | `13.18` | `13.56` | `14.38` |

## Decision

```text
SAM3.1 towel init: pass
EdgeTAM object+controller live path: pass
object fused PCD: non-empty
controller fused PCD: non-empty
15 FPS fused-PCD target: fail
```

Compared with the single-object run, the two-object towel path is slower:

```text
single-object raw/filter FPS:      ~8.33
object+controller raw/filter FPS:  ~6.37
```

The slowdown is expected from two-object HF session tracking plus controller
PCD/filter work. The next useful correctness step is a real SAM3.1 replay IoU
run with non-empty towel masks, not just this live FPS sanity run.
