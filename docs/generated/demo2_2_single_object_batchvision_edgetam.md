# Demo 2.2 Single-Object Batch-Vision EdgeTAM

Status: integration contract implemented; no-render hardware profile completed.

This adds a QQTT Demo 2.2 entry surface for the validated external
EdgeTAM-HF-batched result:

```text
backend: hf_batch_vision_seq_session
track_mode: object-only
object_prompt: stuffed animal
compile_mode: vision-reduce-overhead
mask_postprocess: cuda-inline
batch_vision_encoder: true
depth_source: ffs
FFS TensorRT batch: 3
```

The backend name is intentionally precise. It means:

```text
batch=3 HF EdgeTAM vision encoder
-> split image features
-> independent per-camera HF EdgeTAM video sessions
```

It does **not** mean true `hf_batched_multisession`; batched memory attention,
decoder, memory encoder, and state scatter remain external research work.

## External Validation Source

External fork / draft PR:

```text
/home/zhangxinjie/EdgeTAM-HF-batched
https://github.com/BaochaiXue/transformers/pull/1
```

Validated result on `data/different_types/sloth_set_2_motion_ffs`:

```text
single-object stuffed animal
backend: hf_batch_vision_seq_session
compile: reduce-overhead
correctness: pass
stage_wall_p50: 31.30548 ms
stage_wall_p90: 32.98849 ms
stage_wall_p95: 33.75403 ms
p50-derived group FPS: 31.94
```

Controller/hand is not enabled by this preset because compiled two-object
results still have low-IoU hand outliers.

## New CLI

Validated preset:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-single-object-batchvision-edgetam \
  --dry-run
```

Public alias:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --single-object-batchvision-edgetam \
  --edgetam-external-path /home/zhangxinjie/EdgeTAM-HF-batched \
  --dry-run
```

No-render profile command for the local FFS path:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-single-object-batchvision-edgetam \
  --edgetam-external-path /home/zhangxinjie/EdgeTAM-HF-batched \
  --render-mode none \
  --duration-s 120 \
  --warmup-s 30 \
  --profile-edgetam-stages \
  --profile-cuda-events \
  --profile-json-output docs/generated/demo22_single_object_batchvision_stuffed_animal_no_render_profile.json \
  --debug
```

## Hardware Profile

Run:

```text
profile:
  docs/generated/demo22_single_object_batchvision_stuffed_animal_no_render_profile.json
  docs/generated/demo22_single_object_batchvision_stuffed_animal_no_render_profile.md
mode:
  no-render
  no-parallel-init
depth:
  local FFS TensorRT batch=3
```

The first parallel-init attempt hit a known SAM3.1/torchvision import race:

```text
ImportError: cannot import name 'StochasticDepth' from partially initialized module 'torchvision.ops'
```

The formal run was repeated with `--no-parallel-init` and completed.

Warmup-excluded results:

| Metric | Value |
| --- | ---: |
| capture group FPS | `13.80` |
| raw fusion FPS | `8.33` |
| filter output FPS | `8.33` |
| complete fused groups | `785` |
| complete group ratio | `0.602` |
| render FPS | `0.00` |
| bottleneck class | `upstream_supply` |

Selected timing after warmup:

| Metric | p50 ms | p90 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| EdgeTAM batch-vision total | `22.74` | `24.74` | `25.52` | `33.48` |
| EdgeTAM batch-vision model | `18.44` | `20.33` | `21.34` | `29.34` |
| GPU owner total | `112.95` | `126.05` | `135.92` | `322.96` |
| FFS cycle | `56.82` | `61.81` | `71.24` | `243.77` |
| raw fusion | `5.34` | `6.06` | `6.49` | `8.64` |
| async filter | `22.69` | `25.45` | `27.01` | `215.62` |

Init profile:

| Stage | Value |
| --- | ---: |
| camera startup | `4359.35 ms` |
| SAM3.1 model load | `7970.53 ms` |
| SAM3.1 cam0 segment | `8227.91 ms` |
| SAM3.1 cam1 segment | `122.96 ms` |
| SAM3.1 cam2 segment | `124.80 ms` |
| FFS runner init | `6052.57 ms` |
| FFS first batch run | `954.56 ms` |
| EdgeTAM model load | `811.17 ms` |
| EdgeTAM compile/prewarm | `1108.97 ms` |
| time to first complete fused group | `23.40 s` |

This run proves the single-object batch-vision backend is live in the Demo 2.2
local FFS pipeline and produces non-empty fused object PCD packets. It does not
hit a 15 FPS fused-PCD output target in this no-render configuration; the
current limiter is upstream supply, mainly local FFS + single-owner scheduling,
not the EdgeTAM batch-vision stage.

## Contract Notes

- `hf_batched_multisession` is accepted as a symbolic backend name but rejected
  by live validation because it is not integrated.
- `--parallel-edgetam` is now accepted by the Demo 2.2 wrapper as a compatibility
  alias for `hf_batch_vision_seq_session`; it keeps the single-owner GPU
  pipeline and enables batch vision.
- Demo 2.2 remains the local FFS fused-PCD path. `ffs_remote` flags are accepted
  by the public wrapper for command compatibility, but the current Demo 2.2
  runtime does not execute remote FFS.

## Verification

```text
tests.test_demo_v2_2_async_filtered_fused_pcd_smoke: pass
dry-run preset contract: pass
dry-run public alias contract: pass
hardware no-render profile: pass with --no-parallel-init
```
