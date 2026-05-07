# Demo 2.1 Single EdgeTAM GPU Saturation Probe

Question: can one HF EdgeTAM streaming worker saturate the 5090 Laptop GPU?

Short answer: no. In the tested live single-camera path, one EdgeTAM worker leaves substantial GPU headroom. The Demo 2.1 bottleneck is the aggregate cost and scheduling of three EdgeTAM camera passes plus three FFS passes, not one EdgeTAM worker consuming the whole GPU.

Important correction: the `29.3 FPS` number below is not a single-object EdgeTAM maximum. It is the measured live throughput for a two-object `controller-object` path under the test command. For GPU saturation, the stronger indicators are `cuda_event_model_ms` and sustained GPU utilization.

## Test Setup

- Host: WSL-5090 laptop
- Camera: D455 `239222300412`
- Resolution: `848x480`
- Depth: disabled with `--depth-source none`
- PCD/render: disabled with `--pcd-mode none --render-mode none`
- Init: live `--init-mode sam31-first-frame`
- EdgeTAM: HF EdgeTAMVideo streaming, `--compile-mode vision-reduce-overhead`, `--dtype bfloat16`
- GPU monitoring: `nvidia-smi` sampled at 1 Hz

Logs:

- Object-only log: `docs/generated/demo2_1_edgetam_gpu_saturation_probe/single_edgetam_towel_depth_none_object_only_60s.log`
- Object-only GPU CSV: `docs/generated/demo2_1_edgetam_gpu_saturation_probe/single_edgetam_towel_gpu_1hz.csv`
- Controller-object log: `docs/generated/demo2_1_edgetam_gpu_saturation_probe/single_edgetam_towel_stuffed_controller_object_60s.log`
- Controller-object GPU CSV: `docs/generated/demo2_1_edgetam_gpu_saturation_probe/single_edgetam_towel_stuffed_controller_object_gpu_1hz.csv`
- Parsed summary: `docs/generated/demo2_1_edgetam_gpu_saturation_probe/summary.json`

## Results

| Mode | Prompt(s) | Seg FPS median | CUDA model ms median / p90 / max | E2E latency ms median / p90 | GPU util median / p90 / max | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| object-only | `towel` | 34.9 | 12.44 / 22.02 / 29.46 | 16.05 / 25.09 | steady post-load 27.5 / 29 / 30 | does not saturate GPU |
| controller-object | controller=`towel`, object=`stuffed animal` | 29.3 | 31.41 / 42.18 / 56.46 | 50.16 / 65.36 | nonzero samples 38 / 44 / 47 | heavier, still not saturated |

Notes:

- The first `stuffed animal` object-only run failed during SAM3.1 initialization and was not used as a GPU saturation result.
- `Seg FPS` is live pipeline throughput for the specific command, not the model-only maximum. A live camera FPS cap, SAM3.1 init behavior, prompt quality, and dropped frames can all lower this number.
- `nvidia-smi` at 1 Hz is coarse and can miss sub-second kernel bursts. It is still useful here because a saturated worker would show sustained high utilization near the card limit, which did not happen.
- The object-only steady GPU utilization stayed around 27-30 percent after model load.
- The controller-object run is heavier because EdgeTAM tracks two object ids. Even then, utilization stayed roughly in the 30-45 percent range, not near saturation.

## Interpretation

One EdgeTAM worker does not eat the whole GPU. It has enough headroom that the 5090 can theoretically overlap or interleave other work, but overlap is not free: FFS TensorRT, three EdgeTAM sessions, CUDA stream scheduling, allocator behavior, and same-group synchronization can still inflate p95 latency.

For Demo 2.1, the relevant bottleneck is:

```text
3x FFS depth passes
+ 3x EdgeTAM streaming passes
+ temporal group completeness
+ object/controller fusion and filtering
```

The current single-owner path is therefore reasonable: it avoids partial-group timeouts and makes the total group cost explicit. The next optimization question is not "does one EdgeTAM saturate the GPU?", but "how should we schedule three EdgeTAM passes and three FFS passes so p95 stays low while complete-group ratio stays high?"
