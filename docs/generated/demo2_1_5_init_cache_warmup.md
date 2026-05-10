# Demo 2.1.5 Init Cache Warmup Probe

Date: 2026-05-10

## Command

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py \
  --warm-cache-only \
  --warm-cache-repeat 1 \
  --warm-cache-json-output docs/generated/demo2_1_5_init_cache_warmup_probe.json \
  --debug
```

## Result

The warm-cache path completed without opening RealSense cameras.

| Step | Time |
| --- | ---: |
| Total warm-cache wall time | 29390.44 ms |
| EdgeTAM total | 21468.64 ms |
| EdgeTAM runtime deps | 2572.80 ms |
| EdgeTAM processor load | 3064.71 ms |
| EdgeTAM model load | 2380.19 ms |
| EdgeTAM compile wrap | 448.66 ms |
| EdgeTAM first compiled prewarm forward | 10038.98 ms |
| SAM3.1 preload total | 7913.28 ms |
| SAM3.1 model load | 7855.41 ms |

## Interpretation

Demo 2.1.5's slow first-render timing is dominated by cold-start work rather than RealSense native depth. The largest single item in this probe is the first EdgeTAM compiled prewarm forward. SAM3.1 model load is also a large fixed cost.

This warm-cache command is useful as an operator preflight step. It moves the expensive EdgeTAM/HF/torch.compile and SAM3.1 loading work before the formal live-camera run, but it does not remove the cost from a cold Python process. A strict low-init demo should keep the process alive after this preflight or run the live demo in a persistent process that has already loaded the models.

## Artifact

- JSON: `docs/generated/demo2_1_5_init_cache_warmup_probe.json`
