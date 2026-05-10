# Demo 2.1.5 RealSense-Depth Fused PCD

Demo 2.1.5 mirrors the current Demo 2.2 async filtered fused PCD wrapper, but uses native aligned RealSense depth instead of FFS depth.

Run the default RealSense-depth path:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py
```

Inspect the resolved runtime contract without opening cameras:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py --dry-run
```

The default preset is `demo2.1.5-async-filter-5fps`. It keeps the Demo 2.2 single-owner EdgeTAM + async PCD filter schedule, while setting `depth_source=realsense` and using RealSense RGB-D capture.

Warm EdgeTAM/HF/torch.compile and SAM3.1 caches before an operator-facing run:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v2_1_5/realtime_three_view_async_filtered_fused_pcd.py \
  --warm-cache-only \
  --warm-cache-repeat 1 \
  --warm-cache-json-output docs/generated/demo2_1_5_init_cache_warmup_probe.json
```

This does not open RealSense cameras. It is meant to pay the cold-start cost before the formal live demo, so the later first-camera group is less likely to also carry EdgeTAM compile and SAM3.1 model-load work.
