# Demo 2.1.5 parallel EdgeTAM GPU profile

## Summary

- Public option: `--parallel-edgetam`
- Result: the staged three-camera EdgeTAM path is functional, but slower than the current Demo 2.1.5 single-owner baseline on this RTX 5090 Laptop.
- Parallel render FPS: `4.72`
- Previous single-owner render FPS: `7.29`
- Parallel GPU util median/p95/max: `21 / 23 / 27 %`
- Previous single-owner GPU util median/p95/max: `30 / 37 / 39 %`
- Interpretation: thread-level parallel EdgeTAM loses the faster `vision-reduce-overhead` CUDA graph path and currently should remain a profiling option, not the default.

- preset: `demo2.1.5-staged-parallel-5fps`
- canonical preset: `demo2.1.5-staged-parallel-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- render FPS after warmup: `4.72`
- raw fusion FPS after warmup: `4.72`
- filter output FPS after warmup: `4.72`
- fusion FPS after warmup: `4.72`
- groups after warmup: `603`
- complete fused groups after warmup: `215`
- rendered groups after warmup: `215`
- complete group ratio after warmup: `0.357`
- target deficit: `10.28`
- bottleneck class: `upstream_supply`
- GPU pipeline: `staged`
- single-owner order: `ffs-then-parallel-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `all`
- FFS input staging: `pinned`
- H2D stream mode: `dedicated`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `18422.32` |
| camera startup ms | `4732.43` |
| EdgeTAM model load ms | `4027.94` |
| EdgeTAM compile wrap ms | `585.01` |
| EdgeTAM compile prewarm ms | `13962.29` |
| EdgeTAM warmup/first forward ms | `14713.50` |
| SAM3.1 model load ms | `34348.70` |
| SAM3.1 cam0 segment ms | `8022.49` |
| SAM3.1 cam1 segment ms | `16976.88` |
| SAM3.1 cam2 segment ms | `10028.62` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `8.00` |
| SAM3.1 release cleanup ms | `n/a` |
| time to first complete group s | `62.61` |
| time to first rendered group s | `62.62` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.100`
- samples after warmup: `434`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `21.00` | `23.00` | `23.00` | `27.00` |
| `memory_util_pct` | `6.00` | `7.00` | `7.00` | `8.00` |
| `memory_used_mb` | `2186.48` | `2320.45` | `2336.52` | `2361.37` |
| `power_w` | `110.54` | `115.10` | `116.80` | `121.78` |
| `sm_clock_mhz` | `232.00` | `232.00` | `1110.00` | `1110.00` |
| `mem_clock_mhz` | `9001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `60.00` | `63.00` | `65.00` | `66.00` |

- sampler errors: `nvml: NVMLError_Unknown: Unknown Error; nvml: NVMLError_Unknown: Unknown Error; nvml: NVMLError_Unknown: Unknown Error`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `22.77` | `48.71` | `55.69` | `65.53` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `184.03` | `198.65` | `204.33` | `537.61` |
| `edgetam_cam1_model_ms` | `182.49` | `199.48` | `204.58` | `544.07` |
| `edgetam_cam2_model_ms` | `182.16` | `199.39` | `204.76` | `529.70` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_pin_copy_ms` | `1.16` | `2.74` | `3.24` | `5.05` |
| `edge_h2d_wait_ms` | `0.01` | `0.31` | `0.82` | `4.02` |
| `edge_cam0_pin_copy_ms` | `1.15` | `2.77` | `3.49` | `4.49` |
| `edge_cam1_pin_copy_ms` | `1.20` | `2.64` | `3.04` | `4.29` |
| `edge_cam2_pin_copy_ms` | `1.14` | `2.72` | `3.19` | `5.05` |
| `edge_cam0_h2d_wait_ms` | `0.01` | `0.27` | `1.06` | `4.02` |
| `edge_cam1_h2d_wait_ms` | `0.02` | `0.28` | `0.57` | `3.79` |
| `edge_cam2_h2d_wait_ms` | `0.01` | `0.41` | `0.82` | `1.95` |
| `ffs_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `203.06` | `219.51` | `224.21` | `558.34` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `raw_fusion_total_ms` | `31.09` | `37.03` | `38.98` | `48.12` |
| `fusion_total_ms` | `89.65` | `98.46` | `103.02` | `450.50` |
| `filter_total_ms` | `58.61` | `65.41` | `67.71` | `413.70` |
| `filter_input_age_ms` | `59.09` | `66.23` | `69.13` | `414.43` |
| `object_enhanced_pt_ms` | `38.75` | `44.27` | `46.20` | `395.78` |
| `controller_pt_filter_ms` | `19.46` | `22.31` | `23.61` | `29.16` |
| `render_total_ms` | `0.40` | `0.72` | `0.98` | `2.52` |
| `open3d_object_update_geometry_ms` | `0.03` | `0.03` | `0.04` | `1.08` |
| `open3d_controller_update_geometry_ms` | `0.01` | `0.02` | `0.04` | `2.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1128` | `395.78` | `46587` | `8507` |
| `1276` | `378.38` | `46614` | `8494` |
| `835` | `376.66` | `46613` | `8511` |
| `983` | `370.64` | `46623` | `8528` |
| `1272` | `55.08` | `46595` | `8541` |
| `945` | `51.35` | `46533` | `8474` |
| `1162` | `48.43` | `46645` | `8534` |
| `1270` | `47.86` | `46608` | `8482` |
| `1342` | `47.84` | `46627` | `8485` |
| `1116` | `46.67` | `46596` | `8564` |
