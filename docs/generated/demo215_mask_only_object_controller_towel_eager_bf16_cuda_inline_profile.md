# Demo 2.1 performance profile

- preset: `demo2.1.5-mask-only-debug`
- canonical preset: `demo2.1.5-mask-only-debug`
- target FPS: `60.00`
- capture group target FPS: `15.00`
- compile mode: `none`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `cuda-inline`
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `0.00`
- filter output FPS after warmup: `0.00`
- fusion FPS after warmup: `0.00`
- groups after warmup: `690`
- complete fused groups after warmup: `0`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.000`
- target deficit: `60.00`
- bottleneck class: `upstream_supply`
- GPU pipeline: `separate-workers`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `async`
- render filtered only: `False`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

Warning: this profile has no complete fused groups after warmup. Treat it as an initialization or missing-packet run, not as a valid visual FPS comparison.

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `3531.63` |
| camera startup ms | `4902.03` |
| EdgeTAM model load ms | `5848.54` |
| EdgeTAM compile wrap ms | `0.00` |
| EdgeTAM compile prewarm ms | `n/a` |
| EdgeTAM warmup/first forward ms | `1177.28` |
| SAM3.1 model load ms | `9249.22` |
| SAM3.1 cam0 segment ms | `n/a` |
| SAM3.1 cam1 segment ms | `n/a` |
| SAM3.1 cam2 segment ms | `n/a` |
| FFS runner init ms | `n/a` |
| FFS first run ms | `n/a` |
| session init + prompt add ms | `5.40` |
| SAM3.1 release cleanup ms | `n/a` |
| time to first complete group s | `n/a` |
| time to first rendered group s | `n/a` |

## GPU Sampling

- backend requested: `nvml`
- backend used: `nvml`
- device index: `0`
- interval s: `0.200`
- samples after warmup: `257`

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `gpu_util_pct` | `28.00` | `30.00` | `31.00` | `33.00` |
| `memory_util_pct` | `5.00` | `6.00` | `6.00` | `9.00` |
| `memory_used_mb` | `12353.10` | `14511.10` | `14861.10` | `14861.10` |
| `power_w` | `136.47` | `137.91` | `138.39` | `190.31` |
| `sm_clock_mhz` | `180.00` | `180.00` | `180.00` | `195.00` |
| `mem_clock_mhz` | `14001.00` | `14001.00` | `14001.00` | `14001.00` |
| `temperature_c` | `70.00` | `73.00` | `74.00` | `76.00` |

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `16.80` | `50.64` | `57.44` | `65.93` |
| `edgetam_model_ms` | `189.39` | `254.16` | `285.19` | `410.98` |
| `edgetam_preprocess_ms` | `2.02` | `2.91` | `3.32` | `7.20` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.25` | `0.45` | `0.54` | `1.73` |
| `edgetam_mask_resize_ms` | `0.16` | `0.33` | `0.37` | `0.93` |
| `edgetam_mask_threshold_ms` | `0.08` | `0.18` | `0.24` | `0.83` |
| `edgetam_mask_to_cpu_ms` | `0.25` | `0.38` | `0.44` | `2.65` |
| `edgetam_total_ms` | `174.72` | `233.13` | `262.66` | `376.45` |
| `ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_batch_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_preprocess_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam0_model_ms` | `180.55` | `250.11` | `283.06` | `407.02` |
| `edgetam_cam1_model_ms` | `199.96` | `262.31` | `282.97` | `386.65` |
| `edgetam_cam2_model_ms` | `193.51` | `251.39` | `285.95` | `410.98` |
| `edgetam_cam0_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam1_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_cam2_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_pin_copy_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edge_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_stage_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_ffs_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_edgetam_cycle_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `raw_fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `fusion_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `filter_input_age_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `object_enhanced_pt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `0` | `0.00` | `0` | `0` |
| `1` | `0.00` | `0` | `0` |
| `2` | `0.00` | `0` | `0` |
| `3` | `0.00` | `0` | `0` |
| `4` | `0.00` | `0` | `0` |
| `5` | `0.00` | `0` | `0` |
| `6` | `0.00` | `0` | `0` |
| `7` | `0.00` | `0` | `0` |
| `8` | `0.00` | `0` | `0` |
| `9` | `0.00` | `0` | `0` |
