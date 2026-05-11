# Demo 2.1 performance profile

- preset: `demo2.2-single-object-batchvision-edgetam`
- canonical preset: `demo2.2-single-object-batchvision-edgetam`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `cuda-inline`
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `8.33`
- filter output FPS after warmup: `8.33`
- fusion FPS after warmup: `8.33`
- groups after warmup: `1303`
- complete fused groups after warmup: `785`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.602`
- target deficit: `15.00`
- bottleneck class: `upstream_supply`
- GPU pipeline: `single-owner`
- single-owner order: `ffs-then-edgetam`
- filter scheduler: `async`
- render filtered only: `True`
- pin memory mode: `off`
- FFS input staging: `pinned`
- H2D stream mode: `default`

## Init Profile

| Stage | value |
| --- | ---: |
| parallel init max wait ms | `n/a` |
| camera startup ms | `4359.35` |
| EdgeTAM model load ms | `811.17` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1108.97` |
| EdgeTAM warmup/first forward ms | `51.60` |
| SAM3.1 model load ms | `7970.53` |
| SAM3.1 cam0 segment ms | `8227.91` |
| SAM3.1 cam1 segment ms | `122.96` |
| SAM3.1 cam2 segment ms | `124.80` |
| FFS runner init ms | `6052.57` |
| FFS first run ms | `954.56` |
| session init + prompt add ms | `11.15` |
| SAM3.1 release cleanup ms | `228.58` |
| time to first complete group s | `23.40` |
| time to first rendered group s | `n/a` |

## GPU Sampling

GPU sampling disabled for this run.

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `30.79` | `49.87` | `50.03` | `66.34` |
| `edgetam_model_ms` | `11.42` | `14.91` | `16.32` | `64.86` |
| `edgetam_preprocess_ms` | `0.83` | `0.98` | `1.04` | `1.38` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.06` | `0.10` | `0.12` | `0.68` |
| `edgetam_mask_resize_ms` | `0.03` | `0.06` | `0.08` | `0.62` |
| `edgetam_mask_threshold_ms` | `0.02` | `0.04` | `0.05` | `0.39` |
| `edgetam_mask_to_cpu_ms` | `0.14` | `0.18` | `0.19` | `0.62` |
| `edgetam_total_ms` | `10.69` | `13.89` | `15.20` | `58.77` |
| `ffs_cycle_ms` | `56.82` | `61.81` | `71.24` | `243.77` |
| `ffs_batch_ms` | `41.21` | `45.54` | `54.85` | `225.64` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `18.44` | `20.33` | `21.34` | `29.34` |
| `edgetam_batch_vision_total_ms` | `22.74` | `24.74` | `25.52` | `33.48` |
| `edgetam_batch_vision_preprocess_ms` | `2.49` | `2.93` | `3.10` | `4.13` |
| `edgetam_cam0_model_ms` | `11.69` | `14.98` | `16.39` | `36.88` |
| `edgetam_cam1_model_ms` | `11.23` | `15.00` | `16.62` | `26.28` |
| `edgetam_cam2_model_ms` | `11.17` | `14.78` | `15.93` | `64.86` |
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
| `ffs_stage_ms` | `2.05` | `2.93` | `3.22` | `10.72` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.05` | `2.93` | `3.22` | `10.72` |
| `ffs_cam1_stage_ms` | `2.05` | `2.93` | `3.22` | `10.72` |
| `ffs_cam2_stage_ms` | `2.05` | `2.93` | `3.22` | `10.72` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `112.95` | `126.05` | `135.92` | `322.96` |
| `gpu_owner_ffs_cycle_ms` | `56.82` | `61.81` | `71.24` | `243.77` |
| `gpu_owner_edgetam_cycle_ms` | `55.74` | `62.30` | `66.19` | `101.80` |
| `raw_fusion_total_ms` | `5.34` | `6.06` | `6.49` | `8.64` |
| `fusion_total_ms` | `28.10` | `31.19` | `32.57` | `221.68` |
| `filter_total_ms` | `22.69` | `25.45` | `27.01` | `215.62` |
| `filter_input_age_ms` | `23.27` | `25.93` | `27.52` | `216.98` |
| `object_enhanced_pt_ms` | `22.69` | `25.45` | `27.00` | `215.61` |
| `controller_pt_filter_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `1349` | `215.61` | `45742` | `7924` |
| `820` | `200.48` | `45706` | `7922` |
| `1470` | `198.26` | `45743` | `7922` |
| `1054` | `197.23` | `45754` | `7888` |
| `1585` | `196.30` | `45726` | `7934` |
| `585` | `195.62` | `45719` | `7830` |
| `995` | `193.45` | `45769` | `7949` |
| `1286` | `192.09` | `45725` | `7887` |
| `1527` | `191.85` | `45703` | `7885` |
| `1411` | `191.79` | `45755` | `7866` |
