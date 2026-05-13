# Demo 2.2 performance profile

- preset: `demo2.2-async-filter-5fps`
- canonical preset: `demo2.2-async-filter-5fps`
- target FPS: `15.00`
- capture group target FPS: `15.00`
- compile mode: `vision-reduce-overhead`
- dtype: `bfloat16`
- EdgeTAM input path: `pil`
- mask postprocess: `cuda-inline`
- render FPS after warmup: `0.00`
- raw fusion FPS after warmup: `4.74`
- filter output FPS after warmup: `4.74`
- fusion FPS after warmup: `4.74`
- Demo 2.2 PASS threshold: `14.40 FPS`
- Demo 2.2 result: `FAIL`
- groups after warmup: `673`
- complete fused groups after warmup: `306`
- rendered groups after warmup: `0`
- complete group ratio after warmup: `0.455`
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
| camera startup ms | `4545.67` |
| EdgeTAM model load ms | `936.92` |
| EdgeTAM compile wrap ms | `0.01` |
| EdgeTAM compile prewarm ms | `1314.43` |
| EdgeTAM warmup/first forward ms | `79.50` |
| SAM3.1 model load ms | `8723.95` |
| SAM3.1 cam0 segment ms | `9015.46` |
| SAM3.1 cam1 segment ms | `178.80` |
| SAM3.1 cam2 segment ms | `180.11` |
| FFS runner init ms | `6893.06` |
| FFS first run ms | `1283.57` |
| session init + prompt add ms | `16.03` |
| SAM3.1 release cleanup ms | `229.31` |
| time to first complete group s | `26.18` |
| time to first rendered group s | `n/a` |

## GPU Sampling

GPU sampling disabled for this run.

| Metric | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| `capture_temporal_skew_ms` | `21.07` | `46.83` | `57.89` | `64.24` |
| `edgetam_model_ms` | `21.82` | `26.06` | `27.90` | `40.07` |
| `edgetam_preprocess_ms` | `0.99` | `1.16` | `1.28` | `1.44` |
| `edgetam_prompt_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_postprocess_ms` | `0.06` | `0.10` | `0.13` | `0.30` |
| `edgetam_mask_resize_ms` | `0.03` | `0.05` | `0.07` | `0.26` |
| `edgetam_mask_threshold_ms` | `0.03` | `0.04` | `0.05` | `0.24` |
| `edgetam_mask_to_cpu_ms` | `0.21` | `2.53` | `4.50` | `15.57` |
| `edgetam_total_ms` | `22.81` | `27.06` | `28.79` | `41.40` |
| `ffs_cycle_ms` | `71.60` | `76.85` | `80.10` | `341.70` |
| `ffs_batch_ms` | `52.12` | `56.53` | `58.85` | `325.05` |
| `ffs_gate_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `edgetam_batch_vision_model_ms` | `11.39` | `14.36` | `15.20` | `32.83` |
| `edgetam_batch_vision_total_ms` | `18.45` | `21.88` | `23.01` | `42.31` |
| `edgetam_batch_vision_preprocess_ms` | `2.97` | `3.47` | `3.82` | `4.31` |
| `edgetam_cam0_model_ms` | `22.29` | `26.21` | `28.02` | `40.07` |
| `edgetam_cam1_model_ms` | `21.50` | `25.72` | `27.70` | `37.91` |
| `edgetam_cam2_model_ms` | `21.42` | `26.05` | `27.94` | `34.05` |
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
| `ffs_stage_ms` | `2.70` | `4.58` | `5.38` | `10.09` |
| `ffs_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam0_stage_ms` | `2.70` | `4.55` | `5.37` | `10.09` |
| `ffs_cam1_stage_ms` | `2.70` | `4.55` | `5.37` | `10.09` |
| `ffs_cam2_stage_ms` | `2.70` | `4.55` | `5.37` | `10.09` |
| `ffs_cam0_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam1_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `ffs_cam2_h2d_wait_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `gpu_owner_total_ms` | `159.72` | `171.74` | `178.81` | `422.61` |
| `gpu_owner_ffs_cycle_ms` | `71.60` | `76.85` | `80.10` | `341.70` |
| `gpu_owner_edgetam_cycle_ms` | `87.33` | `95.83` | `99.70` | `122.06` |
| `raw_fusion_total_ms` | `10.70` | `12.57` | `12.95` | `14.25` |
| `fusion_total_ms` | `52.88` | `59.65` | `62.12` | `251.01` |
| `filter_total_ms` | `42.03` | `48.04` | `49.68` | `240.30` |
| `filter_input_age_ms` | `42.48` | `48.65` | `50.42` | `240.96` |
| `object_enhanced_pt_ms` | `30.15` | `35.55` | `38.00` | `227.35` |
| `controller_pt_filter_ms` | `11.68` | `13.15` | `13.65` | `15.28` |
| `render_total_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_object_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |
| `open3d_controller_update_geometry_ms` | `0.00` | `0.00` | `0.00` | `0.00` |

## Top slowest object enhanced-PT groups

| group | ms | input points | kept points |
| ---: | ---: | ---: | ---: |
| `402` | `227.35` | `49054` | `13422` |
| `948` | `221.62` | `49072` | `13479` |
| `607` | `217.43` | `49037` | `13471` |
| `810` | `215.30` | `49044` | `13434` |
| `880` | `213.95` | `49012` | `13443` |
| `746` | `210.22` | `49062` | `13486` |
| `540` | `208.12` | `49073` | `13525` |
| `676` | `208.01` | `49034` | `13402` |
| `324` | `205.59` | `49046` | `13445` |
| `471` | `204.81` | `49050` | `13460` |
