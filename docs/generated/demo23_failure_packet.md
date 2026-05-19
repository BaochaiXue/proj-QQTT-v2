# Demo 2.3 Failure Packet

## Inputs
- `profile_json`: `/home/xinjie/proj-QQTT-v2/docs/generated/demo23_rendered_profile_latest.json`
- `summary_json`: `/home/xinjie/proj-QQTT-v2/result/demo2_1_three_view_fused_pcd/session_20260519_143352_summary.json`
- `calibration_report`: `None`
- `calibration_preflight`: `/home/xinjie/proj-QQTT-v2/docs/generated/calibration_debug/20260519-182349-color-only-preflight-after-option-guard/detection_report.json`

## Contract
- pipeline: `dual-gpu-split`
- render mode: `unknown`
- target FPS: `30.0`
- FFS batch: `0`
- FFS builderOptimizationLevel: `0`
- FFS TRT dir: ``

## Throughput
- `capture_group` FPS: `0.0`
- `raw_fusion` FPS: `0.0`
- `filter_output` FPS: `0.0`
- `fusion` FPS: `0.0`
- `render` FPS: `0.0`
- `complete_group_ratio` FPS: `0.0`

## Runtime Summary
- fatal: `dual-gpu-edgetam: RuntimeError: RuntimeError: SAM3.1 live initialization failed for cam0 after 1 attempt(s); no fallback is allowed`
- latest group: `None`
- object/controller points: `None` / `None`

## Risks
- `high` `ffs_contract_not_batch3_opt5`: FFS contract is not batch=3 builderOptimizationLevel=5.
- `medium` `latest_only_drop_pressure`: Latest-only worker or join queues dropped groups; expect jumps even without mismatched joins.
