# Demo 2.2 BatchTam TRT Memory Path

## Goal

Wire the externally validated BatchTam `memory_path_all` TensorRT component runtime into Demo 2.2 and use only full Demo 2.2 profiles as FPS evidence.

## Scope

- Add CLI/report gates for `--edgetam-component-runtime trt`, `--edgetam-trt-engine-dir`, `--edgetam-trt-report`, and `--edgetam-trt-scope memory_path_all`.
- Require the external BatchTam report to mark closed-loop TRT correctness and Demo 2.2 integration allowed.
- Pass TRT runtime settings into the external `hf_batched_multisession` scheduler without fallback.
- Record BatchTam runtime fields in Demo 2.2 profile JSON.
- Generate a final comparison report after no-render and pointcloud profiles.

## Validation

- `python -m py_compile demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py demo_v2_2/runtime.py demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `python -m unittest -v tests.test_demo22_batchtam_trt_gate tests.test_demo22_final_profile_source`
- `python scripts/harness/check_all.py`
- Demo 2.2 no-render and pointcloud profile commands with BatchTam TRT flags.
