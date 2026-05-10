# Demo 2.2 CLI Simplification

## Goal

Make the Demo 2.2 public command line small and readable without removing the underlying Demo 2.1 runtime flags that older benchmark scripts may still use.

## Plan

1. Add a thin Demo 2.2-specific parser in `demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py`.
2. Expose only daily-operation flags: duration/warmup/profile output, prompts, camera ids/serials, calibration path, FPS, debug/dry-run, and a small set of explicit experimental toggles.
3. Keep unknown/legacy flags as passthrough to the Demo 2.1 parser so existing commands keep working.
4. Rename confusing public concepts at the wrapper layer:
   - `--warmup-s` -> `--profile-warmup-exclude-s`
   - `--min-depth-m` / `--max-depth-m` -> `--depth-min-m` / `--depth-max-m`
   - `--object-only` / `--controller-object` -> `--track-mode`
   - `--experimental-edgetam-batch-vision` -> `--edgetam-batch-vision-encoder`
5. Add `--advanced-help` for the full underlying Demo 2.1 runtime help.
6. Update deterministic tests and generated docs.
7. Run targeted tests and `scripts/harness/check_all.py`.

## Result

- Added a Demo 2.2-specific public parser.
- Kept unknown legacy flags as passthrough to the Demo 2.1 runtime.
- Added clearer public aliases for warmup, depth limits, track mode, and explicit experiments.
- Added generated documentation at `docs/generated/demo2_2_cli_simplification.md`.
- Updated Demo 2.2 smoke tests for the reduced help surface and passthrough compatibility.

## Validation

```bash
conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke
conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py
```

Outcome:

```text
Demo 2.2 targeted smoke: 19 tests passed
check_all quick: 157 tests passed
```
