## Goal

Add a visualization-only semantic-world display transform so professor-facing 3D compare outputs render in a human-intuitive table-up / cameras-above frame.

## Non-Goals

- No `calibrate.pkl` format changes
- No calibration producer changes
- No downstream calibration semantic changes
- No new broad visualization package
- No 2D image-flip hack as the main fix

## Files To Modify

- `data_process/visualization/semantic_world.py`
- `data_process/visualization/turntable_compare.py`
- `data_process/visualization/stereo_audit.py`
- `scripts/harness/visual_compare_turntable.py`
- `scripts/harness/visual_compare_stereo_order_pcd.py`
- targeted tests/docs

## Plan

1. Infer `T_semantic_from_calibration` from the fitted tabletop plane plus camera centers.
2. Apply it in memory only to visualization scene data, camera poses, crop bounds, and view planning.
3. Default professor-facing turntable and stereo-order board outputs to `semantic_world`, with an explicit `--display_frame` override.
4. Keep calibration-world debug outputs alongside semantic-world debug outputs.

## Validation

- Add synthetic semantic-world inference tests
- Update workflow/frame-contract tests
- Run `python scripts/harness/check_all.py`

## Acceptance Criteria

- `calibrate.pkl` semantics unchanged
- professor-facing defaults use `semantic_world`
- top/front/side views are rebuilt in semantic world
- calibration and semantic debug overviews both exist
