# Sloth Set 2 HF EdgeTAM Object + Two Hands 3x3 PCD GIF

## Goal

Generate a 3x3 fused PCD GIF for `sloth_set_2_motion_ffs` that tracks `stuffed animal`, `left hand`, and `right hand` as separate HF EdgeTAM objects.

## Scope

- Keep the existing 2x3 `stuffed animal` + unioned `hand` outputs unchanged.
- Add a three-object mode or wrapper around the current HF EdgeTAM hand/object experiment.
- Use frame-by-frame PNG streaming for EdgeTAM, not offline video input.
- Use a canonical SAM3.1 frame-0 init root whose mask schema is `1=stuffed animal`, `2=left hand`, `3=right hand`.
- Render a 3-row by 3-column fused PCD GIF, with optional enhanced-PT postprocess available.

## Validation

- Add focused tests for the three-object mapping and 3x3 panel output.
- Run catalog and focused tests.
- Run `scripts/harness/check_all.py`.

## Outcome

- Extended the Sloth Set 2 HF EdgeTAM hand/object experiment with a three-object `--track-two-hands` mode:
  - `1=stuffed animal`
  - `2=left hand`
  - `3=right hand`
- Added a canonical frame-0 init-root builder that reuses existing stuffed-animal SAM3.1 masks and the two raw SAM3.1 hand instances, splitting the hands by frame-0 image x-centroid per camera.
- Ran real HF EdgeTAM frame-by-frame PNG streaming over `93` frames for `cam0/1/2` with `vision-reduce-overhead`.
- Generated raw and enhanced-PT 3x3 fused PCD GIF panels under `result/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd/`.
- Wrote generated reports:
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_streaming_results.json`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_benchmark.md`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_results.json`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_enhanced_pt_benchmark.md`
  - `docs/generated/sloth_set_2_motion_ffs_hf_edgetam_object_two_hands_pcd_enhanced_pt_results.json`
- Validation passed:
  - `python -m py_compile scripts/harness/experiments/run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py`
  - focused three-object mapping/init-root/3x3 panel tests
  - `python scripts/harness/check_harness_catalog.py`
  - `python scripts/harness/check_all.py`
