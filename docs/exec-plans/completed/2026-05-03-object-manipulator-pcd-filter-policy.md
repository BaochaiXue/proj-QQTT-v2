# Object/Manipulator PCD Filter Policy

## Goal

Make the Sloth Set 2 object/controller/hand PCD GIF renderer consistently use:

- `enhanced-pt` for object rows.
- `pt-filter` for controller/hand manipulator rows.

## Scope

- Update the row-level PCD postprocess selection in `run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py`.
- Keep the existing `--controller-pcd-postprocess-mode` flag compatible, but document it as applying to controller/hand rows.
- Add smoke coverage for `controller`, `hand`, `left hand`, and `right hand` labels.
- Refresh report wording so generated docs do not imply the rule is controller-only.

## Validation

- `python -m py_compile scripts/harness/experiments/run_sloth_set2_hf_edgetam_hand_object_pcd_gif.py`
- Targeted smoke test covering the renderer postprocess policy.
- `python scripts/harness/check_harness_catalog.py`

## Outcome

- Added semantic row selection: `enhanced-pt` remains the object-row default, while `controller`, `hand`, `left hand`, and `right hand` rows fall back to `pt-filter` when the global mode is `enhanced-pt`.
- Kept `--controller-pcd-postprocess-mode` as a backward-compatible explicit override for controller/hand rows.
- Refreshed the SAM3.1 object/controller PCD report wording to describe controller/hand effective modes.
- Validation passed in `FFS-SAM-RS`; base Python `check_all.py` still lacks `cv2`, so the default check must be run from a repo-capable env.
