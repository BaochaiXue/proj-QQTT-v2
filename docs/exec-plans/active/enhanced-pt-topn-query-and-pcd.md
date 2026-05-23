# Enhanced PT Top-N Query And PCD

## Goal

Upgrade the existing 3D enhanced PhysTwin-style PCD filter from largest-component
plus near-main retention to class-specific top-N 3D connected-component retention.
The same survivor semantics should feed tracker query masks and rendered/fused
semantic PCD surfaces.

## Scope

- Keep enhanced PT as a 3D voxel connected-component filter.
- Add component policies: `main-plus-gap`, `largest-n`, and
  `largest-n-plus-gap`.
- Default object top-N to 1 and controller top-N to 2 for Demo 3.x.
- Ensure controller `<5000` trackable cap happens after component filtering.
- Apply enhanced component filtering to PCD before render density controls.
- Surface contract/profile stats for component selection and removed points.
- Add deterministic tests and a small profile harness.

## Out Of Scope

- Tracker backend model changes.
- FFS inference changes.
- Lowering query count or image size as a quality tradeoff.
- Replacing 3D voxel components with 2D connected components.

## Validation

- Focused unit tests for enhanced PT policy and Demo 3.x trackable masks.
- `py_compile` for changed runtime modules.
- `python -m unittest -v` for affected tests.
- `scripts/harness/check_all.py` before completion.

## Progress

- Added class-specific enhanced PT component policy in `qqtt/demo/pcd_postprocess.py`.
- Added reusable semantic surface survivor helper in `qqtt/demo/semantic_surface_filter.py`.
- Wired object top-1 and controller top-2 enhanced PT defaults into Demo 3.x query masks.
- Moved controller trackable cap after enhanced PT top-N component filtering.
- Applied enhanced PT top-N filtering to rendered/fused semantic PCD before density controls.
- Added CLI and dry-run/profile contract fields for top-N policy, min thresholds, and PCD filtering.
- Added deterministic tests for component policies, controller cap order, and Demo 3.x contract forwarding.
- Added `scripts/harness/profile_enhanced_pt_topn_surface_filter.py`.

## Validation Results

- PASS: `conda run --no-capture-output -n demo_3_1_max python -m unittest -v tests.test_enhanced_pt_topn_surface_filter tests.test_demo32_trackable_mask_filter tests.test_demo31_dual_gpu_contract`
- PASS: `conda run --no-capture-output -n demo_3_1_max python -m unittest -v tests.test_profile_schema`
- PASS: `conda run --no-capture-output -n demo_3_1_max python scripts/harness/check_harness_catalog.py`
- PASS: `conda run --no-capture-output -n demo_3_1_max python scripts/harness/profile_enhanced_pt_topn_surface_filter.py --repeats 1`
- PASS: `conda run --no-capture-output -n demo_3_1_max python scripts/harness/check_all.py`
