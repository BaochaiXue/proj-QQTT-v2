# Demo 3.2 Mask-Cleaned Offline Tracking Render Design

## Summary

Demo 3.2 headless offline `tracking` render should remove table and background
distractions while preserving the PhysTwin-style tracker view. The rendered
video keeps the current `RGB frame + stable rainbow query points` layout, but
the RGB frame is masked by the same-frame EdgeTAM target masks before query
points are drawn.

The default cleanup policy is:

```text
clean_mask = object_mask | controller_mask
```

Pixels outside `clean_mask` are set to black. This keeps the object, the
controller/hands, and the tracking points, while removing unrelated table and
background regions.

## Goals

- Make offline tracking videos look closer to the clean pointcloud/overlay
  results by removing background clutter.
- Preserve the PhysTwin-like tracking visualization: RGB target appearance plus
  stable `gist_rainbow` query dots, with no trajectory lines.
- Keep exact same-seq behavior. The RGB frame, masks, and query trajectory must
  all correspond to the same saved frame.
- Leave offline `pcd` rendering unchanged.

## Non-Goals

- Do not change live Open3D rendering in this step.
- Do not change TAPNext++ query selection, query identity colors, EdgeTAM mask
  propagation, or headless capture generation.
- Do not add refined erode/dilate mask cleanup yet. The first version uses the
  raw saved union mask so the behavior is easy to inspect.
- Do not fall back to older masks or older query trajectories.

## Public Interface

Update `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py`:

- Add `--tracking-background-mask {target-union,rgb}`.
- Default: `target-union`.
- `target-union` applies `object_mask | controller_mask` to the RGB background
  in `--demo-visual-mode tracking`.
- `rgb` preserves the old full-RGB tracking render for comparison.

The existing `--demo-visual-mode pcd` path ignores this option because it does
not use RGB-frame tracking background rendering.

## Data Flow

For each row in `frames.jsonl` during `tracking` render:

1. Load `rgb_path` as the background image.
2. Load `mask_path` from the same frame row.
3. Read `object_mask` and `controller_mask`.
4. Validate mask shape matches the output image size.
5. Compute `clean_mask = object_mask | controller_mask`.
6. Set all background pixels outside `clean_mask` to black.
7. Load the exact same-seq query trajectory.
8. Draw current visible query dots using stored `marker_rgb_u8` or the existing
   PhysTwin color fallback.

No nearest-previous fallback is allowed for masks or query trajectories.

## Error Handling

- If `--demo-visual-mode tracking --tracking-background-mask target-union` is
  used and `mask_path` is missing from `frames.jsonl`, raise a clear error.
- If the mask file does not exist, raise a clear error.
- If `object_mask` or `controller_mask` is missing from the mask payload, raise
  a clear error.
- If mask dimensions do not match the render dimensions, raise a clear error.

Fail-fast behavior is intentional because headless capture is expected to save
mask artifacts. Silent fallback would hide incomplete captures and make visual
debugging misleading.

## Summary Output

The render summary JSON should include:

- `tracking_background_mask`: `target-union` or `rgb`
- `tracking_background_mask_source`: `object_mask|controller_mask` for
  `target-union`, otherwise `full_rgb`
- `missing_query_frames`: existing exact-query count remains unchanged
- per-frame `tracking_background_mask_pixels` when a mask is applied
- total `tracking_background_mask_pixels` across rendered frames

This makes it easy to verify that the video was actually rendered with target
mask cleanup.

## Testing

Add or update tests for the offline render helper:

- `tracking + target-union` blackens pixels outside `object_mask |
  controller_mask`.
- Query dots are still drawn after masking and can appear on retained target
  regions.
- `tracking + rgb` preserves the old full-RGB background behavior.
- `pcd` mode is unaffected by `--tracking-background-mask`.
- Missing `mask_path`, missing mask file, missing mask arrays, and mismatched
  mask dimensions fail with clear errors.
- Render summary records the selected mask policy and mask pixel counts.

## Validation

Run focused tests first:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo32_headless_render_helper
```

Then run the smoke validation profile:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

For a visual check, render a saved headless capture twice:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_query_masked.mp4 \
  --fps 30 \
  --demo-visual-mode tracking
```

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_query_full_rgb_compare.mp4 \
  --fps 30 \
  --demo-visual-mode tracking \
  --tracking-background-mask rgb
```

## Acceptance Criteria

- The default offline tracking render removes table/background pixels using the
  saved same-frame target union mask.
- The object, controller/hands, and tracking query dots remain visible.
- The old full-RGB tracking render remains available through an explicit CLI
  option.
- PCD-only offline rendering output does not change.
- Tests and smoke validation pass.
