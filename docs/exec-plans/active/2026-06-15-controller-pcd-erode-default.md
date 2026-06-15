# Controller PCD Erosion Default

## Goal

Stop Demo 3.2/3.3 from applying the 3px FFS surface mask erosion to the
controller by default. The object can keep the existing default erosion for
edge cleanup, but controller masks should preserve small hand regions unless the
operator explicitly asks otherwise.

## Implementation Notes

- Add object/controller-specific PCD mask erosion controls while keeping the
  existing `--pcd-mask-erode-pixels` as the legacy common fallback.
- For FFS filtered Demo 3.2/3.3 defaults, set object erosion to 3 and controller
  erosion to 0 when no explicit erosion option is supplied.
- Preserve legacy behavior when the user explicitly passes
  `--pcd-mask-erode-pixels`: both object and controller inherit that value
  unless a per-class value is also provided.
- Record effective object/controller erosion values in dry-run contracts,
  runtime metadata, and headless mask artifacts.

## Validation

- Update runtime and writer unit tests for the new defaults/schema.
- Run focused demo runtime/filter/writer tests and the quick harness checks.
