# Camera-Visible Turntable Video

## Goal

Make the 3 real camera positions clearly visible inside each professor-facing object-only turntable video frame without changing the object-only fusion semantics.

## Why

The current object-only workflow is conceptually correct, but the right-bottom overview inset is still too small. The camera frusta and labels are present, yet not readable enough in the exported mp4.

## Plan

1. Increase the rendered overview resolution and the side-by-side footer allocation used by the main turntable board.
2. Keep the existing object-only cloud, coverage-aware orbit, and Native/FFS synchronization logic unchanged.
3. Update overview-related smoke tests if the new layout changes expected dimensions.
4. Re-render the latest object-only professor-facing comparison with the larger overview strip.
5. Re-run deterministic checks.
