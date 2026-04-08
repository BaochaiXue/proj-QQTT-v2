# Unify Depth Colorization

## Goal

Make live RealSense preview depth colorization visually consistent with the repo's aligned-case depth diagnostics.

## Problem

- `cameras_viewer.py` still uses `COLORMAP_JET` with a raw `convertScaleAbs(alpha=0.03)` path.
- panel and comparison diagnostics use a metric-depth-aware `COLORMAP_TURBO` path.
- this makes it harder to compare what users see in preview versus later diagnostic outputs.

## Plan

1. Add a small shared depth-colorization helper with:
   - metric-depth input
   - `TURBO` colormap
   - explicit invalid color
   - shared default display range
2. Update `cameras_viewer.py` to use the shared helper and expose preview depth-range flags.
3. Update other lightweight RealSense preview-style utilities that still use ad hoc depth colorization.
4. Add a software-only regression test to keep the shared helper and diagnostic panel path aligned.
5. Update user-facing docs and run deterministic checks.
