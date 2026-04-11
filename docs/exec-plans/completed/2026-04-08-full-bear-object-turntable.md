# Full-Bear Object Turntable

## Goal

Relax the object-only render bounds so the professor-facing turntable keeps the full teddy-bear silhouette, especially the head, while still excluding most non-object clutter.

## Problem

The current object-only workflow still applies a second tight 3D render crop inside `build_single_frame_scene()`. That crop is too aggressive for the real teddy+box example and trims the upper part of the bear.

## Plan

1. Broaden the render bounds used for the main object-only render, instead of recentering to a tight XY/Z box.
2. Make the render filtering respect the configured object-height band.
3. Add a regression assertion that render bounds cover the object ROI in the turntable smoke test.
4. Re-render the real `native_30_static` vs `ffs_30_static` full-bear output and rerun deterministic checks.
