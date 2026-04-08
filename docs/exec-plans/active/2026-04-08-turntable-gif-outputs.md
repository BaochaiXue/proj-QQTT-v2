# Turntable GIF Outputs

## Goal

Make the turntable comparison workflow emit GIF versions alongside the existing mp4 outputs for geom, rgb, and support renders.

## Scope

- reuse the existing per-frame PNG sequence
- add a shared GIF writer
- expose GIF output through the turntable CLI/workflow
- generate GIF artifacts for the current professor-facing outputs

## Plan

1. Add a deterministic `write_gif()` helper using the already written frame PNGs.
2. Extend the turntable workflow to optionally emit `orbit_compare_*.gif` files and record them in metadata.
3. Update smoke tests to expect `gif_path` fields when GIF export is enabled.
4. Generate GIFs for the latest full-bear turntable output and rerun deterministic checks.
