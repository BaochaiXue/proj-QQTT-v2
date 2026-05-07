# Demo 2.1 15 FPS Capture Presets

Status: active

## Goal

Relax Demo 2.1 three-camera live capture presets from `848x480@30` to
`848x480@15` per camera so hardware startup and temporal grouping have more
headroom.

## Scope

- Change Demo 2.1 preset defaults only; explicit `--fps` and no-preset CLI
  behavior remain available.
- Keep FFS official depth, EdgeTAM streaming, SAM3.1 live init, semantic
  filtering, and render/fusion quality contracts unchanged.
- Update dry-run contract tests and operator-facing docs.

## Validation

- `python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke`
