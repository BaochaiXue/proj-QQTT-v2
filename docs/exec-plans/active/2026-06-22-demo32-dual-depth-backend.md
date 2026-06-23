# Demo 3.2 Dual Depth Backend

## Goal

Expose a Demo 3.2 user-facing depth backend switch so each run can choose
between the current IR stereo FFS path and native RealSense color-aligned depth,
while preserving IR-FFS as the default behavior.

## Guardrails

- Only Demo 3.2 exposes `--depth-backend {ir-ffs,native-realsense}`.
- Keep `--depth-source` hidden at the Demo 3.x wrapper layer.
- Do not change Demo 3.1 or Demo 3.3 public depth behavior.
- Do not add automatic fallback between depth backends.
- Keep EdgeTAM, TAPNext++, query identity, table-world conversion, PCD filters,
  and PhysTwin-like finalization shared downstream of selected depth.
- Keep old headless captures with `ffs_depth_path` readable.

## Implementation Steps

- Add Demo 3.2 backend constants, parser option, and a resolver that maps
  `ir-ffs` to internal `ffs` and `native-realsense` to internal `realsense`.
- Use the resolver in preset defaults, validation, contract generation, and
  delegate argv construction.
- Reject explicitly supplied FFS model/path arguments when Demo 3.2 is run with
  `--depth-backend native-realsense`.
- Relax runtime validation so panel and fake-live headless capture allow both
  internal `ffs` and `realsense` depth sources.
- Rename new headless depth artifacts to `depth_color_m/<seq>.npy` with
  `depth_color_m_path` rows, while retaining compatibility with old
  `ffs_depth_path` rows.
- Update the PhysTwin strict finalizer to consume the canonical depth path and
  emit public/internal depth backend manifest fields; confirm the offline
  headless renderer remains path-agnostic because it consumes saved PCD, mask,
  RGB, and query artifacts rather than depth rows directly.
- Update Demo 3.2 documentation and focused unit tests.

## Validation

```bash
conda run -n demo_2_max --no-capture-output python -m unittest \
  tests.test_single_demo_v3_runtime \
  tests.test_single_demo_tapnextpp_overlay \
  tests.test_demo32_headless_render_helper \
  tests.test_phystwin_strict_product

conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```
