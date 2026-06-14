# Demo 3.1 Open3D VRAM Stability

## Goal
Reduce Open3D GUI VRAM spikes in Single Demo 3.1 when the user rotates the live
masked point cloud.

## Plan
1. Confirm the Demo 3.1 entrypoint delegates rendering to
   `qqtt.demo.realtime_masked_edgetam_pcd`.
2. Replace the local per-frame tensor rebinding Open3D geometry state with the
   shared inplace tensor render layer.
3. Expose and forward practical point-cloud load controls from the public
   Single Demo 3.x launcher.
4. Add deterministic tests for the new CLI contract and delegate argv.
5. Run targeted unit tests and the deterministic harness.

## Validation
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_v3_runtime`
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_tapnextpp_overlay`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`

## Outcome
- Replaced the Demo 3.1 Open3D GUI geometry state with the shared inplace
  tensor render layer.
- The inplace render layer now keeps a capacity buffer so point-count jitter no
  longer forces geometry recreation; Demo 3.1 seeds controller/object capacity
  from `--pcd-max-points` and tracker capacity from the overlay cap.
- Exposed public Single Demo 3.x point-cloud load controls:
  `--pcd-max-points`, `--pcd-stride`, `--depth-min-m`, `--depth-max-m`,
  `--pcd-color-mode`, and `--enable-pcd-filter`.
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_render_fastpath`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay`
- PASS: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
