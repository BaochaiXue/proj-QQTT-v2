# Demo 3.x Render Point Cap

## Goal
Limit the final Open3D visualization load for Single Demo 3.1/3.2/3.3 by
showing at most 5000 object points and 5000 controller points per rendered
frame.

## Plan
1. Add a render-only per-layer point cap to the masked PCD delegate.
2. Apply the cap immediately before Open3D geometry updates so upstream PCD
   generation, filtering, logs, and tracker inputs keep their existing data.
3. Expose and forward the cap through the Single Demo 3.x launcher.
4. Update demo READMEs and deterministic tests.
5. Run targeted tests and `scripts/harness/check_all.py`.

## Validation
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`

## Outcome
- Added `--render-max-points-per-layer` with default `5000`.
- Applied the cap only in the final Open3D semantic layer updates, so each
  frame displays at most 5000 controller points and 5000 object points while
  upstream PCD packets keep their original counts.
- Forwarded the setting through Single Demo 3.1/3.2/3.3 public launchers and
  documented the default in their READMEs.
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_v3_runtime`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_tapnextpp_overlay`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_render_fastpath`
- PASS: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
