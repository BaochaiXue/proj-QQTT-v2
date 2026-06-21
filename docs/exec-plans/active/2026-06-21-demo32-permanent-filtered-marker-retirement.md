# Demo 3.2 Permanent Filtered Marker Retirement

**Goal:** Make Demo 3.2 tracking markers default to monotonic filtered-residual/table-Z retirement, with visible remaining-query counts.

**Plan**

1. Add default-on CLI and Demo 3.x contract metadata for `tracker_retire_filtered_markers`.
2. Add failing tests for retirement persistence, disabled compatibility, non-retirement on TAPNext++ invisibility, overlay-cap independence, and remaining-count breakdowns.
3. Maintain a full-query alive mask in the tracker marker path; update it only from the active PCD residual/table-Z gate and apply it before marker selection/lift.
4. Save alive/remaining fields in headless trajectory payloads and frame metadata, while keeping old captures renderable.
5. Add a top-left remaining-query legend to the Open3D panel HUD and 2D side-by-side renderer.
6. Update Demo 3.2 docs and run focused unit tests plus smoke validation.

**Validation**

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay tests.test_demo32_headless_render_helper`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`

**Notes**

- Retirement means failing the PCD residual/table-Z marker gate, not tracker model invisibility, occlusion, display scope, or overlay cap.
- `original` still participates in table-Z retirement when table-Z filtering is enabled.
