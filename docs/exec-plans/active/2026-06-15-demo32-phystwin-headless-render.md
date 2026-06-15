# Demo 3.2 PhysTwin Headless Render

## Goal

Make Demo 3.2 headless output produce two separate videos: enhanced-pt filtered PCD only, and PhysTwin-style query tracking only.

## Planned Changes

- Save RGB frames in headless capture artifacts and reference them from `frames.jsonl`.
- Use PhysTwin-style stable query colors: `gist_rainbow` keyed by each query point's initial y coordinate.
- Render `pcd` mode as filtered PCD only.
- Render `tracking` mode as RGB background plus stable rainbow query points only, without PCD.
- Keep strict exact-seq query matching and write per-output render summaries.

## Validation

- Updated unit tests for headless capture payloads, PhysTwin colors, and offline render semantics.
- Passed targeted unittest coverage:
  `python -m unittest tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay tests.test_demo32_headless_render_helper`
- Passed `scripts/harness/check_all.py` and `scripts/harness/check_all.py --full`.
- Ran a full headless capture and rendered both requested videos:
  `video_pcd_only.mp4` and `video_query_phystwin.mp4`.
