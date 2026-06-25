# Demo v4 35-Frame Online Window Plan

Goal: make Demo v4 chunks and realtime_phystwin online segment defaults use the same 35-frame window.

Steps:
- [x] Add failing coverage for Demo v4 default chunk frame count and realtime_phystwin online parser defaults.
- [x] Change Demo v4 default chunk frame count to 35 frames while preserving explicit `--chunk-frame-count` and legacy `--chunk-seconds` override behavior.
- [x] Change realtime_phystwin online CMA/train wrapper defaults to `segment_len=35`.
- [x] Run focused tests and syntax checks.

Validation:
- RED: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks.FuturePhysTwinChunkWriterTest.test_demo_v4_parser_defaults_to_fake_live_35_frame_chunks_and_shape_prior tests.test_realtime_phystwin_online_defaults`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks tests.test_realtime_phystwin_online_defaults`
- PASS: `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v4/realtime_futurephystwin_chunks.py realtime_phystwin/optimize_online_cma.py realtime_phystwin/train_online_warp.py realtime_phystwin/train_online_zero_then_first.py tests/test_realtime_phystwin_online_defaults.py tests/test_demo_v4_futurephystwin_chunks.py`
- PASS: `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
