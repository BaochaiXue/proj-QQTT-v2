# Demo 3.x Fake Live Camera

## Goal

Make Demo 3 / 3.1 / 3.2 / 3.3 use the `demo_v3*` public entrypoints on the
single-camera branch and support a shared fake-live camera mode backed by a
single-camera `data_collect` recording.

## Scope

- Replace stale three-view `demo_v3*` wrappers with single-camera wrappers
  around `qqtt.demo.single_demo_v3_runtime`.
- Add `--input-source fake-live` while keeping `recording` as a compatibility
  alias.
- Use `data_collect/sloth_both_eval_2min_e45_g35_20260614_155543` as the
  default fake-live case.
- Replay the first complete recorded step as runtime `seq=0`, wait for
  first-frame segmentation, then emit later frames at metadata or CLI FPS.
- Support RealSense-native fake-live from RGB-D and FFS fake-live from IR
  stereo plus calibration.

## Validation

- Unit tests for RGB-D and IR fake-live packets, source pacing, contracts, and
  path guards.
- Focused command:
  `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_recorded_rgbd_replay_source tests.test_single_demo_v3_runtime tests.test_check_all_smoke`
- Default command:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- Full command:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py --full`
