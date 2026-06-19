# Demo 3.x Fake-Live Frame Dropping

## Goal

Make Demo 3.x fake-live preserve the original recording time flow when a lower
`--replay-fps` is requested. For a 30fps recording replayed at 5fps, fake-live
should emit about every sixth source frame instead of playing every source frame
at 5fps.

## Planned Changes

- Keep `--replay-fps` as the fake-live output cadence.
- Add recording timeline FPS metadata to `RecordedRgbdFrameSource`.
- Add source-frame lookup by original recording elapsed time.
- Change only `input_source=fake-live` capture scheduling to drop source frames
  while preserving contiguous runtime sequence numbers.
- Keep `input_source=recording` replay sequential.
- Update runtime metadata/help/contract wording and tests.

## Validation

- Run focused replay/runtime unit tests:
  `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_recorded_rgbd_replay_source.py tests/test_single_demo_v3_runtime.py tests/test_single_demo_tapnextpp_overlay.py`
- Run smoke validation:
  `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`

## Status

- Implemented and validated.
- Focused replay/runtime unit tests: `108 passed`.
- Smoke validation: `smoke checks passed`.
