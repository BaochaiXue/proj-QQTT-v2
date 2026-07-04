# Demo v5.1 Live Latest Sampling

## Requirement

Problem:
Live RealSense capture can deliver frames around the intended 5 FPS cadence
with jitter, for example one frame after 0.18 seconds and the next after
0.21 seconds. When the demo treats each arrived frame as the next formal
frame, this arrival jitter becomes demo timeline jitter.

Required final behavior:
Run the real camera source at 30 FPS by default. When Demo v5.1 is configured
for a 5 FPS output timeline, publish one live frame every 0.2 seconds by
sampling the latest RealSense frame observed at that cadence. The downstream
lossless tracker/PCD/chunk pipeline still receives a strict 5 FPS sequence.

Inputs:
RealSense live color/depth or color/IR frames, `--fps`, `--replay-fps`, and
`--lossless-input-fps`.

Outputs:
Prepared Demo v5.1 headless capture frames whose sequence cadence is governed
by the output FPS, while the camera hardware is free to capture faster.

State changes:
The Demo v5.1 default camera capture FPS becomes 30. Live capture publication
uses fixed-cadence latest-frame sampling when strict lossless mode is active.

Invalid cases:
Non-positive FPS values continue to fail during argument validation. RealSense
frame acquisition failures continue to fail in the capture worker.

Constraints:
Stay on `single-camera`. Preserve fake-live replay semantics and existing
formal chunk contracts. Do not add fallback or compatibility paths.

Unknowns:
No correctness-blocking unknowns. The user asked whether the design is
feasible; implementation will validate the deterministic scheduling behavior
without requiring hardware.

## Plan

- [x] Confirm branch and note that `git pull --ff-only origin main` cannot
  fast-forward the current `single-camera` branch.
- [x] Add a small live cadence sampler that emits fixed output ticks from the
  latest captured RealSense packet.
- [x] Change the Demo v5.1 default camera FPS to 30 while keeping replay FPS at
  5.
- [x] Update docs/tests that describe the live 30 FPS input versus 5 FPS output
  cadence.
- [x] Run focused unit tests and the smoke validation profile.

## Validation

Branch/setup:

- Confirmed current branch is `single-camera`.
- `git pull --ff-only origin main` failed because `single-camera` cannot
  fast-forward to `origin/main`; no destructive sync was attempted.

Focused tests:

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v5_1_default_config.py -q`
- Result: passed, `16` tests and `80` subtests.

Smoke validation:

- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
- Result: passed, `103` tests.
