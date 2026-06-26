# Demo v5 Camera Duration Warmup Fix

## Goal

Keep the RGB input timeline live during Demo v5 warmup and shape-prior waiting.

## Root Cause

The Demo v5 runner converted `--max-chunks` into a fixed camera subprocess
`--duration-s`. That duration started at camera launch, so shape-prior warmup
time consumed the fake/live camera input budget before enough `final_data`
chunks could be published.

## Fix

- Keep the camera subprocess duration unbounded from the Demo v5 runner.
- Let `stream_chunks_from_headless_capture(...)` enforce `max_chunks`.
- Stop the camera subprocess from the runner once the chunk publisher returns.

## Validation

- Add a regression test that `--max-chunks` no longer generates a bounded camera
  `--duration-s`.
- Run Demo v5 focused tests.
- Run the smoke validation profile.
