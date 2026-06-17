# Demo 3.2 Fake-Live Warmup Clock

## Goal
- During fake-live first-frame warmup, treat the recorded camera as still running at replay FPS.
- Keep runtime `seq` continuous for lossless queues, but advance the recorded source frame by elapsed camera time.
- Leave compatibility `recording` replay behavior unchanged.

## Implementation
- Add a `frame_index` override to `RecordedRgbdFrameSource.read_packet()` so runtime seq and source frame index can differ.
- In `_capture_recording_worker()`, start fake-live camera time when frame 0 is published.
- After the first-frame gate opens, compute the next source frame from elapsed time and continue pacing against camera start time.
- Use source frame index for fake-live duration limits, while leaving recording duration based on runtime seq.

## Validation
- Add unit coverage that fake-live skips source frames during the first-frame gate but still publishes continuous runtime seq.
- Keep existing recording replay tests proving compatibility behavior is unchanged.
