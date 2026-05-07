# 2026-05-06 Demo 2.1 Camera Startup Stability

Status: implemented; deterministic validation pending full quick profile.

## Goal

Make Demo 2.1 tolerate RealSense startup transients after USBIP attach:

- initial frame bursts must not kill a `SingleRealsense` process
- temporarily not-ready multi-camera capture should skip groups instead of
  marking Demo 2.1 fatal

## Non-Goals

- no change to camera defaults
- no change to formal recording/alignment semantics
- no fake hardware success
- no fallback from FFS to native RealSense depth

## Plan

1. Catch `SharedMemoryRingBuffer.put(... wait=False)` startup burst failures in
   `SingleRealsense` and retry with `wait=True` instead of letting the child
   process die.
2. In Demo 2.1 capture grouping, check `camera_system.realsense.is_ready`
   before `get_observation()` and skip a tick while cameras restart.
3. Validate deterministic tests.
4. Retry the cloth-controller 30s live sanity run.

## Current Result

- The startup burst path was patched to throttle the shared-memory ring write
  instead of killing the `SingleRealsense` child process.
- Demo 2.1 capture grouping now skips ticks while a real `CameraSystem`
  temporarily reports not-ready.
- Targeted unit tests passed.
- The cloth-controller sanity progressed past camera startup and into SAM3.1
  live initialization. It failed correctly on the semantic init contract:
  cam2 initialized, but cam0 did not produce a `cloth` controller mask.
