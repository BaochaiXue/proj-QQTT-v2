# Runtime Cleanup And Demo 3.1 Warmup

## Goal

Fix the observed Demo 2.3 shutdown rough edge after a successful render run and apply the same startup-overlap idea to Demo 3.1.

## Scope

- Add an explicit `CameraSystem.stop()` cleanup path that stops RealSense workers and releases the shared-memory manager.
- Make Demo runtime shutdown call the camera system cleanup instead of only stopping the nested RealSense object.
- Narrow the RealSense timeout/restart race during shutdown.
- Prewarm Demo 3.1 CoTracker backends inside the isolated GPU1 process before the main shared runtime starts camera capture.
- Record Demo 3.1 CoTracker process ready/warmup timing in snapshots/profile output.

## Validation

The user asked not to use smoke tests for this flow. Validate with the live render demo path after changes.
