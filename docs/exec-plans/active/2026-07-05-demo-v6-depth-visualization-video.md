# Demo v6 Depth Visualization Video

## Requirement

Render Demo v6 online depth frames with a RealSense-style visualization.
The full-frame depth view must use RealSense Dynamic Jet rather than a
controller-crop quantile range. The per-controller crop diagnostics can keep
their ROI-specific range separately.

## Scope

- Add one helper script under `demo_v6/others/`.
- Read Demo v6 `online_data` depth frames and frame mapping.
- Write a full-frame depth MP4.
- Rewrite the first-window full-depth diagnostic grid using the same full-frame
  RealSense Dynamic Jet visualization.

## Validation

- Compile the helper script.
- Run it on the current `outputs_v6/online_data`.
- Verify the produced MP4 opens, has the expected frame count, FPS, and
  dimensions.
