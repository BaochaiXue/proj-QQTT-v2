# D455 Stream Probe Results

Observed results are authoritative for this machine. Official docs define expectations, but support is claimed only when the probe passed.

## Expected From Docs

- `Projection in RealSense SDK 2.0`: Each stream has its own pixel and 3D coordinate system, and extrinsics are rigid transforms in meters between stream coordinate frames. (https://dev.realsenseai.com/docs/projection-in-intel-realsense-sdk-20)
- `Intel RealSense D400 Series Datasheet / D455 product page`: D455 is a D400-family stereo device with a nominal 95 mm baseline and USB3-advertised 848x480 and 640x480 operating points. (https://www.realsenseai.com/wp-content/uploads/2023/10/Intel-RealSense-D400-Series-Datasheet-September-2023.pdf)
- `Fast-FoundationStereo README`: Stereo inputs should be rectified, undistorted, and use the true left and right images. (https://github.com/NVlabs/Fast-FoundationStereo)

## Summary

- Run id: `20260405T232134Z`
- Stable serial order: `239222300433, 239222300781, 239222303506`
- Total cases: `65`
- Passed: `42`
- Failed: `23`
- Primary recommendation: `C` - ir_pair is stable on all 3 cameras, but rgb_ir_pair is not yet stable as a same-take mode.
- Same-take comparison recommendation: `E` - Do not promise same-take depth-vs-FFS comparison yet; depth_ir_pair / rgbd_ir_pair remain unstable or unsupported.

## Single-Camera Results By Serial

### `239222300433`

| Stream Set | Resolution | Emitter | Status | Note |
| --- | --- | --- | --- | --- |
| depth | 848x480@30 | auto | pass | stable |
| color | 848x480@30 | auto | pass | stable |
| ir_left | 848x480@30 | on | pass | stable |
| ir_right | 848x480@30 | on | pass | stable |
| ir_pair | 848x480@30 | on | pass | stable |
| rgbd | 848x480@30 | auto | fail | Case started but failed delivery / fps / stall thresholds. |
| rgb_ir_pair | 848x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| depth_ir_pair | 848x480@30 | on | pass | stable |
| rgbd_ir_pair | 848x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| ir_left | 848x480@30 | off | pass | stable |
| ir_right | 848x480@30 | off | pass | stable |
| ir_pair | 848x480@30 | off | pass | stable |
| rgbd | 640x480@30 | auto | fail | Case started but failed delivery / fps / stall thresholds. |
| rgb_ir_pair | 640x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| depth_ir_pair | 848x480@30 | off | pass | stable |
| rgbd_ir_pair | 640x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |

### `239222300781`

| Stream Set | Resolution | Emitter | Status | Note |
| --- | --- | --- | --- | --- |
| depth | 848x480@30 | auto | pass | stable |
| color | 848x480@30 | auto | pass | stable |
| ir_left | 848x480@30 | on | pass | stable |
| ir_right | 848x480@30 | on | pass | stable |
| ir_pair | 848x480@30 | on | pass | stable |
| rgbd | 848x480@30 | auto | pass | stable |
| rgb_ir_pair | 848x480@30 | on | pass | stable |
| depth_ir_pair | 848x480@30 | on | pass | stable |
| rgbd_ir_pair | 848x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| ir_left | 848x480@30 | off | pass | stable |
| ir_right | 848x480@30 | off | pass | stable |
| ir_pair | 848x480@30 | off | pass | stable |
| rgb_ir_pair | 848x480@30 | off | pass | stable |
| depth_ir_pair | 848x480@30 | off | pass | stable |
| rgbd_ir_pair | 640x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |

### `239222303506`

| Stream Set | Resolution | Emitter | Status | Note |
| --- | --- | --- | --- | --- |
| depth | 848x480@30 | auto | pass | stable |
| color | 848x480@30 | auto | pass | stable |
| ir_left | 848x480@30 | on | pass | stable |
| ir_right | 848x480@30 | on | pass | stable |
| ir_pair | 848x480@30 | on | pass | stable |
| rgbd | 848x480@30 | auto | pass | stable |
| rgb_ir_pair | 848x480@30 | on | pass | stable |
| depth_ir_pair | 848x480@30 | on | pass | stable |
| rgbd_ir_pair | 848x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| ir_left | 848x480@30 | off | pass | stable |
| ir_right | 848x480@30 | off | pass | stable |
| ir_pair | 848x480@30 | off | pass | stable |
| rgb_ir_pair | 848x480@30 | off | pass | stable |
| depth_ir_pair | 848x480@30 | off | pass | stable |
| rgbd_ir_pair | 640x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |

## Three-Camera Results

| Serials | Stream Set | Resolution | Emitter | Status | Note |
| --- | --- | --- | --- | --- | --- |
| 239222300433, 239222300781, 239222303506 | depth | 848x480@30 | auto | pass | stable |
| 239222300433, 239222300781, 239222303506 | color | 848x480@30 | auto | pass | stable |
| 239222300433, 239222300781, 239222303506 | ir_left | 848x480@30 | on | pass | stable |
| 239222300433, 239222300781, 239222303506 | ir_right | 848x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | ir_pair | 848x480@30 | on | pass | stable |
| 239222300433, 239222300781, 239222303506 | rgbd | 848x480@30 | auto | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | rgb_ir_pair | 848x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | depth_ir_pair | 848x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | rgbd_ir_pair | 848x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | ir_left | 848x480@30 | off | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | ir_right | 640x480@30 | on | pass | stable |
| 239222300433, 239222300781, 239222303506 | ir_pair | 848x480@30 | off | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | rgbd | 640x480@30 | auto | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | rgb_ir_pair | 640x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | depth_ir_pair | 640x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | rgbd_ir_pair | 640x480@30 | on | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | ir_left | 640x480@30 | off | fail | Case started but failed delivery / fps / stall thresholds. |
| 239222300433, 239222300781, 239222303506 | ir_right | 640x480@30 | off | pass | stable |
| 239222300433, 239222300781, 239222303506 | ir_pair | 640x480@30 | off | fail | Case started but failed delivery / fps / stall thresholds. |

## Stable Stream Sets

- `single-239222300433-depth-848x480-fps30-emitter-auto`: `depth` on `239222300433` at `848x480@30` with emitter `auto`
- `single-239222300433-color-848x480-fps30-emitter-auto`: `color` on `239222300433` at `848x480@30` with emitter `auto`
- `single-239222300433-ir_left-848x480-fps30-emitter-on`: `ir_left` on `239222300433` at `848x480@30` with emitter `on`
- `single-239222300433-ir_right-848x480-fps30-emitter-on`: `ir_right` on `239222300433` at `848x480@30` with emitter `on`
- `single-239222300433-ir_pair-848x480-fps30-emitter-on`: `ir_pair` on `239222300433` at `848x480@30` with emitter `on`
- `single-239222300433-depth_ir_pair-848x480-fps30-emitter-on`: `depth_ir_pair` on `239222300433` at `848x480@30` with emitter `on`
- `single-239222300781-depth-848x480-fps30-emitter-auto`: `depth` on `239222300781` at `848x480@30` with emitter `auto`
- `single-239222300781-color-848x480-fps30-emitter-auto`: `color` on `239222300781` at `848x480@30` with emitter `auto`
- `single-239222300781-ir_left-848x480-fps30-emitter-on`: `ir_left` on `239222300781` at `848x480@30` with emitter `on`
- `single-239222300781-ir_right-848x480-fps30-emitter-on`: `ir_right` on `239222300781` at `848x480@30` with emitter `on`
- `single-239222300781-ir_pair-848x480-fps30-emitter-on`: `ir_pair` on `239222300781` at `848x480@30` with emitter `on`
- `single-239222300781-rgbd-848x480-fps30-emitter-auto`: `rgbd` on `239222300781` at `848x480@30` with emitter `auto`
- `single-239222300781-rgb_ir_pair-848x480-fps30-emitter-on`: `rgb_ir_pair` on `239222300781` at `848x480@30` with emitter `on`
- `single-239222300781-depth_ir_pair-848x480-fps30-emitter-on`: `depth_ir_pair` on `239222300781` at `848x480@30` with emitter `on`
- `single-239222303506-depth-848x480-fps30-emitter-auto`: `depth` on `239222303506` at `848x480@30` with emitter `auto`
- `single-239222303506-color-848x480-fps30-emitter-auto`: `color` on `239222303506` at `848x480@30` with emitter `auto`
- `single-239222303506-ir_left-848x480-fps30-emitter-on`: `ir_left` on `239222303506` at `848x480@30` with emitter `on`
- `single-239222303506-ir_right-848x480-fps30-emitter-on`: `ir_right` on `239222303506` at `848x480@30` with emitter `on`
- `single-239222303506-ir_pair-848x480-fps30-emitter-on`: `ir_pair` on `239222303506` at `848x480@30` with emitter `on`
- `single-239222303506-rgbd-848x480-fps30-emitter-auto`: `rgbd` on `239222303506` at `848x480@30` with emitter `auto`
- `single-239222303506-rgb_ir_pair-848x480-fps30-emitter-on`: `rgb_ir_pair` on `239222303506` at `848x480@30` with emitter `on`
- `single-239222303506-depth_ir_pair-848x480-fps30-emitter-on`: `depth_ir_pair` on `239222303506` at `848x480@30` with emitter `on`
- `three_camera-239222300433-239222300781-239222303506-depth-848x480-fps30-emitter-auto`: `depth` on `239222300433, 239222300781, 239222303506` at `848x480@30` with emitter `auto`
- `three_camera-239222300433-239222300781-239222303506-color-848x480-fps30-emitter-auto`: `color` on `239222300433, 239222300781, 239222303506` at `848x480@30` with emitter `auto`
- `three_camera-239222300433-239222300781-239222303506-ir_left-848x480-fps30-emitter-on`: `ir_left` on `239222300433, 239222300781, 239222303506` at `848x480@30` with emitter `on`
- `three_camera-239222300433-239222300781-239222303506-ir_pair-848x480-fps30-emitter-on`: `ir_pair` on `239222300433, 239222300781, 239222303506` at `848x480@30` with emitter `on`
- `single-239222300433-ir_left-848x480-fps30-emitter-off`: `ir_left` on `239222300433` at `848x480@30` with emitter `off`
- `single-239222300433-ir_right-848x480-fps30-emitter-off`: `ir_right` on `239222300433` at `848x480@30` with emitter `off`
- `single-239222300433-ir_pair-848x480-fps30-emitter-off`: `ir_pair` on `239222300433` at `848x480@30` with emitter `off`
- `single-239222300433-depth_ir_pair-848x480-fps30-emitter-off`: `depth_ir_pair` on `239222300433` at `848x480@30` with emitter `off`
- `single-239222300781-ir_left-848x480-fps30-emitter-off`: `ir_left` on `239222300781` at `848x480@30` with emitter `off`
- `single-239222300781-ir_right-848x480-fps30-emitter-off`: `ir_right` on `239222300781` at `848x480@30` with emitter `off`
- `single-239222300781-ir_pair-848x480-fps30-emitter-off`: `ir_pair` on `239222300781` at `848x480@30` with emitter `off`
- `single-239222300781-rgb_ir_pair-848x480-fps30-emitter-off`: `rgb_ir_pair` on `239222300781` at `848x480@30` with emitter `off`
- `single-239222300781-depth_ir_pair-848x480-fps30-emitter-off`: `depth_ir_pair` on `239222300781` at `848x480@30` with emitter `off`
- `single-239222303506-ir_left-848x480-fps30-emitter-off`: `ir_left` on `239222303506` at `848x480@30` with emitter `off`
- `single-239222303506-ir_right-848x480-fps30-emitter-off`: `ir_right` on `239222303506` at `848x480@30` with emitter `off`
- `single-239222303506-ir_pair-848x480-fps30-emitter-off`: `ir_pair` on `239222303506` at `848x480@30` with emitter `off`
- `single-239222303506-rgb_ir_pair-848x480-fps30-emitter-off`: `rgb_ir_pair` on `239222303506` at `848x480@30` with emitter `off`
- `single-239222303506-depth_ir_pair-848x480-fps30-emitter-off`: `depth_ir_pair` on `239222303506` at `848x480@30` with emitter `off`
- `three_camera-239222300433-239222300781-239222303506-ir_right-640x480-fps30-emitter-on`: `ir_right` on `239222300433, 239222300781, 239222303506` at `640x480@30` with emitter `on`
- `three_camera-239222300433-239222300781-239222303506-ir_right-640x480-fps30-emitter-off`: `ir_right` on `239222300433, 239222300781, 239222303506` at `640x480@30` with emitter `off`

## Unstable / Failed Stream Sets

- `single-239222300433-rgbd-848x480-fps30-emitter-auto`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `single-239222300433-rgb_ir_pair-848x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `single-239222300433-rgbd_ir_pair-848x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `single-239222300781-rgbd_ir_pair-848x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `single-239222303506-rgbd_ir_pair-848x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-ir_right-848x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-rgbd-848x480-fps30-emitter-auto`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-rgb_ir_pair-848x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-depth_ir_pair-848x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-rgbd_ir_pair-848x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `single-239222300433-rgbd-640x480-fps30-emitter-auto`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `single-239222300433-rgb_ir_pair-640x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `single-239222300433-rgbd_ir_pair-640x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `single-239222300781-rgbd_ir_pair-640x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `single-239222303506-rgbd_ir_pair-640x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-ir_left-848x480-fps30-emitter-off`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-ir_pair-848x480-fps30-emitter-off`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-rgbd-640x480-fps30-emitter-auto`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-rgb_ir_pair-640x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-depth_ir_pair-640x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-rgbd_ir_pair-640x480-fps30-emitter-on`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-ir_left-640x480-fps30-emitter-off`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.
- `three_camera-239222300433-239222300781-239222303506-ir_pair-640x480-fps30-emitter-off`: `StabilityThresholdNotMet` - Case started but failed delivery / fps / stall thresholds.

## Key Errors

- `StabilityThresholdNotMet`: Case started but failed delivery / fps / stall thresholds.

## Recommended Next Move

- Primary: `C` - ir_pair is stable on all 3 cameras, but rgb_ir_pair is not yet stable as a same-take mode.
- Comparison feasibility: `E` - Do not promise same-take depth-vs-FFS comparison yet; depth_ir_pair / rgbd_ir_pair remain unstable or unsupported.
