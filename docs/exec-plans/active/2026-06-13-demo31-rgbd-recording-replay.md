# Demo 3.1 RGB-D Recording Replay

## Goal

Add a single-camera RGB-D recording input source for Single Demo 3 / 3.1 so a
recorded `data_collect/<case>` folder can drive the existing SAM3.1 first-frame
initialization, EdgeTAM propagation, and masked point-cloud rendering path as if
it were a live RealSense camera.

## Scope

- Add `--input-source {live,recording}`, `--recording-case`, and `--replay-fps`
  to the shared single Demo 3.x launcher.
- Superseded on 2026-06-14 by Demo 3.x fake-live replay: RealSense-native Demo
  3 / 3.1 consume RGB-D and FFS Demo 3.2 / 3.3 consume IR stereo.
- Read RGB-D raw recordings from `metadata.json`, `color/0/*.png`, and
  `depth/0/*.npy`; sort camera-0 metadata steps numerically and remap the first
  complete frame to demo `seq=0`.
- Keep segmentation, tracking, and point-cloud code paths unchanged downstream
  of `FramePacket`.

## Validation

- Unit tests for recording source ordering, packet fields, and missing file
  failures.
- Runtime tests for live defaults, recording contracts, delegate argv, and
  serial-check bypass.
- Headless synthetic replay smoke.
- Run `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.
