# Demo 3.2 Default Table-World Calibration Plan

**Goal:** Make Demo 3.2 and the matching Demo 3.3 FFS wrapper require
single-camera table/world calibration by default, so normal rendering and
headless capture use `table_world_z0` unless the operator supplies an explicit
alternate `--table-calibrate` path.

**Scope:** `qqtt.demo.single_demo_v3_runtime` defaults, Demo 3.2 docs, and
focused runtime tests. The shared delegate already applies the loaded
`camera_to_world_c2w` to PCD points and TAPNext++ markers.

- [x] Add failing runtime tests for default table calibration injection and
      missing-default fail-fast behavior.
- [x] Add a Demo 3.2/3.3 default `table_calibrate.pkl` path in the wrapper and
      validate it before dry-run or live execution.
- [x] Keep explicit `--table-calibrate` override behavior unchanged.
- [x] Update operator docs to say Demo 3.2/3.3 now use repo-root
      `table_calibrate.pkl` by default.
- [x] Run focused unit tests and the smoke validation profile.
