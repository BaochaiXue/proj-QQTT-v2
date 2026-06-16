# Single-Camera Table Z0 Calibration Design

## Summary

Add a dedicated single-camera tabletop calibration entrypoint that uses the
existing ChArUco board workflow to produce a separate table-world calibration.
The board is placed flat on the table. The board plane becomes world `Z=0`, and
the output is written to `table_calibrate.pkl` plus
`table_calibrate_metadata.json` in the repo root.

This does not overwrite or change the existing `calibrate.pkl` contract. Downstream
recording, alignment, and demo tools will use the table calibration only when
they are explicitly passed the table calibration path.

## Confirmed User Decisions

- Calibration method: ChArUco board flat on the table.
- Capture behavior: automatic one-shot with strict acceptance thresholds.
- Output location: repo root.
- Output files: separate `table_calibrate.pkl` and
  `table_calibrate_metadata.json`.
- Downstream intent: the table calibration is a real runtime artifact for
  recording, alignment, and demos, not only a human inspection file.

## Existing Context

`cameras_calibrate.py` currently uses `CameraSystem.calibrate()` to detect a
ChArUco board and write `calibrate.pkl` plus `calibrate_metadata.json`. The
existing `calibrate.pkl` schema is a list of `camera_to_world_c2w` 4x4
transforms, with metadata contract `qqtt_calibrate_pkl_c2w_list_v1`.

Visualization code already has a derived `semantic_world` concept and utilities
for table planes, but those are diagnostic visualization concepts. The new table
calibration must live in the camera/runtime layer so it can be loaded by
recording, alignment, and demo tools without making formal runtime code depend
on experiment-only visualization packages.

## Public Interface

Add a new CLI:

```bash
conda run -n demo_2_max --no-capture-output python cameras_calibrate_table.py
```

Supported options:

```bash
--serial SERIAL
--width 1280
--height 720
--fps 5
--calibration-board calibio-12x9-30mm
--max-reprojection-error-px 0.20
--min-corner-fraction 0.60
--output table_calibrate.pkl
--diagnostic-image table_calibrate_diagnostic.png
```

The default command opens one RealSense camera in color mode, detects the
current lab Calib.io ChArUco board, and writes the table calibration only if the
sample passes all strict checks.

## Output Contract

`table_calibrate.pkl` uses the same physical matrix shape as `calibrate.pkl`:

- object type: Python list
- item shape: `(4, 4)`
- transform convention: `camera_to_world_c2w`
- world frame kind: `table_world_z0`
- schema version: `qqtt_table_calibrate_c2w_v1`

For this single-camera branch the list contains one transform by default. The
world origin and XY axes come from the ChArUco board frame, and the board plane
is the table plane. Points transformed by this `c2w` should have `z ~= 0` on the
table surface when the board is flat on the table.

`table_calibrate_metadata.json` records:

- schema version and compatibility contract
- serial numbers and logical camera names
- image width, height, and FPS
- calibration board profile and any CLI overrides
- transform convention and world frame kind
- strict thresholds used for acceptance
- accepted corner count and corner fraction
- mean reprojection error
- color distortion model and coefficients when available
- creation time
- optional diagnostic image path

## Validation And Failure Behavior

The tool is intentionally strict. It should prefer failing without writing files
over writing a questionable table frame.

Acceptance rules:

- exactly one camera is selected unless `--serial` selects one explicitly
- at least `max(existing_min_corners, min_corner_fraction * board_corners)`
  ChArUco corners are detected
- mean reprojection error is at most `--max-reprojection-error-px`, default
  `0.20`
- all output transforms are finite, non-singular 4x4 homogeneous matrices

Failure rules:

- no ArUco markers detected: fail fast and write nothing
- no ChArUco corners detected: fail fast and write nothing
- corner count too low: fail fast and write nothing
- reprojection error too high: fail fast and write nothing
- output path would conflict with `calibrate.pkl`: fail fast unless a future
  explicit promotion command is added

The failed run reports a clear terminal reason, but it does not create partial
calibration artifacts.

## Downstream Integration

The first implementation should define the table calibration file contract and a
small loader that can be reused by downstream tools. The loader should be able
to validate `table_calibrate.pkl` and metadata without affecting the existing
`calibrate.pkl` loader.

Planned explicit consumers:

- recording and aligned case generation can accept
  `--table-calibrate table_calibrate.pkl`, copy the table calibration into the
  case, and record table calibration metadata fields
- Demo 3.x can accept `--table-calibrate table_calibrate.pkl` and use the table
  world frame for point cloud display, crop, and table-relative Z thresholds
- existing tools remain unchanged when no table calibration path is supplied

The first implementation may add the loader and one minimal consumer path, but
it must keep the broader contract ready for recording, alignment, and demos.

## Non-Goals

- Do not overwrite `calibrate.pkl`.
- Do not add depth-plane fitting or hybrid depth refinement in this version.
- Do not import experiment-only visualization modules into formal camera,
  recording, alignment, or demo runtime code.
- Do not change existing recording or alignment defaults.
- Do not add a multi-camera table calibration flow in this branch.

## Testing

Unit and smoke coverage should verify:

- the new CLI defaults to `table_calibrate.pkl`
- the writer does not touch `calibrate.pkl`
- metadata contains schema, world frame kind, serial, board profile, thresholds,
  corner count, and reprojection error
- high reprojection error prevents output writes
- insufficient corners prevent output writes
- the table calibration loader accepts valid table calibration files
- the table calibration loader rejects invalid shapes, missing metadata, and
  wrong schema versions
- docs mention that the board must be flat on the table and that moving the
  camera or table requires rerunning table calibration

## Documentation

Update:

- `docs/WORKFLOWS.md`: add a "Table Z0 Calibration" step near calibration
- `docs/HARDWARE_VALIDATION.md`: add a manual checklist for flat-board table
  calibration
- `docs/ARCHITECTURE.md`: document `table_calibrate.pkl` as a separate table
  world artifact, distinct from the normal ChArUco `calibrate.pkl`

## Acceptance Criteria

- Running `python cameras_calibrate_table.py` with the board flat on the table
  writes `table_calibrate.pkl`, `table_calibrate_metadata.json`, and optionally
  `table_calibrate_diagnostic.png`.
- Existing `calibrate.pkl` is not modified.
- Downstream code can explicitly load the table calibration contract.
- A failing one-shot calibration leaves no partial table calibration output.
- Smoke validation remains green.
