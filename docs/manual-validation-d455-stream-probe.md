# Manual Validation: D455 Stream Capability Probe

These checks are manual and are not run in CI.

## Hardware Inventory

- `239222300433`
- `239222300781`
- `239222303506`

Stable three-camera probe order:

- cam0 = `239222300433`
- cam1 = `239222300781`
- cam2 = `239222303506`

## Expected Behavior Sources

- Intel RealSense D400 projection documentation:
  - expectation: stream intrinsics / extrinsics define the coordinate contracts
- Intel RealSense D400 / D455 datasheet and supported stream tables:
  - expectation: advertised stream profiles may exist in the SDK, but actual simultaneous stability still needs probing
- Fast-FoundationStereo README:
  - expectation: left/right stereo inputs should be rectified, undistorted, and not swapped

## Probe Validation Steps

1. Run the standalone capability probe.
2. Confirm `results.json` and `summary.md` exist under:
   - `data/ffs_proof_of_life/d455_stream_probe/latest/`
3. Confirm `docs/generated/d455_stream_probe_results.json` exists.
4. Confirm `docs/generated/d455_stream_probe_results.md` exists.
5. For successful cases, confirm sample frames exist under:
   - `sample_frames/<case_id>/`
6. For failed cases, confirm a structured log exists under:
   - `logs/<case_id>.json`
7. Confirm the Markdown summary includes:
   - single-camera summary
   - three-camera summary
   - stable stream sets
   - unstable stream sets
   - key errors
   - recommendation for the next step

## What To Check In The Results

- whether `rgbd` is stable for one camera and for all 3 cameras
- whether `ir_pair` is stable for one camera and for all 3 cameras
- whether `rgb_ir_pair` is stable at `848x480@30`
- whether fallback to `640x480@30` is required
- whether `depth_ir_pair` or `rgbd_ir_pair` are blocked or unstable
- whether emitter `off` changes the stability of IR-containing stream sets
