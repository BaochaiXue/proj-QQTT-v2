# Demo v6.2 tracking visualization cleanup

## Goal

Replace the two duplicated object/controller tracking renderers under
`demo_v6_2/others/` with one readable Demo v6.2 entrypoint, then regenerate the
tracking video from the latest complete fake-camera run.

## Required behavior

- Read every contiguous `outputs_v6_1/online_data/chunks/chunk_*.pkl` file.
- Require one stable query schema and stable object/controller point counts.
- Render object points only when both visible and motion-valid.
- Render controller points as red spheres in the same fixed camera view.
- Write one 5 FPS MP4 and one JSON summary under
  `demo_v6_2/others/obj_shape_asap_outputs/`.
- Fail immediately on missing chunks, discontinuous frame ranges, malformed
  arrays, or schema changes.

## Changes

- Keep `visualize_object_controller_tracking.py` as the only tracking-video
  renderer.
- Delete `enhanced_visualize_object_controller_tracking.py` and the two old,
  overlapping tracking video/summary pairs.
- Use a typed tracking-sequence value instead of an unstructured dictionary.
- Update Demo/version labels and default artifact paths to `demo_v6_2`.
- Add focused loader/summary tests and register them in the smoke profile.

## Validation

1. Run the focused visualization tests.
2. Run formatting, compile, and static checks on touched Python files.
3. Render the latest 15 chunks and inspect the MP4 with `ffprobe` plus sampled
   frames.
4. Run the repository smoke profile.

## Result

- One canonical 579-line renderer replaces 1,237 lines across two duplicated
  scripts.
- Four focused loader/summary tests pass; repository smoke passes 189 tests.
- The regenerated MP4 contains 525 frames at 1280x900 and 5 FPS (105 seconds).
- First, middle, and final frames were extracted and visually inspected.
