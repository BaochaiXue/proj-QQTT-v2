# Demo v4 Online Output Format

## Summary

Align Demo v4 default output with `realtime_phystwin/scripts/fake_online_tracker.py`.
The default consumer-facing output becomes:

```text
<base>/online_data/<case-prefix>/manifest.json
<base>/online_data/<case-prefix>/chunks/chunk_000000.pkl
<base>/data/<case-prefix>/final_data.pkl
```

Existing per-window FuturePhysTwin case directories remain as diagnostics and compatibility artifacts.

## Requirements

- Online chunks use the fake producer fields:
  - `case_name`
  - `chunk_id`
  - `start_frame`
  - `end_frame`
  - `source_frame_indices`
  - `object_points`
  - `object_colors`
  - `object_visibilities`
  - `object_motions_valid`
  - `controller_points`
  - optional `asap_object_points_filled`
  - optional `asap_surface_points`
  - optional `asap_interior_points`
- Online manifest uses the fake producer fields:
  - `case_name`
  - `status`
  - `chunk_size`
  - `num_frames_total`
  - `latest_committed_chunk`
  - `latest_committed_frame`
  - `version`
  - source frame range metadata
- Chunk pickle writes and manifest writes are atomic temp-file plus `os.replace`.
- Static `data/<case-prefix>/final_data.pkl` contains concatenated committed chunk arrays and static shape-prior arrays.
- Default chunk timing remains 5 seconds; `--chunk-seconds` and `--chunk-frame-count` retain current semantics.
- Demo v4 summaries expose `online_dir` and `static_data_path`.

## Validation

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
git diff --check
python -m py_compile demo_v4/online_chunk_output.py demo_v4/headless_chunk_bridge.py demo_v4/realtime_futurephystwin_chunks.py
```

## Notes

- The existing dirty file `realtime_phystwin/qqtt/engine/trainer_warp.py` is outside this change and should not be modified.
- This change does not alter Demo 3.2 tracking, mask, depth, SAM3D, or PhysTwin strict finalizer semantics.

