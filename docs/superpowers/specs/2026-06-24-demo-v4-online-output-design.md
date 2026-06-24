# Demo v4 Online Output Design

## Goal

Make Demo v4 default output match the `realtime_phystwin/scripts/fake_online_tracker.py` producer contract while keeping the static `final_data.pkl` path required by realtime_phystwin online consumers.

## Approved Approach

Use Online Primary plus Static Case.

Demo v4 publishes two default product roots under `--futurephystwin-base-path`:

```text
<base>/
  data/<case-prefix>/
    final_data.pkl
    metadata.json
    split.json
    calibrate.pkl

  online_data/<case-prefix>/
    manifest.json
    chunks/
      chunk_000000.pkl
      chunk_000001.pkl
```

The online directory is the primary consumer entrypoint and follows `fake_online_tracker.py`:

```text
manifest.json:
  case_name
  status
  chunk_size
  num_frames_total
  latest_committed_chunk
  latest_committed_frame
  version
  source_num_frames_total
  source_start_frame
  source_end_frame
  source_frame_step
  online_num_frames_total

chunk_N.pkl:
  case_name
  chunk_id
  start_frame
  end_frame
  source_frame_indices
  object_points
  object_colors
  object_visibilities
  object_motions_valid
  controller_points
  optional asap_object_points_filled
  optional asap_surface_points
  optional asap_interior_points
```

The static `data/<case-prefix>/final_data.pkl` exists because `OnlineFrameBuffer` loads static structure points from `--static_data_path` before appending online chunks.

## Compatibility

Current per-window FuturePhysTwin case directories remain available for diagnostics and validation. They are no longer the recommended default consumer entrypoint. They can remain at the current top-level case names while the new online/static roots are added.

Existing `--chunk-seconds` behavior remains unchanged. The default is still 5 seconds, producing 25 frames at the default 5 FPS. `--chunk-frame-count` remains an explicit test/debug override.

## Atomicity

Online chunk files are written through temporary files followed by `os.replace`.
The online manifest is also written through a temporary file followed by `os.replace`, and it is updated only after the chunk pickle is committed. Consumers that use `OnlineChunkReader` therefore never see a committed chunk id without the corresponding pickle.

The static case is written after chunks are available. It uses a temporary `final_data.pkl.tmp` followed by `os.replace`. For the initial implementation, the static case is the latest committed online buffer concatenated so far, with `surface_points` and `interior_points` copied from the strict chunk final data.

## Runtime Summary

Demo v4 run summary records:

```text
online_dir
static_data_path
online_manifest_path
online_chunk_count
online_latest_committed_frame
legacy_chunk_cases
```

The README should show realtime_phystwin invocation using:

```bash
--online_dir <base>/online_data/<case-prefix>
--static_data_path <base>/data/<case-prefix>/final_data.pkl
```

## Testing

Tests should prove:

1. Parser/contract dry-run reports the new default `output_format`.
2. Existing headless capture conversion writes the online directory and static final data.
3. Online manifest and chunk pickle fields match `fake_online_tracker.py`.
4. Multiple chunks commit as `chunk_000000.pkl`, `chunk_000001.pkl` and update `latest_committed_chunk`.
5. The static final data concatenates time arrays and includes `surface_points` and `interior_points`.
6. Current per-window FuturePhysTwin case validation still passes.

