# Demo v5 Warmup Output Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Demo v5 chunking and the side-by-side viewer treat warmup as waiting time, not output timeline time.

**Architecture:** Keep RGB input timeline starting at camera start, but start `frames.jsonl` chunk output from the first realtime data-process frame after startup hold / shape-prior warmup. Preserve real source frame indices in chunk payloads and make the viewer choose the right-side output frame by source timeline instead of by a local playback counter.

**Tech Stack:** Python, JSONL capture rows, NumPy chunk payloads, OpenCV/Open3D viewer, pytest.

---

### Task 1: Reproduce Warmup-Delayed Frame Boundary

**Files:**
- Modify: `tests/test_demo_v5_realtime_phystwin.py`
- Modify: `demo_v5/realtime_data_process_track.py`

- [x] **Step 1: Write the failing test**

Add a test that feeds chunk rows where row 0 is a delayed warmup source frame and row 1 starts the realtime source stream:

```python
def test_trim_warmup_delayed_rows_starts_at_realtime_stream(self):
    rows = [
        {"seq": 0, "source_frame_index": 0, "pipeline_latency_ms": 18234.0, "startup_hold_s": 18.2},
        {"seq": 1, "source_frame_index": 551, "pipeline_latency_ms": 248.0, "startup_hold_s": 18.2},
        {"seq": 2, "source_frame_index": 557, "pipeline_latency_ms": 221.0, "startup_hold_s": 18.2},
    ]
    trimmed, skipped = bridge._trim_warmup_delayed_rows(rows)
    self.assertEqual(skipped, 1)
    self.assertEqual([row["source_frame_index"] for row in trimmed], [551, 557])
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v5_realtime_phystwin.py::DemoV5RealtimeTests::test_trim_warmup_delayed_rows_starts_at_realtime_stream -q
```

Expected: FAIL because `_trim_warmup_delayed_rows` does not exist.

- [x] **Step 3: Implement the minimal helper**

Add `_trim_warmup_delayed_rows(rows)` in `demo_v5/realtime_data_process_track.py`. It should skip a leading row when it has `startup_hold_s > 0`, `pipeline_latency_ms` near that hold, and the next row jumps forward by more than one source step. It must not trim ordinary sequential rows.

- [x] **Step 4: Use the helper in live and offline chunk loops**

Before appending new rows into `row_buffer`, trim only the beginning of the stream while `next_row_idx == 0` / `row_start == 0`. Record the skip count in manifest fields.

### Task 2: Preserve Source Timeline in Chunks

**Files:**
- Modify: `tests/test_demo_v5_realtime_phystwin.py`
- Modify: `demo_v5/realtime_data_process_track.py`

- [x] **Step 1: Write the failing test**

Add a test for `_chunk_payload_from_prepared_frames` or a small helper proving chunk `source_frame_indices` come from prepared frames or row source indices, not local 0..T numbering.

- [x] **Step 2: Verify RED**

Run the single pytest test and confirm the current payload reindexes source frames in the failing fixture.

- [x] **Step 3: Implement minimal source-index propagation**

Keep real `source_frame_indices` in `DataProcessChunk` and online chunk payloads. Add `source_timestamp_s` if available from rows/prepared frames so the viewer can align by time.

### Task 3: Viewer Source-Time Follow

**Files:**
- Modify: `tests/test_demo_v5_realtime_phystwin.py`
- Modify: `demo_v5/visualize_track.py`

- [x] **Step 1: Write failing pure-function tests**

Add tests for selecting the right output frame from input source time:

```python
selected = visualize_track.select_output_frame_for_input_source_time(
    output_source_times=[18.0, 18.2, 18.4, 25.0, 25.2],
    input_source_time=32.1,
    target_latency_s=7.0,
)
self.assertEqual(selected, 3)
```

- [x] **Step 2: Verify RED**

Run the single pytest test and confirm the selection helper is missing.

- [x] **Step 3: Implement helper and connect side-by-side follow mode**

When auto-follow is enabled, derive the right-side index from latest left RGB `source_timestamp_s - chunk_seconds`, falling back to source frame index / output FPS only when timestamps are missing.

### Task 4: Validation and Run

**Files:**
- Modify: docs only if behavior notes need updating.

- [x] **Step 1: Run focused tests**

```bash
conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v5_realtime_phystwin.py -q
```

- [x] **Step 2: Run smoke validation**

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

- [x] **Step 3: Run source-capture regression check**

```bash
conda run -n demo_2_max --no-capture-output python demo_v5/realtime_data_process_sam3d.py --source-headless-capture result/demo_v5/demo_v5_side_by_side_full_20260626_135619/demo_v5_side_by_side_full_20260626_135619_camera_capture_20260626_135620 --base-path result/demo_v5/warmup_trim_verify_<stamp> --case-prefix warmup_trim_verify --max-chunks 1 --optimization-mode disabled --point-viewer-mode disabled
```

Expected: `chunk_000000.pkl` starts at source frame 551, writes `source_timestamps_s`, and manifest records `warmup_skipped_rows=1`.

- [x] **Step 4: Run short live fake-camera regression check**

Run:

```bash
conda run -n demo_2_max --no-capture-output python demo_v5/realtime_data_process_sam3d.py --base-path result/demo_v5/warmup_trim_live_verify_<stamp> --case-prefix warmup_trim_live_verify --max-chunks 1 --capture-extra-seconds 70 --optimization-mode disabled --point-viewer-mode disabled
```

Expected: live `chunk_000000.pkl` records `warmup_skipped_rows=1`, preserves post-warmup source timestamps, and publishes a normal chunk.
