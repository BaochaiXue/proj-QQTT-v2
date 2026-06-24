# Demo v4 Online Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Demo v4 default output compatible with `realtime_phystwin/scripts/fake_online_tracker.py` while preserving the static `final_data.pkl` required by realtime_phystwin online consumers.

**Architecture:** Add a focused Demo v4 online output writer that converts each produced `final_data.pkl` chunk into an atomic online chunk pickle and online manifest. The headless bridge keeps writing existing diagnostic case directories, then mirrors the same final data into `online_data/<case-prefix>/chunks/` and updates `data/<case-prefix>/final_data.pkl` as the static path.

**Tech Stack:** Python standard library `pickle/json/os.replace`, NumPy arrays, existing Demo v4 `FuturePhysTwinChunk` and chunk writer, `unittest`.

---

### Task 1: Online Writer Contract

**Files:**
- Create: `demo_v4/online_chunk_output.py`
- Modify: `tests/test_demo_v4_futurephystwin_chunks.py`

- [ ] **Step 1: Write the failing online writer tests**

Add tests that instantiate the writer with a temporary base path, commit two `_final_data_payload`-style dictionaries, and assert:

```python
manifest = json.loads((base / "online_data" / "demo_v4" / "manifest.json").read_text())
self.assertEqual(manifest["case_name"], "demo_v4")
self.assertEqual(manifest["chunk_size"], 2)
self.assertEqual(manifest["latest_committed_chunk"], 1)
self.assertEqual(manifest["latest_committed_frame"], 4)
self.assertEqual(manifest["status"], "recording")
self.assertTrue((base / "online_data" / "demo_v4" / "chunks" / "chunk_000000.pkl").is_file())
self.assertTrue((base / "online_data" / "demo_v4" / "chunks" / "chunk_000001.pkl").is_file())
with (base / "online_data" / "demo_v4" / "chunks" / "chunk_000000.pkl").open("rb") as handle:
    chunk = pickle.load(handle)
self.assertEqual(chunk["chunk_id"], 0)
self.assertEqual(chunk["start_frame"], 0)
self.assertEqual(chunk["end_frame"], 2)
self.assertIn("object_points", chunk)
self.assertIn("controller_points", chunk)
self.assertNotIn("surface_points", chunk)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks.FuturePhysTwinChunkWriterTest.test_online_chunk_output_writes_fake_tracker_contract
```

Expected: FAIL because `demo_v4.online_chunk_output` does not exist.

- [ ] **Step 3: Implement minimal online writer**

Create `demo_v4/online_chunk_output.py` with:

```python
TIME_KEYS = (...)
STATIC_KEYS = (...)
atomic_pickle_dump(obj, path)
atomic_json_dump(obj, path)
take_source_frames(value, indices)
build_online_chunk(data, case_name, chunk_id, start_frame, end_frame, source_frame_indices)
class DemoV4OnlineOutputWriter:
    commit_final_data_chunk(...)
    finish(...)
```

The writer uses `os.replace` for chunk pickle and manifest updates.

- [ ] **Step 4: Run focused test and verify GREEN**

Run the same focused test. Expected: PASS.

### Task 2: Static Final Data Aggregation

**Files:**
- Modify: `demo_v4/online_chunk_output.py`
- Modify: `tests/test_demo_v4_futurephystwin_chunks.py`

- [ ] **Step 1: Write failing static data test**

Assert that committing two 2-frame final data chunks writes:

```text
<base>/data/demo_v4/final_data.pkl
```

with 4 frames in each required time array and static `surface_points`/`interior_points` from the latest chunk.

- [ ] **Step 2: Verify RED**

Run the focused static test and confirm it fails because static aggregation is missing.

- [ ] **Step 3: Implement aggregation**

Maintain in-memory lists for all `TIME_KEYS`, concatenate them after each commit, merge static keys, and atomically write `data/<case>/final_data.pkl`. Also write small `metadata.json` with `case_name`, `online_dir`, `chunk_size`, and `latest_committed_frame`.

- [ ] **Step 4: Verify GREEN**

Run both online writer tests.

### Task 3: Bridge Integration

**Files:**
- Modify: `demo_v4/headless_chunk_bridge.py`
- Modify: `tests/test_demo_v4_futurephystwin_chunks.py`

- [ ] **Step 1: Write failing bridge test**

Extend the existing headless capture conversion test to assert:

```python
online_dir = base_path / "online_data" / "demo_v4_capture"
static_path = base_path / "data" / "demo_v4_capture" / "final_data.pkl"
self.assertTrue((online_dir / "manifest.json").is_file())
self.assertTrue((online_dir / "chunks" / "chunk_000000.pkl").is_file())
self.assertTrue(static_path.is_file())
self.assertEqual(manifests[0]["online_dir"], str(online_dir))
self.assertEqual(manifests[0]["static_data_path"], str(static_path))
```

- [ ] **Step 2: Verify RED**

Run the focused bridge test and confirm these files/manifest fields are missing.

- [ ] **Step 3: Integrate writer**

Add optional parameters to `write_chunks_from_headless_capture` and `stream_chunks_from_headless_capture`:

```python
write_online_output: bool = True
online_case_name: str | None = None
```

Create one `DemoV4OnlineOutputWriter` per run and call it after each existing chunk case is written.

- [ ] **Step 4: Verify GREEN**

Run the bridge tests that cover multiple chunks and streaming tailing.

### Task 4: CLI Contract And README

**Files:**
- Modify: `demo_v4/realtime_futurephystwin_chunks.py`
- Modify: `demo_v4/README.md`
- Modify: `tests/test_demo_v4_futurephystwin_chunks.py`

- [ ] **Step 1: Write failing CLI tests**

Assert `_contract(args)` includes:

```python
self.assertEqual(contract["output_format"], "online-primary-static-case")
self.assertEqual(contract["online_dir"], "<base>/online_data/demo_v4")
self.assertEqual(contract["static_data_path"], "<base>/data/demo_v4/final_data.pkl")
```

Assert CLI summaries include the same fields after source-headless and mocked realtime paths.

- [ ] **Step 2: Verify RED**

Run the CLI tests and confirm missing contract fields.

- [ ] **Step 3: Implement CLI metadata**

Add helper functions resolving online/static paths. Update summaries and README output section to document `online_dir` and `static_data_path`.

- [ ] **Step 4: Verify GREEN**

Run `tests.test_demo_v4_futurephystwin_chunks`.

### Task 5: Regression Validation

**Files:**
- Modify only if failures reveal a directly related issue.

- [ ] **Step 1: Run targeted tests**

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks
```

- [ ] **Step 2: Run smoke validation**

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

- [ ] **Step 3: Run diff hygiene**

```bash
git diff --check
python -m py_compile demo_v4/online_chunk_output.py demo_v4/futurephystwin_chunk_writer.py demo_v4/headless_chunk_bridge.py demo_v4/realtime_futurephystwin_chunks.py
```

- [ ] **Step 4: Final audit**

Confirm:

```bash
git status --short
```

Only expected files changed, plus any pre-existing dirty files left untouched.

