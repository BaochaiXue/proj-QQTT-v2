# Demo v5 Interactive Side-by-Side Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for behavior changes and superpowers:verification-before-completion before claiming completion.

**Goal:** Make Demo v5 side-by-side runtime show live RGB input plus an interactive Open3D `final_data` output view, without chunk-boundary frame jumps.

**Architecture:** Keep `demo_v5/visualize_track.py` as the viewer entrypoint. Use OpenCV only for the left RGB input window and Open3D `Visualizer` for the right `final_data` output window when `--layout side-by-side --render-mode sam3d-final-data` runs live. Keep the existing OpenCV composite path for offline video export and fallback compatibility.

**Tech Stack:** Python, OpenCV, Open3D, NumPy, pytest.

---

### Task 1: Reproduce And Guard The Timeline Bug

**Files:**
- Modify: `tests/test_demo_v5_realtime_phystwin.py`
- Modify: `demo_v5/visualize_track.py`

- [x] Add tests that show side-by-side output playback advances sequentially at 5fps and does not jump from the last frame of one newly available chunk to the last frame of the next chunk.
- [x] Run the targeted pytest and verify the new test fails against the current source-time jump behavior.
- [x] Add a small cursor helper in `demo_v5/visualize_track.py` so the live viewer can step output frames in stream order without skipping available frames.
- [x] Run the targeted pytest and verify the new timeline test passes.

### Task 2: Restore Interactive Open3D Right View

**Files:**
- Modify: `tests/test_demo_v5_realtime_phystwin.py`
- Modify: `demo_v5/visualize_track.py`

- [x] Add tests that the default side-by-side live backend is interactive Open3D for `sam3d-final-data`, while `--output-video` keeps the composite image renderer.
- [x] Run the targeted pytest and verify the test fails before implementation.
- [x] Add an interactive final-data renderer using a visible Open3D window, following `data_process_sam3d/data_process_track.py::visualize_track`: object point cloud, rainbow colors, red controller spheres, initial view set once, and `poll_events/update_renderer` every frame.
- [x] Keep left RGB in an OpenCV window and update it continuously while the Open3D right window plays the output stream.
- [x] Run the targeted pytest and verify the interactive backend tests pass.

### Task 3: Validate The Full Demo Path

**Files:**
- Modify: `docs/exec-plans/active/2026-06-26-demo-v5-interactive-side-by-side-viewer.md`

- [x] Run `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo_v5_realtime_phystwin.py -q`.
- [x] Run `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`.
- [x] Run a short fake-live side-by-side check with the default interactive backend and confirm both windows start.
- [x] Update this plan with observed results.

**Observed Results**

- Red tests failed first for missing `OutputStreamPlaybackCursor` and missing `use_interactive_side_by_side`.
- After implementation, `tests/test_demo_v5_realtime_phystwin.py` passed: `35 passed, 19 subtests passed`.
- Smoke validation passed: `302 tests in 3.856s, OK`.
- Short fake-live interactive side-by-side with `--max-chunks 2 --capture-extra-seconds 70` completed with 2 normal chunks.
- Full fake-live interactive side-by-side completed with 23 normal chunks, 901 input RGB frames, 23 online chunks, and the viewer left running for inspection.
