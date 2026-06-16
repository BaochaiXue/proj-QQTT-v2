# Demo 3.2 Mask-Cleaned Tracking Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Demo 3.2 offline `tracking` renders remove table/background pixels by default using the same-frame `object_mask | controller_mask`, while preserving PhysTwin-style RGB target appearance and rainbow query dots.

**Architecture:** Keep headless capture unchanged. Add a small mask-loading and mask-application layer inside `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py`, then call it only for `--demo-visual-mode tracking --tracking-background-mask target-union`. Keep `pcd` mode on the existing projected filtered PCD path.

**Tech Stack:** Python, NumPy, OpenCV video writer, `.npz` headless artifacts, `unittest`, existing `demo_2_max` conda environment.

---

## File Structure

- Modify `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py`
  - Owns offline video rendering from Demo 3.2 headless captures.
  - Add `TRACKING_BACKGROUND_MASK_*` constants, a same-frame target-union mask loader, a mutating background-mask helper, CLI parsing, render summary fields, and per-frame mask counts.
- Modify `tests/test_demo32_headless_render_helper.py`
  - Existing focused tests for the offline render helper.
  - Add mask helper tests, update existing synthetic captures to include mask artifacts where default `target-union` is expected, add legacy `rgb` compatibility coverage, and add fail-fast error tests.
- Modify `demo_v3_2/README.md`
  - Document that offline tracking render now defaults to target-union background cleanup and how to request the old full-RGB comparison mode.

## Task 1: Add Target-Union Mask Helper Tests

**Files:**
- Modify: `tests/test_demo32_headless_render_helper.py`

- [ ] **Step 1: Update imports for direct helper testing**

Change the import near the top of `tests/test_demo32_headless_render_helper.py` from:

```python
from scripts.harness.diagnostics.demo.render_demo32_headless_capture import render_capture_to_video
```

to:

```python
from scripts.harness.diagnostics.demo.render_demo32_headless_capture import (
    TRACKING_BACKGROUND_MASK_RGB,
    TRACKING_BACKGROUND_MASK_TARGET_UNION,
    _apply_tracking_background_mask,
    _read_target_union_mask,
    render_capture_to_video,
)
```

- [ ] **Step 2: Add direct helper test for blacking background pixels**

Add this test method to `Demo32HeadlessRenderHelperTest`:

```python
    def test_apply_tracking_background_mask_blacks_pixels_outside_union(self) -> None:
        image = np.full((4, 5, 3), 80, dtype=np.uint8)
        image[1, 2] = np.array([10, 20, 30], dtype=np.uint8)
        mask = np.zeros((4, 5), dtype=bool)
        mask[1, 2] = True
        mask[3, 4] = True

        kept = _apply_tracking_background_mask(image, mask)

        self.assertEqual(kept, 2)
        np.testing.assert_array_equal(image[1, 2], np.array([10, 20, 30], dtype=np.uint8))
        np.testing.assert_array_equal(image[3, 4], np.array([80, 80, 80], dtype=np.uint8))
        np.testing.assert_array_equal(image[0, 0], np.array([0, 0, 0], dtype=np.uint8))
```

- [ ] **Step 3: Add target-union mask loader test**

Add this test method:

```python
    def test_read_target_union_mask_uses_object_or_controller_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            controller_mask = np.zeros((4, 5), dtype=bool)
            object_mask = np.zeros((4, 5), dtype=bool)
            controller_mask[1, 2] = True
            object_mask[3, 4] = True
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=controller_mask,
                object_mask=object_mask,
            )
            frame = {"seq": 0, "mask_path": "masks/000000.npz"}

            union = _read_target_union_mask(capture_dir=capture_dir, frame=frame, width=5, height=4)

            expected = np.logical_or(controller_mask, object_mask)
            np.testing.assert_array_equal(union, expected)
```

- [ ] **Step 4: Run the two new tests and verify they fail**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_demo32_headless_render_helper.Demo32HeadlessRenderHelperTest.test_apply_tracking_background_mask_blacks_pixels_outside_union \
  tests.test_demo32_headless_render_helper.Demo32HeadlessRenderHelperTest.test_read_target_union_mask_uses_object_or_controller_mask
```

Expected before implementation: import failure for `_apply_tracking_background_mask` or `_read_target_union_mask`.

## Task 2: Implement Target-Union Mask Helpers

**Files:**
- Modify: `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py`
- Test: `tests/test_demo32_headless_render_helper.py`

- [ ] **Step 1: Add tracking background mask constants**

Near `DEMO_VISUAL_MODES`, add:

```python
TRACKING_BACKGROUND_MASK_TARGET_UNION = "target-union"
TRACKING_BACKGROUND_MASK_RGB = "rgb"
TRACKING_BACKGROUND_MASK_MODES = (
    TRACKING_BACKGROUND_MASK_TARGET_UNION,
    TRACKING_BACKGROUND_MASK_RGB,
)
```

- [ ] **Step 2: Add same-frame target-union mask loader**

Add this function after `_read_rgb_frame_bgr`:

```python
def _read_target_union_mask(
    *,
    capture_dir: Path,
    frame: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    if "mask_path" not in frame:
        raise RuntimeError(
            "tracking background target-union requires mask_path in frames.jsonl; "
            "rerun headless capture"
        )
    mask_path = _resolve_capture_path(capture_dir, str(frame["mask_path"]))
    if not mask_path.is_file():
        raise RuntimeError(f"tracking background target-union mask file missing: {mask_path}")
    with np.load(mask_path, allow_pickle=False) as payload:
        missing = [name for name in ("object_mask", "controller_mask") if name not in payload.files]
        if missing:
            raise RuntimeError(
                "tracking background target-union mask payload missing "
                + ", ".join(missing)
                + f": {mask_path}"
            )
        object_mask = np.asarray(payload["object_mask"], dtype=bool)
        controller_mask = np.asarray(payload["controller_mask"], dtype=bool)
    expected_shape = (int(height), int(width))
    if object_mask.shape != expected_shape:
        raise RuntimeError(
            f"object_mask shape {tuple(object_mask.shape)} does not match render shape "
            f"{expected_shape}: {mask_path}"
        )
    if controller_mask.shape != expected_shape:
        raise RuntimeError(
            f"controller_mask shape {tuple(controller_mask.shape)} does not match render shape "
            f"{expected_shape}: {mask_path}"
        )
    return np.ascontiguousarray(np.logical_or(object_mask, controller_mask), dtype=bool)
```

- [ ] **Step 3: Add mutating background mask helper**

Add this function after `_read_target_union_mask`:

```python
def _apply_tracking_background_mask(image_bgr: np.ndarray, target_union_mask: np.ndarray) -> int:
    mask = np.asarray(target_union_mask, dtype=bool)
    if mask.ndim != 2:
        raise RuntimeError(f"tracking background mask must be 2D, got shape {tuple(mask.shape)}")
    if image_bgr.shape[:2] != mask.shape:
        raise RuntimeError(
            f"tracking background mask shape {tuple(mask.shape)} does not match image shape "
            f"{tuple(image_bgr.shape[:2])}"
        )
    image_bgr[~mask] = 0
    return int(np.count_nonzero(mask))
```

- [ ] **Step 4: Run helper tests and verify they pass**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_demo32_headless_render_helper.Demo32HeadlessRenderHelperTest.test_apply_tracking_background_mask_blacks_pixels_outside_union \
  tests.test_demo32_headless_render_helper.Demo32HeadlessRenderHelperTest.test_read_target_union_mask_uses_object_or_controller_mask
```

Expected after implementation: both tests pass.

- [ ] **Step 5: Commit helper implementation**

```bash
git add scripts/harness/diagnostics/demo/render_demo32_headless_capture.py tests/test_demo32_headless_render_helper.py
git commit -m "test: cover tracking background mask helpers"
```

## Task 3: Wire Mask Cleanup Into Tracking Render

**Files:**
- Modify: `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py`
- Modify: `tests/test_demo32_headless_render_helper.py`

- [ ] **Step 1: Update existing synthetic tracking test to include mask artifacts**

In `test_render_synthetic_capture_to_video_summary`, after creating `query_trajectory`, create `masks` and write same-frame masks:

```python
            (capture_dir / "masks").mkdir()
            controller_mask = np.zeros((24, 32), dtype=bool)
            object_mask = np.zeros((24, 32), dtype=bool)
            controller_mask[11:14, 15:18] = True
            object_mask[11:14, 17:20] = True
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=controller_mask,
                object_mask=object_mask,
            )
```

Add `"mask_path": "masks/000000.npz"` to that test's `row` dict.

Add these assertions after the existing summary assertions:

```python
            self.assertEqual(summary["tracking_background_mask"], TRACKING_BACKGROUND_MASK_TARGET_UNION)
            self.assertEqual(summary["tracking_background_mask_source"], "object_mask|controller_mask")
            self.assertEqual(
                summary["rendered_counts"][0]["tracking_background_mask_pixels"],
                int(np.count_nonzero(np.logical_or(controller_mask, object_mask))),
            )
            self.assertEqual(
                summary["tracking_background_mask_pixel_total"],
                int(np.count_nonzero(np.logical_or(controller_mask, object_mask))),
            )
```

- [ ] **Step 2: Update query fallback test to include masks for both frames**

In `test_render_does_not_fallback_to_previous_query_trajectory`, create the masks dir before the `for seq in (0, 1)` loop:

```python
            (capture_dir / "masks").mkdir()
```

Inside the same loop, after saving RGB, add:

```python
                controller_mask = np.zeros((24, 32), dtype=bool)
                object_mask = np.zeros((24, 32), dtype=bool)
                controller_mask[10:13, 14:18] = True
                object_mask[10:13, 18:21] = True
                np.savez(
                    capture_dir / "masks" / f"{seq:06d}.npz",
                    controller_mask=controller_mask,
                    object_mask=object_mask,
                )
```

Add `"mask_path": "masks/000000.npz"` and `"mask_path": "masks/000001.npz"` to the two row dictionaries.

- [ ] **Step 3: Add full-RGB compatibility test**

Add this test method:

```python
    def test_tracking_rgb_background_mask_does_not_require_mask_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            metadata = {
                "width": 16,
                "height": 12,
                "saved_pcd_source": "enhanced_pt_filtered",
                "intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 6.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.empty((0, 3), dtype=np.float32),
                controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
                object_xyz_m=np.empty((0, 3), dtype=np.float32),
                object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((12, 16), dtype=np.float32))
            Image.fromarray(np.full((12, 16, 3), 90, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                tracks_yx=np.array([[6.0, 8.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([True], dtype=bool),
                query_is_controller=np.array([False], dtype=bool),
                query_count=np.array([1], dtype=np.int64),
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=capture_dir / "video.mp4",
                fps=30.0,
                tracking_background_mask=TRACKING_BACKGROUND_MASK_RGB,
            )

            self.assertEqual(summary["tracking_background_mask"], TRACKING_BACKGROUND_MASK_RGB)
            self.assertEqual(summary["tracking_background_mask_source"], "full_rgb")
            self.assertEqual(summary["tracking_background_mask_pixel_total"], 0)
            self.assertEqual(summary["rendered_counts"][0]["tracking_background_mask_pixels"], 0)
            self.assertEqual(summary["rendered_counts"][0]["query_points"], 1)
```

- [ ] **Step 4: Run render wiring tests and verify they fail before implementation**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo32_headless_render_helper
```

Expected before render wiring: failures for missing `tracking_background_mask` parameter or missing summary keys.

- [ ] **Step 5: Add render function parameter and validation**

Change `render_capture_to_video` signature to:

```python
def render_capture_to_video(
    *,
    capture_dir: Path,
    output: Path,
    fps: float,
    point_size: int = 2,
    max_render_points: int = 0,
    query_point_radius: int = 3,
    demo_visual_mode: str = "tracking",
    tracking_background_mask: str = TRACKING_BACKGROUND_MASK_TARGET_UNION,
) -> dict[str, Any]:
```

After validating `demo_visual_mode`, add:

```python
    if str(tracking_background_mask) not in TRACKING_BACKGROUND_MASK_MODES:
        raise ValueError(f"tracking_background_mask must be one of {TRACKING_BACKGROUND_MASK_MODES}")
```

- [ ] **Step 6: Apply target-union mask before drawing query points**

Inside the `for frame in frames:` loop, add this initialization near the other per-frame counts:

```python
            tracking_background_mask_pixels = 0
```

Inside `if str(demo_visual_mode) == "tracking":`, immediately after reading RGB and before resolving the query path, add:

```python
                if str(tracking_background_mask) == TRACKING_BACKGROUND_MASK_TARGET_UNION:
                    target_union_mask = _read_target_union_mask(
                        capture_dir=capture_dir,
                        frame=frame,
                        width=width,
                        height=height,
                    )
                    tracking_background_mask_pixels = _apply_tracking_background_mask(image, target_union_mask)
```

Add this field to each `rendered_counts.append` dict:

```python
                    "tracking_background_mask_pixels": int(tracking_background_mask_pixels),
```

- [ ] **Step 7: Add summary fields**

Before building `summary`, compute:

```python
    tracking_background_mask_source = "none"
    if str(demo_visual_mode) == "tracking":
        tracking_background_mask_source = (
            "object_mask|controller_mask"
            if str(tracking_background_mask) == TRACKING_BACKGROUND_MASK_TARGET_UNION
            else "full_rgb"
        )
```

Add these keys to `summary`:

```python
        "tracking_background_mask": str(tracking_background_mask),
        "tracking_background_mask_source": tracking_background_mask_source,
        "tracking_background_mask_pixel_total": int(
            sum(item["tracking_background_mask_pixels"] for item in rendered_counts)
        ),
```

- [ ] **Step 8: Add CLI argument and pass it through**

In `build_parser`, add:

```python
    parser.add_argument(
        "--tracking-background-mask",
        choices=TRACKING_BACKGROUND_MASK_MODES,
        default=TRACKING_BACKGROUND_MASK_TARGET_UNION,
        help="Tracking render RGB background policy: target-union masks table/background, rgb preserves full RGB.",
    )
```

In `main`, pass:

```python
        tracking_background_mask=str(args.tracking_background_mask),
```

- [ ] **Step 9: Run full focused helper tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo32_headless_render_helper
```

Expected: all tests in `tests.test_demo32_headless_render_helper` pass.

- [ ] **Step 10: Commit render wiring**

```bash
git add scripts/harness/diagnostics/demo/render_demo32_headless_capture.py tests/test_demo32_headless_render_helper.py
git commit -m "feat: mask offline tracking render background"
```

## Task 4: Add Fail-Fast Error Coverage

**Files:**
- Modify: `tests/test_demo32_headless_render_helper.py`
- Modify: `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py` only if tests expose message gaps

- [ ] **Step 1: Add a helper for minimal tracking captures**

Add this private method to `Demo32HeadlessRenderHelperTest`:

```python
    def _write_minimal_tracking_capture(self, capture_dir: Path, *, row_extra: dict[str, str] | None = None) -> dict[str, object]:
        (capture_dir / "pcd").mkdir(parents=True)
        (capture_dir / "ffs_depth").mkdir()
        (capture_dir / "rgb").mkdir()
        (capture_dir / "query_trajectory").mkdir()
        metadata = {
            "width": 8,
            "height": 6,
            "saved_pcd_source": "enhanced_pt_filtered",
            "intrinsics": {"fx": 8.0, "fy": 8.0, "cx": 4.0, "cy": 3.0},
        }
        (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        np.savez(
            capture_dir / "pcd" / "000000.npz",
            controller_xyz_m=np.empty((0, 3), dtype=np.float32),
            controller_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_rgb_u8=np.empty((0, 3), dtype=np.uint8),
        )
        np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((6, 8), dtype=np.float32))
        Image.fromarray(np.full((6, 8, 3), 50, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
        np.savez(
            capture_dir / "query_trajectory" / "000000.npz",
            tracks_yx=np.empty((0, 2), dtype=np.float32),
            visibility=np.empty((0,), dtype=np.float32),
            query_indices=np.empty((0,), dtype=np.int64),
        )
        row = {
            "seq": 0,
            "pcd_path": "pcd/000000.npz",
            "ffs_depth_path": "ffs_depth/000000.npy",
            "rgb_path": "rgb/000000.png",
            "query_trajectory_path": "query_trajectory/000000.npz",
        }
        if row_extra:
            row.update(row_extra)
        (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return row
```

- [ ] **Step 2: Add missing `mask_path` error test**

Add:

```python
    def test_tracking_target_union_requires_mask_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            self._write_minimal_tracking_capture(capture_dir)

            with self.assertRaisesRegex(RuntimeError, "requires mask_path"):
                render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)
```

- [ ] **Step 3: Add missing mask file error test**

Add:

```python
    def test_tracking_target_union_requires_existing_mask_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            self._write_minimal_tracking_capture(capture_dir, row_extra={"mask_path": "masks/000000.npz"})

            with self.assertRaisesRegex(RuntimeError, "mask file missing"):
                render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)
```

- [ ] **Step 4: Add missing mask arrays error test**

Add:

```python
    def test_tracking_target_union_requires_object_and_controller_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            self._write_minimal_tracking_capture(capture_dir, row_extra={"mask_path": "masks/000000.npz"})
            np.savez(capture_dir / "masks" / "000000.npz", object_mask=np.zeros((6, 8), dtype=bool))

            with self.assertRaisesRegex(RuntimeError, "controller_mask"):
                render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)
```

- [ ] **Step 5: Add mismatched mask dimensions error test**

Add:

```python
    def test_tracking_target_union_rejects_wrong_mask_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "masks").mkdir(parents=True)
            self._write_minimal_tracking_capture(capture_dir, row_extra={"mask_path": "masks/000000.npz"})
            np.savez(
                capture_dir / "masks" / "000000.npz",
                object_mask=np.zeros((5, 8), dtype=bool),
                controller_mask=np.zeros((5, 8), dtype=bool),
            )

            with self.assertRaisesRegex(RuntimeError, "does not match render shape"):
                render_capture_to_video(capture_dir=capture_dir, output=capture_dir / "video.mp4", fps=30.0)
```

- [ ] **Step 6: Run error coverage tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo32_headless_render_helper
```

Expected: all helper tests pass. If a regex fails because the implementation message is less clear, update the implementation message to match the exact wording in Tasks 2 and 3.

- [ ] **Step 7: Commit fail-fast coverage**

```bash
git add scripts/harness/diagnostics/demo/render_demo32_headless_capture.py tests/test_demo32_headless_render_helper.py
git commit -m "test: cover tracking render mask artifact errors"
```

## Task 5: Update Demo 3.2 Documentation

**Files:**
- Modify: `demo_v3_2/README.md`

- [ ] **Step 1: Update offline tracking render description**

In the "Render the saved artifacts offline" section, replace the tracking-mode paragraph with:

```markdown
Render the saved artifacts offline. In `pcd` mode, the helper draws only the
enhanced-pt filtered RGB point cloud. In `tracking` mode, the helper follows the
FuturePhysTwin 2D tracker view: same-frame RGB target regions plus current-frame
query points only, with stable `gist_rainbow` colors assigned from each query
point's initial y coordinate. By default the tracking renderer applies
`object_mask | controller_mask` to the RGB frame and blacks out table/background
pixels before drawing query points. No PCD and no historical trajectory lines
are drawn in the offline tracking video. It only uses exact same-seq query
trajectory files, so missing query frames are counted rather than silently
matched to an older tracker output.
```

- [ ] **Step 2: Add full-RGB comparison command**

After the existing tracking render command, add:

````markdown
For comparison with the old full-RGB tracking background, pass:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_query_full_rgb_compare.mp4 \
  --fps 30 \
  --demo-visual-mode tracking \
  --tracking-background-mask rgb
```
````

- [ ] **Step 3: Run markdown/path smoke checks for changed docs**

Run:

```bash
git diff --check -- demo_v3_2/README.md
```

Expected: no output, exit 0.

- [ ] **Step 4: Commit docs**

```bash
git add demo_v3_2/README.md
git commit -m "docs: document mask-cleaned tracking render"
```

## Task 6: Final Validation

**Files:**
- Validate all modified files.

- [ ] **Step 1: Run focused render helper tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo32_headless_render_helper
```

Expected: all tests pass.

- [ ] **Step 2: Run py_compile on the render helper**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m py_compile scripts/harness/diagnostics/demo/render_demo32_headless_capture.py
```

Expected: no output, exit 0.

- [ ] **Step 3: Run smoke validation profile**

Run:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

Expected: `[validation] smoke checks passed`.

- [ ] **Step 4: Run diff whitespace check**

Run:

```bash
git diff --check
```

Expected: no output, exit 0.

- [ ] **Step 5: Inspect final changed files**

Run:

```bash
git status --short --untracked-files=all
git log --oneline -5
```

Expected: only intentional changes remain unstaged, if any user changes existed before this work. The new commits should be visible in the recent log.

- [ ] **Step 6: Push single-camera branch after validation**

Run:

```bash
git push origin single-camera
```

Expected: push succeeds to `origin/single-camera`.
