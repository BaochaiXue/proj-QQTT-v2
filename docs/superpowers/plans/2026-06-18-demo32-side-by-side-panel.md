# Demo 3.2 Side-by-Side Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Demo 3.2 1x3 side-by-side panel that works both from fake-live headless captures and during fake-live runtime, with left latest RGB and strict same-seq filtered PCD/tracking columns.

**Architecture:** Add a pure shared renderer in `qqtt/demo/demo32_side_by_side_panel.py`. Extend the Demo 3.2 runtime to save an independent input RGB timeline during headless capture, render side-by-side MP4s offline from capture artifacts, and run a realtime OpenCV panel mode from latest capture plus strict paired packets. Keep PCD/tracking algorithms unchanged.

**Tech Stack:** Python, dataclasses, NumPy, OpenCV, PIL image writing through existing helpers, `unittest`, existing Demo 3.2 fake-live/headless artifacts.

---

## File Structure

- Create: `qqtt/demo/demo32_side_by_side_panel.py`
  - Pure rendering utilities and dataclasses for 1x3 panel composition.
  - Shared projected PCD and tracking overlay helpers used by offline and runtime callers.
- Create: `tests/test_demo32_side_by_side_panel.py`
  - Focused unit tests for panel composition, HUD drawing, projection, and query marker overlay.
- Modify: `qqtt/demo/realtime_masked_edgetam_pcd.py`
  - Add source timing fields to frame/mask/PCD packets.
  - Add headless input RGB timeline writer.
  - Add `panel` render mode and runtime OpenCV panel loop.
  - Add panel CLI flags and validation.
- Modify: `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py`
  - Add `--panel-mode {single,side-by-side}`.
  - Render side-by-side MP4 from headless captures.
  - Write `.panel_summary.json`.
- Modify: `qqtt/demo/single_demo_v3_runtime.py`
  - Forward panel CLI options and expose them in dry-run contract for Demo 3.2 fake-live.
- Modify: `tests/test_recorded_rgbd_replay_source.py`
  - Verify source timestamp/index/step metadata.
- Modify: `tests/test_single_demo_tapnextpp_overlay.py`
  - Verify headless input timeline writing and runtime panel assembly behavior.
- Modify: `tests/test_demo32_headless_render_helper.py`
  - Verify side-by-side offline rendering.
- Modify: `tests/test_single_demo_v3_runtime.py`
  - Verify wrapper contract and panel flag forwarding.
- Modify docs:
  - `demo_v3_2/README.md`
  - `docs/WORKFLOWS.md`
  - `docs/ARCHITECTURE.md`
  - `scripts/harness/README.md`

## Implementation Notes

- Before editing, run:

```bash
git branch --show-current
git pull --ff-only origin main
```

Expected: current branch is `single-camera`; pull reports the branch is up to date or fast-forwards cleanly.

- There are existing uncommitted changes in this workspace. Each task must stage only the files listed in that task.
- Use the full `conda run -n demo_2_max --no-capture-output` commands listed in each task for validation.
- Keep existing `demo_visual_mode=pcd` and `demo_visual_mode=tracking` behavior unchanged.
- Keep `render_mode=pointcloud` as the default. The new runtime panel is selected explicitly with `--render-mode panel`.

---

### Task 1: Shared Side-by-Side Panel Renderer

**Files:**
- Create: `qqtt/demo/demo32_side_by_side_panel.py`
- Create: `tests/test_demo32_side_by_side_panel.py`

- [ ] **Step 1: Write failing tests for panel composition and HUD**

Create `tests/test_demo32_side_by_side_panel.py` with:

```python
from __future__ import annotations

import unittest

import numpy as np

from qqtt.demo.demo32_side_by_side_panel import (
    CAMERA_COLOR_FRAME,
    SideBySidePanelHud,
    SideBySidePanelInputs,
    compute_rgb_ahead_frames,
    render_projected_pcd_panel,
    render_side_by_side_panel,
    render_tracking_overlay_panel,
)


class Demo32SideBySidePanelTest(unittest.TestCase):
    def test_compute_rgb_ahead_frames_clamps_negative_values(self) -> None:
        self.assertEqual(compute_rgb_ahead_frames(rgb_seq=8, paired_seq=5), 3)
        self.assertEqual(compute_rgb_ahead_frames(rgb_seq=5, paired_seq=8), 0)

    def test_render_side_by_side_panel_stacks_three_columns_and_draws_hud(self) -> None:
        left = np.full((4, 5, 3), (10, 20, 30), dtype=np.uint8)
        middle = np.full((4, 5, 3), (40, 50, 60), dtype=np.uint8)
        right = np.full((4, 5, 3), (70, 80, 90), dtype=np.uint8)
        hud = SideBySidePanelHud(
            rgb_seq=9,
            paired_seq=7,
            input_time_s=1.4,
            pipeline_latency_ms=230.0,
            display_latency_ms=245.0,
            startup_hold_s=2.5,
            filter_preset="enhanced-pt",
            marker_count=12,
            tracking_background="target-union",
            object_point_count=3,
            controller_point_count=4,
        )

        panel = render_side_by_side_panel(
            SideBySidePanelInputs(
                rgb_image_bgr=left,
                pcd_panel_bgr=middle,
                tracking_panel_bgr=right,
                hud=hud,
            )
        )

        self.assertEqual(panel.shape, (4, 15, 3))
        self.assertGreater(int(panel.sum()), int(left.sum() + middle.sum() + right.sum()))

    def test_render_side_by_side_panel_resizes_inputs_to_output_cell(self) -> None:
        left = np.full((4, 5, 3), 20, dtype=np.uint8)
        middle = np.full((8, 10, 3), 40, dtype=np.uint8)
        right = np.full((2, 3, 3), 60, dtype=np.uint8)
        hud = SideBySidePanelHud(
            rgb_seq=1,
            paired_seq=1,
            input_time_s=0.0,
            pipeline_latency_ms=1.0,
            display_latency_ms=2.0,
            startup_hold_s=0.0,
            filter_preset="pt",
            marker_count=0,
        )

        panel = render_side_by_side_panel(
            SideBySidePanelInputs(left, middle, right, hud),
            cell_size=(6, 4),
        )

        self.assertEqual(panel.shape, (4, 18, 3))

    def test_render_projected_pcd_panel_draws_camera_frame_points(self) -> None:
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        panel, count = render_projected_pcd_panel(
            width=8,
            height=6,
            intrinsics={"fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0},
            controller_xyz_m=points[:1],
            controller_rgb_u8=colors[:1],
            object_xyz_m=points[1:],
            object_rgb_u8=colors[1:],
            point_size=1,
            max_render_points=0,
            coordinate_frame=CAMERA_COLOR_FRAME,
            camera_to_world_c2w=None,
        )

        self.assertEqual(panel.shape, (6, 8, 3))
        self.assertEqual(count["controller_points"], 1)
        self.assertEqual(count["object_points"], 1)
        self.assertGreater(int(panel.sum()), 0)

    def test_render_tracking_overlay_panel_draws_visible_query_points(self) -> None:
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        tracks_yx = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        visibility = np.array([1.0, 0.0], dtype=np.float32)
        marker_rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

        panel, counts = render_tracking_overlay_panel(
            image_bgr=image,
            tracks_yx=tracks_yx,
            visibility=visibility,
            marker_rgb_u8=marker_rgb,
            query_is_object=np.array([True, False], dtype=bool),
            query_is_controller=np.array([False, True], dtype=bool),
            query_controller_instance_id=np.array([0, 1], dtype=np.int64),
            marker_radius=1,
        )

        self.assertEqual(panel.shape, (6, 8, 3))
        self.assertEqual(counts["query_points"], 1)
        self.assertEqual(counts["query_object_points"], 1)
        self.assertEqual(counts["query_controller_points"], 0)
        self.assertGreater(int(panel.sum()), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo32_side_by_side_panel
```

Expected: FAIL with `ModuleNotFoundError: No module named 'qqtt.demo.demo32_side_by_side_panel'`.

- [ ] **Step 3: Add the shared renderer module**

Create `qqtt/demo/demo32_side_by_side_panel.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


CAMERA_COLOR_FRAME = "camera_color_frame"
TABLE_WORLD_FRAME_KIND = "table_world_z0"


@dataclass(frozen=True)
class SideBySidePanelHud:
    rgb_seq: int
    paired_seq: int
    input_time_s: float | None
    pipeline_latency_ms: float
    display_latency_ms: float
    startup_hold_s: float
    filter_preset: str
    marker_count: int
    tracking_background: str = "target-union"
    object_point_count: int = 0
    controller_point_count: int = 0

    @property
    def rgb_ahead_frames(self) -> int:
        return compute_rgb_ahead_frames(rgb_seq=self.rgb_seq, paired_seq=self.paired_seq)


@dataclass(frozen=True)
class SideBySidePanelInputs:
    rgb_image_bgr: np.ndarray
    pcd_panel_bgr: np.ndarray
    tracking_panel_bgr: np.ndarray
    hud: SideBySidePanelHud


def compute_rgb_ahead_frames(*, rgb_seq: int, paired_seq: int) -> int:
    return max(0, int(rgb_seq) - int(paired_seq))


def _as_bgr_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"panel image must be HxWx3, got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _resize_to_cell(image: np.ndarray, cell_size: tuple[int, int]) -> np.ndarray:
    width, height = int(cell_size[0]), int(cell_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("cell_size must contain positive width and height")
    image = _as_bgr_u8(image)
    if image.shape[:2] == (height, width):
        return image.copy()
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _draw_text_lines(image: np.ndarray, lines: list[str], *, origin: tuple[int, int]) -> None:
    if not lines:
        return
    x, y = int(origin[0]), int(origin[1])
    line_height = 16
    width = min(image.shape[1] - x, max(1, max(len(line) for line in lines) * 8 + 12))
    height = min(image.shape[0] - y, line_height * len(lines) + 8)
    if width <= 0 or height <= 0:
        return
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, dst=image)
    cursor_y = y + 15
    for line in lines:
        cv2.putText(
            image,
            line,
            (x + 6, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cursor_y += line_height


def _hud_lines(hud: SideBySidePanelHud) -> list[str]:
    input_time = "none" if hud.input_time_s is None else f"{float(hud.input_time_s):.2f}s"
    return [
        f"rgb_seq={int(hud.rgb_seq)} paired_seq={int(hud.paired_seq)} rgb_ahead={hud.rgb_ahead_frames}f",
        f"input_t={input_time} pipeline={float(hud.pipeline_latency_ms):.1f}ms display={float(hud.display_latency_ms):.1f}ms",
        f"startup_hold={float(hud.startup_hold_s):.2f}s filter={hud.filter_preset} markers={int(hud.marker_count)}",
    ]


def render_side_by_side_panel(
    inputs: SideBySidePanelInputs,
    *,
    cell_size: tuple[int, int] | None = None,
) -> np.ndarray:
    left = _as_bgr_u8(inputs.rgb_image_bgr)
    if cell_size is None:
        cell_size = (int(left.shape[1]), int(left.shape[0]))
    left = _resize_to_cell(inputs.rgb_image_bgr, cell_size)
    middle = _resize_to_cell(inputs.pcd_panel_bgr, cell_size)
    right = _resize_to_cell(inputs.tracking_panel_bgr, cell_size)
    _draw_text_lines(left, ["RGB latest"], origin=(4, 4))
    _draw_text_lines(middle, [f"Filtered PCD obj={inputs.hud.object_point_count} ctrl={inputs.hud.controller_point_count}"], origin=(4, 4))
    _draw_text_lines(right, [f"Tracking bg={inputs.hud.tracking_background}"], origin=(4, 4))
    panel = np.concatenate([left, middle, right], axis=1)
    _draw_text_lines(panel, _hud_lines(inputs.hud), origin=(4, max(4, panel.shape[0] - 56)))
    return np.ascontiguousarray(panel, dtype=np.uint8)


def _world_to_camera(points_xyz: np.ndarray, camera_to_world_c2w: Any) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    c2w = np.asarray(camera_to_world_c2w, dtype=np.float32).reshape(4, 4)
    if len(points) == 0:
        return points
    w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return np.ascontiguousarray((w2c @ homogeneous.T).T[:, :3], dtype=np.float32)


def _project_points(
    points_xyz: np.ndarray,
    intrinsics: dict[str, Any],
    *,
    width: int,
    height: int,
    coordinate_frame: str,
    camera_to_world_c2w: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=bool)
    if str(coordinate_frame) == TABLE_WORLD_FRAME_KIND:
        if camera_to_world_c2w is None:
            raise RuntimeError("table_world_z0 projection requires camera_to_world_c2w")
        points = _world_to_camera(points, camera_to_world_c2w)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > np.float32(1e-6))
    fx = np.float32(intrinsics["fx"])
    fy = np.float32(intrinsics["fy"])
    cx = np.float32(intrinsics["cx"])
    cy = np.float32(intrinsics["cy"])
    u = np.rint(points[:, 0] * fx / z + cx).astype(np.int32)
    v = np.rint(points[:, 1] * fy / z + cy).astype(np.int32)
    valid &= (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    return np.stack([u, v], axis=1), valid


def _draw_projected_points(
    image_bgr: np.ndarray,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    intrinsics: dict[str, Any],
    *,
    point_size: int,
    max_points: int,
    coordinate_frame: str,
    camera_to_world_c2w: Any | None,
) -> int:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
    if len(points) == 0:
        return 0
    if int(max_points) > 0 and len(points) > int(max_points):
        indices = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
        points = points[indices]
        colors = colors[indices]
    pixels, valid = _project_points(
        points,
        intrinsics,
        width=image_bgr.shape[1],
        height=image_bgr.shape[0],
        coordinate_frame=coordinate_frame,
        camera_to_world_c2w=camera_to_world_c2w,
    )
    radius = max(1, int(point_size))
    drawn = 0
    for (u, v), ok, rgb in zip(pixels, valid, colors, strict=False):
        if not bool(ok):
            continue
        cv2.circle(image_bgr, (int(u), int(v)), radius, tuple(int(value) for value in rgb[::-1]), thickness=-1)
        drawn += 1
    return drawn


def render_projected_pcd_panel(
    *,
    width: int,
    height: int,
    intrinsics: dict[str, Any],
    controller_xyz_m: np.ndarray,
    controller_rgb_u8: np.ndarray,
    object_xyz_m: np.ndarray,
    object_rgb_u8: np.ndarray,
    point_size: int,
    max_render_points: int,
    coordinate_frame: str,
    camera_to_world_c2w: Any | None,
) -> tuple[np.ndarray, dict[str, int]]:
    image = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    controller_count = _draw_projected_points(
        image,
        controller_xyz_m,
        controller_rgb_u8,
        intrinsics,
        point_size=point_size,
        max_points=max_render_points,
        coordinate_frame=coordinate_frame,
        camera_to_world_c2w=camera_to_world_c2w,
    )
    object_count = _draw_projected_points(
        image,
        object_xyz_m,
        object_rgb_u8,
        intrinsics,
        point_size=point_size,
        max_points=max_render_points,
        coordinate_frame=coordinate_frame,
        camera_to_world_c2w=camera_to_world_c2w,
    )
    return image, {"controller_points": controller_count, "object_points": object_count}


def render_tracking_overlay_panel(
    *,
    image_bgr: np.ndarray,
    tracks_yx: np.ndarray,
    visibility: np.ndarray,
    marker_rgb_u8: np.ndarray,
    query_is_object: np.ndarray,
    query_is_controller: np.ndarray,
    query_controller_instance_id: np.ndarray,
    marker_radius: int,
) -> tuple[np.ndarray, dict[str, int]]:
    image = _as_bgr_u8(image_bgr).copy()
    tracks = np.asarray(tracks_yx, dtype=np.float32).reshape(-1, 2)
    visible = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0.0
    colors = np.asarray(marker_rgb_u8, dtype=np.uint8).reshape(-1, 3)
    is_object = np.asarray(query_is_object, dtype=bool).reshape(-1)
    is_controller = np.asarray(query_is_controller, dtype=bool).reshape(-1)
    controller_instance = np.asarray(query_controller_instance_id, dtype=np.int64).reshape(-1)
    count = min(len(tracks), len(visible), len(colors), len(is_object), len(is_controller), len(controller_instance))
    if count <= 0:
        return image, {
            "query_points": 0,
            "query_object_points": 0,
            "query_controller_points": 0,
            "query_hand_a_points": 0,
            "query_hand_b_points": 0,
        }
    radius = max(1, int(marker_radius))
    visible_count = object_count = controller_count = hand_a_count = hand_b_count = 0
    for index in range(count):
        if not bool(visible[index]):
            continue
        y, x = float(tracks[index, 0]), float(tracks[index, 1])
        if not np.isfinite([y, x]).all():
            continue
        yy, xx = int(round(y)), int(round(x))
        if yy < 0 or yy >= image.shape[0] or xx < 0 or xx >= image.shape[1]:
            continue
        color_bgr = tuple(int(value) for value in colors[index, ::-1])
        cv2.circle(image, (xx, yy), radius, color_bgr, thickness=-1)
        visible_count += 1
        if bool(is_controller[index]):
            controller_count += 1
            if int(controller_instance[index]) == 1:
                hand_a_count += 1
            elif int(controller_instance[index]) == 2:
                hand_b_count += 1
        elif bool(is_object[index]):
            object_count += 1
    return image, {
        "query_points": visible_count,
        "query_object_points": object_count,
        "query_controller_points": controller_count,
        "query_hand_a_points": hand_a_count,
        "query_hand_b_points": hand_b_count,
    }
```

- [ ] **Step 4: Run renderer tests and verify they pass**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo32_side_by_side_panel
```

Expected: all tests in `tests.test_demo32_side_by_side_panel` pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add qqtt/demo/demo32_side_by_side_panel.py tests/test_demo32_side_by_side_panel.py
git commit -m "Add Demo 3.2 side-by-side panel renderer"
```

---

### Task 2: Source Timing and Headless Input RGB Timeline

**Files:**
- Modify: `qqtt/demo/realtime_masked_edgetam_pcd.py`
- Modify: `tests/test_recorded_rgbd_replay_source.py`
- Modify: `tests/test_single_demo_tapnextpp_overlay.py`

- [ ] **Step 1: Add failing source metadata assertions to recording replay tests**

In `tests/test_recorded_rgbd_replay_source.py`, inside `test_numeric_step_order_remaps_first_complete_frame_to_seq_zero`, after `self.assertEqual(packet.depth_u16[0, 0].item(), 2)`, add:

```python
            self.assertEqual(packet.source_frame_index, 0)
            self.assertEqual(packet.source_step, 2)
            self.assertAlmostEqual(packet.source_timestamp_s, 2.0)
```

Inside the same test, after `self.assertEqual(remapped_packet.seq, 7)`, add:

```python
            self.assertEqual(remapped_packet.source_frame_index, 1)
            self.assertEqual(remapped_packet.source_step, 10)
            self.assertAlmostEqual(remapped_packet.source_timestamp_s, 10.0)
```

- [ ] **Step 2: Add failing headless input timeline assertions**

In `tests/test_single_demo_tapnextpp_overlay.py`, in `test_headless_capture_writer_saves_filtered_pcd_depth_and_query_payloads`, immediately after the `demo.HeadlessCaptureWriter` construction block, add:

```python
            input_packet = self._frame_packet(seq=0)
            writer.write_input_frame(input_packet)
```

After the existing metadata assertions for `saved_rgb_source`, add:

```python
            self.assertTrue(metadata["panel_supported"])
            self.assertEqual(metadata["panel_sync_policy"], "left_latest_rgb_right_strict_same_seq")
            self.assertEqual(metadata["input_rgb_timeline"], "input_frames.jsonl")
```

After reading `rows`, add:

```python
            input_rows = [
                json.loads(line)
                for line in (output_dir / "input_frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(input_rows), 1)
            self.assertEqual(input_rows[0]["seq"], 0)
            self.assertEqual(input_rows[0]["input_rgb_path"], "input_rgb/000000.png")
            self.assertTrue((output_dir / input_rows[0]["input_rgb_path"]).is_file())
```

After `self.assertEqual(rows[0]["object_query_count"], 1)`, add:

```python
            self.assertEqual(rows[0]["marker_count"], 1)
            self.assertEqual(rows[0]["filter_preset"], "pt_filter_filtered")
            self.assertGreaterEqual(rows[0]["pipeline_latency_ms"], 0.0)
```

- [ ] **Step 3: Run targeted tests and verify they fail**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_recorded_rgbd_replay_source.RecordedRgbdReplaySourceTest.test_numeric_step_order_remaps_first_complete_frame_to_seq_zero \
  tests.test_single_demo_tapnextpp_overlay.RealtimeMaskedEdgeTamPcdTest.test_headless_capture_writer_saves_filtered_pcd_depth_and_query_payloads
```

Expected: FAIL because `FramePacket.source_frame_index` and `HeadlessCaptureWriter.write_input_frame` do not exist yet.

- [ ] **Step 4: Extend packet dataclasses with source fields**

In `qqtt/demo/realtime_masked_edgetam_pcd.py`, add these fields to `FramePacket`, `MaskPacket`, and `MaskedPcdPacket` dataclasses:

```python
    source_timestamp_s: float | None = None
    source_frame_index: int | None = None
    source_step: int | None = None
```

In `RecordedRgbdFrameSource.read_packet`, add these keyword arguments to the `FramePacket` constructor:

```python
            source_timestamp_s=float(ref.timestamp_s),
            source_frame_index=int(source_index),
            source_step=int(ref.step),
```

In `_run_segmentation_frame`, add these keyword arguments to the `MaskPacket` constructor:

```python
            source_timestamp_s=frame.source_timestamp_s,
            source_frame_index=frame.source_frame_index,
            source_step=frame.source_step,
```

In the functions that build `MaskedPcdPacket`, pass the source fields from the active `mask_packet`:

```python
            source_timestamp_s=mask_packet.source_timestamp_s,
            source_frame_index=mask_packet.source_frame_index,
            source_step=mask_packet.source_step,
```

There are two `MaskedPcdPacket` construction paths in this file: the normal dense PCD path and the sparse remote FFS return path. Update both constructors.

- [ ] **Step 5: Add input RGB timeline writer**

In `HeadlessCaptureWriter.__init__`, add:

```python
        self.input_rgb_dir = self.output_dir / "input_rgb"
        self.input_frames_path = self.output_dir / "input_frames.jsonl"
```

After `self.mask_dir.mkdir(parents=True, exist_ok=True)`, add:

```python
        self.input_rgb_dir.mkdir(parents=True, exist_ok=True)
```

After `self.frames_path.write_text("", encoding="utf-8")`, add:

```python
        self.input_frames_path.write_text("", encoding="utf-8")
```

Before writing `metadata.json`, add these metadata fields to `payload`:

```python
        payload["panel_supported"] = True
        payload["panel_sync_policy"] = "left_latest_rgb_right_strict_same_seq"
        payload["tracking_background_default"] = "target-union"
        payload["input_rgb_timeline"] = "input_frames.jsonl"
        payload["startup_hold_s"] = float(payload.get("startup_hold_s") or 0.0)
```

Immediately before the existing metadata file write call, add:

```python
        self._metadata_payload = payload
```

Add this method to `HeadlessCaptureWriter` below `_relative`:

```python
    def update_metadata(self, values: dict[str, Any]) -> None:
        payload = dict(self._metadata_payload)
        payload.update(values)
        with self._lock:
            self._metadata_payload = payload
            self.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_input_frame(self, packet: FramePacket) -> None:
        seq_name = f"{int(packet.seq):06d}"
        rgb_path = self.input_rgb_dir / f"{seq_name}.png"
        _bgr_to_pil_rgb(packet.color_bgr).save(rgb_path)
        row = {
            "seq": int(packet.seq),
            "input_rgb_path": self._relative(rgb_path),
            "source_timestamp_s": (
                None if packet.source_timestamp_s is None else float(packet.source_timestamp_s)
            ),
            "source_frame_index": (
                None if packet.source_frame_index is None else int(packet.source_frame_index)
            ),
            "source_step": None if packet.source_step is None else int(packet.source_step),
            "receive_perf_s": float(packet.receive_perf_s),
        }
        with self._lock:
            with self.input_frames_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
```

In `_publish_capture_packet`, after `self.capture_slot.put(packet)`, add:

```python
        if self.headless_capture_writer is not None and _is_replay_input_source(str(self.args.input_source)):
            self.headless_capture_writer.write_input_frame(packet)
```

- [ ] **Step 6: Add paired row metadata to `write_pcd`**

Update the `HeadlessCaptureWriter.write_pcd` signature to include:

```python
        startup_hold_s: float = 0.0,
```

In `HeadlessCaptureWriter.write_pcd`, add these fields to `row`:

```python
            "source_timestamp_s": (
                None if packet.source_timestamp_s is None else float(packet.source_timestamp_s)
            ),
            "source_frame_index": (
                None if packet.source_frame_index is None else int(packet.source_frame_index)
            ),
            "source_step": None if packet.source_step is None else int(packet.source_step),
            "startup_hold_s": float(startup_hold_s),
            "pipeline_latency_ms": float(
                max(packet.process_done_perf_s, tracker_packet.process_done_perf_s)
                - float(packet.receive_perf_s)
            )
            * 1000.0
            if tracker_packet is not None
            else float(packet.process_done_perf_s - packet.receive_perf_s) * 1000.0,
            "filter_preset": self.saved_pcd_source,
            "marker_count": int(tracker_packet.marker_count) if tracker_packet is not None else 0,
```

In `_write_headless_pcd_result`, pass the runtime value:

```python
            startup_hold_s=float(getattr(self, "_startup_hold_s", 0.0)),
```

- [ ] **Step 7: Run targeted tests and verify they pass**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_recorded_rgbd_replay_source.RecordedRgbdReplaySourceTest.test_numeric_step_order_remaps_first_complete_frame_to_seq_zero \
  tests.test_single_demo_tapnextpp_overlay.RealtimeMaskedEdgeTamPcdTest.test_headless_capture_writer_saves_filtered_pcd_depth_and_query_payloads
```

Expected: both tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add qqtt/demo/realtime_masked_edgetam_pcd.py tests/test_recorded_rgbd_replay_source.py tests/test_single_demo_tapnextpp_overlay.py
git commit -m "Add Demo 3.2 headless input RGB timeline"
```

---

### Task 3: Offline Side-by-Side MP4 Renderer

**Files:**
- Modify: `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py`
- Modify: `tests/test_demo32_headless_render_helper.py`

- [ ] **Step 1: Add failing offline side-by-side render test**

In `tests/test_demo32_headless_render_helper.py`, update imports:

```python
from scripts.harness.diagnostics.demo.render_demo32_headless_capture import (
    PANEL_MODE_SIDE_BY_SIDE,
    TRACKING_BACKGROUND_MASK_RGB,
    TRACKING_BACKGROUND_MASK_TARGET_UNION,
    _apply_tracking_background_mask,
    _project_points,
    _read_target_union_mask,
    render_capture_to_video,
    render_table_z_filter_overlay_sweep,
)
```

Add this test method to `Demo32HeadlessRenderHelperTest`:

```python
    def test_render_side_by_side_panel_prefers_input_rgb_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp) / "capture"
            (capture_dir / "pcd").mkdir(parents=True)
            (capture_dir / "ffs_depth").mkdir()
            (capture_dir / "rgb").mkdir()
            (capture_dir / "input_rgb").mkdir()
            (capture_dir / "query_trajectory").mkdir()
            (capture_dir / "masks").mkdir()
            metadata = {
                "width": 32,
                "height": 24,
                "saved_pcd_source": "enhanced_pt_filtered",
                "pcd_filter_preset": "enhanced-pt",
                "replay_fps": 5.0,
                "startup_hold_s": 2.0,
                "input_rgb_timeline": "input_frames.jsonl",
                "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 12.0},
            }
            (capture_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            for seq, value in ((0, 30), (2, 120)):
                Image.fromarray(np.full((24, 32, 3), value, dtype=np.uint8)).save(
                    capture_dir / "input_rgb" / f"{seq:06d}.png"
                )
            Image.fromarray(np.full((24, 32, 3), 60, dtype=np.uint8)).save(capture_dir / "rgb" / "000000.png")
            np.savez(
                capture_dir / "pcd" / "000000.npz",
                controller_xyz_m=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
                controller_rgb_u8=np.array([[255, 0, 0]], dtype=np.uint8),
                object_xyz_m=np.array([[0.05, 0.0, 0.6]], dtype=np.float32),
                object_rgb_u8=np.array([[0, 255, 0]], dtype=np.uint8),
            )
            np.save(capture_dir / "ffs_depth" / "000000.npy", np.ones((24, 32), dtype=np.float32))
            np.savez(
                capture_dir / "query_trajectory" / "000000.npz",
                marker_rgb_u8=np.array([[255, 32, 32]], dtype=np.uint8),
                tracks_yx=np.array([[12.0, 16.0]], dtype=np.float32),
                visibility=np.ones((1,), dtype=np.float32),
                query_indices=np.array([0], dtype=np.int64),
                query_is_object=np.array([True], dtype=bool),
                query_is_controller=np.array([False], dtype=bool),
                query_controller_instance_id=np.array([0], dtype=np.int64),
                query_count=np.array([1], dtype=np.int64),
            )
            controller_mask = np.zeros((24, 32), dtype=bool)
            object_mask = np.zeros((24, 32), dtype=bool)
            object_mask[10:15, 14:20] = True
            np.savez(
                capture_dir / "masks" / "000000.npz",
                controller_mask=controller_mask,
                object_mask=object_mask,
            )
            input_rows = [
                {
                    "seq": 0,
                    "input_rgb_path": "input_rgb/000000.png",
                    "source_timestamp_s": 0.0,
                    "receive_perf_s": 10.0,
                },
                {
                    "seq": 2,
                    "input_rgb_path": "input_rgb/000002.png",
                    "source_timestamp_s": 0.4,
                    "receive_perf_s": 10.4,
                },
            ]
            (capture_dir / "input_frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in input_rows) + "\n",
                encoding="utf-8",
            )
            row = {
                "seq": 0,
                "pcd_path": "pcd/000000.npz",
                "ffs_depth_path": "ffs_depth/000000.npy",
                "rgb_path": "rgb/000000.png",
                "query_trajectory_path": "query_trajectory/000000.npz",
                "mask_path": "masks/000000.npz",
                "source_timestamp_s": 0.0,
                "receive_perf_s": 10.0,
                "process_done_perf_s": 10.4,
                "pipeline_latency_ms": 400.0,
                "filter_preset": "enhanced-pt",
                "marker_count": 1,
            }
            (capture_dir / "frames.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            output = capture_dir / "side_by_side.mp4"
            summary = render_capture_to_video(
                capture_dir=capture_dir,
                output=output,
                fps=30.0,
                demo_visual_mode="tracking",
                panel_mode=PANEL_MODE_SIDE_BY_SIDE,
            )

            self.assertTrue(output.is_file())
            self.assertTrue((capture_dir / "side_by_side.panel_summary.json").is_file())
            self.assertEqual(summary["panel_mode"], PANEL_MODE_SIDE_BY_SIDE)
            self.assertEqual(summary["left_rgb_policy"], "latest_input_rgb")
            self.assertEqual(summary["rendered_counts"][0]["rgb_seq"], 2)
            self.assertEqual(summary["rendered_counts"][0]["paired_seq"], 0)
            self.assertEqual(summary["rendered_counts"][0]["rgb_ahead_frames"], 2)
            self.assertEqual(summary["rendered_counts"][0]["query_points"], 1)
```

- [ ] **Step 2: Run the side-by-side render test and verify it fails**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_demo32_headless_render_helper.Demo32HeadlessRenderHelperTest.test_render_side_by_side_panel_prefers_input_rgb_timeline
```

Expected: FAIL because `PANEL_MODE_SIDE_BY_SIDE` and `panel_mode` support do not exist.

- [ ] **Step 3: Add panel mode constants and input frame helpers**

In `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py`, add import:

```python
from qqtt.demo.demo32_side_by_side_panel import (
    SideBySidePanelHud,
    SideBySidePanelInputs,
    compute_rgb_ahead_frames,
    render_projected_pcd_panel,
    render_side_by_side_panel,
    render_tracking_overlay_panel,
)
```

Add constants near `DEMO_VISUAL_MODES`:

```python
PANEL_MODE_SINGLE = "single"
PANEL_MODE_SIDE_BY_SIDE = "side-by-side"
PANEL_MODES = (PANEL_MODE_SINGLE, PANEL_MODE_SIDE_BY_SIDE)
```

Add helper functions after `_read_frames`:

```python
def _read_input_frames(capture_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = str(metadata.get("input_rgb_timeline") or "input_frames.jsonl")
    path = capture_dir / timeline
    if not path.is_file():
        return []
    return _read_frames(path)


def _read_input_rgb_frame_bgr(*, capture_dir: Path, input_frame: dict[str, Any], width: int, height: int) -> np.ndarray:
    if "input_rgb_path" not in input_frame:
        raise RuntimeError("input RGB timeline row requires input_rgb_path")
    path = _resolve_capture_path(capture_dir, str(input_frame["input_rgb_path"]))
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read input RGB frame: {path}")
    if image.shape[:2] != (int(height), int(width)):
        image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _latest_input_frame_for_paired_row(
    *,
    paired_row: dict[str, Any],
    input_frames: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not input_frames:
        return None
    paired_done_s = paired_row.get("process_done_perf_s")
    if paired_done_s is None:
        paired_done_s = paired_row.get("receive_perf_s")
    if paired_done_s is None:
        return None
    candidates = [
        row for row in input_frames
        if row.get("receive_perf_s") is not None and float(row["receive_perf_s"]) <= float(paired_done_s)
    ]
    if not candidates:
        return input_frames[0]
    return max(candidates, key=lambda row: int(row.get("seq", -1)))
```

- [ ] **Step 4: Add side-by-side render branch**

Update `render_capture_to_video` signature:

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
    panel_mode: str = PANEL_MODE_SINGLE,
) -> dict[str, Any]:
```

After validating `demo_visual_mode`, add:

```python
    if str(panel_mode) not in PANEL_MODES:
        raise ValueError(f"panel_mode must be one of {PANEL_MODES}")
```

Change `VideoWriter` size:

```python
    writer_size = (width * 3, height) if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE else (width, height)
```

and pass `writer_size` to `cv2.VideoWriter`.

Before the loop, add:

```python
    input_frames = _read_input_frames(capture_dir, metadata)
    left_rgb_policy = "latest_input_rgb" if input_frames else "same_seq_fallback"
    missing_rgb_frames = 0
```

Inside the loop, after the existing single-mode image is produced, add this branch before the video writer writes the frame:

```python
            if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE:
                paired_seq = int(frame["seq"])
                input_frame = _latest_input_frame_for_paired_row(paired_row=frame, input_frames=input_frames)
                if input_frame is None:
                    rgb_seq = paired_seq
                    rgb_image = _read_rgb_frame_bgr(capture_dir=capture_dir, frame=frame, width=width, height=height)
                    missing_rgb_frames += 1
                else:
                    rgb_seq = int(input_frame["seq"])
                    rgb_image = _read_input_rgb_frame_bgr(
                        capture_dir=capture_dir,
                        input_frame=input_frame,
                        width=width,
                        height=height,
                    )
                pcd_path = _resolve_capture_path(capture_dir, str(frame["pcd_path"]))
                with np.load(pcd_path, allow_pickle=False) as pcd:
                    pcd_panel, pcd_counts = render_projected_pcd_panel(
                        width=width,
                        height=height,
                        intrinsics=intrinsics,
                        controller_xyz_m=pcd["controller_xyz_m"],
                        controller_rgb_u8=pcd["controller_rgb_u8"],
                        object_xyz_m=pcd["object_xyz_m"],
                        object_rgb_u8=pcd["object_rgb_u8"],
                        point_size=int(point_size),
                        max_render_points=int(max_render_points),
                        coordinate_frame=pcd_coordinate_frame,
                        camera_to_world_c2w=camera_to_world_c2w,
                    )
                tracking_image = _read_rgb_frame_bgr(capture_dir=capture_dir, frame=frame, width=width, height=height)
                if str(tracking_background_mask) == TRACKING_BACKGROUND_MASK_TARGET_UNION:
                    target_union_mask = _read_target_union_mask(
                        capture_dir=capture_dir,
                        frame=frame,
                        width=width,
                        height=height,
                    )
                    tracking_background_mask_pixels = _apply_tracking_background_mask(tracking_image, target_union_mask)
                query_path = _trajectory_path_for_frame(
                    capture_dir=capture_dir,
                    frame=frame,
                    trajectory_by_seq=trajectory_by_seq,
                )
                if query_path is None or not query_path.is_file():
                    missing_query_frames += 1
                    tracking_panel = tracking_image
                    tracking_counts = {
                        "query_points": 0,
                        "query_object_points": 0,
                        "query_controller_points": 0,
                        "query_hand_a_points": 0,
                        "query_hand_b_points": 0,
                    }
                else:
                    with np.load(query_path, allow_pickle=False) as query_payload:
                        marker_rgb = (
                            query_payload["marker_rgb_u8"]
                            if "marker_rgb_u8" in query_payload.files
                            else query_payload["query_rgb_u8"]
                        )
                        tracking_panel, tracking_counts = render_tracking_overlay_panel(
                            image_bgr=tracking_image,
                            tracks_yx=query_payload["tracks_yx"],
                            visibility=query_payload["visibility"],
                            marker_rgb_u8=marker_rgb,
                            query_is_object=query_payload["query_is_object"],
                            query_is_controller=query_payload["query_is_controller"],
                            query_controller_instance_id=query_payload["query_controller_instance_id"],
                            marker_radius=int(query_point_radius),
                        )
                display_done_s = time.perf_counter()
                display_latency_ms = (
                    float(display_done_s - float(frame["receive_perf_s"])) * 1000.0
                    if frame.get("receive_perf_s") is not None
                    else 0.0
                )
                hud = SideBySidePanelHud(
                    rgb_seq=rgb_seq,
                    paired_seq=paired_seq,
                    input_time_s=frame.get("source_timestamp_s"),
                    pipeline_latency_ms=float(frame.get("pipeline_latency_ms", 0.0) or 0.0),
                    display_latency_ms=display_latency_ms,
                    startup_hold_s=float(frame.get("startup_hold_s", metadata.get("startup_hold_s", 0.0)) or 0.0),
                    filter_preset=str(frame.get("filter_preset", metadata.get("pcd_filter_preset", metadata.get("saved_pcd_source", "unknown")))),
                    marker_count=int(frame.get("marker_count", tracking_counts["query_points"])),
                    tracking_background=str(tracking_background_mask),
                    object_point_count=int(pcd_counts["object_points"]),
                    controller_point_count=int(pcd_counts["controller_points"]),
                )
                image = render_side_by_side_panel(
                    SideBySidePanelInputs(
                        rgb_image_bgr=rgb_image,
                        pcd_panel_bgr=pcd_panel,
                        tracking_panel_bgr=tracking_panel,
                        hud=hud,
                    )
                )
                controller_count = int(pcd_counts["controller_points"])
                object_count = int(pcd_counts["object_points"])
                query_count = int(tracking_counts["query_points"])
                query_object_count = int(tracking_counts["query_object_points"])
                query_controller_count = int(tracking_counts["query_controller_points"])
                query_hand_a_count = int(tracking_counts["query_hand_a_points"])
                query_hand_b_count = int(tracking_counts["query_hand_b_points"])
```

At the dictionary passed to `rendered_counts.append`, add:

```python
                    "rgb_seq": int(rgb_seq if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE else frame["seq"]),
                    "paired_seq": int(frame["seq"]),
                    "rgb_ahead_frames": int(
                        compute_rgb_ahead_frames(
                            rgb_seq=int(rgb_seq if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE else frame["seq"]),
                            paired_seq=int(frame["seq"]),
                        )
                    ),
```

Add panel fields to `summary`:

```python
        "panel_mode": str(panel_mode),
        "left_rgb_policy": left_rgb_policy,
        "input_rgb_frame_count": int(len(input_frames)),
        "missing_rgb_frames": int(missing_rgb_frames),
        "sync_policy": "left_latest_rgb_right_strict_same_seq"
        if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE
        else "single_view",
```

Write an additional summary for side-by-side mode:

```python
    if str(panel_mode) == PANEL_MODE_SIDE_BY_SIDE:
        panel_summary_path = output.with_suffix(".panel_summary.json")
        panel_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

- [ ] **Step 5: Add CLI option**

In `build_parser`, add:

```python
    parser.add_argument("--panel-mode", choices=PANEL_MODES, default=PANEL_MODE_SINGLE)
```

When calling `render_capture_to_video` from `main`, pass:

```python
        panel_mode=str(args.panel_mode),
```

- [ ] **Step 6: Run offline renderer tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo32_headless_render_helper
```

Expected: all tests in `tests.test_demo32_headless_render_helper` pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add scripts/harness/diagnostics/demo/render_demo32_headless_capture.py tests/test_demo32_headless_render_helper.py
git commit -m "Render Demo 3.2 side-by-side headless panels"
```

---

### Task 4: Runtime Realtime Panel Mode

**Files:**
- Modify: `qqtt/demo/realtime_masked_edgetam_pcd.py`
- Modify: `tests/test_single_demo_tapnextpp_overlay.py`

- [ ] **Step 1: Add failing validation and helper tests**

In `tests/test_single_demo_tapnextpp_overlay.py`, add:

```python
    def test_panel_render_mode_requires_fake_live_lossless_tracking(self) -> None:
        args = demo.build_parser().parse_args(
            [
                "--render-mode",
                "panel",
                "--input-source",
                "live",
                "--track-mode",
                "controller-object",
                "--pcd-mode",
                "masked",
                "--tracker-backend",
                "tapnextpp",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--render-mode panel requires --input-source fake-live"):
            demo.validate_args(args)

    def test_panel_hud_from_runtime_pair_uses_latest_rgb_and_paired_seq(self) -> None:
        args = self._tracker_args()
        args.input_source = demo.INPUT_SOURCE_FAKE_LIVE
        args.render_mode = "panel"
        runtime = demo.RealtimeMaskedEdgeTamPcdDemo(args)
        rgb_frame = self._frame_packet(seq=5)
        pair = demo.PairedRenderPacket(
            seq=3,
            pcd_packet=self._pcd_packet(seq=3),
            tracker_packet=self._tracker_packet(seq=3),
            mask_packet=self._mask_packet(seq=3),
        )

        hud = runtime._build_panel_hud(rgb_frame=rgb_frame, pair=pair, display_time_s=pair.pcd_packet.process_done_perf_s + 0.1)

        self.assertEqual(hud.rgb_seq, 5)
        self.assertEqual(hud.paired_seq, 3)
        self.assertEqual(hud.rgb_ahead_frames, 2)
        self.assertEqual(hud.marker_count, pair.tracker_packet.marker_count)
```

- [ ] **Step 2: Run the new runtime tests and verify they fail**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_single_demo_tapnextpp_overlay.RealtimeMaskedEdgeTamPcdTest.test_panel_render_mode_requires_fake_live_lossless_tracking \
  tests.test_single_demo_tapnextpp_overlay.RealtimeMaskedEdgeTamPcdTest.test_panel_hud_from_runtime_pair_uses_latest_rgb_and_paired_seq
```

Expected: FAIL because `panel` is not in `RENDER_MODES`, `PairedRenderPacket.mask_packet` does not exist, and `_build_panel_hud` does not exist.

- [ ] **Step 3: Add panel render mode constants and parser flags**

In `qqtt/demo/realtime_masked_edgetam_pcd.py`, update render constants:

```python
RENDER_MODE_POINTCLOUD = "pointcloud"
RENDER_MODE_NONE = "none"
RENDER_MODE_PANEL = "panel"
RENDER_MODES = (RENDER_MODE_POINTCLOUD, RENDER_MODE_NONE, RENDER_MODE_PANEL)
DEFAULT_RENDER_MODE = RENDER_MODE_POINTCLOUD
PANEL_LAYOUT_SIDE_BY_SIDE = "side-by-side"
PANEL_LAYOUTS = (PANEL_LAYOUT_SIDE_BY_SIDE,)
```

Add parser arguments after `--render-mode`:

```python
    parser.add_argument(
        "--panel-layout",
        choices=PANEL_LAYOUTS,
        default=PANEL_LAYOUT_SIDE_BY_SIDE,
        help="Runtime panel layout. side-by-side shows latest RGB, filtered PCD, and tracking overlay.",
    )
    parser.add_argument(
        "--panel-video-output",
        type=Path,
        default=None,
        help="Optional MP4 path for saving the runtime panel frames.",
    )
    parser.add_argument(
        "--tracking-background-mask",
        choices=("target-union", "rgb"),
        default="target-union",
        help="Runtime/offline tracking panel background policy.",
    )
```

- [ ] **Step 4: Extend `PairedRenderPacket` with mask packet**

Update the dataclass:

```python
@dataclass(frozen=True)
class PairedRenderPacket:
    seq: int
    pcd_packet: MaskedPcdPacket
    tracker_packet: TrackerMarkerPacket
    mask_packet: MaskPacket | None = None
```

In `_publish_strict_render_pair`, construct it with:

```python
        pair = PairedRenderPacket(
            seq=int(pcd_result.packet.seq),
            pcd_packet=pcd_result.packet,
            tracker_packet=tracker_packet,
            mask_packet=pcd_result.mask_packet,
        )
```

- [ ] **Step 5: Add validation for panel mode**

In `validate_args`, update render validation:

```python
    if args.render_mode in {"pointcloud", "panel"} and args.pcd_mode == "none":
        raise ValueError(f"--render-mode {args.render_mode} requires --pcd-mode masked")
```

Add this block after headless validation:

```python
    if args.render_mode == "panel":
        if args.input_source != INPUT_SOURCE_FAKE_LIVE:
            raise ValueError("--render-mode panel requires --input-source fake-live")
        if args.depth_source != "ffs":
            raise ValueError("--render-mode panel requires --depth-source ffs")
        if args.track_mode != TRACK_MODE_CONTROLLER_OBJECT:
            raise ValueError("--render-mode panel requires --track-mode controller-object")
        if args.pcd_mode != "masked":
            raise ValueError("--render-mode panel requires --pcd-mode masked")
        if args.tracker_backend != TRACKER_BACKEND_TAPNEXTPP:
            raise ValueError("--render-mode panel requires --tracker-backend tapnextpp")
```

Update tracker validation:

```python
        if args.render_mode not in {"pointcloud", "panel"} and not headless_capture_enabled(args):
            raise ValueError("--tracker-backend tapnextpp requires --render-mode pointcloud or panel")
```

- [ ] **Step 6: Add runtime panel helpers**

Import the shared renderer near the other demo imports:

```python
from qqtt.demo.demo32_side_by_side_panel import (
    SideBySidePanelHud,
    SideBySidePanelInputs,
    render_projected_pcd_panel,
    render_side_by_side_panel,
    render_tracking_overlay_panel,
)
```

Add method `_build_panel_hud` to `RealtimeMaskedEdgeTamPcdDemo`:

```python
    def _build_panel_hud(
        self,
        *,
        rgb_frame: FramePacket,
        pair: PairedRenderPacket,
        display_time_s: float,
    ) -> SideBySidePanelHud:
        pcd_packet = pair.pcd_packet
        pipeline_done_s = max(float(pcd_packet.process_done_perf_s), float(pair.tracker_packet.process_done_perf_s))
        return SideBySidePanelHud(
            rgb_seq=int(rgb_frame.seq),
            paired_seq=int(pair.seq),
            input_time_s=pcd_packet.source_timestamp_s,
            pipeline_latency_ms=float(pipeline_done_s - float(pcd_packet.receive_perf_s)) * 1000.0,
            display_latency_ms=float(display_time_s - float(pcd_packet.receive_perf_s)) * 1000.0,
            startup_hold_s=float(getattr(self, "_startup_hold_s", 0.0)),
            filter_preset=str(getattr(self.args, "pcd_filter_preset", None) or headless_capture_saved_pcd_source(self.args)),
            marker_count=int(pair.tracker_packet.marker_count),
            tracking_background=str(getattr(self.args, "tracking_background_mask", "target-union")),
            object_point_count=int(pcd_packet.object_point_count),
            controller_point_count=int(pcd_packet.controller_point_count),
        )
```

Add method `_render_runtime_panel_frame`:

```python
    def _render_runtime_panel_frame(self, *, rgb_frame: FramePacket, pair: PairedRenderPacket) -> np.ndarray:
        if pair.mask_packet is None:
            raise RuntimeError("runtime panel requires mask_packet in PairedRenderPacket")
        pcd_packet = pair.pcd_packet
        intrinsics = {
            "fx": float(pcd_packet.intrinsics.fx),
            "fy": float(pcd_packet.intrinsics.fy),
            "cx": float(pcd_packet.intrinsics.cx),
            "cy": float(pcd_packet.intrinsics.cy),
        }
        pcd_panel, pcd_counts = render_projected_pcd_panel(
            width=self.width,
            height=self.height,
            intrinsics=intrinsics,
            controller_xyz_m=pcd_packet.controller_xyz_m,
            controller_rgb_u8=pcd_packet.controller_colors_rgb_u8,
            object_xyz_m=pcd_packet.object_xyz_m,
            object_rgb_u8=pcd_packet.object_colors_rgb_u8,
            point_size=int(max(1.0, float(self.args.point_size))),
            max_render_points=int(self.args.render_max_points_per_layer),
            coordinate_frame=str(pcd_packet.coordinate_frame),
            camera_to_world_c2w=None if self.table_c2w is None else self.table_c2w,
        )
        tracking_image = np.ascontiguousarray(pair.mask_packet.color_bgr.copy(), dtype=np.uint8)
        if str(self.args.tracking_background_mask) == "target-union":
            target_union = np.logical_or(pair.mask_packet.object_mask, pair.mask_packet.controller_mask)
            tracking_image[~target_union] = 0
        tracker = pair.tracker_packet
        tracking_panel, tracking_counts = render_tracking_overlay_panel(
            image_bgr=tracking_image,
            tracks_yx=tracker.tracks_yx,
            visibility=tracker.visibility,
            marker_rgb_u8=tracker.marker_colors_rgb_u8,
            query_is_object=tracker.query_is_object,
            query_is_controller=tracker.query_is_controller,
            query_controller_instance_id=tracker.query_controller_instance_id,
            marker_radius=max(1, int(round(float(self.args.tracker_marker_point_size)))),
        )
        display_time_s = time.perf_counter()
        hud = self._build_panel_hud(rgb_frame=rgb_frame, pair=pair, display_time_s=display_time_s)
        hud = SideBySidePanelHud(
            rgb_seq=hud.rgb_seq,
            paired_seq=hud.paired_seq,
            input_time_s=hud.input_time_s,
            pipeline_latency_ms=hud.pipeline_latency_ms,
            display_latency_ms=hud.display_latency_ms,
            startup_hold_s=hud.startup_hold_s,
            filter_preset=hud.filter_preset,
            marker_count=int(tracking_counts["query_points"]),
            tracking_background=hud.tracking_background,
            object_point_count=int(pcd_counts["object_points"]),
            controller_point_count=int(pcd_counts["controller_points"]),
        )
        return render_side_by_side_panel(
            SideBySidePanelInputs(
                rgb_image_bgr=rgb_frame.color_bgr,
                pcd_panel_bgr=pcd_panel,
                tracking_panel_bgr=tracking_panel,
                hud=hud,
            )
        )
```

- [ ] **Step 7: Add runtime OpenCV panel loop**

Add method `_run_panel_viewer`:

```python
    def _run_panel_viewer(self) -> None:
        self._start_threads()
        writer: cv2.VideoWriter | None = None
        window_name = "Demo 3.2 side-by-side panel"
        last_pair_seq = -1
        try:
            while not self.stop_event.is_set():
                fatal = self._fatal_error_snapshot()
                if fatal is not None:
                    raise RuntimeError(fatal.log_message())
                if self._lossless_enabled() and self._lossless_pipeline_active:
                    pair = self.lossless_paired_render_queue.get_nowait()
                else:
                    pair = self.paired_render_slot.get_latest_after(last_pair_seq)
                if pair is None:
                    if self._lossless_processing_done.is_set():
                        break
                    time.sleep(0.005)
                    continue
                last_pair_seq = int(pair.seq)
                rgb_frame = self.capture_slot.get_latest_after(-1)
                if rgb_frame is None:
                    time.sleep(0.005)
                    continue
                panel = self._render_runtime_panel_frame(rgb_frame=rgb_frame, pair=pair)
                if writer is None and self.args.panel_video_output is not None:
                    output = _resolve_path(self.args.panel_video_output)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(self._lossless_input_fps()),
                        (int(panel.shape[1]), int(panel.shape[0])),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open panel video writer: {output}")
                if writer is not None:
                    writer.write(panel)
                cv2.imshow(window_name, panel)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    self.stop_event.set()
                    break
                if self._lossless_enabled() and self._lossless_processing_done.is_set():
                    break
        finally:
            if writer is not None:
                writer.release()
            cv2.destroyWindow(window_name)
```

In `run`, replace:

```python
            if self.args.render_mode == "none":
                self._run_headless()
            else:
                self._run_open3d_viewer()
```

with:

```python
            if self.args.render_mode == "none":
                self._run_headless()
            elif self.args.render_mode == "panel":
                self._run_panel_viewer()
            else:
                self._run_open3d_viewer()
```

- [ ] **Step 8: Record startup hold seconds**

In `__init__`, initialize:

```python
        self._startup_hold_s = 0.0
```

In `_capture_recording_worker`, after `gate_done_s = time.perf_counter()`, add:

```python
        self._startup_hold_s = max(0.0, float(gate_done_s - camera_start_s))
        if self.headless_capture_writer is not None:
            self.headless_capture_writer.update_metadata({"startup_hold_s": float(self._startup_hold_s)})
```

In `_build_headless_capture_metadata`, set:

```python
            "startup_hold_s": float(getattr(self, "_startup_hold_s", 0.0)),
```

- [ ] **Step 9: Run runtime panel tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_single_demo_tapnextpp_overlay.RealtimeMaskedEdgeTamPcdTest.test_panel_render_mode_requires_fake_live_lossless_tracking \
  tests.test_single_demo_tapnextpp_overlay.RealtimeMaskedEdgeTamPcdTest.test_panel_hud_from_runtime_pair_uses_latest_rgb_and_paired_seq
```

Expected: both tests pass.

- [ ] **Step 10: Commit Task 4**

```bash
git add qqtt/demo/realtime_masked_edgetam_pcd.py tests/test_single_demo_tapnextpp_overlay.py
git commit -m "Add Demo 3.2 realtime side-by-side panel mode"
```

---

### Task 5: Demo 3.2 Wrapper Contract and CLI Forwarding

**Files:**
- Modify: `qqtt/demo/single_demo_v3_runtime.py`
- Modify: `tests/test_single_demo_v3_runtime.py`

- [ ] **Step 1: Add failing wrapper dry-run contract test**

In `tests/test_single_demo_v3_runtime.py`, add:

```python
    def test_demo32_panel_dry_run_contract_exposes_side_by_side_panel(self) -> None:
        result = runtime.main(
            [
                "--dry-run",
                "--input-source",
                "fake-live",
                "--render-mode",
                "panel",
                "--panel-layout",
                "side-by-side",
                "--panel-video-output",
                "result/panel.mp4",
                "--tracking-background-mask",
                "rgb",
            ],
            demo_version=runtime.DEMO_VERSION_3_2,
            connected_serials_provider=lambda: ["239222300781"],
        )

        self.assertEqual(result, 0)
```

If the existing test module captures stdout through a helper, use that helper and assert:

```python
        self.assertEqual(contract["render_mode"], "panel")
        self.assertEqual(contract["panel_layout"], "side-by-side")
        self.assertEqual(contract["panel_video_output"], "result/panel.mp4")
        self.assertEqual(contract["tracking_background_mask"], "rgb")
        self.assertEqual(contract["panel_sync_policy"], "left_latest_rgb_right_strict_same_seq")
```

- [ ] **Step 2: Run the wrapper test and verify it fails**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_v3_runtime
```

Expected: FAIL because wrapper parser does not expose panel flags yet.

- [ ] **Step 3: Add wrapper parser flags**

In `build_arg_parser` after `--render-mode`, add:

```python
    parser.add_argument(
        "--panel-layout",
        choices=masked_pcd.PANEL_LAYOUTS,
        default=masked_pcd.PANEL_LAYOUT_SIDE_BY_SIDE,
        help="Runtime panel layout forwarded to the masked PCD delegate.",
    )
    parser.add_argument(
        "--panel-video-output",
        type=Path,
        default=None,
        help="Optional MP4 output path for --render-mode panel.",
    )
    parser.add_argument(
        "--tracking-background-mask",
        choices=("target-union", "rgb"),
        default="target-union",
        help="Tracking overlay background for side-by-side panel and offline render parity.",
    )
```

- [ ] **Step 4: Forward panel flags to delegate args**

In the function that builds the delegate argv or namespace for `masked_pcd.main`, add:

```python
    delegate_args.extend(["--panel-layout", str(args.panel_layout)])
    delegate_args.extend(["--tracking-background-mask", str(args.tracking_background_mask)])
    if args.panel_video_output is not None:
        delegate_args.extend(["--panel-video-output", str(args.panel_video_output)])
```

Use the local variable name used by the existing wrapper for forwarded arguments. If the wrapper builds a dict instead of a list, add equivalent keys:

```python
        "panel_layout": str(args.panel_layout),
        "tracking_background_mask": str(args.tracking_background_mask),
        "panel_video_output": None if args.panel_video_output is None else str(args.panel_video_output),
```

- [ ] **Step 5: Add dry-run contract fields**

In the dry-run contract dictionary, add:

```python
        "panel_layout": str(getattr(args, "panel_layout", masked_pcd.PANEL_LAYOUT_SIDE_BY_SIDE)),
        "panel_video_output": None if getattr(args, "panel_video_output", None) is None else str(args.panel_video_output),
        "tracking_background_mask": str(getattr(args, "tracking_background_mask", "target-union")),
        "panel_sync_policy": (
            "left_latest_rgb_right_strict_same_seq"
            if str(getattr(args, "render_mode", "")) == "panel"
            else "none"
        ),
```

- [ ] **Step 6: Run wrapper tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_single_demo_v3_runtime
```

Expected: all tests in `tests.test_single_demo_v3_runtime` pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add qqtt/demo/single_demo_v3_runtime.py tests/test_single_demo_v3_runtime.py
git commit -m "Expose Demo 3.2 side-by-side panel contract"
```

---

### Task 6: Documentation and Harness Map

**Files:**
- Modify: `demo_v3_2/README.md`
- Modify: `docs/WORKFLOWS.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `scripts/harness/README.md`

- [ ] **Step 1: Update Demo 3.2 README**

In `demo_v3_2/README.md`, add this section after the existing headless render commands:

```markdown
### Side-by-side panel

Demo 3.2 can render a 1x3 panel for fake-live review:

1. original latest RGB input
2. filtered PCD projected into the camera view
3. tracking overlay with current-frame query markers

The left RGB column follows the latest fake-live input frame and may lead the
processed output. The PCD and tracking columns always use the same strict
same-seq paired frame. The HUD reports `rgb_seq`, `paired_seq`,
`rgb_ahead`, source input time, pipeline latency, display latency, startup hold,
filter preset, and marker count.

Offline from a headless capture:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_side_by_side.mp4 \
  --fps 30 \
  --panel-mode side-by-side \
  --tracking-background-mask target-union
```

Realtime fake-live panel:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --mode demo \
  --demo-visual-mode tracking \
  --render-mode panel \
  --panel-layout side-by-side \
  --tracking-background-mask target-union \
  --panel-video-output result/single_demo_v3_2_ffs_masked_pcd/realtime_panel.mp4
```
```

- [ ] **Step 2: Update workflows and architecture docs**

In `docs/WORKFLOWS.md`, add a Demo 3.2 panel workflow with the same two commands from Step 1 and one sentence explaining the left latest RGB vs right strict pair sync.

In `docs/ARCHITECTURE.md`, add:

```markdown
- `qqtt/demo/demo32_side_by_side_panel.py`: pure Demo 3.2 side-by-side panel
  renderer shared by offline headless rendering and runtime fake-live panel
  display. It composes latest RGB, projected filtered PCD, tracking overlay,
  and timing HUD frames; it does not own camera capture, FFS, EdgeTAM, or
  runtime queues.
```

In `scripts/harness/README.md`, add `render_demo32_headless_capture.py --panel-mode side-by-side` to the Demo diagnostics list.

- [ ] **Step 3: Run doc grep checks**

Run:

```bash
rg -n "side-by-side|panel-mode|panel-video-output|demo32_side_by_side_panel" demo_v3_2/README.md docs/WORKFLOWS.md docs/ARCHITECTURE.md scripts/harness/README.md
```

Expected: each edited doc appears in the output.

- [ ] **Step 4: Commit Task 6**

```bash
git add demo_v3_2/README.md docs/WORKFLOWS.md docs/ARCHITECTURE.md scripts/harness/README.md
git commit -m "Document Demo 3.2 side-by-side panel workflow"
```

---

### Task 7: Full Focused Validation and Smoke

**Files:**
- Modify only if validation exposes a bug in files changed by Tasks 1-6.

- [ ] **Step 1: Run focused unit suite**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_demo32_side_by_side_panel \
  tests.test_demo32_headless_render_helper \
  tests.test_recorded_rgbd_replay_source \
  tests.test_single_demo_tapnextpp_overlay \
  tests.test_single_demo_v3_runtime
```

Expected: all tests pass.

- [ ] **Step 2: Run py_compile for changed runtime and harness files**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m py_compile \
  qqtt/demo/demo32_side_by_side_panel.py \
  qqtt/demo/realtime_masked_edgetam_pcd.py \
  qqtt/demo/single_demo_v3_runtime.py \
  scripts/harness/diagnostics/demo/render_demo32_headless_capture.py
```

Expected: command exits 0 with no output.

- [ ] **Step 3: Run Demo 3.2 dry-run contract**

Run:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --dry-run \
  --input-source fake-live \
  --render-mode panel \
  --panel-layout side-by-side \
  --tracking-background-mask target-union
```

Expected: command exits 0 and printed JSON includes:

```json
{
  "render_mode": "panel",
  "panel_layout": "side-by-side",
  "tracking_background_mask": "target-union",
  "panel_sync_policy": "left_latest_rgb_right_strict_same_seq"
}
```

- [ ] **Step 4: Run smoke validation**

Run:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

Expected: smoke profile exits 0. If unrelated dirty-worktree changes cause failure, capture the failing command and inspect whether the failure is in files touched by this plan before editing.

- [ ] **Step 5: Commit validation fixes if needed**

If Step 1-4 expose implementation bugs, stage only the files changed for those fixes:

```bash
git add qqtt/demo/demo32_side_by_side_panel.py qqtt/demo/realtime_masked_edgetam_pcd.py qqtt/demo/single_demo_v3_runtime.py scripts/harness/diagnostics/demo/render_demo32_headless_capture.py tests/test_demo32_side_by_side_panel.py tests/test_demo32_headless_render_helper.py tests/test_recorded_rgbd_replay_source.py tests/test_single_demo_tapnextpp_overlay.py tests/test_single_demo_v3_runtime.py
git commit -m "Fix Demo 3.2 side-by-side panel validation issues"
```

Expected: commit is created only when validation fixes were necessary.

---

## Final Handoff Checklist

- [ ] `git branch --show-current` prints `single-camera`.
- [ ] `git status --short` shows no unstaged changes from this feature. Existing unrelated user changes may remain, but feature files should be committed.
- [ ] Focused unit suite passes.
- [ ] `py_compile` command passes.
- [ ] `scripts/harness/validation/run.py --profile smoke` passes or a concrete unrelated failure is documented.
- [ ] The final response reports the MP4 commands for:
  - offline side-by-side panel from headless capture
  - runtime fake-live side-by-side panel with optional MP4 output
