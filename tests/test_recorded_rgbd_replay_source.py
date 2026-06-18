from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np
from PIL import Image

from qqtt.demo import realtime_masked_edgetam_pcd as masked_demo


class RecordedRgbdReplaySourceTest(unittest.TestCase):
    def _write_case(
        self,
        root: Path,
        *,
        steps: tuple[int, ...] = (10, 2),
        fps: float = 30,
        include_ir: bool = False,
    ) -> Path:
        case_dir = root / "case"
        (case_dir / "color" / "0").mkdir(parents=True)
        (case_dir / "depth" / "0").mkdir(parents=True)
        if include_ir:
            (case_dir / "ir_left" / "0").mkdir(parents=True)
            (case_dir / "ir_right" / "0").mkdir(parents=True)
        streams_present = ["color", "depth"]
        if include_ir:
            streams_present.extend(["ir_left", "ir_right"])
        metadata = {
            "schema_version": "qqtt_recording_v2",
            "serial_numbers": ["s0"],
            "capture_mode": "both_eval" if include_ir else "rgbd",
            "streams_present": streams_present,
            "fps": fps,
            "WH": [3, 2],
            "K_color": [
                [
                    [2.0, 0.0, 1.0],
                    [0.0, 4.0, 0.5],
                    [0.0, 0.0, 1.0],
                ]
            ],
            "depth_scale_m_per_unit": [0.001],
            "recording": {"0": {str(step): float(step) for step in steps}},
        }
        if include_ir:
            metadata.update(
                {
                    "K_ir_left": [
                        [
                            [3.0, 0.0, 1.0],
                            [0.0, 3.0, 0.5],
                            [0.0, 0.0, 1.0],
                        ]
                    ],
                    "T_ir_left_to_color": [
                        [
                            [1.0, 0.0, 0.0, -0.05],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ],
                    "ir_baseline_m": [0.095],
                }
            )
        (case_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        for step in steps:
            rgb = np.zeros((2, 3, 3), dtype=np.uint8)
            rgb[:, :] = (step, step + 1, step + 2)
            Image.fromarray(rgb, mode="RGB").save(case_dir / "color" / "0" / f"{step}.png")
            np.save(case_dir / "depth" / "0" / f"{step}.npy", np.full((2, 3), step, dtype=np.uint16))
            if include_ir:
                Image.fromarray(np.full((2, 3), step, dtype=np.uint8), mode="L").save(
                    case_dir / "ir_left" / "0" / f"{step}.png"
                )
                Image.fromarray(np.full((2, 3), step + 1, dtype=np.uint8), mode="L").save(
                    case_dir / "ir_right" / "0" / f"{step}.png"
                )
        return case_dir

    def test_numeric_step_order_remaps_first_complete_frame_to_seq_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir))

            source = masked_demo.RecordedRgbdFrameSource(case_dir, replay_fps=0.0)
            self.assertEqual(source.steps, [2, 10])
            self.assertEqual(source.effective_fps, 30.0)
            self.assertEqual((source.width, source.height), (3, 2))
            self.assertEqual(source.serial, "s0")

            packet = source.read_packet(seq=0, wait_ms=1.25, receive_perf_s=123.0, frame_copy_ms=0.5)
            self.assertEqual(packet.seq, 0)
            self.assertEqual(packet.depth_source, "realsense")
            self.assertEqual(packet.intrinsics.fx, 2.0)
            self.assertEqual(packet.intrinsics.fy, 4.0)
            self.assertEqual(packet.depth_scale_m_per_unit, 0.001)
            self.assertEqual(packet.timing.wait_ms, 1.25)
            self.assertEqual(packet.timing.frame_copy_ms, 0.5)
            np.testing.assert_array_equal(packet.k_color, np.array(source.k_color, dtype=np.float32))
            self.assertEqual(packet.color_bgr[0, 0].tolist(), [4, 3, 2])
            self.assertEqual(packet.depth_u16[0, 0].item(), 2)
            self.assertEqual(packet.source_frame_index, 0)
            self.assertEqual(packet.source_step, 2)
            self.assertAlmostEqual(packet.source_timestamp_s, 2.0)
            self.assertIsNone(packet.ir_left_u8)

            remapped_packet = source.read_packet(seq=7, frame_index=1)
            self.assertEqual(remapped_packet.seq, 7)
            self.assertEqual(remapped_packet.source_frame_index, 1)
            self.assertEqual(remapped_packet.source_step, 10)
            self.assertAlmostEqual(remapped_packet.source_timestamp_s, 10.0)
            self.assertEqual(remapped_packet.color_bgr[0, 0].tolist(), [12, 11, 10])

    def test_missing_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                masked_demo.RecordedRgbdFrameSource(Path(tmp_dir) / "missing", replay_fps=30.0)

    def test_missing_listed_depth_frame_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir), steps=(2,))
            (case_dir / "depth" / "0" / "2.npy").unlink()

            with self.assertRaises(FileNotFoundError):
                masked_demo.RecordedRgbdFrameSource(case_dir, replay_fps=30.0)

    def test_metadata_fps_zero_defaults_to_30fps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir), steps=(2,), fps=0)

            source = masked_demo.RecordedRgbdFrameSource(case_dir, replay_fps=0.0)

            self.assertEqual(source.effective_fps, 30.0)

    def test_ffs_fake_live_packet_uses_ir_stereo_without_native_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir), include_ir=True)

            source = masked_demo.RecordedRgbdFrameSource(case_dir, replay_fps=0.0, depth_source="ffs")
            packet = source.read_packet(seq=0)

            self.assertEqual(packet.depth_source, "ffs")
            self.assertIsNone(packet.depth_u16)
            self.assertIsNotNone(packet.ir_left_u8)
            self.assertIsNotNone(packet.ir_right_u8)
            self.assertEqual(packet.ir_left_u8[0, 0].item(), 2)
            self.assertEqual(packet.ir_right_u8[0, 0].item(), 3)
            self.assertAlmostEqual(packet.ir_baseline_m, 0.095)
            np.testing.assert_array_equal(packet.k_ir_left, np.asarray(source.k_ir_left, dtype=np.float32))
            np.testing.assert_array_equal(packet.t_ir_left_to_color, np.asarray(source.t_ir_left_to_color, dtype=np.float32))

    def test_ffs_fake_live_requires_ir_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir))

            with self.assertRaisesRegex(ValueError, "ir_left and ir_right"):
                masked_demo.RecordedRgbdFrameSource(case_dir, replay_fps=30.0, depth_source="ffs")

    def test_headless_recording_replay_smoke_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir), steps=(2, 3))

            code = masked_demo.main(
                [
                    "--input-source",
                    "recording",
                    "--recording-case",
                    str(case_dir),
                    "--replay-fps",
                    "120",
                    "--depth-source",
                    "realsense",
                    "--render-mode",
                    "none",
                    "--track-mode",
                    "none",
                    "--pcd-mode",
                    "none",
                    "--duration-s",
                    "1",
                ]
            )

            self.assertEqual(code, 0)

    def test_recording_capture_holds_after_seq_zero_until_first_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir), steps=(2, 3, 4))
            args = masked_demo.build_parser().parse_args(
                [
                    "--input-source",
                    "recording",
                    "--recording-case",
                    str(case_dir),
                    "--replay-fps",
                    "100",
                    "--depth-source",
                    "realsense",
                    "--render-mode",
                    "none",
                    "--track-mode",
                    "object-only",
                    "--pcd-mode",
                    "none",
                ]
            )
            demo = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
            demo.recording_source = masked_demo.RecordedRgbdFrameSource(case_dir, replay_fps=100)

            thread = threading.Thread(target=demo._capture_recording_worker, daemon=True)
            thread.start()
            time.sleep(0.05)
            self.assertEqual(demo.capture_slot.latest_seq(), 0)

            demo._recording_first_frame_segmented.set()
            deadline = time.time() + 1.0
            while time.time() < deadline and demo.capture_slot.latest_seq() < 1:
                time.sleep(0.01)
            self.assertGreaterEqual(demo.capture_slot.latest_seq(), 1)
            demo.stop_event.set()
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())

    def test_lossless_recording_capture_offers_all_frames_without_latest_wins_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir), steps=(2, 3, 4))
            args = masked_demo.build_parser().parse_args(
                [
                    "--input-source",
                    "recording",
                    "--recording-case",
                    str(case_dir),
                    "--replay-fps",
                    "100",
                    "--depth-source",
                    "realsense",
                    "--render-mode",
                    "none",
                    "--track-mode",
                    "object-only",
                    "--pcd-mode",
                    "masked",
                    "--tracker-backend",
                    "tapnextpp",
                ]
            )
            demo = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
            demo._reset_lossless_state()
            demo.recording_source = masked_demo.RecordedRgbdFrameSource(case_dir, replay_fps=100)

            thread = threading.Thread(target=demo._capture_recording_worker, daemon=True)
            thread.start()

            seen: list[int] = []
            first = demo.lossless_frame_queue.get(stop_event=demo.stop_event)
            self.assertIsNotNone(first)
            seen.append(first.seq)
            demo._recording_first_frame_segmented.set()

            deadline = time.time() + 1.0
            while time.time() < deadline:
                packet = demo.lossless_frame_queue.get(stop_event=demo.stop_event)
                if packet is None:
                    break
                seen.append(packet.seq)

            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(seen, [0, 1, 2])
            self.assertEqual(demo._lossless_offered_frames, 3)
            self.assertTrue(demo.lossless_frame_queue.is_closed_and_empty())

    def test_lossless_fake_live_warmup_advances_source_frame_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir), steps=tuple(range(10, 22)))
            args = masked_demo.build_parser().parse_args(
                [
                    "--input-source",
                    "fake-live",
                    "--recording-case",
                    str(case_dir),
                    "--replay-fps",
                    "100",
                    "--depth-source",
                    "realsense",
                    "--render-mode",
                    "none",
                    "--track-mode",
                    "object-only",
                    "--pcd-mode",
                    "masked",
                    "--tracker-backend",
                    "tapnextpp",
                ]
            )
            demo = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
            demo._reset_lossless_state()
            demo._lossless_input_fps = lambda: 20.0  # type: ignore[method-assign]
            demo.recording_source = masked_demo.RecordedRgbdFrameSource(case_dir, replay_fps=100)

            thread = threading.Thread(target=demo._capture_recording_worker, daemon=True)
            thread.start()

            first = demo.lossless_frame_queue.get(stop_event=demo.stop_event)
            self.assertIsNotNone(first)
            assert first is not None
            self.assertEqual(first.seq, 0)
            self.assertEqual(first.color_bgr[0, 0, 2].item(), 10)

            time.sleep(0.12)
            demo._recording_first_frame_segmented.set()
            second = demo.lossless_frame_queue.get(stop_event=demo.stop_event)
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(second.seq, 1)
            self.assertGreater(second.color_bgr[0, 0, 2].item(), 11)

            demo.stop_event.set()
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())

    def test_lossless_recording_duration_uses_fixed_5fps_task_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = self._write_case(Path(tmp_dir), steps=(2, 3, 4, 5))
            args = masked_demo.build_parser().parse_args(
                [
                    "--input-source",
                    "recording",
                    "--recording-case",
                    str(case_dir),
                    "--replay-fps",
                    "100",
                    "--duration-s",
                    "0.21",
                    "--depth-source",
                    "realsense",
                    "--render-mode",
                    "none",
                    "--track-mode",
                    "object-only",
                    "--pcd-mode",
                    "masked",
                    "--tracker-backend",
                    "tapnextpp",
                ]
            )
            demo = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
            demo._reset_lossless_state()
            demo.recording_source = masked_demo.RecordedRgbdFrameSource(case_dir, replay_fps=100)

            thread = threading.Thread(target=demo._capture_recording_worker, daemon=True)
            thread.start()

            seen: list[int] = []
            first = demo.lossless_frame_queue.get(stop_event=demo.stop_event)
            self.assertIsNotNone(first)
            seen.append(first.seq)
            demo._recording_first_frame_segmented.set()
            deadline = time.time() + 1.0
            while time.time() < deadline:
                packet = demo.lossless_frame_queue.get(stop_event=demo.stop_event)
                if packet is None:
                    break
                seen.append(packet.seq)

            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(seen, [0, 1])
            self.assertEqual(demo._lossless_offered_frames, 2)


if __name__ == "__main__":
    unittest.main()
