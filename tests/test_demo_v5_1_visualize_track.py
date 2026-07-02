from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest

import cv2
import numpy as np

from demo_v5_1 import visualize_track


class DemoV51VisualizeTrackTests(unittest.TestCase):
    def test_output_cursor_advances_at_configured_fps(self) -> None:
        cursor = visualize_track.OutputStreamPlaybackCursor(fps=5.0)

        self.assertEqual(0, cursor.advance(latest=10, now_s=0.0, paused=False))
        self.assertEqual(0, cursor.advance(latest=10, now_s=0.19, paused=False))
        self.assertEqual(1, cursor.advance(latest=10, now_s=0.2, paused=False))
        self.assertEqual(1, cursor.advance(latest=10, now_s=0.39, paused=False))
        self.assertEqual(2, cursor.advance(latest=10, now_s=0.4, paused=False))

    def test_target_latency_cli_is_removed(self) -> None:
        parser = visualize_track.build_parser()

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            parser.parse_args(
                [
                    "--online-dir",
                    "outputs/online_data",
                    "--target-latency-s",
                    "7",
                ]
            )

        self.assertEqual(2, error.exception.code)

    def test_load_camera_model_accepts_fake_live_metadata(self) -> None:
        with self.subTest("metadata intrinsics dict and c2w pose"):
            with tempfile.TemporaryDirectory() as temp_dir:
                case_dir = Path(temp_dir)
                (case_dir / "input_rgb").mkdir(parents=True)
                cv2.imwrite(
                    str(case_dir / "input_rgb" / "000000.png"),
                    np.zeros((13, 17, 3), dtype=np.uint8),
                )
                c2w = np.eye(4, dtype=float)
                c2w[0, 3] = 1.25
                metadata = {
                    "intrinsics": {"fx": 10.0, "fy": 11.0, "cx": 5.0, "cy": 6.0},
                    "camera_to_world_c2w": c2w.tolist(),
                    "replay_fps": 5.0,
                }
                (case_dir / "metadata.json").write_text(
                    json.dumps(metadata),
                    encoding="utf-8",
                )

                camera = visualize_track.load_camera_model(case_dir, cam_idx=0)

                np.testing.assert_allclose(
                    camera.intrinsic,
                    np.asarray(
                        [[10.0, 0.0, 5.0], [0.0, 11.0, 6.0], [0.0, 0.0, 1.0]]
                    ),
                )
                np.testing.assert_allclose(camera.camera_to_world, c2w)
                self.assertEqual((17, 13), camera.image_size)
                self.assertEqual(5.0, camera.metadata_fps)

    def test_read_background_uses_fake_live_input_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir)
            (case_dir / "input_rgb").mkdir(parents=True)
            expected = np.zeros((8, 9, 3), dtype=np.uint8)
            expected[:, :, 1] = 127
            cv2.imwrite(str(case_dir / "input_rgb" / "000123.png"), expected)

            image = visualize_track.read_background(
                case_dir,
                cam_idx=0,
                source_frame=123,
                image_size=(9, 8),
                use_background=True,
            )

            self.assertEqual((8, 9, 3), image.shape)
            self.assertGreater(int(image[:, :, 1].mean()), 120)

    def test_background_lookup_uses_source_frame_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir)
            (case_dir / "input_rgb").mkdir(parents=True)
            expected = np.zeros((8, 9, 3), dtype=np.uint8)
            expected[:, :, 2] = 180
            cv2.imwrite(str(case_dir / "input_rgb" / "000007.png"), expected)
            row = {
                "input_rgb_path": "input_rgb/000007.png",
                "seq": 7,
                "source_frame_index": 456,
            }
            (case_dir / "input_frames.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )

            lookup = visualize_track.load_input_rgb_background_paths(
                case_dir / "input_frames.jsonl",
                capture_dir=case_dir,
            )
            image = visualize_track.read_background(
                case_dir,
                cam_idx=0,
                source_frame=456,
                image_size=(9, 8),
                use_background=True,
                frame_path=lookup[456],
            )

            self.assertEqual((8, 9, 3), image.shape)
            self.assertGreater(int(image[:, :, 2].mean()), 170)
