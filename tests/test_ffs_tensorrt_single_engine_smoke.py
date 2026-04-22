from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.depth_backends.fast_foundation_stereo import (
    finalize_tensorrt_disparity_batch_outputs,
    finalize_single_engine_tensorrt_output,
    load_tensorrt_model_config,
    normalize_single_engine_tensorrt_image,
    resolve_single_engine_tensorrt_model_path,
    resolve_tensorrt_model_config_path,
    select_tensorrt_disparity_output,
    split_disparity_batch_output_maps,
)


class FfsTensorrtSingleEngineSmokeTest(unittest.TestCase):
    def test_resolve_tensorrt_model_config_prefers_engine_stem_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            model_path = model_dir / "fast_foundationstereo.engine"
            model_path.write_bytes(b"stub")
            (model_dir / "fast_foundationstereo.yaml").write_text(
                "image_size: [480, 848]\nvalid_iters: 4\nmax_disp: 192\n",
                encoding="utf-8",
            )
            (model_dir / "config.yaml").write_text(
                "image_size: [1, 1]\nvalid_iters: 1\nmax_disp: 1\n",
                encoding="utf-8",
            )
            (model_dir / "onnx.yaml").write_text(
                "image_size: [2, 2]\nvalid_iters: 2\nmax_disp: 2\n",
                encoding="utf-8",
            )

            cfg_path = resolve_tensorrt_model_config_path(model_dir, model_path=model_path)
            cfg = load_tensorrt_model_config(model_dir, model_path=model_path)

        self.assertEqual(cfg_path.name, "fast_foundationstereo.yaml")
        self.assertEqual(cfg["image_size"], [480, 848])
        self.assertEqual(cfg["valid_iters"], 4)
        self.assertEqual(cfg["max_disp"], 192)

    def test_resolve_tensorrt_model_config_falls_back_to_config_then_onnx_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            model_path = model_dir / "fast_foundationstereo.engine"
            model_path.write_bytes(b"stub")
            (model_dir / "config.yaml").write_text(
                "image_size: [480, 640]\nvalid_iters: 5\nmax_disp: 96\n",
                encoding="utf-8",
            )
            cfg_path = resolve_tensorrt_model_config_path(model_dir, model_path=model_path)
            cfg = load_tensorrt_model_config(model_dir, model_path=model_path)
            self.assertEqual(cfg_path.name, "config.yaml")
            self.assertEqual(cfg["image_size"], [480, 640])

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            model_path = model_dir / "fast_foundationstereo.engine"
            model_path.write_bytes(b"stub")
            (model_dir / "onnx.yaml").write_text(
                "image_size: [448, 640]\nvalid_iters: 6\nmax_disp: 128\n",
                encoding="utf-8",
            )
            cfg_path = resolve_tensorrt_model_config_path(model_dir, model_path=model_path)
            cfg = load_tensorrt_model_config(model_dir, model_path=model_path)

        self.assertEqual(cfg_path.name, "onnx.yaml")
        self.assertEqual(cfg["image_size"], [448, 640])
        self.assertEqual(cfg["valid_iters"], 6)
        self.assertEqual(cfg["max_disp"], 128)

    def test_resolve_single_engine_tensorrt_model_path_requires_exactly_one_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            with self.assertRaises(FileNotFoundError):
                resolve_single_engine_tensorrt_model_path(model_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "a.engine").write_bytes(b"stub")
            (model_dir / "b.engine").write_bytes(b"stub")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                resolve_single_engine_tensorrt_model_path(model_dir)

    def test_normalize_single_engine_tensorrt_image_matches_imagenet_contract(self) -> None:
        image = np.array([[[0, 255, 127]]], dtype=np.uint8)
        normalized = normalize_single_engine_tensorrt_image(image)
        expected = np.array(
            [[[
                (0.0 - 0.485) / 0.229,
                (1.0 - 0.456) / 0.224,
                ((127.0 / 255.0) - 0.406) / 0.225,
            ]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-6)

    def test_select_tensorrt_disparity_output_prefers_named_or_single_output(self) -> None:
        self.assertEqual(select_tensorrt_disparity_output({"disparity": 3}), 3)
        self.assertEqual(select_tensorrt_disparity_output({"disp": 4}), 4)
        self.assertEqual(select_tensorrt_disparity_output({"only": 5}), 5)

    def test_finalize_single_engine_tensorrt_output_preserves_disparity_contract(self) -> None:
        disparity_raw = np.ones((1, 1, 2, 3), dtype=np.float32)
        transform = {
            "mode": "match",
            "engine_height": 2,
            "engine_width": 3,
            "output_height": 2,
            "output_width": 3,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
        }
        k_ir_left = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        result = finalize_single_engine_tensorrt_output(
            disparity_raw,
            transform=transform,
            K_ir_left=k_ir_left,
            baseline_m=0.1,
            valid_iters=4,
            max_disp=192,
            audit_mode=False,
        )

        self.assertEqual(result["disparity"].shape, (2, 3))
        self.assertEqual(result["disparity"].dtype, np.float32)
        self.assertEqual(result["depth_ir_left_m"].shape, (2, 3))
        self.assertEqual(result["depth_ir_left_m"].dtype, np.float32)
        np.testing.assert_allclose(result["K_ir_left_used"], k_ir_left)
        self.assertEqual(result["valid_iters"], 4)
        self.assertEqual(result["max_disp"], 192)

    def test_split_disparity_batch_output_maps_preserves_batch_order(self) -> None:
        disparity_raw = np.arange(3 * 1 * 2 * 4, dtype=np.float32).reshape(3, 1, 2, 4)
        disparity_maps = split_disparity_batch_output_maps(disparity_raw, expected_batch_size=3)
        self.assertEqual(len(disparity_maps), 3)
        np.testing.assert_array_equal(disparity_maps[0], disparity_raw[0, 0])
        np.testing.assert_array_equal(disparity_maps[1], disparity_raw[1, 0])
        np.testing.assert_array_equal(disparity_maps[2], disparity_raw[2, 0])

    def test_finalize_tensorrt_disparity_batch_outputs_preserves_per_sample_contract(self) -> None:
        disparity_raw = np.stack(
            [
                np.full((2, 3), 1.0, dtype=np.float32),
                np.full((2, 3), 2.0, dtype=np.float32),
                np.full((2, 3), 3.0, dtype=np.float32),
            ],
            axis=0,
        )[:, None, :, :]
        transform = {
            "mode": "match",
            "engine_height": 2,
            "engine_width": 3,
            "output_height": 2,
            "output_width": 3,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
        }
        batch_samples = [
            {
                "K_ir_left": np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                "baseline_m": 0.10,
                "audit_mode": False,
            },
            {
                "K_ir_left": np.array([[110.0, 0.0, 1.0], [0.0, 110.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                "baseline_m": 0.20,
                "audit_mode": False,
            },
            {
                "K_ir_left": np.array([[120.0, 0.0, 1.0], [0.0, 120.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                "baseline_m": 0.30,
                "audit_mode": False,
            },
        ]

        outputs = finalize_tensorrt_disparity_batch_outputs(
            disparity_raw,
            transform=transform,
            batch_samples=batch_samples,
            valid_iters=4,
            max_disp=192,
        )

        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0]["disparity"].shape, (2, 3))
        self.assertEqual(outputs[1]["disparity"].shape, (2, 3))
        self.assertEqual(outputs[2]["disparity"].shape, (2, 3))
        self.assertAlmostEqual(float(outputs[0]["disparity"][0, 0]), 1.0)
        self.assertAlmostEqual(float(outputs[1]["disparity"][0, 0]), 2.0)
        self.assertAlmostEqual(float(outputs[2]["disparity"][0, 0]), 3.0)
        np.testing.assert_allclose(outputs[1]["K_ir_left_used"], batch_samples[1]["K_ir_left"])
        self.assertAlmostEqual(float(outputs[2]["baseline_m"]), 0.30)


if __name__ == "__main__":
    unittest.main()
