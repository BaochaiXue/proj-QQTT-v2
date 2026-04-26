from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from data_process.depth_backends.confidence_filtering import build_confidence_filtered_depth_uint16
from data_process.depth_backends.fast_foundation_stereo import compute_confidence_proxies_from_logits
from data_process.record_data_align import align_case
from tests.test_record_data_align_ffs_smoke import make_v2_case


class FakeConfidenceRunner:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)

    def run_pair_with_confidence(self, left_image, right_image, *, K_ir_left, baseline_m):
        depth_ir = np.full((2, 2), 0.5, dtype=np.float32)
        confidence = np.array([[0.9, 0.1], [0.9, 0.9]], dtype=np.float32)
        return {
            "disparity": np.full((2, 2), 10.0, dtype=np.float32),
            "depth_ir_left_m": depth_ir,
            "K_ir_left_used": np.asarray(K_ir_left, dtype=np.float32),
            "baseline_m": float(baseline_m),
            "scale": 1.0,
            "valid_iters": 4,
            "max_disp": 64,
            "confidence_margin_ir_left": confidence,
            "confidence_max_softmax_ir_left": confidence,
            "confidence_entropy_ir_left": confidence,
            "confidence_variance_ir_left": confidence,
        }


def _confidence_args(*, base_path: Path, output_path: Path, write_debug: bool = False) -> Namespace:
    return Namespace(
        base_path=base_path,
        case_name="sample_case",
        output_path=output_path,
        start=10,
        end=10,
        fps=None,
        write_mp4=False,
        depth_backend="ffs",
        ffs_repo="C:/external/fake",
        ffs_model_path="C:/external/fake/model.pth",
        ffs_scale=1.0,
        ffs_valid_iters=4,
        ffs_max_disp=64,
        ffs_radius_outlier_filter=False,
        ffs_radius_outlier_radius_m=0.01,
        ffs_radius_outlier_nb_points=40,
        ffs_native_like_postprocess=False,
        ffs_confidence_mode="max_softmax",
        ffs_confidence_threshold=0.5,
        ffs_confidence_depth_min_m=0.2,
        ffs_confidence_depth_max_m=1.5,
        write_ffs_confidence_debug=write_debug,
        write_ffs_valid_mask_debug=write_debug,
        write_ffs_float_m=False,
        fail_if_no_ir_stereo=True,
    )


class FfsConfidenceFilteringSmokeTest(unittest.TestCase):
    def test_confidence_proxies_return_four_bounded_float32_maps(self) -> None:
        logits = np.array([[[[10.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]], dtype=np.float32)

        confidence = compute_confidence_proxies_from_logits(logits)

        self.assertEqual(set(confidence), {"margin", "max_softmax", "entropy", "variance"})
        for name, confidence_map in confidence.items():
            self.assertEqual(confidence_map.shape, (1, 1, 2), name)
            self.assertEqual(confidence_map.dtype, np.float32, name)
            self.assertTrue(np.all(confidence_map >= 0.0), name)
            self.assertTrue(np.all(confidence_map <= 1.0), name)
            self.assertGreater(float(confidence_map[0, 0, 0]), float(confidence_map[0, 0, 1]), name)

    def test_build_confidence_filtered_depth_uint16_rejects_bad_pixels(self) -> None:
        depth_m = np.array([[0.5, 0.5], [np.nan, 2.0]], dtype=np.float32)
        confidence = np.array([[0.9, 0.1], [0.9, 0.9]], dtype=np.float32)

        result = build_confidence_filtered_depth_uint16(
            depth_m=depth_m,
            confidence=confidence,
            confidence_threshold=0.5,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.2,
            depth_max_m=1.5,
        )

        depth_uint16 = np.asarray(result["depth_uint16"])
        self.assertEqual(depth_uint16.dtype, np.uint16)
        np.testing.assert_array_equal(depth_uint16, np.array([[500, 0], [0, 0]], dtype=np.uint16))
        self.assertEqual(np.asarray(result["valid_mask_uint8"]).dtype, np.uint8)
        self.assertEqual(np.asarray(result["confidence_uint8"]).dtype, np.uint8)
        stats = result["stats"]
        self.assertAlmostEqual(float(stats["valid_depth_ratio_before_confidence"]), 0.5)
        self.assertAlmostEqual(float(stats["valid_ratio_after_confidence"]), 0.25)

    def test_align_case_confidence_filter_writes_uint16_without_debug_or_float_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "data_collect"
            case_dir = base_path / "sample_case"
            make_v2_case(case_dir)

            output_path = tmp_root / "data"
            align_case(_confidence_args(base_path=base_path, output_path=output_path), runner_factory=FakeConfidenceRunner)

            aligned_case = output_path / "sample_case"
            depth = np.load(aligned_case / "depth" / "0" / "0.npy")
            self.assertEqual(depth.dtype, np.uint16)
            self.assertEqual(int(depth[0, 0]), 500)
            self.assertEqual(int(depth[0, 1]), 0)
            self.assertEqual(int(depth[1, 0]), 500)
            self.assertFalse((aligned_case / "depth_ffs_float_m").exists())
            self.assertFalse((aligned_case / "confidence_ffs").exists())
            self.assertFalse((aligned_case / "valid_mask_ffs").exists())

            metadata_ext = json.loads((aligned_case / "metadata_ext.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata_ext["ffs_confidence_filter"]["enabled"])
            self.assertEqual(metadata_ext["ffs_confidence_filter"]["mode"], "max_softmax")

    def test_align_case_confidence_debug_flags_write_uint8_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "data_collect"
            case_dir = base_path / "sample_case"
            make_v2_case(case_dir)

            output_path = tmp_root / "data"
            args = _confidence_args(base_path=base_path, output_path=output_path, write_debug=True)
            align_case(args, runner_factory=FakeConfidenceRunner)

            aligned_case = output_path / "sample_case"
            confidence_debug = cv2.imread(str(aligned_case / "confidence_ffs" / "0" / "0.png"), cv2.IMREAD_UNCHANGED)
            valid_mask_debug = cv2.imread(str(aligned_case / "valid_mask_ffs" / "0" / "0.png"), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(confidence_debug)
            self.assertIsNotNone(valid_mask_debug)
            self.assertEqual(confidence_debug.dtype, np.uint8)
            self.assertEqual(valid_mask_debug.dtype, np.uint8)
            self.assertEqual(int(valid_mask_debug[0, 0]), 255)
            self.assertEqual(int(valid_mask_debug[0, 1]), 0)


if __name__ == "__main__":
    unittest.main()
