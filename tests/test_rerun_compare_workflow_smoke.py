from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_process.visualization.rerun_compare import run_rerun_compare_workflow
from tests.visualization_test_utils import make_rerun_compare_cases


class FakeRunner:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run_pair(self, left_image, right_image, *, K_ir_left, baseline_m, audit_mode=False):
        del right_image, audit_mode
        height, width = left_image.shape[:2]
        disparity_raw = np.full((height, width), 5.0, dtype=np.float32)
        disparity = disparity_raw.clip(0, None)
        fx_ir = float(np.asarray(K_ir_left, dtype=np.float32)[0, 0])
        depth_ir_left_m = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity > 0
        depth_ir_left_m[valid] = (fx_ir * float(baseline_m)) / disparity[valid]
        return {
            "disparity_raw": disparity_raw,
            "disparity": disparity,
            "depth_ir_left_m": depth_ir_left_m,
            "K_ir_left_used": np.asarray(K_ir_left, dtype=np.float32),
            "baseline_m": float(baseline_m),
            "scale": 1.0,
            "valid_iters": 8,
            "max_disp": 192,
        }


class FakeRerun:
    def __init__(self) -> None:
        self.init_calls = []
        self.save_calls = []
        self.time_calls = []
        self.log_calls = []

    def init(self, application_id, spawn=False):
        self.init_calls.append({"application_id": application_id, "spawn": bool(spawn)})

    def save(self, path):
        self.save_calls.append(str(path))

    def set_time_sequence(self, timeline, value):
        self.time_calls.append((str(timeline), int(value)))

    def log(self, entity_path, payload):
        self.log_calls.append((str(entity_path), payload))

    def Points3D(self, positions, colors=None):
        return {
            "positions": np.asarray(positions, dtype=np.float32),
            "colors": np.asarray(colors, dtype=np.uint8) if colors is not None else None,
        }


class RerunCompareWorkflowSmokeTest(unittest.TestCase):
    def test_writes_fused_plys_summary_and_rerun_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            make_rerun_compare_cases(aligned_root, frame_num=2)
            output_dir = tmp_root / "rerun_output"
            fake_rerun = FakeRerun()

            summary = run_rerun_compare_workflow(
                aligned_root=aligned_root,
                realsense_case="native_case",
                ffs_case="ffs_case",
                output_dir=output_dir,
                ffs_repo="C:/external/fake",
                ffs_model_path="C:/external/fake/model.pth",
                rerun_output="viewer_and_rrd",
                runner_factory=FakeRunner,
                rerun_module=fake_rerun,
            )

            self.assertTrue((output_dir / "pointcloud_compare.rrd").parent.is_dir())
            self.assertTrue((output_dir / "ply_fullscene" / "native_frame_0000_fused_fullscene.ply").is_file())
            self.assertTrue((output_dir / "ply_fullscene" / "ffs_remove_1_frame_0000_fused_fullscene.ply").is_file())
            self.assertTrue((output_dir / "ply_fullscene" / "ffs_remove_0_frame_0001_fused_fullscene.ply").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())

            on_disk_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["frames"]), 2)
            self.assertEqual(len(on_disk_summary["frames"]), 2)
            self.assertEqual(fake_rerun.init_calls, [{"application_id": "qqtt_pointcloud_compare", "spawn": True}])
            self.assertEqual(len(fake_rerun.save_calls), 1)
            self.assertEqual(fake_rerun.time_calls, [("frame", 0), ("frame", 1)])
            self.assertEqual(len(fake_rerun.log_calls), 6)

            entity_paths = [item[0] for item in fake_rerun.log_calls]
            self.assertEqual(entity_paths.count("native"), 2)
            self.assertEqual(entity_paths.count("ffs_remove_1"), 2)
            self.assertEqual(entity_paths.count("ffs_remove_0"), 2)

            first_frame = on_disk_summary["frames"][0]["variants"]
            self.assertGreater(first_frame["ffs_remove_1"]["remove_invisible_pixel_count"], 0)
            self.assertEqual(first_frame["ffs_remove_0"]["remove_invisible_pixel_count"], 0)
            self.assertLess(
                int(first_frame["ffs_remove_1"]["fused_point_count"]),
                int(first_frame["ffs_remove_0"]["fused_point_count"]),
            )


if __name__ == "__main__":
    unittest.main()
