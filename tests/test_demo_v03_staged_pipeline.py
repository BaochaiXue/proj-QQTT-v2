from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from demo_v0_3 import staged_remote_ffs_triplet_client as client
from services.ffs_remote import async_protocol_v03 as proto
from services.ffs_remote import ffs_depth_staged_server_v03 as server


def _camera_metadata(camera_idx: int) -> dict:
    return {
        "camera_idx": camera_idx,
        "serial": f"s{camera_idx}",
        "width": 4,
        "height": 3,
        "K_ir_left": np.eye(3, dtype=np.float32).tolist(),
        "K_color": np.eye(3, dtype=np.float32).tolist(),
        "T_ir_left_to_color": np.eye(4, dtype=np.float32).tolist(),
        "baseline_m": 0.095 + camera_idx * 0.001,
    }


def _write_replay(root: Path, *, kit_count: int = 4) -> None:
    metadata = {
        "mode": "demo_v0_3_ir_triplet_100kits",
        "camera_count": 3,
        "kit_count": kit_count,
        "width": 4,
        "height": 3,
        "capture_kit_fps": 15.0,
        "cameras": [_camera_metadata(idx) for idx in range(3)],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (root / "kits.jsonl").open("w", encoding="utf-8") as handle:
        for kit_idx in range(kit_count):
            cameras = []
            for cam_idx in range(3):
                left = np.full((3, 4), kit_idx + cam_idx, dtype=np.uint8)
                right = np.full((3, 4), kit_idx + cam_idx + 10, dtype=np.uint8)
                left_path = root / f"cam{cam_idx}" / "left" / f"{kit_idx:06d}.png"
                right_path = root / f"cam{cam_idx}" / "right" / f"{kit_idx:06d}.png"
                left_path.parent.mkdir(parents=True, exist_ok=True)
                right_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(left).save(left_path)
                Image.fromarray(right).save(right_path)
                cameras.append(
                    {
                        "camera_idx": cam_idx,
                        "left_ir_path": f"cam{cam_idx}/left/{kit_idx:06d}.png",
                        "right_ir_path": f"cam{cam_idx}/right/{kit_idx:06d}.png",
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "kit_idx": kit_idx,
                        "source_kit_idx": kit_idx,
                        "capture_time_s": kit_idx / 15.0,
                        "cameras": cameras,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _request_for_ffs() -> proto.StagedFfsRequest:
    cameras = []
    for idx in range(3):
        metadata = _camera_metadata(idx)
        cameras.append(
            proto.StagedCameraRequest(
                camera_idx=idx,
                serial=str(metadata["serial"]),
                width=4,
                height=3,
                k_ir_left=np.asarray(metadata["K_ir_left"], dtype=np.float32),
                k_color=np.asarray(metadata["K_color"], dtype=np.float32),
                t_ir_left_to_color=np.asarray(metadata["T_ir_left_to_color"], dtype=np.float32),
                baseline_m=float(metadata["baseline_m"]),
                ir_left_u8=np.full((3, 4), idx, dtype=np.uint8),
                ir_right_u8=np.full((3, 4), idx + 1, dtype=np.uint8),
            )
        )
    return proto.StagedFfsRequest(
        header={
            "protocol": proto.PROTOCOL_NAME,
            "version": proto.PROTOCOL_VERSION,
            "mode": "triplet-replay",
            "request_id": "measured-000000",
            "kit_idx": 0,
            "phase": "measured",
        },
        cameras=cameras,
    )


class FakeBatch3Runner:
    def __init__(self) -> None:
        self.batch_calls = 0

    def run_batch(self, samples: list[dict]) -> list[dict]:
        self.batch_calls += 1
        return [
            {
                "depth_ir_left_m": np.full(sample["left_image"].shape, 1.0 + idx, dtype=np.float32),
                "K_ir_left_used": sample["K_ir_left"],
            }
            for idx, sample in enumerate(samples)
        ]


class FakeBatch1Runner:
    def __init__(self) -> None:
        self.pair_calls = 0

    def run_pair(self, left: np.ndarray, right: np.ndarray, *, K_ir_left: np.ndarray, baseline_m: float) -> dict:
        self.pair_calls += 1
        return {
            "depth_ir_left_m": np.full(left.shape, 2.0, dtype=np.float32),
            "K_ir_left_used": K_ir_left,
        }


class DemoV03StagedPipelineTest(unittest.TestCase):
    def test_server_validate_args_enforces_single_ffs_worker(self) -> None:
        args = server.build_parser().parse_args(["--ffs-workers", "2"])

        with self.assertRaisesRegex(ValueError, "must equal 1"):
            server.validate_args(args)

    def test_client_schedule_excludes_warmup_from_measured_ordinals(self) -> None:
        kits = [
            client.ReplayKit(kit_idx=idx, source_kit_idx=idx, capture_time_s=idx / 15.0, cameras=[])
            for idx in range(4)
        ]

        schedule = client.build_request_schedule(
            kits=kits,
            warmup_kits=2,
            measure_kits=4,
            replay_once_measured=True,
        )

        self.assertEqual([task.phase for task in schedule], ["warmup", "warmup", "measured", "measured", "measured", "measured"])
        self.assertEqual([task.ordinal for task in schedule if task.phase == "measured"], [0, 1, 2, 3])
        self.assertEqual([task.kit.kit_idx for task in schedule if task.phase == "warmup"], [0, 1])

    def test_replay_loader_and_payloads_read_v03_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay_dir = Path(tmp) / "replay"
            _write_replay(replay_dir, kit_count=2)

            dataset = client.load_replay_dataset(replay_dir)
            payload = client._camera_payload(dataset, dataset.kits[1].cameras[2])

            self.assertEqual(len(dataset.kits), 2)
            self.assertEqual(payload["camera_idx"], 2)
            self.assertEqual(payload["ir_left_u8"].shape, (3, 4))
            self.assertEqual(int(payload["ir_left_u8"][0, 0]), 3)

    def test_run_ffs_models_uses_batch3_single_batch_call(self) -> None:
        request = _request_for_ffs()
        runner = FakeBatch3Runner()
        args = argparse.Namespace(ffs_mode="batch3")

        outputs, metrics = server.run_ffs_models(request=request, runner=runner, args=args)

        self.assertEqual(runner.batch_calls, 1)
        self.assertEqual(len(outputs), 3)
        self.assertGreater(metrics["server_ffs_batch3_ms"], 0.0)
        self.assertEqual(metrics["server_ffs_cam0_ms"], 0.0)

    def test_run_ffs_models_uses_three_sequential_pair_calls(self) -> None:
        request = _request_for_ffs()
        runner = FakeBatch1Runner()
        args = argparse.Namespace(ffs_mode="sequential_batch1")

        outputs, metrics = server.run_ffs_models(request=request, runner=runner, args=args)

        self.assertEqual(runner.pair_calls, 3)
        self.assertEqual(len(outputs), 3)
        self.assertGreater(metrics["server_ffs_triplet_ms"], 0.0)
        self.assertGreater(metrics["server_ffs_cam2_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
