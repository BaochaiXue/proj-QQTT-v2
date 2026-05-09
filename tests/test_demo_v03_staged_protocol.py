from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from services.ffs_remote import async_protocol_v03 as proto


try:
    HAS_LZ4 = importlib.util.find_spec("lz4") is not None and importlib.util.find_spec("lz4.frame") is not None
except ModuleNotFoundError:
    HAS_LZ4 = False


def _camera_payload(camera_idx: int) -> dict:
    left = np.arange(12, dtype=np.uint8).reshape(3, 4) + np.uint8(camera_idx)
    right = np.flip(left, axis=1).copy()
    return {
        "camera_idx": camera_idx,
        "serial": f"cam{camera_idx}",
        "ir_left_u8": left,
        "ir_right_u8": right,
        "K_ir_left": np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        "K_color": np.array([[101.0, 0.0, 2.0], [0.0, 101.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        "T_ir_left_to_color": np.eye(4, dtype=np.float32),
        "baseline_m": 0.095,
    }


class DemoV03StagedProtocolTest(unittest.TestCase):
    @unittest.skipUnless(HAS_LZ4, "lz4 is required for v0.3 protocol payload roundtrip")
    def test_triplet_request_roundtrip_has_seven_frames_and_ordered_cameras(self) -> None:
        parts = proto.build_request_parts(
            request_id="measured-000001",
            kit_idx=1,
            camera_payloads=[_camera_payload(0), _camera_payload(1), _camera_payload(2)],
            capture_kit_fps=15.0,
            phase="measured",
        )

        self.assertEqual(len(parts), 7)
        request = proto.parse_request_parts(parts)
        self.assertEqual(request.header["protocol"], proto.PROTOCOL_NAME)
        self.assertEqual(request.header["mode"], "triplet-replay")
        self.assertEqual(request.header["phase"], "measured")
        self.assertEqual([camera.camera_idx for camera in request.cameras], [0, 1, 2])
        self.assertGreater(request.header["request_kb"], 0.0)
        np.testing.assert_array_equal(request.cameras[0].ir_left_u8, _camera_payload(0)["ir_left_u8"])

    @unittest.skipUnless(HAS_LZ4, "lz4 is required for v0.3 protocol payload roundtrip")
    def test_reply_roundtrip_carries_required_flat_metrics(self) -> None:
        request = proto.parse_request_parts(
            proto.build_request_parts(
                request_id="measured-000002",
                kit_idx=2,
                camera_payloads=[_camera_payload(0), _camera_payload(1), _camera_payload(2)],
                capture_kit_fps=15.0,
            )
        )
        depths = [np.full((3, 4), 100 + idx, dtype=np.uint16) for idx in range(3)]
        metrics = {
            "server_decode_ms": 1.0,
            "server_ffs_cam0_ms": 10.0,
            "server_ffs_cam1_ms": 11.0,
            "server_ffs_cam2_ms": 12.0,
            "server_ffs_triplet_ms": 33.0,
            "server_postprocess_encode_ms": 4.0,
            "server_total_ms": 40.0,
            "depth_nonzero_cam0": 12,
            "depth_nonzero_cam1": 12,
            "depth_nonzero_cam2": 12,
            "request_kb": 3.0,
        }

        reply = proto.parse_reply_parts(proto.build_reply_parts(request=request, depths=depths, metrics=metrics))

        self.assertEqual(reply.header["request_id"], "measured-000002")
        self.assertEqual(reply.header["status"], "ok")
        self.assertEqual(reply.header["server_ffs_triplet_ms"], 33.0)
        self.assertEqual(reply.header["depth_nonzero_cam2"], 12.0)
        self.assertGreater(reply.header["reply_kb"], 0.0)
        for key in proto.METRIC_KEYS:
            self.assertIn(key, reply.header)
        self.assertEqual(len(reply.depths), 3)
        self.assertEqual(reply.depths[2].depth_u16.dtype, np.uint16)

    def test_rejects_non_triplet_and_wrong_camera_order(self) -> None:
        with self.assertRaisesRegex(proto.StagedFfsProtocolError, "expected 3"):
            proto.build_request_parts(
                request_id="bad",
                kit_idx=0,
                camera_payloads=[_camera_payload(0), _camera_payload(1)],
                capture_kit_fps=15.0,
            )
        if not HAS_LZ4:
            self.skipTest("lz4 is required for camera-order validation after compression")
        with self.assertRaisesRegex(proto.StagedFfsProtocolError, "camera order"):
            proto.build_request_parts(
                request_id="bad-order",
                kit_idx=0,
                camera_payloads=[_camera_payload(1), _camera_payload(0), _camera_payload(2)],
                capture_kit_fps=15.0,
            )

    def test_error_reply_has_no_depth_frames(self) -> None:
        if not HAS_LZ4:
            self.skipTest("lz4 is required for v0.3 protocol payload roundtrip")
        reply = proto.parse_reply_parts(
            proto.build_error_reply_parts(
                request_id="bad",
                kit_idx=7,
                phase="measured",
                error="queue full",
                metrics={"raw_queue_size": 64, "request_kb": 12.5},
            )
        )

        self.assertEqual(reply.header["status"], "error")
        self.assertEqual(reply.header["kit_idx"], 7)
        self.assertEqual(reply.header["raw_queue_size"], 64.0)
        self.assertEqual(reply.depths, [])


if __name__ == "__main__":
    unittest.main()
