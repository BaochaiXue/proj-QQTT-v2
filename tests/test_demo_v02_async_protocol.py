from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from services.ffs_remote import async_protocol_v02 as proto


try:
    HAS_LZ4 = importlib.util.find_spec("lz4") is not None and importlib.util.find_spec("lz4.frame") is not None
except ModuleNotFoundError:
    HAS_LZ4 = False


def _camera_payload(camera_idx: int, *, serial: str = "cam") -> dict:
    left = np.arange(12, dtype=np.uint8).reshape(3, 4) + np.uint8(camera_idx)
    right = np.flip(left, axis=1).copy()
    return {
        "camera_idx": camera_idx,
        "serial": f"{serial}{camera_idx}",
        "ir_left_u8": left,
        "ir_right_u8": right,
        "K_ir_left": np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        "K_color": np.array([[101.0, 0.0, 2.0], [0.0, 101.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        "T_ir_left_to_color": np.eye(4, dtype=np.float32),
        "baseline_m": 0.055,
    }


class DemoV02AsyncProtocolTest(unittest.TestCase):
    @unittest.skipUnless(HAS_LZ4, "lz4 is required for v0.2 protocol payload roundtrip")
    def test_single_request_roundtrip(self) -> None:
        parts = proto.build_request_parts(
            request_id="single-1",
            mode="single",
            camera_payloads=[_camera_payload(0)],
            target_kit_fps=45,
        )

        self.assertEqual(len(parts), 3)
        request = proto.parse_request_parts(parts)
        self.assertEqual(request.header["protocol"], proto.PROTOCOL_NAME)
        self.assertEqual(request.header["mode"], "single")
        self.assertEqual(len(request.cameras), 1)
        self.assertEqual(request.cameras[0].serial, "cam0")
        np.testing.assert_array_equal(request.cameras[0].ir_left_u8, _camera_payload(0)["ir_left_u8"])

    @unittest.skipUnless(HAS_LZ4, "lz4 is required for v0.2 protocol payload roundtrip")
    def test_triplet_request_frame_count_and_bytes(self) -> None:
        parts = proto.build_request_parts(
            request_id="triplet-1",
            mode="triplet",
            camera_payloads=[_camera_payload(0), _camera_payload(1), _camera_payload(2)],
            target_kit_fps=15,
        )

        self.assertEqual(len(parts), 7)
        request = proto.parse_request_parts(parts)
        self.assertEqual(len(request.cameras), 3)
        self.assertGreater(request.header["request_uncompressed_bytes"], 0)
        self.assertGreater(request.header["request_encoded_bytes"], 0)
        self.assertEqual([camera.camera_idx for camera in request.cameras], [0, 1, 2])

    @unittest.skipUnless(HAS_LZ4, "lz4 is required for v0.2 protocol payload roundtrip")
    def test_reply_roundtrip_depth_u16_shape_and_dtype(self) -> None:
        request = proto.parse_request_parts(
            proto.build_request_parts(
                request_id="triplet-2",
                mode="triplet",
                camera_payloads=[_camera_payload(0), _camera_payload(1), _camera_payload(2)],
                target_kit_fps=15,
            )
        )
        depths = [
            np.full((3, 4), 100 + idx, dtype=np.uint16)
            for idx in range(3)
        ]

        reply = proto.parse_reply_parts(
            proto.build_reply_parts(
                request=request,
                depths=depths,
                per_camera_stats=[
                    {"camera_idx": idx, "server_ffs_ms": 10.0, "server_align_ms": 2.0}
                    for idx in range(3)
                ],
                server_total_ms=39.0,
            )
        )

        self.assertEqual(reply.header["request_id"], "triplet-2")
        self.assertEqual(reply.header["return_type"], "depth_u16")
        self.assertEqual(reply.header["status"], "ok")
        self.assertEqual(len(reply.depths), 3)
        for idx, depth in enumerate(reply.depths):
            self.assertEqual(depth.depth_u16.dtype, np.uint16)
            self.assertEqual(depth.depth_u16.shape, (3, 4))
            self.assertEqual(int(depth.depth_u16[0, 0]), 100 + idx)

    def test_error_reply_has_no_depth_frames(self) -> None:
        reply = proto.parse_reply_parts(
            proto.build_error_reply_parts(
                request_id="bad",
                mode="triplet",
                error="queue full",
            )
        )

        self.assertEqual(reply.header["status"], "error")
        self.assertEqual(reply.header["request_id"], "bad")
        self.assertEqual(reply.depths, [])

    def test_rejects_sparse_return_and_wrong_triplet_size(self) -> None:
        with self.assertRaisesRegex(proto.AsyncFfsProtocolError, "unsupported return_type"):
            proto.build_request_parts(
                request_id="bad-return",
                mode="single",
                camera_payloads=[_camera_payload(0)],
                target_kit_fps=45,
                return_type="masked_uv_depth",
            )
        with self.assertRaisesRegex(proto.AsyncFfsProtocolError, "expected 3"):
            proto.build_request_parts(
                request_id="bad-triplet",
                mode="triplet",
                camera_payloads=[_camera_payload(0), _camera_payload(1)],
                target_kit_fps=15,
            )


if __name__ == "__main__":
    unittest.main()
