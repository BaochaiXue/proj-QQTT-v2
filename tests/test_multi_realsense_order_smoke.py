from __future__ import annotations

import unittest

try:
    from qqtt.env.camera.realsense.multi_realsense import MultiRealsense
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local RealSense install.
    MultiRealsense = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _FakeCamera:
    def __init__(self, serial_number: str):
        self.serial_number = serial_number
        self.is_ready = True

    def get(self, k=None, out=None):
        _ = (k, out)
        return {"serial": self.serial_number}

    def get_intrinsics(self):
        return self.serial_number

    def get_depth_scale(self):
        return self.serial_number

    def get_stream_metadata(self):
        return {"serial": self.serial_number}


@unittest.skipIf(MultiRealsense is None, f"pyrealsense2 unavailable: {IMPORT_ERROR}")
class MultiRealsenseOrderSmokeTest(unittest.TestCase):
    def test_get_uses_serial_numbers_order_not_dict_insertion_order(self) -> None:
        rig = object.__new__(MultiRealsense)
        rig.serial_numbers = ["cam_b", "cam_a", "cam_c"]
        rig.cameras = {
            "cam_a": _FakeCamera("cam_a"),
            "cam_b": _FakeCamera("cam_b"),
            "cam_c": _FakeCamera("cam_c"),
        }

        frames = rig.get()
        self.assertEqual([frames[idx]["serial"] for idx in range(3)], ["cam_b", "cam_a", "cam_c"])
        self.assertEqual(rig.get(index=1)["serial"], "cam_a")
        self.assertEqual(rig.get_stream_metadata(), [{"serial": "cam_b"}, {"serial": "cam_a"}, {"serial": "cam_c"}])

    def test_duplicate_serials_fail_before_camera_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate RealSense serial numbers"):
            MultiRealsense(serial_numbers=["cam_a", "cam_a"], shm_manager=object())


if __name__ == "__main__":
    unittest.main()
