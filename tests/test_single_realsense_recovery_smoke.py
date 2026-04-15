from __future__ import annotations

import threading
import unittest

from qqtt.env.camera.realsense.single_realsense import SingleRealsense


class _PipelineThatShouldNotBeTouched:
    def get_active_profile(self):
        raise AssertionError("get_active_profile should not be called after stop")

    def stop(self):
        raise AssertionError("pipeline.stop should not be called after stop")


class _ProfileLookupFailsPipeline:
    def __init__(self) -> None:
        self.stop_called = False

    def get_active_profile(self):
        raise RuntimeError("inactive pipeline")

    def stop(self):
        self.stop_called = True


class SingleRealsenseRecoverySmokeTest(unittest.TestCase):
    def test_stop_requested_skips_recovery_after_wait_error(self) -> None:
        worker = SingleRealsense.__new__(SingleRealsense)
        worker.serial_number = "test-serial"
        worker.verbose = False
        worker.stop_event = threading.Event()
        worker.stop_event.set()
        worker.ready_event = threading.Event()
        worker.pipeline = _PipelineThatShouldNotBeTouched()

        init_calls: list[str] = []

        should_continue = worker._handle_wait_for_frames_error(
            RuntimeError("Device disconnected"), lambda: init_calls.append("init")
        )

        self.assertFalse(should_continue)
        self.assertEqual(init_calls, [])

    def test_recovery_tolerates_missing_active_profile(self) -> None:
        worker = SingleRealsense.__new__(SingleRealsense)
        worker.serial_number = "test-serial"
        worker.verbose = False
        worker.stop_event = threading.Event()
        worker.ready_event = threading.Event()
        worker.pipeline = _ProfileLookupFailsPipeline()

        init_calls: list[str] = []

        should_continue = worker._handle_wait_for_frames_error(
            RuntimeError("Device disconnected"), lambda: init_calls.append("init")
        )

        self.assertTrue(should_continue)
        self.assertEqual(init_calls, ["init"])
        self.assertTrue(worker.pipeline.stop_called)


if __name__ == "__main__":
    unittest.main()
