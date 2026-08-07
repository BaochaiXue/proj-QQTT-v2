"""Round-trip tests: FakeLiveCaseRecorder output IS a fake-live case.

The strongest hardware-free contract check: record synthetic FramePackets,
then open the produced directory with the UNCHANGED v6.2 fake-live reader
(``RecordedRgbdFrameSource``) and verify frames come back bit-identical and
the runtime metadata (intrinsics/scale/serial/WH) survives.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from demo_v6_2.mdp.capture_source import RecordedRgbdFrameSource
from demo_v6_2.mdp.packets import CameraIntrinsics, FramePacket, PipelineTiming

from demo_v7.service.recorder import FakeLiveCaseRecorder

W, H = 64, 48
K = np.array([[52.0, 0.0, 31.5], [0.0, 51.0, 23.5], [0.0, 0.0, 1.0]], np.float32)


def make_packet(seq: int, rng: np.random.Generator) -> FramePacket:
    color = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    depth = rng.integers(0, 3000, size=(H, W), dtype=np.uint16)
    depth[0, :] = 0  # invalid band, like real sensors
    return FramePacket(
        seq=seq,
        color_bgr=np.ascontiguousarray(color),
        depth_source="realsense",
        intrinsics=CameraIntrinsics(fx=52.0, fy=51.0, cx=31.5, cy=23.5),
        depth_scale_m_per_unit=0.001,
        receive_perf_s=time.perf_counter(),
        timing=PipelineTiming(),
        depth_u16=np.ascontiguousarray(depth),
        k_color=K.copy(),
    )


def wait_written(recorder: FakeLiveCaseRecorder, count: int, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while recorder.written < count:
        assert time.monotonic() < deadline, (
            f"recorder wrote {recorder.written}/{count} before timeout"
        )
        time.sleep(0.01)


class TestRoundTrip:
    def test_recorded_case_replays_bit_identical(self, tmp_path) -> None:
        rng = np.random.default_rng(3)
        packets = [make_packet(i, rng) for i in range(5)]
        recorder = FakeLiveCaseRecorder(tmp_path / "case")
        recorder.serial = "unit-test-cam"
        for packet in packets:
            recorder.submit(packet)
        wait_written(recorder, len(packets))
        summary = recorder.close()
        assert summary["frames_written"] == 5
        assert summary["frames_dropped"] == 0
        assert summary["error"] is None

        source = RecordedRgbdFrameSource(
            tmp_path / "case", depth_source="realsense"
        )
        assert source.frame_count == 5
        assert source.steps == [0, 1, 2, 3, 4]
        runtime = source.make_runtime()
        assert runtime.serial == "unit-test-cam"
        assert runtime.intrinsics.fx == pytest.approx(52.0)
        assert runtime.intrinsics.cy == pytest.approx(23.5)
        assert runtime.depth_scale_m_per_unit == pytest.approx(0.001)
        for i, sent in enumerate(packets):
            got = source.read_packet(seq=i, frame_index=i)
            assert np.array_equal(got.color_bgr, sent.color_bgr)
            assert np.array_equal(got.depth_u16, sent.depth_u16)
            assert got.source_step == i

    def test_timestamps_monotonic_and_fps_positive(self, tmp_path) -> None:
        rng = np.random.default_rng(4)
        recorder = FakeLiveCaseRecorder(tmp_path / "case")
        for i in range(3):
            recorder.submit(make_packet(i, rng))
        wait_written(recorder, 3)
        recorder.close()
        import json

        meta = json.loads((tmp_path / "case" / "metadata.json").read_text())
        ts = [meta["recording"]["0"][str(i)] for i in range(3)]
        assert ts == sorted(ts)
        assert meta["fps"] > 0
        assert meta["WH"] == [W, H]
        assert meta["schema_version"] == "qqtt_recording_v2"
        assert meta["streams_present"] == ["color", "depth"]
        assert meta["K_color"][0][0][0] == pytest.approx(52.0)

    def test_refuses_nonempty_dir(self, tmp_path) -> None:
        target = tmp_path / "case"
        target.mkdir()
        (target / "stale.txt").write_text("x")
        with pytest.raises(FileExistsError, match="not empty"):
            FakeLiveCaseRecorder(target)

    def test_color_only_packets_skipped_and_close_idempotent(self, tmp_path) -> None:
        rng = np.random.default_rng(5)
        recorder = FakeLiveCaseRecorder(tmp_path / "case")
        stub = make_packet(0, rng)
        object.__setattr__(stub, "depth_u16", None)
        recorder.submit(stub)  # warmup-gate color-only stub: not a step
        real = make_packet(1, rng)
        recorder.submit(real)
        wait_written(recorder, 1)
        first = recorder.close()
        assert first["frames_written"] == 1
        recorder.submit(make_packet(2, rng))  # after close: silent no-op
        assert recorder.close() == first

    def test_periodic_metadata_flush_yields_replayable_truncated_case(
        self, tmp_path, monkeypatch
    ) -> None:
        # A SIGTERM'd process never reaches close(); the periodic flush must
        # leave a valid metadata.json so the recording survives truncated.
        import demo_v7.service.recorder as recorder_mod

        monkeypatch.setattr(recorder_mod, "_META_FLUSH_EVERY", 2)
        rng = np.random.default_rng(6)
        recorder = FakeLiveCaseRecorder(tmp_path / "case")
        recorder.serial = "unit-test-cam"
        for i in range(3):
            recorder.submit(make_packet(i, rng))
        wait_written(recorder, 3)
        # No close(): simulate a killed process.
        source = RecordedRgbdFrameSource(tmp_path / "case", depth_source="realsense")
        assert source.frame_count == 2  # metadata knows the first flush's steps
        got = source.read_packet(seq=0, frame_index=0)
        assert got.color_bgr is not None
        recorder.close()

    def test_zero_frame_close_removes_scaffolding(self, tmp_path) -> None:
        target = tmp_path / "case"
        recorder = FakeLiveCaseRecorder(target)
        summary = recorder.close()
        assert summary["frames_written"] == 0
        assert not target.exists()  # next run can reuse the path
        # And the path is immediately reusable:
        FakeLiveCaseRecorder(target).close()
