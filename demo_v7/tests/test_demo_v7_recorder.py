"""Round-trip tests: FakeLiveCaseRecorder output IS a fake-live case.

The strongest hardware-free contract check: record synthetic FramePackets,
then open the produced directory with the UNCHANGED v6.2 fake-live reader
(``RecordedRgbdFrameSource``) and verify frames come back bit-identical and
the runtime metadata (intrinsics/scale/serial/WH) survives.
"""

from __future__ import annotations

import time

from pathlib import Path
import numpy as np
import pytest

from demo_v7.runtime.mdp.capture_source import RecordedRgbdFrameSource
from demo_v7.runtime.mdp.packets import CameraIntrinsics, FramePacket, PipelineTiming

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
        """The metadata contract the v6.2 reader replays against.

        Guards recorder.py:256-268 (``_write_metadata``: K_color /
        serial_numbers / streams_present / WH / depth_scale) and :148-156
        (color PNG + depth npy writes). A hardcoded serial or a transposed
        K_color breaks replay, and only this test notices — it is the one
        that re-opens the output with the UNCHANGED v6.2 reader.
        """
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

    def test_refuses_nonempty_dir(self, tmp_path) -> None:
        # recorder.py:67-71 — refuse a non-empty record dir; without it two
        # runs interleave their steps into one case and the reader silently
        # replays a Frankenstein recording. (Coupled with the 0-frame cleanup
        # below, which is what keeps a retried path out of this refusal.)
        target = tmp_path / "case"
        target.mkdir()
        (target / "stale.txt").write_text("x")
        with pytest.raises(FileExistsError):
            FakeLiveCaseRecorder(target)

    def test_color_only_packets_skipped(self, tmp_path) -> None:
        # recorder.py:99-102 — submit() drops the warmup gate's color-only
        # preview stubs. Without the depth-None check np.save(None) latches
        # an error on the writer thread and the WHOLE recording dies
        # (frames_written 1 -> 0), not merely the stub frame.
        rng = np.random.default_rng(5)
        recorder = FakeLiveCaseRecorder(tmp_path / "case")
        stub = make_packet(0, rng)
        object.__setattr__(stub, "depth_u16", None)
        recorder.submit(stub)  # warmup-gate color-only stub: not a step
        real = make_packet(1, rng)
        recorder.submit(real)
        wait_written(recorder, 1)
        assert recorder.close()["frames_written"] == 1

    def test_periodic_metadata_flush_yields_replayable_truncated_case(
        self, tmp_path, monkeypatch
    ) -> None:
        # recorder.py:172-173 (`if self.written % _META_FLUSH_EVERY == 0:
        # self._write_metadata()`). A SIGTERM'd process never reaches
        # close(); before the periodic flush, close() was the only metadata
        # writer, so a killed/timed-out run lost the ENTIRE recording instead
        # of leaving a truncated-but-replayable case.
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
        # recorder.py:210-211 (the `else: self._remove_empty_scaffolding()`
        # branch) and the rmdir loop at :222-233. A 0-frame run used to leave
        # color/0 + depth/0 behind, which then trips the non-empty refusal at
        # :67 and poisons the path for the next run.
        target = tmp_path / "case"
        recorder = FakeLiveCaseRecorder(target)
        summary = recorder.close()
        assert summary["frames_written"] == 0
        assert not target.exists()  # next run can reuse the path
        # And the path is immediately reusable:
        FakeLiveCaseRecorder(target).close()


class TestCaseTableCalibrateSnapshot:
    """fake-live replays with the case's record-time c2w when present."""

    def test_none_without_snapshot_pair(self, tmp_path) -> None:
        # session.py:134-149, all three clauses as one table of case layouts:
        # the pkl counts only together with its metadata sidecar; the legacy
        # per-case calibrate.pkl (a different pipeline, undefined world frame)
        # must never be mistaken for one; and the complete pair must actually
        # come back (that last row is what rules out "always return None").
        from demo_v7.orchestration.session import case_table_calibrate_snapshot

        # case_dir=None is production-reachable: session.py:452 passes
        # self._args.fake_live_case, which session.py:320 shows can be None
        # while input_source == "fake-live" — without the clause: TypeError.
        assert case_table_calibrate_snapshot(None) is None

        layouts = [
            ((), None),                                    # empty case dir
            (("calibrate.pkl",), None),                    # legacy per-case pkl
            (("table_calibrate.pkl",), None),              # pkl, no sidecar
            (("table_calibrate.pkl", "table_calibrate_metadata.json"),
             "table_calibrate.pkl"),                       # the real snapshot
        ]
        for names, expected in layouts:
            case_dir = tmp_path / ("_".join(names) or "empty")
            case_dir.mkdir()
            for name in names:
                (case_dir / name).write_text("{}")
            found = case_table_calibrate_snapshot(case_dir)
            assert found == (case_dir / expected if expected else None), names

    def test_recorder_output_carries_usable_snapshot(self, tmp_path) -> None:
        # The sole test linking the two halves of the 25a7686 policy: the
        # recorder snapshots the record-time c2w at close (recorder.py:307-312
        # — the "table_calibrate.pkl"/"table_calibrate_metadata.json" entries
        # of the _copy_repo_calibration name list at :310-311) and the session
        # must then pick exactly that file up for replay.
        from demo_v7.orchestration.session import case_table_calibrate_snapshot

        rng = np.random.default_rng(7)
        recorder = FakeLiveCaseRecorder(tmp_path / "case")
        recorder.submit(make_packet(0, rng))
        wait_written(recorder, 1)
        recorder.close()
        found = case_table_calibrate_snapshot(tmp_path / "case")
        # Precondition, declared rather than assumed: the recorder copies the
        # REPO-ROOT calibration pair, so this half of the contract is only
        # checkable on a box that has one (a clean CI runner does not).
        repo_root = Path(__file__).resolve().parents[2]
        if not (repo_root / "table_calibrate.pkl").is_file():
            pytest.skip("no repo-root table_calibrate.pkl to snapshot")
        assert found is not None
        assert found.is_file()
