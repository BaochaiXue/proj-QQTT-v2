from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


TRACK_STATUS_SEQUENCE = ("normal", "invalid", "degraded")


def _minimal_chunk_data_window(
    chunk_data_payload,
    *,
    frame_count: int = 1,
    chunk_index: int = 0,
    source_frame_indices: list[int] | None = None,
    track_process_status: str | None = None,
):
    object_points = np.asarray(
        [
            [[0.05 + 0.01 * float(frame_idx), 0.0, 1.0]]
            for frame_idx in range(int(frame_count))
        ],
        dtype=np.float32,
    )
    controller_points = np.asarray(
        [
            [[0.20 + 0.01 * float(frame_idx), 0.0, 1.0]]
            for frame_idx in range(int(frame_count))
        ],
        dtype=np.float32,
    )
    track_process_data = {
        "controller_mask": np.asarray([True], dtype=bool),
        "controller_points": controller_points,
        "object_colors": np.tile(
            np.asarray([[[0.7, 0.2, 0.1]]], dtype=np.float32),
            (int(frame_count), 1, 1),
        ),
        "object_motions_valid": np.ones((int(frame_count), 1), dtype=bool),
        "object_points": object_points,
        "object_visibilities": np.ones((int(frame_count), 1), dtype=bool),
        "query_ids": np.asarray([10, 20], dtype=np.int64),
        "query_semantic_labels": np.asarray([1, 2], dtype=np.int8),
        "object_query_indices": np.asarray([10], dtype=np.int64),
        "controller_query_indices": np.asarray([20], dtype=np.int64),
        "object_track_query_indices": np.asarray([10], dtype=np.int64),
        "controller_track_query_indices": np.asarray([20], dtype=np.int64),
    }
    if track_process_status is not None:
        track_process_data["track_process_status"] = np.asarray(
            str(track_process_status)
        )
    return chunk_data_payload.ChunkDataWindow(
        track_process_data=track_process_data,
        fps=5,
        serial_number="test-camera",
        depth_backend="test-depth",
        depth_source_internal="test-depth",
        chunk_index=int(chunk_index),
        source_frame_indices=(
            list(range(int(frame_count)))
            if source_frame_indices is None
            else source_frame_indices
        ),
    )


def _status_window_factory(
    chunk_data_payload,
):
    next_chunk_index = 0

    def fake_window(*args: object, **kwargs: object):
        del args, kwargs
        nonlocal next_chunk_index
        chunk_index = next_chunk_index
        next_chunk_index += 1
        return _minimal_chunk_data_window(
            chunk_data_payload,
            chunk_index=chunk_index,
            source_frame_indices=[chunk_index],
            track_process_status=TRACK_STATUS_SEQUENCE[chunk_index],
        )

    return fake_window


def _assert_status_stream_output(
    test_case: unittest.TestCase,
    *,
    manifests: list[dict[str, object]],
    callbacks: list[dict[str, object]],
    root: Path,
) -> None:
    expected_chunk_count = len(TRACK_STATUS_SEQUENCE)
    test_case.assertEqual(expected_chunk_count, len(manifests))
    test_case.assertEqual(expected_chunk_count, len(callbacks))
    test_case.assertEqual(
        list(TRACK_STATUS_SEQUENCE),
        [manifest["track_process_status"] for manifest in manifests],
    )
    test_case.assertEqual(
        [False] * expected_chunk_count,
        [manifest["online_publish_skipped"] for manifest in manifests],
    )
    skip_reasons = [
        manifest for manifest in manifests if "online_publish_skip_reason" in manifest
    ]
    test_case.assertEqual([], skip_reasons)
    test_case.assertEqual(
        list(range(expected_chunk_count)),
        [manifest["online_chunk_id"] for manifest in manifests],
    )
    for idx in range(expected_chunk_count):
        test_case.assertTrue(
            (root / "online_data" / "chunks" / f"chunk_{idx:06d}.pkl").is_file()
        )
    with (root / "data" / "final_data.pkl").open("rb") as handle:
        static_data = pickle.load(handle)
    test_case.assertEqual(
        expected_chunk_count,
        int(np.asarray(static_data["object_points"]).shape[0]),
    )


class DemoV51ChunkDataTest(unittest.TestCase):
    def test_chunk_data_payload_builder_is_memory_only(self) -> None:
        from demo_v5_1 import chunk_data_payload

        self.assertFalse(hasattr(chunk_data_payload, "write_data_process_chunk_case"))
        self.assertFalse(hasattr(chunk_data_payload, "validate_data_process_case"))
        final_data, track_process, manifest = (
            chunk_data_payload.build_chunk_data_payload(
                _minimal_chunk_data_window(chunk_data_payload)
            )
        )

        self.assertIn("object_points", final_data)
        self.assertIn("controller_points", final_data)
        self.assertEqual(
            final_data["query_schema_hash"], track_process["query_schema_hash"]
        )
        self.assertNotIn("data_process_case_root", manifest)
        self.assertEqual("online_final_data_chunk", manifest["publish_contract"])

    def test_chunk_data_writer_does_not_emit_data_process_chunk_case(self) -> None:
        from demo_v5_1 import chunk_data_output
        from demo_v5_1 import chunk_data_payload

        final_data, track_process, _ = chunk_data_payload.build_chunk_data_payload(
            _minimal_chunk_data_window(chunk_data_payload)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = chunk_data_output.ChunkDataWriter(
                base_path=tmpdir,
                case_name="demo_v5_1",
                chunk_size=1,
                num_frames_total=1,
            )
            result = output.commit_chunk_data(
                final_data,
                track_process,
                source_frame_indices=[0],
                status="recording",
            )
            output.finish()

            root = Path(tmpdir)
            self.assertEqual(
                root / "online_data" / "chunks" / "chunk_000000.pkl",
                Path(result["online_chunk_path"]),
            )
            self.assertEqual(
                root / "online_data" / "manifest.json",
                Path(result["online_manifest_path"]),
            )
            self.assertEqual(
                root / "data" / "final_data.pkl",
                Path(result["static_data_path"]),
            )
            self.assertTrue(Path(result["online_chunk_path"]).is_file())
            self.assertTrue(Path(result["online_manifest_path"]).is_file())
            self.assertTrue(Path(result["static_data_path"]).is_file())
            with Path(result["online_chunk_path"]).open("rb") as handle:
                chunk = pickle.load(handle)
            self.assertEqual(0, chunk["chunk_id"])
            self.assertEqual(0, chunk["start_frame"])
            self.assertEqual(1, chunk["end_frame"])
            self.assertEqual([], sorted(root.glob("demo_v5_1_chunk_*")))
            self.assertFalse((root / "online_data" / "demo_v5_1").exists())
            self.assertFalse((root / "data" / "demo_v5_1").exists())

    def test_track_process_status_is_metadata_only_for_writer(self) -> None:
        from demo_v5_1 import chunk_data_output
        from demo_v5_1 import chunk_data_payload

        final_data, track_process, _ = chunk_data_payload.build_chunk_data_payload(
            _minimal_chunk_data_window(chunk_data_payload)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = chunk_data_output.ChunkDataWriter(
                base_path=tmpdir,
                case_name="demo_v5_1",
                chunk_size=1,
                num_frames_total=2,
            )
            first_track = dict(track_process)
            first_track["track_process_status"] = "degraded"
            first = output.commit_chunk_data(
                final_data,
                first_track,
                source_frame_indices=[0],
                status="recording",
            )
            second_track = dict(track_process)
            second_track["track_process_status"] = "invalid"
            second = output.commit_chunk_data(
                final_data,
                second_track,
                source_frame_indices=[1],
                status="recording",
            )
            output.finish()

            root = Path(tmpdir)
            with Path(first["online_chunk_path"]).open("rb") as handle:
                first_chunk = pickle.load(handle)
            with Path(second["online_chunk_path"]).open("rb") as handle:
                second_chunk = pickle.load(handle)
            with (root / "data" / "final_data.pkl").open("rb") as handle:
                static_data = pickle.load(handle)

            self.assertEqual("degraded", first_chunk["track_process_status"])
            self.assertEqual("invalid", second_chunk["track_process_status"])
            self.assertEqual(0, first_chunk["start_frame"])
            self.assertEqual(1, first_chunk["end_frame"])
            self.assertEqual(1, second_chunk["start_frame"])
            self.assertEqual(2, second_chunk["end_frame"])
            self.assertEqual([0], first_chunk["source_frame_indices"])
            self.assertEqual([1], second_chunk["source_frame_indices"])
            self.assertEqual(
                2,
                int(np.asarray(static_data["object_points"]).shape[0]),
            )
            self.assertEqual("invalid", static_data["track_process_status"])

    def test_headless_conversion_publishes_all_track_statuses(self) -> None:
        from demo_v5_1 import chunk_data_payload
        from demo_v5_1 import chunk_data_stream

        callbacks: list[dict[str, object]] = []
        fake_window = _status_window_factory(chunk_data_payload)

        rows = [
            {"seq": idx, "source_frame_index": idx, "source_timestamp_s": idx * 0.2}
            for idx in range(len(TRACK_STATUS_SEQUENCE))
        ]
        metadata = {
            "serial_numbers": ["test-camera"],
            "depth_backend": "test-depth",
        }
        shape_points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(
                    chunk_data_stream,
                    "_read_json_file_stable",
                    return_value=metadata,
                ),
                mock.patch.object(
                    chunk_data_stream,
                    "_shape_points_from_capture",
                    return_value=(shape_points, shape_points),
                ),
                mock.patch.object(
                    chunk_data_stream,
                    "_iter_jsonl",
                    return_value=iter(rows),
                ),
                mock.patch.object(
                    chunk_data_stream,
                    "_prepared_frame_from_row",
                    return_value=None,
                ),
                mock.patch.object(
                    chunk_data_stream,
                    "_chunk_data_window_from_rows",
                    side_effect=fake_window,
                ),
            ):
                manifests = chunk_data_stream.write_chunk_data_from_headless_capture(
                    Path(tmpdir) / "capture",
                    base_path=tmpdir,
                    case_prefix="demo_v5_1",
                    chunk_frame_count=1,
                    fps=5,
                    max_chunks=3,
                    on_chunk_written=lambda manifest: callbacks.append(
                        dict(manifest)
                    ),
                )

            root = Path(tmpdir)
            _assert_status_stream_output(
                self,
                manifests=manifests,
                callbacks=callbacks,
                root=root,
            )

    def test_live_stream_publishes_all_track_statuses(self) -> None:
        from demo_v5_1 import chunk_data_payload
        from demo_v5_1 import chunk_data_stream

        callbacks: list[dict[str, object]] = []
        fake_window = _status_window_factory(chunk_data_payload)

        rows = [
            {"seq": idx, "source_frame_index": idx, "source_timestamp_s": idx * 0.2}
            for idx in range(len(TRACK_STATUS_SEQUENCE))
        ]
        read_calls = 0

        def fake_read_jsonl_from_offset(*args: object, **kwargs: object):
            nonlocal read_calls
            read_calls += 1
            if read_calls == 1:
                return rows, 123
            return [], 123

        metadata = {
            "serial_numbers": ["test-camera"],
            "depth_backend": "test-depth",
        }
        shape_points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(
                    chunk_data_stream,
                    "_wait_for_metadata",
                    return_value=metadata,
                ),
                mock.patch.object(
                    chunk_data_stream,
                    "_read_jsonl_from_offset",
                    side_effect=fake_read_jsonl_from_offset,
                ),
                mock.patch.object(
                    chunk_data_stream,
                    "_filter_warmup_start_rows",
                    side_effect=lambda state, new_rows, capture_finished: new_rows,
                ),
                mock.patch.object(
                    chunk_data_stream,
                    "_shape_points_for_chunk",
                    return_value=(metadata, shape_points, shape_points),
                ),
                mock.patch.object(
                    chunk_data_stream,
                    "_prepared_frame_from_row",
                    return_value=None,
                ),
                mock.patch.object(
                    chunk_data_stream,
                    "_chunk_data_window_from_rows",
                    side_effect=fake_window,
                ),
            ):
                manifests = chunk_data_stream.stream_chunk_data_from_headless_capture(
                    Path(tmpdir) / "capture",
                    base_path=tmpdir,
                    case_prefix="demo_v5_1",
                    chunk_frame_count=1,
                    fps=5,
                    max_chunks=3,
                    capture_finished=lambda: True,
                    on_chunk_written=lambda manifest: callbacks.append(
                        dict(manifest)
                    ),
                )

            root = Path(tmpdir)
            _assert_status_stream_output(
                self,
                manifests=manifests,
                callbacks=callbacks,
                root=root,
            )

    def test_warmup_trim_returns_delayed_source_frame_zero(self) -> None:
        from demo_v5_1 import chunk_data_stream

        result = chunk_data_stream._trim_warmup_delayed_rows(
            [
                {
                    "seq": 0,
                    "source_frame_index": 0,
                    "source_timestamp_s": 0.0,
                    "startup_hold_s": 4.0,
                    "pipeline_latency_ms": 2500.0,
                    "controller_point_count": 30,
                    "object_point_count": 1,
                },
                {
                    "seq": 42,
                    "source_frame_index": 42,
                    "source_timestamp_s": 4.5,
                    "startup_hold_s": 4.0,
                    "pipeline_latency_ms": 30.0,
                    "controller_point_count": 30,
                    "object_point_count": 1,
                },
                {
                    "seq": 43,
                    "source_frame_index": 43,
                    "source_timestamp_s": 4.7,
                    "startup_hold_s": 4.0,
                    "pipeline_latency_ms": 30.0,
                    "controller_point_count": 30,
                    "object_point_count": 1,
                },
            ]
        )

        self.assertEqual(0, result.skipped_count)
        self.assertIsNotNone(result.warmup_row)
        self.assertEqual(0, result.warmup_row["source_frame_index"])
        self.assertEqual(
            [0, 42, 43],
            [row["source_frame_index"] for row in result.rows],
        )

    def test_warmup_trim_keeps_shape_prior_gated_timeline(self) -> None:
        # design_spec.md warmup/formal split: the producer writes only warmup
        # frame 0 during the shape-prior wait, so the first formal row jumps
        # far ahead in source index and wall time. Frame 0 must stay first and
        # nothing may be dropped.
        from demo_v5_1 import chunk_data_stream

        def _row(seq: int, source: int, ts: float, latency_ms: float) -> dict:
            return {
                "seq": seq,
                "source_frame_index": source,
                "source_timestamp_s": ts,
                "startup_hold_s": 4.0,
                "pipeline_latency_ms": latency_ms,
                "controller_point_count": 30,
                "object_point_count": 1,
            }

        result = chunk_data_stream._trim_warmup_delayed_rows(
            [
                _row(0, 0, 0.0, 17900.0),
                _row(350, 395, 78.6, 220.0),
                _row(351, 396, 78.8, 220.0),
                _row(352, 397, 79.0, 220.0),
            ]
        )

        self.assertEqual(0, result.skipped_count)
        self.assertIsNotNone(result.warmup_row)
        self.assertEqual(
            [0, 395, 396, 397],
            [row["source_frame_index"] for row in result.rows],
        )

    def test_formal_chunk_rows_gate_truth_table(self) -> None:
        from demo_v5_1 import shape_prior_warmup
        from demo_v5_1.main_data_processing import _formal_chunk_rows_gated

        # Rows always write until a chunk-ready warmup anchor row has landed,
        # whatever the prior status is (invalid startup rows must not gate).
        for status in (
            shape_prior_warmup.STATUS_DISABLED,
            shape_prior_warmup.STATUS_PENDING,
            shape_prior_warmup.STATUS_RUNNING,
            shape_prior_warmup.STATUS_READY,
            shape_prior_warmup.STATUS_FAILED,
        ):
            self.assertFalse(
                _formal_chunk_rows_gated(
                    warmup_anchor_written=False, shape_prior_status=status
                ),
                status,
            )

        # After the anchor, rows are held back only while the prior computes.
        for status in (
            shape_prior_warmup.STATUS_PENDING,
            shape_prior_warmup.STATUS_RUNNING,
        ):
            self.assertTrue(
                _formal_chunk_rows_gated(
                    warmup_anchor_written=True, shape_prior_status=status
                ),
                status,
            )
        # Terminal states lift the gate: ready starts the formal timeline,
        # disabled keeps the legacy ungated behavior, failed lets the chunk
        # bridge surface its shape-prior error instead of stalling silently.
        for status in (
            shape_prior_warmup.STATUS_READY,
            shape_prior_warmup.STATUS_DISABLED,
            shape_prior_warmup.STATUS_FAILED,
        ):
            self.assertFalse(
                _formal_chunk_rows_gated(
                    warmup_anchor_written=True, shape_prior_status=status
                ),
                status,
            )

    def test_headless_gate_deadline_expires_and_stays_lifted(self) -> None:
        # The demo-level gate enforces --shape-prior-timeout-ms: a prior stuck
        # in 'running' may withhold rows only until the deadline, after which
        # rows resume permanently so the bridge can fail loudly.
        import time as time_module
        from types import SimpleNamespace

        from demo_v5_1.main_data_processing import MainDataProcessingDemo

        stub = SimpleNamespace(
            headless_capture_writer=SimpleNamespace(saved_pcd_count=1),
            args=SimpleNamespace(shape_prior_timeout_ms=50),
            _warmup_anchor_row_written=True,
            _formal_timeline_gate_started_s=None,
            _formal_timeline_gate_expired=False,
            _shape_prior_profile=lambda: {"shape_prior_status": "running"},
        )
        gated = MainDataProcessingDemo._headless_product_rows_gated
        self.assertTrue(gated(stub))
        # Simulate the deadline having passed.
        stub._formal_timeline_gate_started_s = time_module.perf_counter() - 1.0
        self.assertFalse(gated(stub))
        self.assertTrue(stub._formal_timeline_gate_expired)
        # Expiry is permanent even though the prior is still 'running'.
        self.assertFalse(gated(stub))

    def test_incomplete_formal_timeline_records_fatal_error(self) -> None:
        import threading
        from types import SimpleNamespace

        from demo_v5_1.main_data_processing import MainDataProcessingDemo

        metadata_updates: list[dict[str, object]] = []
        stop_event = threading.Event()
        stub = SimpleNamespace(
            stop_event=stop_event,
            _close_lossless_queues=lambda: None,
            _threads=[],
            runtime=None,
            recording_source=None,
            ffs_remote_client=None,
            remote_quality_client=None,
            _run_deferred_shape_prior_after_teardown=lambda: None,
            _write_shape_prior_profile_json=lambda: None,
            headless_capture_writer=SimpleNamespace(
                update_metadata=metadata_updates.append,
            ),
            _formal_timeline_gated_frames=3,
            _formal_timeline_metadata_written=False,
            _fatal_error_lock=threading.Lock(),
            _fatal_error=None,
            _request_render_update=lambda: None,
            filter_worker=None,
            _local_ffs_lock=threading.Lock(),
            _local_ffs_depth_cache={},
        )
        stub._record_fatal_worker_error = (  # type: ignore[attr-defined]
            MainDataProcessingDemo._record_fatal_worker_error.__get__(
                stub,
                type(stub),
            )
        )

        MainDataProcessingDemo.stop(stub)

        self.assertTrue(stop_event.is_set())
        self.assertEqual(
            [
                {
                    "formal_timeline_incomplete": True,
                    "formal_timeline_gated_frame_count": 3,
                }
            ],
            metadata_updates,
        )
        fatal = MainDataProcessingDemo._fatal_error_snapshot(stub)
        self.assertIsNotNone(fatal)
        assert fatal is not None
        self.assertEqual("formal chunk timeline", fatal.stage)
        self.assertEqual("RuntimeError", fatal.exc_type)
        self.assertIn("no formal timeline", fatal.message)

    def test_warmup_frame_is_first_frame_of_chunk_zero(self) -> None:
        from demo_v5_1 import chunk_data_output
        from demo_v5_1 import chunk_data_payload

        final_data, track_process, _ = chunk_data_payload.build_chunk_data_payload(
            _minimal_chunk_data_window(
                chunk_data_payload,
                frame_count=2,
                chunk_index=0,
                source_frame_indices=[0, 42],
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = chunk_data_output.ChunkDataWriter(
                base_path=tmpdir,
                case_name="demo_v5_1",
                chunk_size=2,
                num_frames_total=2,
            )
            result = output.commit_chunk_data(
                final_data,
                track_process,
                source_frame_indices=[0, 42],
                source_timestamps_s=[1.25, 4.5],
                status="recording",
            )
            output.finish()

            root = Path(tmpdir)
            chunk_path = root / "online_data" / "chunks" / "chunk_000000.pkl"
            static_data_path = root / "data" / "final_data.pkl"
            self.assertEqual(chunk_path, Path(result["online_chunk_path"]))
            self.assertTrue(chunk_path.is_file())
            self.assertTrue(static_data_path.is_file())
            self.assertFalse(
                (root / "online_data" / "chunks" / "chunk_warmup.pkl").exists()
            )
            with chunk_path.open("rb") as handle:
                chunk = pickle.load(handle)
            with static_data_path.open("rb") as handle:
                static_data = pickle.load(handle)
            self.assertNotIn("chunk_role", chunk)
            self.assertNotIn("is_warmup_chunk", chunk)
            self.assertEqual(0, chunk["chunk_id"])
            self.assertEqual(0, chunk["start_frame"])
            self.assertEqual(2, chunk["end_frame"])
            self.assertEqual([0, 42], chunk["source_frame_indices"])
            self.assertEqual([1.25, 4.5], chunk["source_timestamps_s"])
            self.assertEqual(2, int(np.asarray(chunk["object_points"]).shape[0]))
            np.testing.assert_allclose(
                np.asarray(chunk["object_points"])[0],
                np.asarray(final_data["object_points"])[0],
            )
            np.testing.assert_allclose(
                np.asarray(static_data["object_points"])[0],
                np.asarray(final_data["object_points"])[0],
            )


if __name__ == "__main__":
    unittest.main()
