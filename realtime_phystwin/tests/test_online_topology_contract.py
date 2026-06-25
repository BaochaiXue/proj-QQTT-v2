import importlib.util
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _import_realtime_online_frame_buffer():
    realtime_root = Path(__file__).resolve().parents[1]
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "qqtt" or name.startswith("qqtt.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(realtime_root))
    try:
        from qqtt.data.online_stream import OnlineFrameBuffer as buffer_cls
    finally:
        try:
            sys.path.remove(str(realtime_root))
        except ValueError:
            pass
        for name in list(sys.modules):
            if name == "qqtt" or name.startswith("qqtt."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
    return buffer_cls


OnlineFrameBuffer = _import_realtime_online_frame_buffer()


TOPOLOGY_KEYS = (
    "query_ids",
    "query_semantic_labels",
    "object_sample_query_ids",
    "controller_sample_query_ids",
    "topology_version",
    "topology_hash",
)


def _topology():
    controller_ids = np.arange(100, 130, dtype=np.int64)
    object_ids = np.array([10, 11], dtype=np.int64)
    query_ids = np.concatenate([object_ids, controller_ids])
    labels = np.concatenate(
        [
            np.full((len(object_ids),), 1, dtype=np.int8),
            np.full((len(controller_ids),), 2, dtype=np.int8),
        ]
    )
    return {
        "query_ids": query_ids,
        "query_semantic_labels": labels,
        "object_sample_query_ids": object_ids.copy(),
        "controller_sample_query_ids": controller_ids.copy(),
        "topology_version": "demo_v4_session_topology_v1",
        "topology_hash": "a" * 64,
    }


def _online_payload(frame_count=2, *, start_frame=0, topology=None, object_count=2, controller_count=30):
    if topology is None:
        topology = _topology()
    object_points = np.zeros((frame_count, object_count, 3), dtype=np.float32)
    object_points[:, :, 0] = np.linspace(0.01, 0.02, object_count)
    object_points[:, :, 2] = -0.1
    controller_points = np.zeros((frame_count, controller_count, 3), dtype=np.float32)
    controller_points[:, :, 0] = np.linspace(0.05, 0.15, controller_count)
    controller_points[:, :, 2] = -0.2
    payload = {
        "case_name": "demo_v4",
        "chunk_id": int(start_frame // max(1, frame_count)),
        "start_frame": int(start_frame),
        "end_frame": int(start_frame + frame_count),
        "source_frame_indices": list(range(start_frame, start_frame + frame_count)),
        "object_points": object_points,
        "object_colors": np.ones_like(object_points, dtype=np.float32),
        "object_visibilities": np.ones((frame_count, object_count), dtype=bool),
        "object_motions_valid": np.ones((frame_count, object_count), dtype=bool),
        "controller_points": controller_points,
    }
    payload.update({key: value.copy() if isinstance(value, np.ndarray) else value for key, value in topology.items()})
    return payload


def _write_static(path: Path, topology=None):
    payload = _online_payload(frame_count=1, topology=topology)
    payload["surface_points"] = np.array([[0.0, 0.0, -0.03]], dtype=np.float32)
    payload["interior_points"] = np.array([[0.0, 0.0, -0.04]], dtype=np.float32)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def _fake_online_tracker_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "fake_online_tracker.py"
    spec = importlib.util.spec_from_file_location("fake_online_tracker_for_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OnlineTopologyContractTest(unittest.TestCase):
    def test_online_frame_buffer_accepts_identical_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_path = Path(tmp) / "final_data.pkl"
            topology = _topology()
            _write_static(static_path, topology)
            buffer = OnlineFrameBuffer(static_data_path=static_path, device="cpu")

            added = buffer.append_chunks(
                [
                    _online_payload(start_frame=0, topology=topology),
                    _online_payload(start_frame=2, topology=topology),
                ]
            )

            self.assertEqual(added, 4)
            self.assertEqual(buffer.frame_len, 4)

    def test_online_frame_buffer_rejects_changed_topology_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_path = Path(tmp) / "final_data.pkl"
            topology = _topology()
            _write_static(static_path, topology)
            buffer = OnlineFrameBuffer(static_data_path=static_path, device="cpu")
            buffer.append_chunks([_online_payload(start_frame=0, topology=topology)])

            changed = _topology()
            changed["topology_hash"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "topology_hash"):
                buffer.append_chunks([_online_payload(start_frame=2, topology=changed)])

    def test_online_frame_buffer_rejects_changed_sample_ids_even_when_shapes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_path = Path(tmp) / "final_data.pkl"
            topology = _topology()
            _write_static(static_path, topology)
            buffer = OnlineFrameBuffer(static_data_path=static_path, device="cpu")
            buffer.append_chunks([_online_payload(start_frame=0, topology=topology)])

            changed = _topology()
            changed["controller_sample_query_ids"] = changed["controller_sample_query_ids"].copy()
            changed["controller_sample_query_ids"][0] += 1
            with self.assertRaisesRegex(ValueError, "controller_sample_query_ids"):
                buffer.append_chunks([_online_payload(start_frame=2, topology=changed)])

    def test_online_frame_buffer_rejects_point_count_sample_id_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_path = Path(tmp) / "final_data.pkl"
            topology = _topology()
            _write_static(static_path, topology)
            buffer = OnlineFrameBuffer(static_data_path=static_path, device="cpu")

            with self.assertRaisesRegex(ValueError, "object_sample_query_ids"):
                buffer.append_chunks([
                    _online_payload(
                        start_frame=0,
                        topology=topology,
                        object_count=3,
                    )
                ])

    def test_fake_online_tracker_synthesizes_legacy_topology_and_writes_it_to_chunks(self):
        fake = _fake_online_tracker_module()
        frame_count = 3
        legacy = {
            "object_points": np.zeros((frame_count, 2, 3), dtype=np.float32),
            "object_colors": np.zeros((frame_count, 2, 3), dtype=np.float32),
            "object_visibilities": np.ones((frame_count, 2), dtype=bool),
            "object_motions_valid": np.ones((frame_count, 2), dtype=bool),
            "controller_points": np.zeros((frame_count, 30, 3), dtype=np.float32),
            "surface_points": np.zeros((1, 3), dtype=np.float32),
            "interior_points": np.zeros((1, 3), dtype=np.float32),
        }

        fake.ensure_topology_contract(legacy)
        chunk = fake.build_chunk(
            legacy,
            case_name="legacy_case",
            chunk_id=0,
            start_frame=0,
            end_frame=2,
            include_static=False,
            source_frame_indices=[0, 1],
        )

        for key in TOPOLOGY_KEYS:
            self.assertIn(key, legacy)
            self.assertIn(key, chunk)
        self.assertEqual(legacy["topology_version"], "demo_v4_session_topology_v1")
        self.assertRegex(legacy["topology_hash"], r"^[0-9a-f]{64}$")
        np.testing.assert_array_equal(legacy["object_sample_query_ids"], np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(legacy["controller_sample_query_ids"], np.arange(2, 32, dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
