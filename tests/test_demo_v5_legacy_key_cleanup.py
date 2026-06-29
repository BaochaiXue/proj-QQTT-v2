from __future__ import annotations

import importlib
import unittest
from pathlib import Path
from typing import Any

import numpy as np


DEMO_PACKAGES = ("demo_v5_1",)
REMOVED_WRAPPER_EXPORTS = (
    "CONTROLLER_ANCHOR_STATIC_KEYS",
    "CONTROLLER_ANCHOR_TIME_KEYS",
    "CONTROLLER_QUALITY_STATUSES",
    "FUTUREPHYSTWIN_TOPOLOGY_KEYS",
    "FUTUREPHYSTWIN_TOPOLOGY_VERSION",
    "build_topology_payload",
)
REMOVED_SOURCE_TOKENS = (
    "LEGACY_TO_CANONICAL_KEYS",
    "CANONICAL_TO_LEGACY_KEYS",
    "LEGACY_QUERY_SCHEMA_KEYS",
    "normalize_data_process_keys",
    "canonical_key",
    "legacy_key",
    "add_legacy_aliases",
    "controller_anchor_query_indices",
    "controller_anchor_active_query_indices",
    "controller_anchor_status",
    "controller_anchor_bundle_query_ids",
    "controller_anchor_source_query_id",
    "controller_anchor_observation_mode",
    "controller_anchor_confidence",
    "controller_anchor_failure_reason",
    "controller_anchor_bundle_support_count",
    "controller_anchor_bundle_raw_visible_count",
    "controller_anchor_bundle_depth_valid_count",
    "controller_anchor_bundle_processed_mask_valid_count",
    "controller_anchor_bundle_motion_valid_count",
    "controller_anchor_recovery_residual",
    "controller_anchor_mode",
    "controller_anchor_count",
    "object_anchor_query_indices",
    "object_anchor_active_query_indices",
    "object_anchor_status",
    "object_anchor_mode",
    "object_anchor_count",
    "controller_quality_status",
    "controller_fps_indices",
    "topology_version",
    "topology_hash",
    "futurephystwin_case_root",
    "futurephystwin_base_path",
)


def _canonical_final_payload(writer: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "controller_points": np.asarray([[[0.20, 0.0, 1.0]]], dtype=np.float32),
        "controller_final_indices": np.asarray([0], dtype=np.int64),
        "controller_selected_query_ids": np.asarray([20], dtype=np.int64),
        "controller_sample_query_ids": np.asarray([20], dtype=np.int64),
        "object_colors": np.asarray([[[0.7, 0.2, 0.1]]], dtype=np.float32),
        "object_motions_valid": np.asarray([[True]], dtype=bool),
        "object_points": np.asarray([[[0.05, 0.0, 1.0]]], dtype=np.float32),
        "object_sample_indices": np.asarray([0], dtype=np.int64),
        "object_selected_query_ids": np.asarray([10], dtype=np.int64),
        "object_sample_query_ids": np.asarray([10], dtype=np.int64),
        "object_visibilities": np.asarray([[True]], dtype=bool),
        "query_schema_version": writer.DATA_PROCESS_QUERY_SCHEMA_VERSION,
        "query_ids": np.asarray([10, 20], dtype=np.int64),
        "query_semantic_labels": np.asarray([1, 2], dtype=np.int8),
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
    }
    payload["query_schema_hash"] = writer._query_schema_hash(payload)
    return payload


class DemoV5LegacyKeyCleanupTests(unittest.TestCase):
    def test_schema_alias_modules_are_removed(self) -> None:
        for package in DEMO_PACKAGES:
            with self.subTest(package=package):
                importlib.invalidate_caches()
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(f"{package}.data_process_schema")

    def test_demo_v51_atomic_io_helpers_live_under_utils(self) -> None:
        importlib.invalidate_caches()
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("demo_v5_1.atomic_io")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("demo_v5_1.tools.atomic_io")

        helpers = importlib.import_module("demo_v5_1.utils.atomic_io")
        self.assertTrue(hasattr(helpers, "atomic_json_dump"))
        self.assertTrue(hasattr(helpers, "atomic_pickle_dump"))

    def test_demo_v51_shape_prior_warmup_lives_with_demo_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        old_worker_path = root / "services" / "shape_prior_remote" / "server.py"

        self.assertTrue((root / "demo_v5_1" / "shape_prior_warmup.py").is_file())
        self.assertFalse((root / "demo_v5_1" / "shape_prior_worker.py").exists())
        self.assertFalse(old_worker_path.exists())

    def test_demo_v51_shape_prior_warmup_uses_demo_relative_repo_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "demo_v5_1" / "shape_prior_warmup.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("Path(__file__).resolve().parents[1]", source)
        self.assertNotIn("QQTT_REPO_ROOT", source)
        self.assertNotIn('(root / "services").is_dir()', source)

    def test_demo_v51_shape_prior_warmup_uses_local_shape_prior_helpers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "demo_v5_1" / "shape_prior_warmup.py").read_text(
            encoding="utf-8"
        )

        self.assertFalse((root / "demo_v5_1" / "shape_prior.py").exists())
        self.assertFalse((root / "demo_v5_1" / "shape_prior_runtime.py").exists())
        self.assertFalse((root / "demo_v5_1" / "shape_prior_rpc.py").exists())
        self.assertFalse((root / "demo_v5_1" / "single_view_shape_align.py").exists())
        self.assertFalse(
            (root / "demo_v5_1" / "single_view_shape_prior_sampling.py").exists()
        )
        self.assertNotIn("from qqtt.demo", source)
        self.assertNotIn("import qqtt.demo", source)
        self.assertNotIn("services.shape_prior_remote", source)
        self.assertIn("demo_v5_1.shape_prior_generate", source)
        self.assertIn("demo_v5_1.shape_prior_align", source)
        self.assertIn("demo_v5_1.shape_prior_sample", source)

    def test_legacy_final_data_key_is_rejected_instead_of_normalized(self) -> None:
        for package in DEMO_PACKAGES:
            writer = importlib.import_module(f"{package}.data_process_chunk_writer")
            payload = _canonical_final_payload(writer)
            payload["controller_fps_indices"] = payload.pop("controller_final_indices")

            with self.subTest(package=package):
                with self.assertRaisesRegex(ValueError, "controller_final_indices"):
                    writer._validate_final_shapes(payload)

    def test_futurephystwin_wrapper_does_not_export_old_alias_names(self) -> None:
        for package in DEMO_PACKAGES:
            try:
                wrapper = importlib.import_module(
                    f"{package}.futurephystwin_chunk_writer"
                )
            except ModuleNotFoundError:
                continue
            with self.subTest(package=package):
                for name in REMOVED_WRAPPER_EXPORTS:
                    self.assertFalse(hasattr(wrapper, name), name)

    def test_demo_v5_sources_do_not_contain_removed_legacy_key_tokens(self) -> None:
        root = Path(__file__).resolve().parents[1]
        violations: list[str] = []
        for package in DEMO_PACKAGES:
            for path in sorted((root / package).glob("*.py")):
                source = path.read_text(encoding="utf-8")
                for token in REMOVED_SOURCE_TOKENS:
                    if token in source:
                        violations.append(f"{path.relative_to(root)}: {token}")
        self.assertEqual([], violations)

    def test_demo_v51_main_data_processing_is_local_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source_path = root / "demo_v5_1" / "main_data_processing.py"
        source = source_path.read_text(encoding="utf-8")

        forbidden_tokens = (
            "from qqtt.demo import realtime_masked_edgetam_pcd",
            "import qqtt.demo.realtime_masked_edgetam_pcd",
            "from demo_v5 import realtime_dense_track",
            "import demo_v5.realtime_dense_track",
            "from demo_v5 import main_data_processing",
            "import demo_v5.main_data_processing",
            "RealtimeMaskedEdgeTamPcdDemo",
            "masked_pcd.main",
            "thin wrapper",
        )
        for token in forbidden_tokens:
            with self.subTest(token=token):
                self.assertNotIn(token, source)
        self.assertIn("class MainDataProcessingDemo", source)
        self.assertIn("def build_parser(", source)
        self.assertIn("def main(", source)


if __name__ == "__main__":
    unittest.main()
