from __future__ import annotations

import importlib
import unittest
from pathlib import Path
from typing import Any

import numpy as np


DEMO_PACKAGES = ("demo_v5", "demo_v5_1")
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
                wrapper = importlib.import_module(f"{package}.futurephystwin_chunk_writer")
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


if __name__ == "__main__":
    unittest.main()
