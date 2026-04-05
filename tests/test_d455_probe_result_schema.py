from __future__ import annotations

import unittest

from scripts.harness.probe_d455_stream_capability import build_run_document, make_case, normalize_case_result


class D455ProbeResultSchemaTest(unittest.TestCase):
    def test_normalized_case_result_has_required_fields(self) -> None:
        case = make_case(
            topology_type="single",
            serials=["239222300433"],
            stream_set="depth",
            width=848,
            height=480,
            fps=30,
            emitter_request="auto",
            warmup_s=2.0,
            duration_s=10.0,
        )
        result = normalize_case_result(case, {"error_type": "RuntimeError", "error_message": "frame timeout"})

        for key in (
            "schema_version",
            "case_id",
            "topology_type",
            "serials",
            "stream_set",
            "requested_streams",
            "width",
            "height",
            "fps",
            "emitter_request",
            "start_success",
            "success",
            "status",
            "error_type",
            "error_message",
            "per_camera",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["error_message"], "frame timeout")

    def test_run_document_contains_required_sections(self) -> None:
        run_document = build_run_document(
            run_id="test-run",
            output_root=None,  # type: ignore[arg-type]
            cases=[],
            recommendation={
                "primary_case": "F",
                "primary_statement": "Only single-camera support is stable.",
                "comparison_case": "E",
                "comparison_statement": "Do not promise same-take comparison.",
                "evidence_case_ids": [],
            },
        )

        for key in (
            "schema_version",
            "generated_at",
            "run_id",
            "host",
            "stable_serial_order",
            "expectation_sources",
            "stability_thresholds",
            "cases",
            "recommendation",
        ):
            self.assertIn(key, run_document)


if __name__ == "__main__":
    unittest.main()
