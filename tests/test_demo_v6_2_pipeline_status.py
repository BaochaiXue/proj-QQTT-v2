"""Tests for the Demo v6.2 live pipeline-status stream + renderer (Q23)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from demo_v6_2.pipeline_status import (
    STAGE_CAPTURE_START,
    STAGE_FATAL,
    STAGE_RUN_START,
    PipelineStatusWriter,
    read_status_events,
    status_path,
)


class StatusWriterTests(unittest.TestCase):
    def test_emit_roundtrips_through_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            writer = PipelineStatusWriter(base, "camera")
            writer.emit(STAGE_RUN_START, "input=fake-live")
            writer.emit(STAGE_CAPTURE_START, "capturing", frame_index=0)
            writer.emit(STAGE_FATAL, "warmup boom", ok=False, exc_type="ValueError")
            events = read_status_events(base)
            self.assertEqual([e["stage"] for e in events],
                             [STAGE_RUN_START, STAGE_CAPTURE_START, STAGE_FATAL])
            self.assertEqual(events[0]["source"], "camera")
            self.assertFalse(events[2]["ok"])
            self.assertEqual(events[2]["exc_type"], "ValueError")
            self.assertTrue(status_path(base).is_file())

    def test_none_base_is_noop(self) -> None:
        writer = PipelineStatusWriter(None, "camera")
        writer.emit(STAGE_RUN_START, "should not raise or write")  # no exception

    def test_reader_tolerates_missing_file_and_torn_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.assertEqual(read_status_events(base), [])
            status_path(base).write_text(
                '{"stage": "run_start", "ok": true}\n{"stage": "capture_', # torn last line
                encoding="utf-8",
            )
            events = read_status_events(base)
            self.assertEqual([e["stage"] for e in events], ["run_start"])

    def test_emit_never_raises_on_bad_path(self) -> None:
        # base_path points at a file, so opening <file>/pipeline_status.jsonl for
        # append fails; emit must swallow it.
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "a_file"
            bad.write_text("x", encoding="utf-8")
            PipelineStatusWriter(bad, "camera").emit(STAGE_RUN_START, "boom")


class StatusRendererTests(unittest.TestCase):
    def test_band_drawn_and_fatal_is_red(self) -> None:
        from demo_v6_2.viz_panels import draw_pipeline_status

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        # Fixed now_s keeps the drawn text deterministic.
        ok_events = [{"t": 100.0, "source": "camera", "stage": STAGE_CAPTURE_START,
                      "detail": "capturing", "ok": True}]
        out = draw_pipeline_status(image.copy(), ok_events, now_s=101.0)
        band = out[240 - 34:, :, :]
        self.assertGreater(int(band.sum()), 0, "status band should draw pixels")
        # A fatal event paints the band red (BGR: high red channel).
        fatal_events = ok_events + [{"t": 102.0, "source": "camera",
                                     "stage": STAGE_FATAL, "detail": "warmup boom",
                                     "ok": False}]
        red = draw_pipeline_status(image.copy(), fatal_events, now_s=103.0)
        band_red = red[240 - 34:, :, :]
        mean_bgr = band_red.reshape(-1, 3).mean(axis=0)
        self.assertGreater(mean_bgr[2], mean_bgr[0], "fatal band should be red-dominant")
        self.assertGreater(mean_bgr[2], mean_bgr[1], "fatal band should be red-dominant")

    def test_empty_events_draws_waiting_band(self) -> None:
        from demo_v6_2.viz_panels import draw_pipeline_status

        image = np.zeros((120, 200, 3), dtype=np.uint8)
        out = draw_pipeline_status(image, [], now_s=0.0)
        self.assertGreater(int(out[120 - 34:, :, :].sum()), 0)


if __name__ == "__main__":
    unittest.main()
