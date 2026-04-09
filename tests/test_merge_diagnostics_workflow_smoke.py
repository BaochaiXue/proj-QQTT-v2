from __future__ import annotations

import unittest

from data_process.visualization.types import RenderOutputSpec
from data_process.visualization.workflows.merge_diagnostics import build_render_output_spec_models


class MergeDiagnosticsWorkflowSmokeTest(unittest.TestCase):
    def test_render_output_spec_models_include_merge_diagnostics(self) -> None:
        specs = build_render_output_spec_models(
            geom_render_mode="neutral_gray_shaded",
            render_both_modes=True,
        )
        self.assertTrue(all(isinstance(item, RenderOutputSpec) for item in specs))
        self.assertEqual([item.name for item in specs], ["geom", "rgb", "support", "source", "mismatch"])
