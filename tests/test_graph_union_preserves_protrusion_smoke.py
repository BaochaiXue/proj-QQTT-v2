from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.object_roi import estimate_object_roi_bounds


class GraphUnionPreservesProtrusionSmokeTest(unittest.TestCase):
    def test_graph_union_keeps_torso_head_and_ear_via_transitive_closure(self) -> None:
        xx, yy = np.meshgrid(np.linspace(-0.35, 0.35, 40), np.linspace(-0.35, 0.35, 40), indexing="xy")
        table = np.stack([xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)], axis=1).astype(np.float32)

        torso_xx, torso_yy, torso_zz = np.meshgrid(
            np.linspace(-0.05, 0.05, 10),
            np.linspace(-0.05, 0.05, 10),
            np.linspace(0.05, 0.11, 6),
            indexing="xy",
        )
        torso = np.stack([torso_xx.reshape(-1), torso_yy.reshape(-1), torso_zz.reshape(-1)], axis=1).astype(np.float32)

        head_xx, head_yy, head_zz = np.meshgrid(
            np.linspace(-0.03, 0.03, 6),
            np.linspace(-0.03, 0.03, 6),
            np.linspace(0.16, 0.21, 4),
            indexing="xy",
        )
        head = np.stack([head_xx.reshape(-1), head_yy.reshape(-1), head_zz.reshape(-1)], axis=1).astype(np.float32)

        ear = np.stack(
            [
                np.linspace(0.055, 0.075, 16),
                np.linspace(-0.01, 0.01, 16),
                np.linspace(0.22, 0.27, 16),
            ],
            axis=1,
        ).astype(np.float32)

        points = np.concatenate([table, torso, head, ear], axis=0)
        bounds = {"min": points.min(axis=0), "max": points.max(axis=0)}

        largest = estimate_object_roi_bounds(
            points,
            fallback_bounds=bounds,
            full_bounds=bounds,
            object_height_min=0.02,
            object_height_max=0.30,
            object_component_mode="largest",
            object_component_topk=2,
            roi_margin_xy=0.02,
            roi_margin_z=0.02,
        )
        graph_union = estimate_object_roi_bounds(
            points,
            fallback_bounds=bounds,
            full_bounds=bounds,
            object_height_min=0.02,
            object_height_max=0.30,
            object_component_mode="graph_union",
            object_component_topk=2,
            roi_margin_xy=0.02,
            roi_margin_z=0.02,
        )

        self.assertGreaterEqual(len(graph_union["selected_component_indices"]), 2)
        self.assertGreater(float(graph_union["object_roi_max"][2]), float(largest["object_roi_max"][2]) + 0.04)


if __name__ == "__main__":
    unittest.main()
