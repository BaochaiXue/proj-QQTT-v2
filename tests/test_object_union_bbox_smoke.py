from __future__ import annotations

import unittest

import numpy as np

from data_process.visualization.object_roi import estimate_object_roi_bounds


class ObjectUnionBboxSmokeTest(unittest.TestCase):
    def test_union_bbox_keeps_sparse_head_above_dense_body(self) -> None:
        xx, yy = np.meshgrid(np.linspace(-0.25, 0.25, 30), np.linspace(-0.25, 0.25, 30), indexing="xy")
        table = np.stack([xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)], axis=1).astype(np.float32)

        body_xx, body_yy, body_zz = np.meshgrid(
            np.linspace(-0.06, 0.06, 10),
            np.linspace(-0.05, 0.05, 10),
            np.linspace(0.05, 0.12, 6),
            indexing="xy",
        )
        body = np.stack([body_xx.reshape(-1), body_yy.reshape(-1), body_zz.reshape(-1)], axis=1).astype(np.float32)

        head_xx, head_yy, head_zz = np.meshgrid(
            np.linspace(-0.03, 0.03, 5),
            np.linspace(-0.03, 0.03, 5),
            np.linspace(0.26, 0.31, 3),
            indexing="xy",
        )
        head = np.stack([head_xx.reshape(-1), head_yy.reshape(-1), head_zz.reshape(-1)], axis=1).astype(np.float32)

        points = np.concatenate([table, body, head], axis=0)
        bounds = {"min": points.min(axis=0), "max": points.max(axis=0)}

        largest = estimate_object_roi_bounds(
            points,
            fallback_bounds=bounds,
            full_bounds=bounds,
            object_height_min=0.02,
            object_height_max=0.40,
            object_component_mode="largest",
            object_component_topk=2,
            roi_margin_xy=0.02,
            roi_margin_z=0.02,
        )
        union = estimate_object_roi_bounds(
            points,
            fallback_bounds=bounds,
            full_bounds=bounds,
            object_height_min=0.02,
            object_height_max=0.40,
            object_component_mode="union",
            object_component_topk=2,
            roi_margin_xy=0.02,
            roi_margin_z=0.02,
        )

        self.assertGreater(len(union["selected_component_indices"]), 1)
        self.assertGreater(float(union["object_roi_max"][2]), float(largest["object_roi_max"][2]) + 0.08)


if __name__ == "__main__":
    unittest.main()
