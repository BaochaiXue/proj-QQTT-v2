from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import trimesh


ROOT = Path(__file__).resolve().parents[1]
ORIGIN = ROOT / "data_process_origin"


def import_origin_module(module_name: str, path: Path):
    old_path = list(sys.path)
    sys.path.insert(0, str(ORIGIN))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


class DataProcessOriginSam3DTests(unittest.TestCase):
    def test_shape_prior_uses_rgba_mask_and_exports_object_glb(self) -> None:
        shape_prior = import_origin_module(
            "data_process_origin_shape_prior_test",
            ORIGIN / "shape_prior.py",
        )

        class FakeMesh:
            def export(self, path: Path) -> None:
                Path(path).write_bytes(b"fake glb")

        class FakeGaussian:
            def save_ply(self, path: Path) -> None:
                Path(path).write_bytes(b"fake ply")

        class FakePipeline:
            def __init__(self) -> None:
                self.calls: list[tuple[np.ndarray, np.ndarray, dict]] = []
                self.rendering_engine = "pytorch3d"

            def run(self, image_rgb, mask, **kwargs):
                self.calls.append((image_rgb, mask, kwargs))
                return {"glb": FakeMesh(), "gaussian": [FakeGaussian()]}

        fake_pipeline = FakePipeline()

        class FakeInference:
            def __init__(self, config: str, compile: bool = False) -> None:
                self.config = config
                self.compile = compile
                self._pipeline = fake_pipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            sam3d_root = tmp / "sam3d"
            (sam3d_root / "notebook").mkdir(parents=True)
            (sam3d_root / "notebook" / "inference.py").write_text("", encoding="utf-8")
            (sam3d_root / "sam3d_objects").mkdir()
            config = sam3d_root / "checkpoints" / "hf" / "pipeline.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("pipeline: fake\n", encoding="utf-8")

            rgba = np.zeros((4, 4, 4), dtype=np.uint8)
            rgba[:, :, :3] = [10, 20, 30]
            rgba[1:3, 1:3, 3] = 255
            image_path = tmp / "object.png"
            Image.fromarray(rgba).save(image_path)

            output_dir = tmp / "shape"
            with mock.patch.object(
                shape_prior,
                "load_inference_class",
                return_value=FakeInference,
            ):
                shape_prior.main(
                    [
                        "--img_path",
                        str(image_path),
                        "--output_dir",
                        str(output_dir),
                        "--sam3d-root",
                        str(sam3d_root),
                        "--skip-visualization",
                    ]
                )

            self.assertTrue((output_dir / "object.glb").is_file())
            self.assertTrue((output_dir / "object.ply").is_file())
            image_rgb, mask, kwargs = fake_pipeline.calls[0]
            self.assertEqual(image_rgb.shape, (4, 4, 3))
            self.assertEqual(sorted(np.unique(mask).tolist()), [0, 255])
            self.assertTrue(kwargs["with_mesh_postprocess"])
            self.assertTrue(kwargs["with_texture_baking"])
            self.assertTrue(kwargs["with_layout_postprocess"])
            self.assertFalse(kwargs["use_vertex_color"])

    def test_align_import_is_cli_safe_and_helpers_work(self) -> None:
        align = import_origin_module("data_process_origin_align_test", ORIGIN / "align.py")
        mesh_points = np.eye(3, dtype=np.float32)
        observed_points = mesh_points * np.float32(2.5)

        scale = float(align.registration_scale(mesh_points, observed_points))
        self.assertAlmostEqual(scale, 2.5, places=4)

        distances = align.line_point_distance(
            np.array([1.0, 0.0, 0.0]),
            np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        )
        np.testing.assert_allclose(distances, np.array([0.0, 3.0]))

    def test_sample_import_is_cli_safe_and_shape_prior_outputs_fields(self) -> None:
        sample = import_origin_module(
            "data_process_origin_sample_test",
            ORIGIN / "data_process_sample.py",
        )

        class FakeViewControl:
            def rotate(self, *_args) -> None:
                return None

        class FakeVisualizer:
            def create_window(self, visible: bool = False) -> None:
                return None

            def capture_screen_float_buffer(self, do_render: bool = True):
                return np.zeros((2, 2, 3), dtype=np.float32)

            def add_geometry(self, *_args) -> None:
                return None

            def get_view_control(self):
                return FakeViewControl()

            def poll_events(self) -> None:
                return None

            def update_renderer(self) -> None:
                return None

            def destroy_window(self) -> None:
                return None

        class FakeWriter:
            def write(self, _frame) -> None:
                return None

        track_data = {
            "object_points": np.array(
                [[[0.0, 0.0, -0.1], [0.0, 0.0, -0.1], [0.2, 0.0, -0.1]]]
            ),
            "object_colors": np.zeros((1, 3, 3), dtype=np.float32),
            "object_visibilities": np.ones((1, 3), dtype=bool),
            "object_motions_valid": np.ones((1, 3), dtype=bool),
            "controller_points": np.zeros((1, 1, 3), dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            case_dir = tmp / "case"
            mesh_dir = case_dir / "shape" / "matching"
            mesh_dir.mkdir(parents=True)
            trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(
                mesh_dir / "final_mesh.glb"
            )

            sample.base_path = str(tmp)
            sample.case_name = "case"
            sample.SHAPE_PRIOR = True
            sample.num_surface_points = 2
            sample.volume_sample_size = 0.001

            with mock.patch.object(
                sample.o3d.visualization,
                "Visualizer",
                FakeVisualizer,
            ), mock.patch.object(
                sample.cv2,
                "VideoWriter",
                return_value=FakeWriter(),
            ), mock.patch.object(
                sample.cv2,
                "VideoWriter_fourcc",
                return_value=0,
            ), mock.patch.object(
                sample.cv2,
                "cvtColor",
                side_effect=lambda frame, _code: frame,
            ), mock.patch.object(
                sample.trimesh.sample,
                "sample_surface",
                return_value=(np.array([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]]), None),
            ), mock.patch.object(
                sample.trimesh.sample,
                "volume_mesh",
                return_value=np.array([[2.0, 0.0, 0.0]]),
            ):
                result = sample.process_unique_points(track_data)

        self.assertIn("surface_points", result)
        self.assertIn("interior_points", result)
        self.assertEqual(result["surface_points"].shape, (2, 3))
        self.assertEqual(result["interior_points"].shape, (1, 3))
        self.assertEqual(result["object_points"].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
