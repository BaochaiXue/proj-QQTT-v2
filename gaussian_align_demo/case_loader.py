"""Reader for a demo_v6_2 shape-prior frame-0 case (external, read-only).

Contract (see demo_v6_2/shape_prior/case.py::write_shape_prior_case):

- ``color/0/0.png``          frame-0 RGB (PIL RGB on disk).
- ``mask/0/0/0.png``         object mask, uint8 {0, 255}.
- ``mask/0/1/0.png``         controller (hand) mask, uint8 {0, 255}.
- ``mask/mask_info_0.json``  {"0": object_name, "1": controller_name}.
- ``pcd/0.npz``              points (1,H,W,3) float32 WORLD-space meters,
                             colors (1,H,W,3) [0,1] RGB, masks (1,H,W) bool
                             depth-valid.
- ``calibrate.pkl``          [c2w] — one 4x4 float32 camera-to-world,
                             OpenCV column-vector convention, table world (z=0
                             is the table plane).
- ``metadata.json``          intrinsics = [ [3x3] ]; NO width/height (take
                             from the PNG).
- ``shape/matching/final_mesh.glb``  optional; SAM3D mesh already aligned to
                             the same world frame (weak reference only).

The trajectory ``<base>/data/final_data.pkl`` lives in the SAME world frame
(demo_v6_2 writes byte-identical calibrate.pkl for both).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Frame0Case:
    case_dir: Path
    rgb_u8: np.ndarray  # (H, W, 3) uint8
    object_mask: np.ndarray  # (H, W) bool
    controller_mask: np.ndarray  # (H, W) bool
    points_world: np.ndarray  # (H, W, 3) float32, world meters
    depth_valid: np.ndarray  # (H, W) bool
    K: np.ndarray  # (3, 3) float64
    c2w: np.ndarray  # (4, 4) float64
    object_name: str
    controller_name: str

    @property
    def width(self) -> int:
        return int(self.rgb_u8.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb_u8.shape[0])

    @property
    def w2c(self) -> np.ndarray:
        return np.linalg.inv(self.c2w)

    @property
    def depth_m(self) -> np.ndarray:
        """(H, W) float32 z-depth in the camera frame; 0 where invalid."""
        w2c = self.w2c
        cam_z = self.points_world @ w2c[2, :3].T + w2c[2, 3]
        return np.where(self.depth_valid, cam_z, 0.0).astype(np.float32)

    @property
    def object_points_world(self) -> np.ndarray:
        """(N, 3) world points of depth-valid object pixels."""
        sel = self.object_mask & self.depth_valid
        return self.points_world[sel].reshape(-1, 3)

    @property
    def final_mesh_path(self) -> Path | None:
        path = self.case_dir / "shape" / "matching" / "final_mesh.glb"
        return path if path.exists() else None

    def object_rgba(self) -> np.ndarray:
        """(H, W, 4) uint8 RGBA — alpha is the object mask (TripoSplat input)."""
        alpha = np.where(self.object_mask, 255, 0).astype(np.uint8)
        return np.concatenate([self.rgb_u8, alpha[..., None]], axis=2)


def _load_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim != 2:
        raise ValueError(f"{path}: expected single-channel mask, got shape {mask.shape}")
    return mask > 127


def load_frame0_case(case_dir: str | Path) -> Frame0Case:
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"case dir not found: {case_dir}")

    rgb = np.asarray(Image.open(case_dir / "color" / "0" / "0.png").convert("RGB"))
    object_mask = _load_binary_mask(case_dir / "mask" / "0" / "0" / "0.png")
    controller_mask = _load_binary_mask(case_dir / "mask" / "0" / "1" / "0.png")

    with np.load(case_dir / "pcd" / "0.npz") as pcd:
        points = np.asarray(pcd["points"][0], dtype=np.float32)
        depth_valid = np.asarray(pcd["masks"][0], dtype=bool)

    with open(case_dir / "calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    c2w = np.asarray(c2ws[0], dtype=np.float64).reshape(4, 4)

    metadata = json.loads((case_dir / "metadata.json").read_text())
    K = np.asarray(metadata["intrinsics"][0], dtype=np.float64).reshape(3, 3)

    mask_info = json.loads((case_dir / "mask" / "mask_info_0.json").read_text())
    controller_name = mask_info.get("1", "controller")
    object_name = mask_info.get("0", "object")

    shapes = {rgb.shape[:2], object_mask.shape, controller_mask.shape,
              points.shape[:2], depth_valid.shape}
    if len(shapes) != 1:
        raise ValueError(f"{case_dir}: inconsistent image shapes across artifacts: {shapes}")

    return Frame0Case(
        case_dir=case_dir,
        rgb_u8=rgb,
        object_mask=object_mask,
        controller_mask=controller_mask,
        points_world=points,
        depth_valid=depth_valid,
        K=K,
        c2w=c2w,
        object_name=object_name,
        controller_name=controller_name,
    )


@dataclass
class Trajectory:
    """Aggregate final_data.pkl trajectory (same world frame as the case)."""

    path: Path
    object_points: np.ndarray  # (T, N, 3) float64 world meters
    controller_points: np.ndarray  # (T, M, 3) float64
    object_visibilities: np.ndarray  # (T, N) bool
    object_motions_valid: np.ndarray  # (T, N) bool

    @property
    def frame_count(self) -> int:
        return int(self.object_points.shape[0])

    @property
    def bone_count(self) -> int:
        return int(self.object_points.shape[1])


def load_trajectory(final_data_path: str | Path) -> Trajectory:
    final_data_path = Path(final_data_path)
    with open(final_data_path, "rb") as f:
        data = pickle.load(f)
    required = ["object_points", "controller_points", "object_visibilities", "object_motions_valid"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{final_data_path}: missing keys {missing}")
    object_points = np.asarray(data["object_points"], dtype=np.float64)
    if object_points.ndim != 3 or object_points.shape[2] != 3:
        raise ValueError(f"object_points has shape {object_points.shape}, expected (T, N, 3)")
    if not np.isfinite(object_points).all():
        raise ValueError(f"{final_data_path}: object_points contains non-finite values")
    return Trajectory(
        path=final_data_path,
        object_points=object_points,
        controller_points=np.asarray(data["controller_points"], dtype=np.float64),
        object_visibilities=np.asarray(data["object_visibilities"], dtype=bool),
        object_motions_valid=np.asarray(data["object_motions_valid"], dtype=bool),
    )
