"""3D Gaussian Splatting PLY I/O and rigid/similarity transforms.

Field semantics follow the standard 3DGS PLY convention (shared by TripoSplat
and PhysTwin exports):

- ``x, y, z``              gaussian centers.
- ``f_dc_0..2``            SH DC coefficients (RGB = 0.5 + C0 * f_dc).
- ``f_rest_*``             optional higher-order SH (TripoSplat writes none).
- ``opacity``              inverse-sigmoid logit of alpha.
- ``scale_0..2``           log of the per-axis linear scale.
- ``rot_0..3``             quaternion in wxyz order (not necessarily normalized
                           on disk; normalize on load).

Because scale is stored in log space and rotation as a quaternion, a
similarity transform x' = s * R @ x + t maps to::

    means'      = s * R @ means + t
    log_scales' = log_scales + log(s)
    quats'      = quat(R) ⊗ quats        (left multiply, wxyz)

Appearance fields (SH / opacity) are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


@dataclass
class GaussianCloud:
    """In-memory 3DGS point set. All arrays are float32, N rows each."""

    means: np.ndarray  # (N, 3)
    sh_dc: np.ndarray  # (N, 3) SH DC coefficients (f_dc_*)
    sh_rest: np.ndarray  # (N, M) flattened f_rest_* coefficients; M may be 0
    opacity_logits: np.ndarray  # (N,)
    log_scales: np.ndarray  # (N, 3)
    quats_wxyz: np.ndarray  # (N, 4), unit norm

    def __post_init__(self) -> None:
        n = self.means.shape[0]
        for name in ("sh_dc", "sh_rest", "opacity_logits", "log_scales", "quats_wxyz"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def colors_rgb(self) -> np.ndarray:
        """View-independent RGB in [0, 1] from the SH DC term."""
        return np.clip(0.5 + SH_C0 * self.sh_dc, 0.0, 1.0)

    @property
    def opacities(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacity_logits))

    @property
    def scales(self) -> np.ndarray:
        return np.exp(self.log_scales)

    def select(self, mask_or_indices: np.ndarray) -> "GaussianCloud":
        return GaussianCloud(
            means=self.means[mask_or_indices],
            sh_dc=self.sh_dc[mask_or_indices],
            sh_rest=self.sh_rest[mask_or_indices],
            opacity_logits=self.opacity_logits[mask_or_indices],
            log_scales=self.log_scales[mask_or_indices],
            quats_wxyz=self.quats_wxyz[mask_or_indices],
        )


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 for wxyz quaternions; broadcasts (4,)x(N,4)."""
    q1 = np.asarray(q1, dtype=np.float32).reshape(-1, 4)
    q2 = np.asarray(q2, dtype=np.float32).reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    ).astype(np.float32)


def rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix (3, 3) to a unit wxyz quaternion."""
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return (quat / np.linalg.norm(quat)).astype(np.float32)


def load_gaussian_ply(path: str | Path) -> GaussianCloud:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = {prop.name for prop in vertex.properties}
    required = {"x", "y", "z", "opacity"} | {f"f_dc_{i}" for i in range(3)}
    required |= {f"scale_{i}" for i in range(3)} | {f"rot_{i}" for i in range(4)}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path}: not a 3DGS PLY, missing fields {missing}")

    def stack(fields: list[str]) -> np.ndarray:
        return np.stack([np.asarray(vertex[f], dtype=np.float32) for f in fields], axis=1)

    rest_names = sorted(
        (n for n in names if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1])
    )
    n = vertex.count
    quats = stack([f"rot_{i}" for i in range(4)])
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.clip(norms, 1e-12, None)
    return GaussianCloud(
        means=stack(["x", "y", "z"]),
        sh_dc=stack([f"f_dc_{i}" for i in range(3)]),
        sh_rest=stack(rest_names) if rest_names else np.zeros((n, 0), dtype=np.float32),
        opacity_logits=np.asarray(vertex["opacity"], dtype=np.float32),
        log_scales=stack([f"scale_{i}" for i in range(3)]),
        quats_wxyz=quats,
    )


def save_gaussian_ply(cloud: GaussianCloud, path: str | Path) -> None:
    n = len(cloud)
    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(3)]
    fields += [f"f_rest_{i}" for i in range(cloud.sh_rest.shape[1])]
    fields += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.empty(n, dtype=[(f, "f4") for f in fields])
    payload = np.concatenate(
        [
            cloud.means,
            np.zeros((n, 3), dtype=np.float32),
            cloud.sh_dc,
            cloud.sh_rest,
            cloud.opacity_logits[:, None],
            cloud.log_scales,
            cloud.quats_wxyz,
        ],
        axis=1,
    ).astype(np.float32)
    for i, f in enumerate(fields):
        data[f] = payload[:, i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def apply_sim3(
    cloud: GaussianCloud,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
) -> GaussianCloud:
    """Return a new cloud under x' = scale * rotation @ x + translation."""
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float32).reshape(3)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale}")
    det = float(np.linalg.det(rotation))
    if abs(det - 1.0) > 1e-3:
        raise ValueError(f"rotation determinant {det:.6f} is not +1 (not a proper rotation)")
    means = (cloud.means @ rotation.T) * scale + translation
    q_rot = rotation_matrix_to_quat_wxyz(rotation)
    quats = quat_multiply_wxyz(q_rot, cloud.quats_wxyz)
    quats /= np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    return replace(
        cloud,
        means=means.astype(np.float32),
        log_scales=(cloud.log_scales + np.float32(np.log(scale))).astype(np.float32),
        quats_wxyz=quats.astype(np.float32),
    )


def sim3_matrix(rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    """4x4 homogeneous matrix of x' = s R x + t (for logging / composing)."""
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = float(scale) * np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    mat[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return mat
