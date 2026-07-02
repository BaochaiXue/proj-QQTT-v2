"""FFS IR-left depth to color-frame alignment (numba fast path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit  # type: ignore

from demo_v5_1.utils.projection import build_projection_grid_from_matrix


@njit
def _align_ir_to_color_numba(
    depth_ir_m: np.ndarray,
    a_x_flat: np.ndarray,
    a_y_flat: np.ndarray,
    a_z_flat: np.ndarray,
    tx: np.float32,
    ty: np.float32,
    tz: np.float32,
    color_fx: np.float32,
    color_fy: np.float32,
    color_cx: np.float32,
    color_cy: np.float32,
    out_flat: np.ndarray,
    out_width: int,
    out_height: int,
    invalid_value: np.float32,
) -> None:
    # Splat every valid IR depth pixel into the color frame, keeping the nearest
    # depth per color pixel (z-buffer): initialize to +inf, take min, then map
    # untouched pixels to invalid_value.
    for i in range(out_flat.shape[0]):
        out_flat[i] = np.inf

    ir_height, ir_width = depth_ir_m.shape
    for y in range(ir_height):
        row_offset = y * ir_width
        for x in range(ir_width):
            z = depth_ir_m[y, x]
            if not np.isfinite(z) or z <= 0.0:
                continue

            # a_{x,y,z} are precomputed per-pixel coefficients (see
            # FfsIrToColorAligner.__init__) so the IR->color transform reduces
            # to coeff * z + t per axis.
            coeff_idx = row_offset + x
            z_color = a_z_flat[coeff_idx] * z + tz
            if not np.isfinite(z_color) or z_color <= 0.0:
                continue

            x_color = a_x_flat[coeff_idx] * z + tx
            y_color = a_y_flat[coeff_idx] * z + ty
            u = int(np.rint((x_color / z_color) * color_fx + color_cx))
            v = int(np.rint((y_color / z_color) * color_fy + color_cy))
            if 0 <= u < out_width and 0 <= v < out_height:
                out_idx = v * out_width + u
                if z_color < out_flat[out_idx]:
                    out_flat[out_idx] = z_color

    for i in range(out_flat.shape[0]):
        if not np.isfinite(out_flat[i]):
            out_flat[i] = invalid_value


def warm_up_numba_ffs_align() -> None:
    """Trigger the numba JIT compile with a 1x1 input so the first real frame is not slow."""
    depth = np.array([[1.0]], dtype=np.float32)
    coeff = np.ones(1, dtype=np.float32)
    output = np.empty(1, dtype=np.float32)
    _align_ir_to_color_numba(
        depth,
        coeff,
        coeff,
        coeff,
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(1.0),
        np.float32(1.0),
        np.float32(0.0),
        np.float32(0.0),
        output,
        1,
        1,
        np.float32(0.0),
    )


class FfsIrToColorAligner:
    """Reusable IR-left metric depth to color-frame depth aligner.

    The returned aligned depth array is reused on the next align call.
    Callers that need to retain it must copy it before calling align again.
    """

    def __init__(
        self,
        *,
        k_ir_left: np.ndarray,
        t_ir_left_to_color: np.ndarray,
        k_color: np.ndarray,
        ir_shape: tuple[int, int],
        color_shape: tuple[int, int],
    ) -> None:
        ir_height, ir_width = (int(ir_shape[0]), int(ir_shape[1]))
        color_height, color_width = (int(color_shape[0]), int(color_shape[1]))
        if ir_height <= 0 or ir_width <= 0 or color_height <= 0 or color_width <= 0:
            raise ValueError("ir_shape and color_shape must be positive")

        k_color_arr = np.asarray(k_color, dtype=np.float32).reshape(3, 3)
        transform = np.asarray(t_ir_left_to_color, dtype=np.float32).reshape(4, 4)
        # Fold intrinsics and rotation into per-pixel coefficients: an IR pixel at
        # depth z maps to color-camera coords (a_x*z + tx, a_y*z + ty, a_z*z + tz),
        # leaving only one multiply-add per axis in the numba hot loop.
        ray_x, ray_y = build_projection_grid_from_matrix(width=ir_width, height=ir_height, K=k_ir_left)
        r = transform[:3, :3]

        self.ir_shape = (ir_height, ir_width)
        self.color_shape = (color_height, color_width)
        self.color_width = color_width
        self.color_fx = np.float32(k_color_arr[0, 0])
        self.color_fy = np.float32(k_color_arr[1, 1])
        self.color_cx = np.float32(k_color_arr[0, 2])
        self.color_cy = np.float32(k_color_arr[1, 2])
        self.tx = np.float32(transform[0, 3])
        self.ty = np.float32(transform[1, 3])
        self.tz = np.float32(transform[2, 3])
        self.a_x = np.ascontiguousarray(r[0, 0] * ray_x + r[0, 1] * ray_y + r[0, 2], dtype=np.float32)
        self.a_y = np.ascontiguousarray(r[1, 0] * ray_x + r[1, 1] * ray_y + r[1, 2], dtype=np.float32)
        self.a_z = np.ascontiguousarray(r[2, 0] * ray_x + r[2, 1] * ray_y + r[2, 2], dtype=np.float32)
        self.a_x_flat = self.a_x.ravel()
        self.a_y_flat = self.a_y.ravel()
        self.a_z_flat = self.a_z.ravel()
        self.nearest = np.empty(color_height * color_width, dtype=np.float32)
        self.output = self.nearest.reshape(color_height, color_width)

    def align(self, depth_ir_m: np.ndarray, *, invalid_value: float = 0.0) -> np.ndarray:
        depth = np.asarray(depth_ir_m, dtype=np.float32)
        if depth.shape != self.ir_shape:
            raise ValueError("depth_ir_m shape does not match aligner ir_shape")

        out_height, out_width = self.color_shape
        _align_ir_to_color_numba(
            depth,
            self.a_x_flat,
            self.a_y_flat,
            self.a_z_flat,
            self.tx,
            self.ty,
            self.tz,
            self.color_fx,
            self.color_fy,
            self.color_cx,
            self.color_cy,
            self.nearest,
            out_width,
            out_height,
            np.float32(invalid_value),
        )
        return self.output


def validate_ffs_paths(*, ffs_repo: Path, model_dir: Path) -> None:
    if not ffs_repo.exists():
        raise ValueError(f"--ffs-repo does not exist: {ffs_repo}")
    if not ffs_repo.is_dir():
        raise ValueError(f"--ffs-repo is not a directory: {ffs_repo}")
    if not model_dir.exists():
        raise ValueError(f"--ffs-trt-model-dir does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"--ffs-trt-model-dir is not a directory: {model_dir}")
    missing = [
        str(path)
        for path in (
            model_dir / "feature_runner.engine",
            model_dir / "post_runner.engine",
            model_dir / "onnx.yaml",
        )
        if not path.is_file()
    ]
    if missing:
        raise ValueError("missing required two-stage TensorRT FFS artifact files: " + ", ".join(missing))
