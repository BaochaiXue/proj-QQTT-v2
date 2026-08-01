"""Camera utilities in a single, tested convention.

Everything in gaussian_align_demo uses the OpenCV pinhole model with column
vectors: camera x right, y down, z forward; ``w2c`` (4x4) maps world points to
camera points; ``c2w = inv(w2c)``. Pixel (u, v) has u along width (x) and v
along height (y). These helpers are deliberately independent from demo_v6_2's
PyTorch3D-flavored pose code — poses produced here are only consumed by
``renderer.render_gaussians_torch`` and the projections below, and
``tests/test_cameras.py`` locks the project/unproject round trip.
"""

from __future__ import annotations

import numpy as np


def look_at_w2c(
    eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray = (0.0, 0.0, 1.0)
) -> np.ndarray:
    """OpenCV w2c for a camera at ``eye`` looking at ``target``.

    ``up_hint`` is the world direction that should point *up* in the image
    (mapped to camera -y). It must not be parallel to the view direction.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up_hint = np.asarray(up_hint, dtype=np.float64).reshape(3)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        raise ValueError("eye and target coincide")
    forward = forward / norm
    right = np.cross(forward, up_hint)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        raise ValueError("up_hint is parallel to the view direction")
    right = right / right_norm
    down = np.cross(forward, right)  # camera +y (down) completes the RH frame
    rotation_c2w = np.stack([right, down, forward], axis=1)  # columns = camera axes
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rotation_c2w.T
    w2c[:3, 3] = -rotation_c2w.T @ eye
    return w2c


def intrinsics_for_fov(
    *, width: int, height: int, fov_x_deg: float | None = None, fov_y_deg: float | None = None
) -> np.ndarray:
    """Square-pixel K from one field-of-view angle (the other follows aspect)."""
    if (fov_x_deg is None) == (fov_y_deg is None):
        raise ValueError("provide exactly one of fov_x_deg / fov_y_deg")
    if fov_x_deg is not None:
        focal = (width / 2.0) / np.tan(np.deg2rad(fov_x_deg) / 2.0)
    else:
        focal = (height / 2.0) / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def sample_orbit_w2c(
    *,
    center: np.ndarray,
    radius: float,
    n_azimuth: int,
    elevations_deg: tuple[float, ...],
    roll_angles_deg: tuple[float, ...] = (0.0,),
    world_up: np.ndarray = (0.0, 0.0, 1.0),
) -> list[np.ndarray]:
    """Candidate orbit poses: azimuth x elevation x in-plane roll.

    Rolls emulate demo_v6_2's multiple up-vectors: a generated object's canonical
    "up" need not match the world up, so candidates cover in-plane rotation too.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    world_up = np.asarray(world_up, dtype=np.float64)
    world_up = world_up / np.linalg.norm(world_up)
    # Build an orthonormal basis (a, b, world_up) for placing the orbit.
    helper = np.array([1.0, 0.0, 0.0]) if abs(world_up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(world_up, helper)
    a /= np.linalg.norm(a)
    b = np.cross(world_up, a)
    poses: list[np.ndarray] = []
    for elevation_deg in elevations_deg:
        el = np.deg2rad(elevation_deg)
        for k in range(n_azimuth):
            az = 2.0 * np.pi * k / n_azimuth
            direction = (
                np.cos(el) * (np.cos(az) * a + np.sin(az) * b) + np.sin(el) * world_up
            )
            eye = center + radius * direction
            base = look_at_w2c(eye, center, up_hint=world_up)
            for roll_deg in roll_angles_deg:
                roll = np.deg2rad(roll_deg)
                cr, sr = np.cos(roll), np.sin(roll)
                # Rotate about the camera z (view) axis: applied in camera frame.
                roll_mat = np.eye(4)
                roll_mat[:2, :2] = np.array([[cr, -sr], [sr, cr]])
                poses.append(roll_mat @ base)
    return poses


def project_points(
    points_world: np.ndarray, K: np.ndarray, w2c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World points -> (pixels (N,2) u,v; depths (N,)). Depth <= 0 means behind."""
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    cam = pts @ np.asarray(w2c)[:3, :3].T + np.asarray(w2c)[:3, 3]
    depth = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.asarray(K)[0, 0] * cam[:, 0] / depth + np.asarray(K)[0, 2]
        v = np.asarray(K)[1, 1] * cam[:, 1] / depth + np.asarray(K)[1, 2]
    return np.stack([u, v], axis=1), depth


def unproject_pixels(
    pixels_uv: np.ndarray, depths: np.ndarray, K: np.ndarray, w2c: np.ndarray
) -> np.ndarray:
    """Pixels + depths -> world points (inverse of :func:`project_points`)."""
    uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(depths, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    x = (uv[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (uv[:, 1] - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], axis=1)
    c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    return cam @ c2w[:3, :3].T + c2w[:3, 3]
