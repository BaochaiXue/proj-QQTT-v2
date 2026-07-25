#!/usr/bin/env python3
"""Offline shape-prior mesh evolution: SAM3D -> each align step -> final mesh.

Renders a turntable comparison of every mesh state the shape-prior chain
produces for one completed case (shape_prior_case/shape_prior_frame0):

  1  object.glb            raw SAM3D generation, canonical coordinates;
  2  rigid PnP + scale     object.glb placed into the world frame;
  3  ARAP keypoint deform  pulled onto the SuperGlue correspondences;
  4  ARAP ray + table      ray-registration ARAP with the above-table clamp
                           (replayed exactly via demo_v6_2/shape_prior/align.py);
  5  final_mesh.glb        the saved mesh Demo v6.2 actually consumes
                           (sample stage points + ASAP augmentation).

World-frame stages overlay the frame-0 object observation point cloud (real
RGB) so each deformation step can be judged against what the camera saw. The
replayed stage-4 mesh is checked against the saved final_mesh.glb and the
maximum vertex deviation is reported in the stats JSON.

Strictly offline: reads saved artifacts only, replays the deterministic
CPU-side alignment steps (imported from align.py, never copied), and never
touches the realtime pipeline. SuperGlue/SAM3D are not rerun.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

from demo_v6_2.shape_prior import align as align_mod  # noqa: E402
from demo_v6_2.visualization.visualize_shape_prior_matches import (  # noqa: E402
    CaseData,
    MatchStages,
    load_case,
    replay_match_stages,
)

CAM_IDX = 0
PANEL_SIZE = 480
TITLE_BAR_PX = 30
FOV_DEG = 40.0
ELEVATION_DEG = 32.0
MAX_POINTS_PER_PANEL = 9000
MESH_COLOR_BGR = (80, 220, 80)
CANONICAL_COLOR_BGR = (200, 200, 60)
GRID_COLOR_BGR = (70, 70, 70)

DEFAULT_OUTPUT_DIR = Path("demo_v6_2/others/shape_prior_mesh_stages_outputs")


@dataclass(frozen=True)
class MeshStage:
    """One renderable mesh state."""

    key: str
    title: str
    vertices: np.ndarray  # Nx3
    world_frame: bool  # False: canonical SAM3D coordinates


@dataclass(frozen=True)
class Observation:
    """Frame-0 object observation point cloud (world frame, real RGB)."""

    points: np.ndarray  # Nx3
    colors_bgr: np.ndarray  # Nx3 uint8


def load_observation(case_dir: Path) -> Observation:
    """Load the masked object observation exactly like align.py main()."""
    data = np.load(case_dir / "pcd" / "0.npz")
    with open(case_dir / "mask" / "processed_masks.pkl", "rb") as f:
        processed_masks = pickle.load(f)
    obs_points = []
    obs_colors = []
    for i in range(int(np.asarray(data["points"]).shape[0])):
        mask = processed_masks[0][i]["object"]
        obs_points.append(data["points"][i][mask])
        obs_colors.append(data["colors"][i][mask])
    points = np.vstack(obs_points)
    colors = np.vstack(obs_colors)
    if colors.dtype != np.uint8:
        colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return Observation(points=points, colors_bgr=colors[:, ::-1].copy())


def replay_mesh_stages(
    case: CaseData, stages: MatchStages, obs: Observation
) -> tuple[list[MeshStage], dict]:
    """Replay align.py's deformation chain and collect every mesh state."""
    canonical_vertices = np.asarray(case.mesh.vertices).copy()

    # Same o3d conversion + trimesh index mapping as align.py main().
    initial_mesh_world = o3d.geometry.TriangleMesh()
    initial_mesh_world.vertices = o3d.utility.Vector3dVector(canonical_vertices)
    initial_mesh_world.triangles = o3d.utility.Vector3iVector(
        np.asarray(case.mesh.faces)
    )
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
    kdtree = KDTree(initial_mesh_world.vertices)
    _, trimesh_indices = kdtree.query(canonical_vertices)
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(stages.mesh2world)
    rigid_vertices = np.asarray(initial_mesh_world.vertices).copy()

    mesh_matching_points_world = (
        stages.mesh2world
        @ np.hstack(
            (
                stages.mesh_matching_points,
                np.ones((stages.mesh_matching_points.shape[0], 1)),
            )
        ).T
    ).T[:, :3]

    deform_kp_mesh_world, mesh_points_indices = align_mod.deform_ARAP(
        initial_mesh_world, mesh_matching_points_world, stages.matching_points_world
    )
    keypoint_vertices = np.asarray(deform_kp_mesh_world.vertices).copy()

    c2ws = [case.c2w]
    w2cs = [case.w2c]
    final_mesh_world = align_mod.deform_ARAP_ray_registration(
        deform_kp_mesh_world,
        obs.points,
        case.mesh,  # mutated in place by the ray step, canonical saved above
        trimesh_indices,
        c2ws,
        w2cs,
        mesh_points_indices,
        stages.matching_points_world,
    )
    replayed_vertices = np.asarray(final_mesh_world.vertices).copy()
    saved_vertices = np.asarray(case.final_mesh.vertices)

    # align.py exports final vertices mapped back through trimesh_indices.
    replay_max_delta_m = float(
        np.abs(replayed_vertices[trimesh_indices] - saved_vertices).max()
    )
    mesh_stages = [
        MeshStage(
            key="sam3d_raw",
            title="1 SAM3D object.glb (canonical)",
            vertices=canonical_vertices,
            world_frame=False,
        ),
        MeshStage(
            key="rigid_pnp_scale",
            title="2 rigid: PnP + scale -> world",
            vertices=rigid_vertices,
            world_frame=True,
        ),
        MeshStage(
            key="arap_keypoint",
            title="3 ARAP keypoint deform",
            vertices=keypoint_vertices,
            world_frame=True,
        ),
        MeshStage(
            key="arap_ray_table",
            title="4 ARAP ray + table clamp (replayed)",
            vertices=replayed_vertices,
            world_frame=True,
        ),
        MeshStage(
            key="final_mesh_glb",
            title="5 final_mesh.glb (used by Demo v6.2)",
            vertices=saved_vertices,
            world_frame=True,
        ),
    ]
    stats = {
        "vertex_counts": {
            stage.key: int(len(stage.vertices)) for stage in mesh_stages
        },
        "observation_points": int(len(obs.points)),
        "optimal_scale": stages.optimal_scale,
        "pnp_reprojection_error_px": stages.reproj_error_px,
        "final_correspondences": int(len(stages.matching_points_world)),
        "replay_vs_saved_final_max_vertex_delta_m": replay_max_delta_m,
    }
    return mesh_stages, stats


def _subsample(points: np.ndarray, limit: int) -> np.ndarray:
    """Deterministically thin a point set for rendering."""
    if len(points) <= limit:
        return points
    stride = int(np.ceil(len(points) / limit))
    return points[::stride]


def _orbit_camera(
    center: np.ndarray, radius: float, azimuth_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (eye, world-to-camera 4x4) orbiting the center."""
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(ELEVATION_DEG)
    eye = center + radius * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    forward = center - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    w2c = np.eye(4)
    w2c[0, :3] = right
    w2c[1, :3] = down
    w2c[2, :3] = forward
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return eye, w2c


def _project_points(points: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Project world points into the PANEL_SIZE viewport; NaN when behind."""
    cam = (w2c @ np.hstack((points, np.ones((len(points), 1)))).T).T[:, :3]
    focal = 0.5 * PANEL_SIZE / np.tan(np.deg2rad(FOV_DEG) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = np.stack(
            [
                focal * cam[:, 0] / cam[:, 2] + PANEL_SIZE / 2,
                focal * cam[:, 1] / cam[:, 2] + PANEL_SIZE / 2,
                cam[:, 2],
            ],
            axis=1,
        )
    pixels[cam[:, 2] <= 1e-6] = np.nan
    return pixels


def _draw_points(
    canvas: np.ndarray, projected: np.ndarray, colors_bgr: np.ndarray
) -> None:
    """Paint projected points far-to-near so near points win."""
    valid = ~np.isnan(projected[:, 2])
    projected = projected[valid]
    colors_bgr = colors_bgr[valid]
    order = np.argsort(-projected[:, 2])
    xs = np.round(projected[order, 0]).astype(int)
    ys = np.round(projected[order, 1]).astype(int)
    inside = (xs >= 0) & (xs < PANEL_SIZE) & (ys >= 1) & (ys < PANEL_SIZE - 1)
    xs, ys = xs[inside], ys[inside]
    ordered_colors = colors_bgr[order][inside]
    for dy in (-1, 0, 1):  # 1x3 splat: readable points without per-point circles
        canvas[ys + dy, xs] = ordered_colors


def _draw_table_grid(canvas: np.ndarray, w2c: np.ndarray, center: np.ndarray, extent: float) -> None:
    """Draw a z=0 reference grid so the table clamp is visible."""
    ticks = np.linspace(-extent, extent, 7)
    for tick in ticks:
        for axis in (0, 1):
            ends = np.zeros((2, 3))
            ends[:, axis] = (-extent, extent)
            ends[:, 1 - axis] = tick
            ends[:, 0] += center[0]
            ends[:, 1] += center[1]
            projected = _project_points(ends, w2c)
            if np.isnan(projected[:, 2]).any():
                continue
            p0 = tuple(np.round(projected[0, :2]).astype(int))
            p1 = tuple(np.round(projected[1, :2]).astype(int))
            cv2.line(canvas, p0, p1, GRID_COLOR_BGR, 1, cv2.LINE_AA)


def render_stage_panel(
    stage: MeshStage,
    obs: Observation,
    azimuth_deg: float,
    *,
    world_center: np.ndarray,
    world_radius: float,
) -> np.ndarray:
    """Render one stage viewport for one turntable azimuth."""
    canvas = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    mesh_points = _subsample(stage.vertices, MAX_POINTS_PER_PANEL)
    if stage.world_frame:
        center, radius = world_center, world_radius
        _, w2c = _orbit_camera(center, radius, azimuth_deg)
        _draw_table_grid(canvas, w2c, center, world_radius * 0.45)
        obs_points = _subsample(obs.points, MAX_POINTS_PER_PANEL)
        obs_colors = _subsample(obs.colors_bgr, MAX_POINTS_PER_PANEL)
        _draw_points(canvas, _project_points(obs_points, w2c), obs_colors)
        mesh_color = MESH_COLOR_BGR
    else:
        extent = stage.vertices.max(axis=0) - stage.vertices.min(axis=0)
        center = stage.vertices.mean(axis=0)
        radius = 1.8 * float(np.linalg.norm(extent))
        _, w2c = _orbit_camera(center, radius, azimuth_deg)
        mesh_color = CANONICAL_COLOR_BGR
    colors = np.full((len(mesh_points), 3), mesh_color, dtype=np.uint8)
    _draw_points(canvas, _project_points(mesh_points, w2c), colors)
    cv2.putText(
        canvas, stage.title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return canvas


def _stats_panel(stats: dict) -> np.ndarray:
    """Render the stats text panel filling the sixth grid cell."""
    canvas = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    lines = [
        "shape-prior mesh stages",
        "",
        f"scale: {stats['optimal_scale']:.4f}",
        f"pnp reproj err: {stats['pnp_reprojection_error_px']:.2f} px",
        f"final correspondences: {stats['final_correspondences']}",
        f"obs points: {stats['observation_points']}",
        "",
        "replay vs saved final mesh:",
        f"max vertex delta {stats['replay_vs_saved_final_max_vertex_delta_m'] * 1000:.3f} mm",
        "",
        "green = mesh | rgb = observation",
        "grid = table plane z=0",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            canvas, line, (14, 34 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (210, 210, 210), 1, cv2.LINE_AA,
        )
    return canvas


def compose_grid(panels: list[np.ndarray]) -> np.ndarray:
    """Arrange the five stage panels + stats panel as a 3x2 grid."""
    row0 = np.hstack(panels[:3])
    row1 = np.hstack(panels[3:6])
    return np.vstack([row0, row1])


def write_outputs(
    mesh_stages: list[MeshStage],
    obs: Observation,
    stats: dict,
    output_dir: Path,
    *,
    video_fps: float,
    orbit_steps: int,
) -> None:
    """Write per-stage PNGs, the grid turntable video, and the stats JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    world_points = [stage.vertices for stage in mesh_stages if stage.world_frame]
    world_points.append(obs.points)
    stacked = np.vstack(world_points)
    world_center = stacked.mean(axis=0)
    world_radius = 1.6 * float(
        np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0))
    )

    stats_panel = _stats_panel(stats)
    still_azimuth = 210.0
    for stage in mesh_stages:
        panel = render_stage_panel(
            stage, obs, still_azimuth,
            world_center=world_center, world_radius=world_radius,
        )
        cv2.imwrite(str(output_dir / f"{stage.key}.png"), panel)

    video_path = output_dir / "mesh_stages_turntable.mp4"
    grid_shape = compose_grid(
        [np.zeros((PANEL_SIZE, PANEL_SIZE, 3), np.uint8)] * 6
    ).shape
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (grid_shape[1], grid_shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {video_path}")
    for step in range(orbit_steps):
        azimuth = 360.0 * step / orbit_steps
        panels = [
            render_stage_panel(
                stage, obs, azimuth,
                world_center=world_center, world_radius=world_radius,
            )
            for stage in mesh_stages
        ]
        panels.append(stats_panel)
        writer.write(compose_grid(panels))
    writer.release()

    grid_still = compose_grid(
        [
            render_stage_panel(
                stage, obs, still_azimuth,
                world_center=world_center, world_radius=world_radius,
            )
            for stage in mesh_stages
        ]
        + [stats_panel]
    )
    cv2.imwrite(str(output_dir / "mesh_stages_overview.png"), grid_still)
    (output_dir / "mesh_stages_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[mesh-stages] wrote {video_path}")
    print(f"[mesh-stages] stats: {json.dumps(stats)}")


def build_parser() -> ArgumentParser:
    """Build the command-line argument parser."""
    from demo_v6_2.orchestration.main_config import (  # noqa: PLC0415
        CONFIG_SHAPE_PRIOR_CONTROLLER_NAME,
        DEFAULT_DATA_PROCESS_BASE_PATH,
        SHAPE_PRIOR_CASE_DIR_NAME,
    )
    from demo_v6_2.shape_prior.warmup import CASE_NAME  # noqa: PLC0415

    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_DATA_PROCESS_BASE_PATH / SHAPE_PRIOR_CASE_DIR_NAME / CASE_NAME,
        help="Completed shape-prior case directory (…/shape_prior_frame0).",
    )
    parser.add_argument(
        "--controller-name",
        default=CONFIG_SHAPE_PRIOR_CONTROLLER_NAME,
        help="Controller label in mask_info; the other entry is the object.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument(
        "--orbit-steps",
        type=int,
        default=96,
        help="Turntable frames per full revolution (one revolution per video).",
    )
    return parser


def main(argv=None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    case_dir = Path(args.case_dir)
    case = load_case(case_dir, str(args.controller_name))
    stages = replay_match_stages(case)
    obs = load_observation(case_dir)
    mesh_stages, stats = replay_mesh_stages(case, stages, obs)
    write_outputs(
        mesh_stages,
        obs,
        stats,
        Path(args.output_dir),
        video_fps=float(args.video_fps),
        orbit_steps=int(args.orbit_steps),
    )


if __name__ == "__main__":
    main()
