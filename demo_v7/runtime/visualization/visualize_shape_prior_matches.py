"""Offline shape-prior match comparison: SuperGlue -> align filter -> final.

Reads one completed shape-prior case (shape_prior_case/shape_prior_frame0)
and renders a staged comparison of what happened to the 2D matches:

  stage 1  raw SuperGlue matches between the best SAM3D render view and the
           masked frame-0 crop (exactly what best_match.pkl recorded);
  stage 2  the depth-valid subset that survives project_2d_to_3d and feeds
           PnP, with the PnP reprojection drawn on the full frame;
  stage 3  the final correspondences ARAP actually uses: every PnP point
           snapped to the nearest object-mask pixel (select_point) whose 3D
           observation becomes the deformation target;
  stage 4  the mesh consequence: PnP+scale mesh projection vs the final
           ARAP-deformed final_mesh.glb projection.

Outputs one PNG per stage, a 2x2 overview PNG, match_compare.mp4 cycling the
stages, and match_compare_stats.json.

This tool is strictly offline: it only reads saved artifacts and replays the
cheap filtering steps from demo_v6_2/shape_prior/align.py (imported, not
copied) -- SuperGlue/SAM3D never rerun and the realtime pipeline is untouched.
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
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

from demo_v7.runtime.orchestration.main_config import (  # noqa: E402
    CONFIG_SHAPE_PRIOR_CONTROLLER_NAME,
    DEFAULT_DATA_PROCESS_BASE_PATH,
    SHAPE_PRIOR_CASE_DIR_NAME,
)
from demo_v7.runtime.shape_prior import align as align_mod  # noqa: E402
from demo_v7.runtime.shape_prior.warmup import CASE_NAME  # noqa: E402
from demo_v7.runtime.utils.align_util import (  # noqa: E402
    as_mesh,
    project_2d_to_3d,
    select_point,
)

CAM_IDX = 0
PANEL_GAP_PX = 12
TITLE_BAR_PX = 56
CANVAS_WIDTH = 1440
MESH_POINT_STRIDE = 3


@dataclass(frozen=True)
class CaseData:
    """Saved artifacts of one completed shape-prior case."""

    raw_img: np.ndarray  # HxWx3 RGB frame 0
    object_mask: np.ndarray  # HxW bool
    intrinsic: np.ndarray  # 3x3
    c2w: np.ndarray  # 4x4
    w2c: np.ndarray  # 4x4
    first_points: np.ndarray  # HxWx3 world points of camera 0
    best_color: np.ndarray  # best render view (HxWx3)
    best_depth: np.ndarray  # best render depth (HxW)
    best_pose: np.ndarray  # 4x4 render camera pose
    match_result: dict  # keypoints0/keypoints1/matches/match_confidence
    camera_intrinsics: np.ndarray  # render intrinsics 3x3
    bbox: tuple  # crop bbox (x0, y0, x1, y1)
    mesh: trimesh.Trimesh  # canonical object.glb
    final_mesh: trimesh.Trimesh  # ARAP result final_mesh.glb


@dataclass(frozen=True)
class MatchStages:
    """Replayed intermediate match sets, align.py order."""

    crop_gray: np.ndarray
    render_gray: np.ndarray
    render_kpts: np.ndarray  # raw-match keypoints on render view
    crop_kpts: np.ndarray  # raw-match keypoints on crop
    conf: np.ndarray  # raw-match confidences
    depth_valid: np.ndarray  # bool per raw match: survives project_2d_to_3d
    mesh_matching_points: np.ndarray  # 3D mesh points of surviving matches
    raw_matching_points: np.ndarray  # surviving matches in full-image coords
    reprojected_points: np.ndarray  # PnP reprojection of mesh points
    reproj_error_px: float
    snapped_points: np.ndarray  # select_point targets in full-image coords
    matching_points_world: np.ndarray  # 3D world targets of the snapped points
    optimal_scale: float
    mesh2world: np.ndarray  # 4x4 PnP+scale transform


def load_case(case_dir: Path, controller_name: str) -> CaseData:
    """Load every artifact the comparison needs; fail fast when incomplete."""
    matching_dir = case_dir / "shape" / "matching"
    best_match_pkl = matching_dir / "best_match.pkl"
    final_mesh_glb = matching_dir / "final_mesh.glb"
    for required in (best_match_pkl, final_mesh_glb):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} is missing; run a shape-prior warmup to completion "
                "before visualizing its matches"
            )

    with open(case_dir / "mask" / f"mask_info_{CAM_IDX}.json") as f:
        mask_info = json.load(f)
    obj_idx = None
    for key, value in mask_info.items():
        if value != controller_name:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    if obj_idx is None:
        raise ValueError(f"no non-{controller_name!r} object in mask_info")

    raw_img = cv2.cvtColor(
        cv2.imread(str(case_dir / "color" / str(CAM_IDX) / "0.png")),
        cv2.COLOR_BGR2RGB,
    )
    mask_img = cv2.imread(
        str(case_dir / "mask" / str(CAM_IDX) / str(obj_idx) / "0.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    with open(case_dir / "metadata.json") as f:
        intrinsic = np.array(json.load(f)["intrinsics"])[CAM_IDX]
    with open(case_dir / "calibrate.pkl", "rb") as f:
        c2w = pickle.load(f)[CAM_IDX]
    first_points = np.load(case_dir / "pcd" / "0.npz")["points"][CAM_IDX]
    with open(best_match_pkl, "rb") as f:
        best_color, best_depth, best_pose, match_result, camera_intrinsics, bbox = (
            pickle.load(f)
        )
    return CaseData(
        raw_img=raw_img,
        object_mask=mask_img > 0,
        intrinsic=np.asarray(intrinsic, dtype=np.float64),
        c2w=np.asarray(c2w, dtype=np.float64),
        w2c=np.linalg.inv(np.asarray(c2w, dtype=np.float64)),
        first_points=np.asarray(first_points),
        best_color=np.asarray(best_color),
        best_depth=np.asarray(best_depth),
        best_pose=np.asarray(best_pose),
        match_result=dict(match_result),
        camera_intrinsics=np.asarray(camera_intrinsics),
        bbox=tuple(int(v) for v in bbox),
        mesh=as_mesh(trimesh.load_mesh(case_dir / "shape" / "object.glb", force="mesh")),
        final_mesh=as_mesh(trimesh.load_mesh(final_mesh_glb, force="mesh")),
    )


def replay_match_stages(case: CaseData) -> MatchStages:
    """Replay align.py's match filtering on the saved raw SuperGlue result."""
    bbox = case.bbox
    crop_img = case.raw_img.copy()
    crop_img[~case.object_mask] = 0
    crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    render_gray = cv2.cvtColor(case.best_color, cv2.COLOR_BGR2GRAY)

    match_result = case.match_result
    matches = np.asarray(match_result["matches"])
    valid_matches = matches > -1
    render_kpts = np.asarray(match_result["keypoints0"])[valid_matches]
    crop_kpts = np.asarray(match_result["keypoints1"])[matches[valid_matches]]
    conf = np.asarray(match_result["match_confidence"])[valid_matches]

    mesh_matching_points, depth_valid = project_2d_to_3d(
        render_kpts, case.best_depth, case.camera_intrinsics, case.best_pose
    )
    raw_matching_points = crop_kpts[depth_valid] + np.array([bbox[0], bbox[1]])

    mesh2raw_camera = align_mod.registration_pnp(
        mesh_matching_points, raw_matching_points, case.intrinsic
    )
    rvec, _ = cv2.Rodrigues(mesh2raw_camera[:3, :3].astype(np.float64))
    reprojected, _ = cv2.projectPoints(
        np.float32(mesh_matching_points),
        rvec,
        mesh2raw_camera[:3, 3].astype(np.float64),
        np.float64(case.intrinsic),
        np.zeros(4),
    )
    reprojected = reprojected.reshape(-1, 2)
    reproj_error_px = float(
        np.linalg.norm(np.float64(raw_matching_points) - reprojected, axis=1).mean()
    )

    snapped_points, matching_points = select_point(
        case.first_points, raw_matching_points, case.object_mask
    )
    mesh_matching_points_cam = (
        mesh2raw_camera
        @ np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T
    ).T[:, :3]
    matching_points_cam = (
        case.w2c
        @ np.hstack((matching_points, np.ones((matching_points.shape[0], 1)))).T
    ).T[:, :3]
    optimal_scale = float(
        align_mod.registration_scale(mesh_matching_points_cam, matching_points_cam)
    )
    scale_matrix = np.eye(4) * optimal_scale
    scale_matrix[3, 3] = 1
    mesh2world = case.c2w @ scale_matrix @ mesh2raw_camera

    return MatchStages(
        crop_gray=crop_gray,
        render_gray=render_gray,
        render_kpts=render_kpts,
        crop_kpts=crop_kpts,
        conf=conf,
        depth_valid=np.asarray(depth_valid, dtype=bool),
        mesh_matching_points=mesh_matching_points,
        raw_matching_points=raw_matching_points,
        reprojected_points=reprojected,
        reproj_error_px=reproj_error_px,
        snapped_points=np.asarray(snapped_points, dtype=np.float64),
        matching_points_world=np.asarray(matching_points, dtype=np.float64),
        optimal_scale=optimal_scale,
        mesh2world=mesh2world,
    )


def _conf_colors(conf: np.ndarray) -> list[tuple[int, int, int]]:
    """Map SuperGlue confidences to JET BGR colors (blue=low, red=high)."""
    values = np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0)
    mapped = cv2.applyColorMap((values * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return [tuple(int(c) for c in row) for row in mapped.reshape(-1, 3)]


def _side_by_side(
    render_gray: np.ndarray, crop_gray: np.ndarray
) -> tuple[np.ndarray, int]:
    """Stack the render view and the crop horizontally on one BGR canvas."""
    height = max(render_gray.shape[0], crop_gray.shape[0])
    width = render_gray.shape[1] + PANEL_GAP_PX + crop_gray.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: render_gray.shape[0], : render_gray.shape[1]] = cv2.cvtColor(
        render_gray, cv2.COLOR_GRAY2BGR
    )
    x_offset = render_gray.shape[1] + PANEL_GAP_PX
    canvas[: crop_gray.shape[0], x_offset : x_offset + crop_gray.shape[1]] = (
        cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    )
    return canvas, x_offset


def draw_match_pair_panel(
    stages: MatchStages, *, subset: np.ndarray | None, show_dropped: bool
) -> np.ndarray:
    """Render-vs-crop match panel; subset selects which matches are kept."""
    canvas, x_offset = _side_by_side(stages.render_gray, stages.crop_gray)
    keep = (
        np.ones(len(stages.render_kpts), dtype=bool) if subset is None else subset
    )
    colors = _conf_colors(stages.conf)
    for idx in range(len(stages.render_kpts)):
        p0 = tuple(np.round(stages.render_kpts[idx]).astype(int))
        p1 = tuple(
            np.round(stages.crop_kpts[idx] + [x_offset, 0]).astype(int)
        )
        if keep[idx]:
            color = colors[idx]
        elif show_dropped:
            color = (64, 64, 255)  # dropped: red, no confidence coloring
        else:
            continue
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)
    return canvas


def draw_pnp_panel(case: CaseData, stages: MatchStages) -> np.ndarray:
    """Full frame with PnP-input points vs their PnP reprojection."""
    canvas = cv2.cvtColor(case.raw_img, cv2.COLOR_RGB2BGR)
    for raw_pt, reproj_pt in zip(
        stages.raw_matching_points, stages.reprojected_points
    ):
        p_raw = tuple(np.round(raw_pt).astype(int))
        p_re = tuple(np.round(reproj_pt).astype(int))
        cv2.line(canvas, p_raw, p_re, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, p_raw, 3, (0, 255, 255), -1, cv2.LINE_AA)  # matched
        cv2.circle(canvas, p_re, 3, (255, 128, 0), -1, cv2.LINE_AA)  # reprojected
    return canvas


def draw_final_panel(case: CaseData, stages: MatchStages) -> np.ndarray:
    """Final ARAP correspondences: PnP points snapped onto the object mask."""
    canvas = cv2.cvtColor(case.raw_img, cv2.COLOR_RGB2BGR)
    canvas[~case.object_mask] //= 3  # dim background, keep object bright
    for raw_pt, snapped_pt in zip(stages.raw_matching_points, stages.snapped_points):
        p_raw = tuple(np.round(raw_pt).astype(int))
        p_snap = tuple(np.round(snapped_pt).astype(int))
        cv2.arrowedLine(
            canvas, p_raw, p_snap, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35
        )
        cv2.circle(canvas, p_raw, 3, (0, 255, 255), -1, cv2.LINE_AA)  # PnP point
        cv2.circle(canvas, p_snap, 3, (0, 220, 0), -1, cv2.LINE_AA)  # ARAP target
    return canvas


def _project_world_points(
    points_world: np.ndarray, case: CaseData
) -> np.ndarray:
    """Project world points to pixel coords, keeping in-frame points only."""
    cam = (
        case.w2c
        @ np.hstack((points_world, np.ones((points_world.shape[0], 1)))).T
    ).T[:, :3]
    in_front = cam[:, 2] > 1e-6
    cam = cam[in_front]
    pixels = (case.intrinsic @ (cam / cam[:, 2:3]).T).T[:, :2]
    height, width = case.raw_img.shape[:2]
    in_frame = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels[in_frame]


def draw_mesh_panel(case: CaseData, stages: MatchStages) -> np.ndarray:
    """Rigid PnP+scale mesh projection vs the final ARAP mesh projection."""
    canvas = cv2.cvtColor(case.raw_img, cv2.COLOR_RGB2BGR)
    rigid_vertices = (
        stages.mesh2world
        @ np.hstack(
            (case.mesh.vertices, np.ones((case.mesh.vertices.shape[0], 1)))
        ).T
    ).T[:, :3]
    for points_world, color in (
        (rigid_vertices[::MESH_POINT_STRIDE], (64, 64, 255)),  # before ARAP: red
        (np.asarray(case.final_mesh.vertices)[::MESH_POINT_STRIDE], (0, 220, 0)),
    ):
        for pixel in _project_world_points(points_world, case):
            cv2.circle(
                canvas, tuple(np.round(pixel).astype(int)), 1, color, -1, cv2.LINE_AA
            )
    return canvas


def compose_frame(panel: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """Scale a panel onto the fixed-width canvas and add the title bar."""
    scale = CANVAS_WIDTH / panel.shape[1]
    scaled = cv2.resize(
        panel, (CANVAS_WIDTH, int(round(panel.shape[0] * scale)))
    )
    frame = np.zeros(
        (TITLE_BAR_PX + scaled.shape[0], CANVAS_WIDTH, 3), dtype=np.uint8
    )
    frame[TITLE_BAR_PX:] = scaled
    cv2.putText(
        frame, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, subtitle, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180),
        1, cv2.LINE_AA,
    )
    return frame


def build_stage_frames(case: CaseData, stages: MatchStages) -> list[np.ndarray]:
    """Build the four titled comparison frames in pipeline order."""
    total = len(stages.render_kpts)
    kept = int(stages.depth_valid.sum())
    return [
        compose_frame(
            draw_match_pair_panel(stages, subset=None, show_dropped=False),
            "stage 1/4  raw SuperGlue matches: SAM3D render view <-> frame-0 crop",
            f"matches {total} "
            "| line color = SuperGlue confidence (blue low -> red high)",
        ),
        compose_frame(
            draw_match_pair_panel(
                stages, subset=stages.depth_valid, show_dropped=True
            ),
            "stage 2/4  align filter: render-depth-valid matches kept for PnP",
            f"kept {kept} / dropped {total - kept} (red = no render depth) "
            f"| PnP reprojection error {stages.reproj_error_px:.2f} px",
        ),
        compose_frame(
            draw_pnp_panel(case, stages),
            "stage 2b/4  PnP check on the full frame",
            "yellow = matched image point, orange = PnP-reprojected mesh point",
        ),
        compose_frame(
            draw_final_panel(case, stages),
            "stage 3/4  final correspondences: PnP points snapped to object mask",
            f"{len(stages.snapped_points)} pairs drive scale "
            f"({stages.optimal_scale:.4f}) + keypoint ARAP "
            "| yellow = PnP point, green = ARAP target",
        ),
        compose_frame(
            draw_mesh_panel(case, stages),
            "stage 4/4  mesh result: rigid PnP+scale vs final ARAP mesh",
            "red = rigid-aligned object.glb, green = final_mesh.glb "
            "(keypoint ARAP + ray registration + table clamp)",
        ),
    ]


def write_outputs(
    frames: list[np.ndarray],
    stages: MatchStages,
    output_dir: Path,
    *,
    video_fps: float,
    seconds_per_stage: float,
) -> None:
    """Write per-stage PNGs, the overview grid, the cycle video, and stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "stage1_raw_superglue.png",
        "stage2_depth_valid.png",
        "stage2b_pnp_reprojection.png",
        "stage3_final_correspondences.png",
        "stage4_mesh_before_after.png",
    ]
    for name, frame in zip(names, frames):
        cv2.imwrite(str(output_dir / name), frame)

    height = max(frame.shape[0] for frame in frames)
    padded = [
        cv2.copyMakeBorder(
            frame, 0, height - frame.shape[0], 0, 0, cv2.BORDER_CONSTANT
        )
        for frame in frames
    ]
    grid = np.vstack([np.hstack(padded[:2]), np.hstack(padded[3:5])])
    cv2.imwrite(str(output_dir / "match_compare_overview.png"), grid)

    video_path = output_dir / "match_compare.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (frames[0].shape[1], height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {video_path}")
    hold_frames = max(1, int(round(video_fps * seconds_per_stage)))
    for frame in padded:
        for _ in range(hold_frames):
            writer.write(frame)
    writer.release()

    total = len(stages.render_kpts)
    stats = {
        "raw_superglue_matches": total,
        "depth_valid_matches": int(stages.depth_valid.sum()),
        "pnp_reprojection_error_px": stages.reproj_error_px,
        "final_correspondences": int(len(stages.snapped_points)),
        "optimal_scale": stages.optimal_scale,
    }
    (output_dir / "match_compare_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[match-compare] wrote {video_path}")
    print(f"[match-compare] stats: {json.dumps(stats)}")


def build_parser() -> ArgumentParser:
    """Build the command-line argument parser."""
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <case-dir>/shape/matching/match_compare.",
    )
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument(
        "--seconds-per-stage",
        type=float,
        default=2.5,
        help="How long the video holds each comparison stage.",
    )
    return parser


def main(argv=None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    case_dir = Path(args.case_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else case_dir / "shape" / "matching" / "match_compare"
    )
    case = load_case(case_dir, str(args.controller_name))
    stages = replay_match_stages(case)
    frames = build_stage_frames(case, stages)
    write_outputs(
        frames,
        stages,
        output_dir,
        video_fps=float(args.video_fps),
        seconds_per_stage=float(args.seconds_per_stage),
    )


if __name__ == "__main__":
    main()
