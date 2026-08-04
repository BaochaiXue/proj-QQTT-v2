"""Visualize demo_v6_2 shape-prior alignment of the deepAlign (wrong-pose) mesh.

    conda run -n demo_2_max python deepAlign/visualize_align.py

Replays the EXACT demo_v6_2/shape_prior/align.py pipeline (same functions,
read-only imports) on a case assembled from the real frame-0 observation
(outputs/shape_prior_case/shape_prior_frame0) with shape/object.glb swapped
for the cache entry generated from the deepAlign frame — the same sloth in a
spread-eagle pose, violating align's core assumption (same part layout /
approximate pose, one global Sim(3) + mild non-rigid deformation).

Outputs under deepAlign/outputs/:
    align_process.mp4   how align works step by step on this mesh:
                        prior vs target, 192-candidate sweep with SuperGlue
                        match counts, winner correspondences, PnP, scale,
                        keypoint-ARAP morph, ray-ARAP morph.
    align_error.mp4     how bad the result is: blink real<->render, silhouette
                        XOR, depth residual, orbit vs observation PCD — always
                        side by side with the good same-pose prior's result.
    metrics.json        IoU / depth error / chamfer for bad vs good.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import KDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo_v6_2.shape_prior import align as A  # noqa: E402
from demo_v6_2.shape_prior.match_pairs import (  # noqa: E402
    extract_superpoint_features,
    get_matching_model,
    image_pair_matching,
)
from demo_v6_2.models.utils import read_image  # noqa: E402
from demo_v6_2.utils.align_util import (  # noqa: E402
    as_mesh,
    project_2d_to_3d,
    render_image,
    render_multi_images,
    select_point,
)

CASE_SRC = REPO_ROOT / "outputs/shape_prior_case/shape_prior_frame0"
BAD_MESH = Path.home() / "qqtt_shape_prior_cache/schema_v1/sloth_deepalign/object.glb"
OUT_DIR = REPO_ROOT / "deepAlign/outputs"
CASE_ROOT = OUT_DIR / "align_case"
CASE_NAME = "shape_prior_frame0"
CONTROLLER_NAME = "hand"
FPS = 24
W, H = 848, 480

GREEN = (60, 200, 60)
RED = (230, 60, 60)  # frames are RGB (imageio), not cv2 BGR
WHITE = (240, 240, 240)


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------


def assemble_case() -> Path:
    case = CASE_ROOT / CASE_NAME
    if case.exists():
        shutil.rmtree(case)
    for rel in ("color", "mask", "pcd", "metadata.json", "calibrate.pkl"):
        src = CASE_SRC / rel
        dst = case / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)
    (case / "shape").mkdir(parents=True)
    shutil.copyfile(BAD_MESH, case / "shape" / "object.glb")
    return case


# ---------------------------------------------------------------------------
# Faithful align replay with intermediates
# ---------------------------------------------------------------------------


def replay_align(case: Path) -> dict:
    """Mirror align.main() step by step, returning every intermediate."""
    raw_img = cv2.cvtColor(cv2.imread(str(case / "color/0/0.png")), cv2.COLOR_BGR2RGB)
    mask_img = cv2.imread(str(case / "mask/0/0/0.png"), cv2.IMREAD_GRAYSCALE)
    metadata = json.loads((case / "metadata.json").read_text())
    intrinsic = np.array(metadata["intrinsics"])[0]
    with open(case / "calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    c2w = c2ws[0]
    w2c = np.linalg.inv(c2w)
    w2cs = [np.linalg.inv(m) for m in c2ws]
    mesh_path = str(case / "shape/object.glb")
    mesh = as_mesh(trimesh.load_mesh(mesh_path, force="mesh"))
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))

    # Crop (identical bbox math to align.main).
    points = np.argwhere(mask_img > 0.8 * 255)
    bx0, by0 = points[:, 1].min(), points[:, 0].min()
    bx1, by1 = points[:, 1].max(), points[:, 0].max()
    center = ((bx0 + bx1) / 2, (by0 + by1) / 2)
    size = int(max(bx1 - bx0, by1 - by0) * 1.2)
    bbox = (
        max(0, int(center[0] - size // 2)),
        max(0, int(center[1] - size // 2)),
        min(raw_img.shape[1], int(center[0] + size // 2)),
        min(raw_img.shape[0], int(center[1] + size // 2)),
    )
    crop_img = raw_img.copy()
    crop_img[~(mask_img > 0)] = 0
    crop_img = cv2.cvtColor(crop_img[bbox[1]:bbox[3], bbox[0]:bbox[2]], cv2.COLOR_RGB2GRAY)

    # Candidate rendering (pose_selection_render_superglue's cold branch).
    bounds = mesh.bounds
    radius = 2 * (np.linalg.norm(bounds[1] - bounds[0]) / 2) / np.tan(fov / 2)
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(
        mesh_path, raw_img.shape[1], raw_img.shape[0], fov,
        radius=radius, num_samples=8, num_ups=4, device="cuda",
    )
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]

    # Formal winner (same call as align) + per-candidate counts for the sweep.
    matching_dir = OUT_DIR / "replay" / "matching"
    matching_dir.mkdir(parents=True, exist_ok=True)
    best_idx, match_result = image_pair_matching(
        grays, crop_img, matching_dir, viz=False, cache=False, save=False, viz_best=False,
    )
    counts = per_candidate_match_counts(grays, crop_img)
    print(f"[replay] best candidate {best_idx}: "
          f"{int(np.sum(match_result['matches'] > -1))} matches "
          f"(max over candidates {int(counts.max())})")

    best_depth = depths[best_idx]
    best_pose = camera_poses[best_idx].cpu().numpy()

    # Keypoints -> 3D mesh points and full-image pixels (align.main lines).
    valid_matches = match_result["matches"] > -1
    render_matching_points = match_result["keypoints0"][valid_matches]
    mesh_matching_points, valid_mask = project_2d_to_3d(
        render_matching_points, best_depth, camera_intrinsics, best_pose
    )
    render_matching_points = render_matching_points[valid_mask]
    raw_matching_points_box = match_result["keypoints1"][
        match_result["matches"][valid_matches]
    ][valid_mask]
    raw_matching_points = raw_matching_points_box + np.array([bbox[0], bbox[1]])

    # registration_pnp's >50px warning print reads align's module-global
    # case_name, which only main() assigns — set it for the replay.
    A.case_name = CASE_NAME
    mesh2raw_camera = A.registration_pnp(mesh_matching_points, raw_matching_points, intrinsic)

    mesh_matching_points_cam = (
        mesh2raw_camera @ np.hstack(
            (mesh_matching_points, np.ones((len(mesh_matching_points), 1)))
        ).T
    ).T[:, :3]

    pcd_data = np.load(case / "pcd/0.npz")
    with open(case / "mask/processed_masks.pkl", "rb") as f:
        processed_masks = pickle.load(f)
    first_points = pcd_data["points"][0]
    first_mask = processed_masks[0][0]["object"]
    obs_points = first_points[first_mask]
    obs_colors = pcd_data["colors"][0][first_mask]

    new_match, matching_points = select_point(first_points, raw_matching_points, first_mask)
    matching_points_cam = (
        w2c @ np.hstack((matching_points, np.ones((len(matching_points), 1)))).T
    ).T[:, :3]

    optimal_scale = A.registration_scale(mesh_matching_points_cam, matching_points_cam)
    scale_matrix = np.eye(4) * optimal_scale
    scale_matrix[3, 3] = 1
    mesh2world = c2w @ scale_matrix @ mesh2raw_camera

    mesh_matching_points_world = (
        mesh2world @ np.hstack(
            (mesh_matching_points, np.ones((len(mesh_matching_points), 1)))
        ).T
    ).T[:, :3]

    initial_mesh_world = o3d.geometry.TriangleMesh()
    initial_mesh_world.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    initial_mesh_world.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
    kdtree = KDTree(np.asarray(initial_mesh_world.vertices))
    _, trimesh_indices = kdtree.query(np.asarray(mesh.vertices))
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(mesh2world)
    verts_initial = np.asarray(initial_mesh_world.vertices).copy()

    deform_kp_mesh_world, mesh_points_indices = A.deform_ARAP(
        initial_mesh_world, mesh_matching_points_world, matching_points
    )
    verts_kp = np.asarray(deform_kp_mesh_world.vertices).copy()

    # NOTE deform_ARAP_ray_registration mutates mesh.vertices (camera frame);
    # align.main() overwrites them right after, and so do we.
    final_mesh_world = A.deform_ARAP_ray_registration(
        deform_kp_mesh_world, obs_points, mesh, trimesh_indices, c2ws, w2cs,
        mesh_points_indices, matching_points,
    )
    verts_final = np.asarray(final_mesh_world.vertices).copy()

    mesh.vertices = verts_final[trimesh_indices]
    replay_dir = OUT_DIR / "replay"
    mesh.export(replay_dir / "final_mesh.glb")

    return {
        "raw_img": raw_img, "mask_img": mask_img, "crop_img": crop_img, "bbox": bbox,
        "intrinsic": intrinsic, "c2w": c2w, "w2c": w2c, "fov": fov,
        "mesh_path": mesh_path, "mesh_faces": np.asarray(mesh.faces),
        "trimesh_indices": trimesh_indices,
        "colors": colors, "grays": grays, "counts": counts, "best_idx": int(best_idx),
        "match_result": match_result,
        "render_matching_points": render_matching_points,
        "raw_matching_points": raw_matching_points,
        "matching_points": matching_points,
        "mesh2raw_camera": mesh2raw_camera, "optimal_scale": float(optimal_scale),
        "mesh2world": mesh2world,
        "verts_initial": verts_initial, "verts_kp": verts_kp, "verts_final": verts_final,
        "obs_points": obs_points, "obs_colors": obs_colors,
        "reproj_error_px": reprojection_error(
            mesh_matching_points, raw_matching_points, mesh2raw_camera, intrinsic
        ),
    }


def per_candidate_match_counts(grays, crop_img) -> np.ndarray:
    """SuperGlue match count per candidate (same model/config as the demo)."""
    device = "cuda"
    matching = get_matching_model(
        nms_radius=4, keypoint_threshold=0.005, max_keypoints=1024,
        superglue="indoor", sinkhorn_iterations=20, match_threshold=0.2, device=device,
    )
    _, ref_tensor, _ = read_image(crop_img, device, [-1], 0, False)
    ref_features = extract_superpoint_features(matching, ref_tensor)
    counts = np.zeros(len(grays), dtype=np.int64)
    for i, gray in enumerate(grays):
        _, cand_tensor, _ = read_image(gray, device, [-1], 0, False)
        cand_features = extract_superpoint_features(matching, cand_tensor)
        data = {"image0": cand_tensor, "image1": ref_tensor}
        data.update({k + "0": v for k, v in cand_features.items()})
        data.update({k + "1": v for k, v in ref_features.items()})
        pred = matching(data)
        counts[i] = int((pred["matches0"][0].cpu().numpy() > -1).sum())
    return counts


def reprojection_error(mesh_pts, raw_pts, mesh2raw_camera, intrinsic) -> float:
    rvec, _ = cv2.Rodrigues(mesh2raw_camera[:3, :3])
    projected, _ = cv2.projectPoints(
        np.float32(mesh_pts), rvec, np.float32(mesh2raw_camera[:3, 3]),
        np.float32(intrinsic), np.zeros(4, dtype=np.float32),
    )
    return float(np.linalg.norm(
        np.float32(raw_pts) - projected.reshape(-1, 2), axis=1).mean())


# ---------------------------------------------------------------------------
# Rendering helpers (align_util conventions)
# ---------------------------------------------------------------------------


def p3d_pose_from_opencv(matrix: np.ndarray) -> np.ndarray:
    """OpenCV column-vector X->camera matrix -> align_util row-vector pose.

    Same recipe as align.main's PnP visualization block.
    """
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.linalg.inv(matrix[:3, :3])
    pose[3, :3] = matrix[:3, 3]
    pose[:, :2] = -pose[:, :2]
    return pose


def render_world_mesh(verts, template, trimesh_indices, w2cs_list, fov,
                      width=W, height=H) -> tuple[np.ndarray, np.ndarray]:
    """Render a world-frame mesh from OpenCV w2c poses; returns (colors, depths).

    ``template`` is the original textured trimesh; the deformed stages share
    its topology, so swapping vertices keeps the texture (pytorch3d's shader
    requires one).
    """
    tm = template.copy()
    tm.vertices = np.asarray(verts)[trimesh_indices]
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as handle:
        tmp_path = handle.name
    tm.export(tmp_path)
    poses = np.stack([p3d_pose_from_opencv(w2c) for w2c in w2cs_list])
    colors, depths = render_image(tmp_path, poses, width, height, fov, "cuda")
    Path(tmp_path).unlink()
    if colors.ndim == 3:
        colors = colors[None]
    if depths.ndim == 2:
        depths = depths[None]
    return colors, depths


def textured_world_render(glb_path, w2cs_list, fov, width=W, height=H):
    poses = np.stack([p3d_pose_from_opencv(w2c) for w2c in w2cs_list])
    colors, depths = render_image(str(glb_path), poses, width, height, fov, "cuda")
    if colors.ndim == 3:
        colors = colors[None]
    if depths.ndim == 2:
        depths = depths[None]
    return colors, depths


# ---------------------------------------------------------------------------
# Video composition
# ---------------------------------------------------------------------------


def label(img, text, y=28, color=WHITE, scale=0.7):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


def to_panel(img, width=W, height=H):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[:2] != (height, width):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        scale = min(width / img.shape[1], height / img.shape[0])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        resized = cv2.resize(img, new_size)
        y0 = (height - new_size[1]) // 2
        x0 = (width - new_size[0]) // 2
        canvas[y0:y0 + new_size[1], x0:x0 + new_size[0]] = resized
        return canvas
    return img.copy()


def two_panel(left, right):
    return np.concatenate([to_panel(left), to_panel(right)], axis=1)


def overlay_on_real(raw_img, color, depth, alpha=0.65):
    out = raw_img.copy()
    visible = depth > 0
    out[visible] = (alpha * color[visible] + (1 - alpha) * raw_img[visible]).astype(np.uint8)
    return out


PROCESS_CANVAS = (H + 60, W * 2)  # every process-video frame, strip or not


def pad_to_canvas(frame, canvas_hw=PROCESS_CANVAS):
    height, width = canvas_hw
    if frame.shape[:2] == (height, width):
        return frame
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    canvas[: frame.shape[0], : frame.shape[1]] = frame[:height, :width]
    return canvas


def hold(writer, frame, seconds, canvas_hw=None):
    if canvas_hw is not None:
        frame = pad_to_canvas(frame, canvas_hw)
    for _ in range(int(seconds * FPS)):
        writer.append_data(frame)


def count_strip(counts, upto, best_idx, width=W * 2, height=60):
    strip = np.full((height, width, 3), 25, dtype=np.uint8)
    peak = max(1, int(counts.max()))
    bar_w = max(1, width // len(counts))
    for i in range(min(upto + 1, len(counts))):
        bar_h = int((counts[i] / peak) * (height - 12))
        x0 = i * bar_w
        color = (70, 170, 245) if i != best_idx else (60, 220, 60)
        cv2.rectangle(strip, (x0, height - 4 - bar_h), (x0 + bar_w - 1, height - 4), color, -1)
    return strip


def make_process_video(rep: dict, deep_frame: np.ndarray, deep_mask: np.ndarray) -> Path:
    path = OUT_DIR / "align_process.mp4"
    writer = imageio.get_writer(path, fps=FPS, macro_block_size=1)
    raw = rep["raw_img"]
    fov = rep["fov"]
    template = as_mesh(trimesh.load_mesh(rep["mesh_path"], force="mesh"))
    tidx = rep["trimesh_indices"]

    # Seg A: the prior's source frame vs our target frame.
    src = deep_frame.copy()
    contours, _ = cv2.findContours((deep_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src, contours, -1, (60, 220, 60), 2)
    frame = two_panel(
        label(src, "deepAlign frame: prior GENERATED from this pose"),
        label(raw.copy(), "target: our fake-camera frame-0 (different pose)"),
    )
    hold(writer, frame, 3.0, PROCESS_CANVAS)

    # Seg B: canonical mesh turntable ring (mid elevation, up 0).
    ring = [((2 * 8 + t) * 4) for t in range(8)]
    for _ in range(2):
        for idx in ring:
            frame = two_panel(
                label(rep["colors"][idx].copy(), "generated shape prior (canonical views)"),
                label(cv2.cvtColor(rep["crop_img"], cv2.COLOR_GRAY2RGB),
                      "masked crop used for SuperGlue"),
            )
            hold(writer, frame, 2.0 / FPS, PROCESS_CANVAS)

    # Seg C: 192-candidate sweep with match counts.
    counts, best_idx = rep["counts"], rep["best_idx"]
    for i in range(len(rep["grays"])):
        left = label(cv2.cvtColor(rep["grays"][i], cv2.COLOR_GRAY2RGB),
                     f"candidate {i + 1}/192   SuperGlue matches: {counts[i]}")
        running_best = int(counts[: i + 1].argmax())
        right = label(cv2.cvtColor(rep["grays"][running_best], cv2.COLOR_GRAY2RGB),
                      f"best so far: #{running_best} ({counts[running_best]} matches)",
                      color=GREEN)
        frame = np.concatenate([two_panel(left, right),
                                count_strip(counts, i, best_idx)], axis=0)
        writer.append_data(pad_to_canvas(frame))

    # Seg D: winner correspondences.
    match = rep["match_result"]
    winner = cv2.cvtColor(rep["grays"][best_idx], cv2.COLOR_GRAY2RGB)
    target = raw.copy()
    n_matches = int((match["matches"] > -1).sum())
    for uv_r, uv_full in zip(rep["render_matching_points"], rep["raw_matching_points"]):
        cv2.circle(winner, (int(uv_r[0]), int(uv_r[1])), 3, (60, 220, 60), -1, cv2.LINE_AA)
        cv2.circle(target, (int(uv_full[0]), int(uv_full[1])), 3, (60, 220, 60), -1, cv2.LINE_AA)
    frame = two_panel(
        label(winner, f"winning candidate #{best_idx}: {n_matches} matches"),
        label(target, "matched keypoints on the real frame"),
    )
    frame = np.concatenate([frame, count_strip(counts, len(counts) - 1, best_idx)], axis=0)
    hold(writer, frame, 3.0, PROCESS_CANVAS)

    def world_overlay(verts, caption, color=WHITE):
        colors, depths = render_world_mesh(verts, template, tidx, [rep["w2c"]], fov)
        return label(overlay_on_real(raw, colors[0], depths[0]), caption, color=color)

    # Seg E: PnP pose (canonical mesh seen from the PnP camera).
    pnp_colors, pnp_depths = textured_world_render(
        rep["mesh_path"], [rep["mesh2raw_camera"]], fov)
    pnp_frame = overlay_on_real(raw, pnp_colors[0], pnp_depths[0])
    frame = two_panel(
        label(pnp_frame, f"PnP pose (reproj err {rep['reproj_error_px']:.1f}px, no scale yet)"),
        label(raw.copy(), "target frame-0"),
    )
    hold(writer, frame, 2.5, PROCESS_CANVAS)

    # Seg F: rigid world placement after scale.
    frame = two_panel(
        world_overlay(rep["verts_initial"],
                      f"rigid mesh2world (scale {rep['optimal_scale']:.3f})"),
        label(raw.copy(), "target frame-0"),
    )
    hold(writer, frame, 2.5, PROCESS_CANVAS)

    # Seg G/H: ARAP morphs.
    for name, v0, v1 in (
        ("keypoint ARAP", rep["verts_initial"], rep["verts_kp"]),
        ("ray-registration ARAP + table clamp", rep["verts_kp"], rep["verts_final"]),
    ):
        for t in np.linspace(0.0, 1.0, 18):
            verts = (1 - t) * v0 + t * v1
            frame = two_panel(
                world_overlay(verts, f"{name}: {int(t * 100)}%"),
                label(raw.copy(), "target frame-0"),
            )
            writer.append_data(pad_to_canvas(frame))
        hold(writer, frame, 0.8, PROCESS_CANVAS)

    # Seg I: final hold.
    frame = two_panel(
        world_overlay(rep["verts_final"], "FINAL aligned mesh", color=RED),
        label(raw.copy(), "target frame-0"),
    )
    hold(writer, frame, 3.0, PROCESS_CANVAS)
    writer.close()
    return path


def trusted_silhouette(verts, faces, K, w2c, width=W, height=H):
    """OpenCV-rasterized silhouette with the TRUE intrinsics (the pytorch3d
    render path assumes a centered principal point; metrics must not)."""
    cam = np.asarray(verts) @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uv = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2],
                   K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=1)
    sil = np.zeros((height, width), np.uint8)
    tris = uv[np.asarray(faces)].astype(np.int32)
    ok = (z[np.asarray(faces)] > 0.05).all(axis=1)
    cv2.fillPoly(sil, list(tris[ok]), 1)
    return sil.astype(bool)


def silhouette_metrics(depth, render_depth_valid, mask_img, case_depth):
    pred = render_depth_valid
    gt = mask_img > 127
    union = np.logical_or(pred, gt).sum()
    iou = float(np.logical_and(pred, gt).sum() / union) if union else 0.0
    sel = pred & gt & (case_depth > 0)
    if sel.any():
        err = np.abs(depth[sel] - case_depth[sel])
        depth_med = float(np.median(err) * 1000)
        depth_p90 = float(np.percentile(err, 90) * 1000)
    else:
        depth_med = depth_p90 = float("nan")
    return iou, depth_med, depth_p90


def make_error_video(rep: dict, good_glb: Path, case_depth: np.ndarray) -> tuple[Path, dict]:
    path = OUT_DIR / "align_error.mp4"
    writer = imageio.get_writer(path, fps=FPS, macro_block_size=1)
    raw, mask_img, fov = rep["raw_img"], rep["mask_img"], rep["fov"]
    template = as_mesh(trimesh.load_mesh(rep["mesh_path"], force="mesh"))
    tidx = rep["trimesh_indices"]

    bad_colors, bad_depths = render_world_mesh(
        rep["verts_final"], template, tidx, [rep["w2c"]], fov)
    good_colors, good_depths = textured_world_render(good_glb, [rep["w2c"]], fov)
    panels = {}
    metrics = {}
    good_mesh = as_mesh(trimesh.load_mesh(good_glb, force="mesh"))
    for tag, colors, depths in (("bad", bad_colors, bad_depths),
                                ("good", good_colors, good_depths)):
        if tag == "bad":
            verts = rep["verts_final"][tidx]
            faces = rep["mesh_faces"]
        else:
            verts = np.asarray(good_mesh.vertices)
            faces = np.asarray(good_mesh.faces)
        sil = trusted_silhouette(verts, faces, rep["intrinsic"], rep["w2c"])
        gt = mask_img > 127
        union = np.logical_or(sil, gt).sum()
        iou = float(np.logical_and(sil, gt).sum() / union) if union else 0.0
        _, med, p90 = silhouette_metrics(depths[0], depths[0] > 0, mask_img, case_depth)
        nn_fwd, _ = KDTree(rep["obs_points"]).query(verts)
        nn_bwd, _ = KDTree(verts).query(rep["obs_points"])
        metrics[tag] = {
            "silhouette_iou": round(iou, 4),
            "depth_median_mm": round(med, 1),
            "depth_p90_mm": round(p90, 1),
            "mesh_to_obs_median_mm": round(float(np.median(nn_fwd)) * 1000, 1),
            "obs_to_mesh_median_mm": round(float(np.median(nn_bwd)) * 1000, 1),
            "obs_to_mesh_p90_mm": round(float(np.percentile(nn_bwd, 90)) * 1000, 1),
        }
        blend = overlay_on_real(raw, colors[0], depths[0])
        xor = raw.copy() // 3
        xor[gt & ~sil] = (40, 200, 40)   # green: real object the mesh missed
        xor[sil & ~gt] = (70, 110, 255)  # blue: mesh where there is no object
        panels[tag] = {"render": colors[0], "depth": depths[0], "blend": blend, "xor": xor}

    def caption(tag):
        m = metrics[tag]
        who = "deepAlign prior (WRONG pose)" if tag == "bad" else "good prior (same pose)"
        return (f"{who}  IoU {m['silhouette_iou']:.2f}  "
                f"mesh->obs {m['mesh_to_obs_median_mm']:.0f}mm  "
                f"obs->mesh {m['obs_to_mesh_median_mm']:.0f}mm")

    # Seg 1: blends side by side.
    frame = two_panel(
        label(panels["bad"]["blend"].copy(), caption("bad"), color=RED),
        label(panels["good"]["blend"].copy(), caption("good"), color=GREEN),
    )
    hold(writer, frame, 4.0)

    # Seg 2: blink real <-> render (2 Hz), synchronized panels.
    for cycle in range(8):
        show_render = cycle % 2 == 1
        left = panels["bad"]["render"] if show_render else raw
        right = panels["good"]["render"] if show_render else raw
        frame = two_panel(
            label(left.copy(), "blink: real <-> BAD render", color=RED),
            label(right.copy(), "blink: real <-> good render", color=GREEN),
        )
        hold(writer, frame, 0.5)

    # Seg 3: silhouette XOR (green = missed object, red = hallucinated).
    frame = two_panel(
        label(panels["bad"]["xor"].copy(),
              "silhouette error: green=missed  blue=extra", color=RED),
        label(panels["good"]["xor"].copy(), "good prior for contrast", color=GREEN),
    )
    hold(writer, frame, 4.0)

    # Seg 4: orbit around the aligned meshes + observation PCD dots.
    center = rep["obs_points"].mean(axis=0)
    orbit_radius = 0.55
    n_orbit = 72
    orbit_w2cs = []
    for k in range(n_orbit):
        az = 2 * np.pi * k / n_orbit
        eye = center + orbit_radius * np.array(
            [np.cos(az) * np.cos(np.deg2rad(35)),
             np.sin(az) * np.cos(np.deg2rad(35)),
             np.sin(np.deg2rad(35))])
        forward = center - eye
        forward /= np.linalg.norm(forward)
        right_v = np.cross(forward, [0.0, 0.0, 1.0])
        right_v /= np.linalg.norm(right_v)
        down = np.cross(forward, right_v)
        w2c_orbit = np.eye(4)
        w2c_orbit[:3, :3] = np.stack([right_v, down, forward], axis=1).T
        w2c_orbit[:3, 3] = -w2c_orbit[:3, :3] @ eye
        orbit_w2cs.append(w2c_orbit)
    bad_orb_c, bad_orb_d = render_world_mesh(rep["verts_final"], template, tidx, orbit_w2cs, fov)
    good_orb_c, good_orb_d = textured_world_render(good_glb, orbit_w2cs, fov)
    fx = 0.5 * W / np.tan(fov / 2)
    K_orbit = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    obs_sub = rep["obs_points"][:: max(1, len(rep["obs_points"]) // 4000)]
    for k in range(n_orbit):
        def with_dots(color_img, depth_img):
            img = color_img.copy()
            cam = (orbit_w2cs[k][:3, :3] @ obs_sub.T).T + orbit_w2cs[k][:3, 3]
            z = cam[:, 2]
            ok = z > 0.05
            u = (K_orbit[0, 0] * cam[ok, 0] / z[ok] + K_orbit[0, 2]).astype(int)
            v = (K_orbit[1, 1] * cam[ok, 1] / z[ok] + K_orbit[1, 2]).astype(int)
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            img[v[inb], u[inb]] = (255, 200, 40)
            return img
        frame = two_panel(
            label(with_dots(bad_orb_c[k], bad_orb_d[k]),
                  "BAD aligned mesh vs observation (yellow)", color=RED),
            label(with_dots(good_orb_c[k], good_orb_d[k]),
                  "good aligned mesh vs observation", color=GREEN),
        )
        hold(writer, frame, 2.0 / FPS)

    writer.close()
    return path, metrics


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BAD_MESH.exists():
        raise SystemExit(f"[visualize] cache mesh missing: {BAD_MESH} "
                         "(run deepAlign/build_cache.py first)")
    good_glb = CASE_SRC / "shape/matching/final_mesh.glb"
    if not good_glb.exists():
        raise SystemExit(f"[visualize] good final mesh missing: {good_glb}")

    case = CASE_ROOT / CASE_NAME
    rep_cache = OUT_DIR / "replay" / "rep.pkl"
    if rep_cache.exists() and "--fresh" not in sys.argv:
        print(f"[visualize] reusing cached replay {rep_cache}")
        with open(rep_cache, "rb") as f:
            rep = pickle.load(f)
    else:
        case = assemble_case()
        print(f"[visualize] case assembled at {case}")
        rep = replay_align(case)
        rep_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_cache, "wb") as f:
            pickle.dump(rep, f)

    deep_frame = cv2.cvtColor(
        cv2.imread(str(REPO_ROOT / "deepAlign/data/color/000001.png")), cv2.COLOR_BGR2RGB)
    deep_mask = cv2.imread(str(REPO_ROOT / "deepAlign/data/mask/000001.png"),
                           cv2.IMREAD_GRAYSCALE)

    process_path = make_process_video(rep, deep_frame, deep_mask)
    print(f"[visualize] wrote {process_path}")

    # Case depth map for metrics (camera-frame z of the world PCD).
    with np.load(case / "pcd/0.npz") as pcd:
        points = pcd["points"][0]
        valid = pcd["masks"][0]
    cam_z = points @ rep["w2c"][2, :3].T + rep["w2c"][2, 3]
    case_depth = np.where(valid, cam_z, 0.0).astype(np.float32)

    error_path, metrics = make_error_video(rep, good_glb, case_depth)
    print(f"[visualize] wrote {error_path}")
    summary = {
        "bad_mesh": str(BAD_MESH),
        "good_mesh": str(good_glb),
        "winner_candidate": rep["best_idx"],
        "winner_matches": int((rep["match_result"]["matches"] > -1).sum()),
        "max_candidate_matches": int(rep["counts"].max()),
        "pnp_reprojection_error_px": round(rep["reproj_error_px"], 2),
        "optimal_scale": round(rep["optimal_scale"], 4),
        "metrics": metrics,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
