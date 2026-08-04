"""Round 0 of "Rigged Shape-Prior Align v2": articulated fit of the wrong-pose mesh.

    CUDA_VISIBLE_DEVICES=1 conda run -n demo_2_max python deepAlign/articulated_fit.py

The deepAlign cache mesh (sloth, spread-eagle) violates legacy align's
one-Sim(3)+mild-ARAP assumption; Task-3 showed ARAP crushing it onto the
observation. This script re-parameterizes the SAME problem the way the
Align-v2 proposal prescribes:

  rest mesh --RigAnything rig (28 joints, skinning)--> cached skeleton
  legacy PnP+scale placement (rep.pkl)              --> root initialization
  root SE(3) refine + per-joint bounded rotations   --> articulated ICP vs
  metric frame-0 obs PCD (LBS skinning, FK about joint pivots)

Rigid limbs are enforced BY CONSTRUCTION (rotations about pivots), so
part-internal edge lengths cannot stretch — the failure mode of legacy ARAP.
Joint limits via tanh caps (root +-30 deg, joints +-45 deg); bone lengths
exactly preserved by FK.

Inputs (must exist):
  deepAlign/outputs/replay/rep.pkl                  Task-3 legacy replay
  /home/xinjie/RigAnything/outputs/sloth_deepalign/sloth_deepalign_simplified.npz

Outputs under deepAlign/outputs/articulated/:
  summary.json         metrics for rigid-only / root-refined / articulated vs
                       legacy ARAP (+ good-prior reference), incl. the new
                       part-rigidity metric and timings
  five_panel.png       live | rigid | root-refined | articulated | legacy ARAP
  parts_skeleton.png   part-colored verts + skeleton on the real frame
  fit_process.mp4      optimization rollout overlaid on the real frame
  turntable.mp4        orbit: part-colored posed mesh + obs PCD + skeleton
  final_mesh_articulated.glb + sample-stage verification (candidates.npz)
"""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "deepAlign"))

import visualize_align as V  # noqa: E402  (heavy: pulls demo_v6_2 modules)
from build_cache import stage_env  # noqa: E402

# match_pairs import (inside visualize_align) disables autograd globally.
torch.set_grad_enabled(True)

RIG_NPZ = Path("/home/xinjie/RigAnything/outputs/sloth_deepalign/sloth_deepalign_simplified.npz")
OUT_DIR = V.OUT_DIR / "articulated"
DEVICE = "cuda"

ROOT_ROT_CAP_DEG = 30.0
JOINT_ROT_CAP_DEG = 45.0
ROOT_ITERS = 150
FULL_ITERS = 400
NN_REFRESH = 10
MESH2OBS_TRIM = 0.03  # only verts already near the observation act as ICP pullers
HUBER_DELTA = 0.01
POSE_PRIOR_W = 5e-4

# 28 visually distinct part colors (RGB), tab20-ish.
PALETTE = np.array([
    (188, 189, 34), (31, 119, 180), (255, 127, 14), (44, 160, 44),
    (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194),
    (127, 127, 127), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
    (57, 59, 121), (82, 84, 163), (107, 110, 207), (156, 158, 222),
    (99, 121, 57), (140, 162, 82), (181, 207, 107), (206, 219, 156),
], dtype=np.uint8)


def axis_angle_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """Rodrigues for a batch of axis-angle vectors [J,3] -> [J,3,3]."""
    theta = r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    axis = r / theta
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], -1),
        torch.stack([z, zero, -x], -1),
        torch.stack([-y, x, zero], -1),
    ], -2)
    theta = theta[..., None]
    eye = torch.eye(3, device=r.device, dtype=r.dtype).expand(K.shape)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def fk_global_transforms(root_rot, root_t, joint_rot, pivots, parents, centroid):
    """World-frame 4x4 per joint: G_j = G_parent . T(a_j) R_j T(-a_j)."""
    n = pivots.shape[0]
    R = axis_angle_to_matrix(joint_rot)  # [n,3,3]
    R_root = axis_angle_to_matrix(root_rot[None])[0]
    G = [None] * n
    root = torch.eye(4, device=pivots.device, dtype=pivots.dtype)
    # root: rotation about the rest centroid + translation
    root = root.clone()
    root[:3, :3] = R_root
    root[:3, 3] = centroid - R_root @ centroid + root_t
    for j in range(n):
        local = torch.eye(4, device=pivots.device, dtype=pivots.dtype).clone()
        local[:3, :3] = R[j]
        local[:3, 3] = pivots[j] - R[j] @ pivots[j]
        G[j] = root @ local if parents[j] == j else G[parents[j]] @ local
    return torch.stack(G)  # [n,4,4]


def lbs(verts_h, weights, G):
    """verts_h [N,4], weights [N,J], G [J,4,4] -> [N,3]."""
    per_joint = torch.einsum("jab,nb->jna", G, verts_h)  # [J,N,4]
    return torch.einsum("nj,jna->na", weights, per_joint)[:, :3]


def huber(d: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(d < delta, 0.5 * d * d, delta * (d - 0.5 * delta))


def project_pts(pts: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    uv = cam @ K.T
    return np.stack([uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2], cam[:, 2]], axis=1)


def mesh_metrics(verts_dedup, rep, edges, intra_mask, rest_len) -> dict:
    tidx, faces = rep["trimesh_indices"], rep["mesh_faces"]
    verts = verts_dedup[tidx]
    sil = V.trusted_silhouette(verts, faces, rep["intrinsic"], rep["w2c"])
    gt = rep["mask_img"] > 127
    union = np.logical_or(sil, gt).sum()
    iou = float(np.logical_and(sil, gt).sum() / union) if union else 0.0
    nn_fwd, _ = KDTree(rep["obs_points"]).query(verts_dedup)
    nn_bwd, _ = KDTree(verts_dedup).query(rep["obs_points"])
    length = np.linalg.norm(verts_dedup[edges[:, 0]] - verts_dedup[edges[:, 1]], axis=1)
    rel = np.abs(length - rest_len) / np.maximum(rest_len, 1e-9)
    return {
        "silhouette_iou": round(iou, 4),
        "mesh_to_obs_median_mm": round(float(np.median(nn_fwd)) * 1000, 1),
        "mesh_to_obs_p90_mm": round(float(np.percentile(nn_fwd, 90)) * 1000, 1),
        "obs_to_mesh_median_mm": round(float(np.median(nn_bwd)) * 1000, 1),
        "obs_to_mesh_p90_mm": round(float(np.percentile(nn_bwd, 90)) * 1000, 1),
        "part_rigid_intra_mean_pct": round(float(rel[intra_mask].mean()) * 100, 2),
        "part_rigid_intra_p95_pct": round(float(np.percentile(rel[intra_mask], 95)) * 100, 2),
        "seam_edge_mean_pct": round(float(rel[~intra_mask].mean()) * 100, 2),
    }


def look_at_w2c(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([right, down, forward])
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict = {}

    with open(V.OUT_DIR / "replay" / "rep.pkl", "rb") as f:
        rep = pickle.load(f)

    mesh2world = rep["mesh2world"]
    verts_initial = rep["verts_initial"]           # deduped verts, world frame
    verts_legacy = rep["verts_final"]              # legacy ARAP result
    obs = rep["obs_points"]
    tidx = rep["trimesh_indices"]
    faces = rep["mesh_faces"]

    # ---- rig transfer: canonical frame -> deduped verts + world pivots ----
    t0 = time.perf_counter()
    rig = np.load(RIG_NPZ, allow_pickle=True)
    joints_c, parents = rig["joints"].astype(np.float64), rig["parents"].astype(int)
    rig_pts, rig_w = rig["pointcloud"], rig["skinning_weights"]

    inv_m2w = np.linalg.inv(mesh2world)
    canon = verts_initial @ inv_m2w[:3, :3].T + inv_m2w[:3, 3]
    nn_dist, nn_idx = KDTree(rig_pts).query(canon)
    print(f"[rig] weight transfer NN dist: median {np.median(nn_dist):.4f}, "
          f"p95 {np.percentile(nn_dist, 95):.4f} (canonical units)")
    assert np.median(nn_dist) < 0.02, "rig pointcloud is not in the mesh frame"
    weights = rig_w[nn_idx]
    weights = weights / np.maximum(weights.sum(1, keepdims=True), 1e-9)
    joints_w = joints_c @ mesh2world[:3, :3].T + mesh2world[:3, 3]
    timings["rig_transfer_s"] = round(time.perf_counter() - t0, 2)

    dominant = weights.argmax(1)

    # ---- part-rigidity edge set (on the deduped vertex ids) ----
    edge_set = np.unique(
        np.sort(np.stack([
            tidx[faces[:, [0, 1]]], tidx[faces[:, [1, 2]]], tidx[faces[:, [2, 0]]],
        ]).reshape(-1, 2), axis=1), axis=0)
    edge_set = edge_set[edge_set[:, 0] != edge_set[:, 1]]
    intra = dominant[edge_set[:, 0]] == dominant[edge_set[:, 1]]
    rest_len = np.linalg.norm(
        verts_initial[edge_set[:, 0]] - verts_initial[edge_set[:, 1]], axis=1)
    print(f"[rig] {len(joints_c)} joints, {len(edge_set)} edges "
          f"({int(intra.sum())} intra-part, {int((~intra).sum())} seam)")

    # ---- torch setup ----
    dev = torch.device(DEVICE)
    vh = torch.from_numpy(
        np.concatenate([verts_initial, np.ones((len(verts_initial), 1))], 1)
    ).float().to(dev)
    w_t = torch.from_numpy(weights).float().to(dev)
    pivots = torch.from_numpy(joints_w).float().to(dev)
    parents_t = parents.tolist()
    centroid = torch.from_numpy(verts_initial.mean(0)).float().to(dev)
    obs_t = torch.from_numpy(obs).float().to(dev)

    u_root_r = torch.zeros(3, device=dev, requires_grad=True)
    u_root_t = torch.zeros(3, device=dev, requires_grad=True)
    u_joints = torch.zeros(len(joints_c), 3, device=dev, requires_grad=True)
    root_cap = np.deg2rad(ROOT_ROT_CAP_DEG)
    joint_cap = np.deg2rad(JOINT_ROT_CAP_DEG)

    def pose_verts():
        G = fk_global_transforms(
            root_cap * torch.tanh(u_root_r), u_root_t,
            joint_cap * torch.tanh(u_joints), pivots, parents_t, centroid)
        return lbs(vh, w_t, G), G

    idx_om = idx_mo = keep_mo = None

    def refresh_nn(verts_np):
        nonlocal idx_om, idx_mo, keep_mo
        tree_v = KDTree(verts_np)
        _, i_om = tree_v.query(obs)
        idx_om = torch.from_numpy(i_om).long().to(dev)
        d_mo, i_mo = KDTree(obs).query(verts_np)
        keep = d_mo < MESH2OBS_TRIM
        idx_mo = torch.from_numpy(i_mo[keep]).long().to(dev)
        keep_mo = torch.from_numpy(np.where(keep)[0]).long().to(dev)

    def loss_fn(verts):
        l_om = huber((verts[idx_om] - obs_t).norm(dim=1), HUBER_DELTA).mean()
        l_mo = huber((verts[keep_mo] - obs_t[idx_mo]).norm(dim=1), HUBER_DELTA).mean()
        prior = POSE_PRIOR_W * (u_joints.square().mean() + u_root_r.square().mean())
        return l_om + 0.3 * l_mo + prior, l_om, l_mo

    snapshots, losses = [], []

    def run_phase(params, iters, tag):
        opt = torch.optim.Adam(params, lr=0.03)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.5)
        for it in range(iters):
            if it % NN_REFRESH == 0:
                with torch.no_grad():
                    refresh_nn(pose_verts()[0].cpu().numpy())
            opt.zero_grad()
            verts, _ = pose_verts()
            loss, l_om, l_mo = loss_fn(verts)
            loss.backward()
            opt.step()
            sched.step()
            if it % 20 == 0 or it == iters - 1:
                snapshots.append((f"{tag} {it}", verts.detach().cpu().numpy()))
                losses.append(float(loss))
                print(f"[fit] {tag} {it:4d} loss {float(loss):.5f} "
                      f"(om {float(l_om):.5f} mo {float(l_mo):.5f})", flush=True)

    t0 = time.perf_counter()
    run_phase([u_root_r, u_root_t], ROOT_ITERS, "root")
    with torch.no_grad():
        verts_root = pose_verts()[0].cpu().numpy()
    timings["fit_root_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    run_phase([u_root_r, u_root_t, u_joints], FULL_ITERS, "joints")
    with torch.no_grad():
        verts_art, G_final = pose_verts()
        verts_art = verts_art.cpu().numpy()
        joints_final = (
            torch.einsum("jab,jb->ja",
                         G_final,
                         torch.cat([pivots, torch.ones(len(joints_c), 1, device=dev)], 1))
            [:, :3].cpu().numpy())
    timings["fit_joints_s"] = round(time.perf_counter() - t0, 2)

    angles = np.rad2deg(joint_cap * np.tanh(u_joints.detach().cpu().numpy()))
    angle_mag = np.linalg.norm(angles, axis=1)
    root_deg = float(np.linalg.norm(
        np.rad2deg(root_cap * np.tanh(u_root_r.detach().cpu().numpy()))))
    bone_len_rest = np.linalg.norm(
        joints_w[1:] - joints_w[parents[1:]], axis=1)
    bone_len_final = np.linalg.norm(
        joints_final[1:] - joints_final[parents[1:]], axis=1)

    # ---- metrics ----
    stages = {
        "rigid_only": verts_initial,
        "root_refined": verts_root,
        "articulated_final": verts_art,
        "legacy_arap": verts_legacy,
    }
    metrics = {name: mesh_metrics(v, rep, edge_set, intra, rest_len)
               for name, v in stages.items()}
    previous = json.loads((V.OUT_DIR / "metrics.json").read_text())

    # ---- final mesh + sample-stage verification ----
    template = V.as_mesh(trimesh.load_mesh(rep["mesh_path"], force="mesh"))
    posed = template.copy()
    posed.vertices = verts_art[tidx]
    posed.export(OUT_DIR / "final_mesh_articulated.glb")

    case = OUT_DIR / "sample_case" / V.CASE_NAME
    if case.exists():
        shutil.rmtree(case)
    for rel in ("color", "mask", "pcd", "metadata.json", "calibrate.pkl"):
        src, dst = V.CASE_SRC / rel, case / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copyfile(src, dst)
    (case / "shape" / "matching").mkdir(parents=True)
    shutil.copyfile(V.BAD_MESH, case / "shape" / "object.glb")
    shutil.copyfile(OUT_DIR / "final_mesh_articulated.glb",
                    case / "shape" / "matching" / "final_mesh.glb")
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "demo_v6_2.shape_prior.sample",
         "--base_path", str(case.parent), "--case_name", V.CASE_NAME,
         "--num_surface_points", "1024"],
        check=True, cwd=REPO_ROOT, env=stage_env())
    timings["sample_s"] = round(time.perf_counter() - t0, 2)
    with np.load(case / "shape" / "candidates.npz") as data:
        sample_counts = {k: int(np.asarray(data[k]).shape[0])
                        for k in ("raw_surface_points", "raw_interior_points")}

    # ---- renders ----
    raw, w2c, fov, K = rep["raw_img"], rep["w2c"], rep["fov"], rep["intrinsic"]

    def overlay(verts_dedup, caption, color):
        colors, depths = V.render_world_mesh(verts_dedup, template, tidx, [w2c], fov)
        return V.label(V.overlay_on_real(raw, colors[0], depths[0]), caption, color=color)

    live = raw.copy()
    contour, _ = cv2.findContours((rep["mask_img"] > 127).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(live, contour, -1, V.GREEN, 2)
    five = np.concatenate([
        V.label(live, "live frame-0 + mask", color=V.GREEN),
        overlay(verts_initial, "1 rigid (PnP+scale)", V.WHITE),
        overlay(verts_root, "2 root SE(3) refined", V.WHITE),
        overlay(verts_art, "3 articulated (joints)", V.RED),
        overlay(verts_legacy, "legacy full ARAP", (255, 170, 60)),
    ], axis=1)
    cv2.imwrite(str(OUT_DIR / "five_panel.png"), cv2.cvtColor(five, cv2.COLOR_RGB2BGR))

    def draw_parts_skeleton(canvas, verts_dedup, joints_world, K_, w2c_, dot=2):
        pts = project_pts(verts_dedup, K_, w2c_)
        order = np.argsort(pts[:, 2])[::-1]
        h_, w_ = canvas.shape[:2]
        for i in order:
            u, v_px = int(round(pts[i, 0])), int(round(pts[i, 1]))
            if 0 <= u < w_ and 0 <= v_px < h_ and pts[i, 2] > 0:
                cv2.circle(canvas, (u, v_px), dot,
                           tuple(int(c) for c in PALETTE[dominant[i] % len(PALETTE)]), -1)
        jp = project_pts(joints_world, K_, w2c_)
        for j, p in enumerate(parents):
            if j == p:
                continue
            a = (int(round(jp[j, 0])), int(round(jp[j, 1])))
            b = (int(round(jp[p, 0])), int(round(jp[p, 1])))
            cv2.line(canvas, a, b, (245, 245, 245), 2, cv2.LINE_AA)
        for j in range(len(jp)):
            cv2.circle(canvas, (int(round(jp[j, 0])), int(round(jp[j, 1]))), 4,
                       (20, 20, 20), -1)
            cv2.circle(canvas, (int(round(jp[j, 0])), int(round(jp[j, 1]))), 3,
                       tuple(int(c) for c in PALETTE[j % len(PALETTE)]), -1)
        return canvas

    parts_rest = draw_parts_skeleton(raw.copy() // 2, verts_initial, joints_w, K, w2c)
    parts_final = draw_parts_skeleton(raw.copy() // 2, verts_art, joints_final, K, w2c)
    parts = np.concatenate([
        V.label(parts_rest, "rig on rigid placement", color=V.WHITE),
        V.label(parts_final, "rig after articulated fit", color=V.RED),
    ], axis=1)
    cv2.imwrite(str(OUT_DIR / "parts_skeleton.png"),
                cv2.cvtColor(parts, cv2.COLOR_RGB2BGR))

    # fit process video
    writer = imageio.get_writer(OUT_DIR / "fit_process.mp4", fps=8)
    for i, (tag, verts_snap) in enumerate(snapshots):
        frame = overlay(verts_snap, f"articulated ICP  [{tag}]  loss {losses[i]:.4f}",
                        V.RED)
        writer.append_data(frame)
    for _ in range(16):
        writer.append_data(frame)
    writer.close()

    # turntable: part-colored dots + obs pcd + skeleton from orbit poses
    center = obs.mean(0)
    radius = 2.5 * np.linalg.norm(obs - center, axis=1).max()
    fov_k = 0.5 * V.W / np.tan(fov / 2)
    K_orbit = np.array([[fov_k, 0, V.W / 2], [0, fov_k, V.H / 2], [0, 0, 1.0]])
    writer = imageio.get_writer(OUT_DIR / "turntable.mp4", fps=24)
    for ang in np.linspace(0, 2 * np.pi, 96, endpoint=False):
        eye = center + np.array(
            [radius * np.cos(ang), radius * np.sin(ang), 0.55 * radius])
        w2c_o = look_at_w2c(eye, center)
        frame = np.full((V.H, V.W, 3), 12, dtype=np.uint8)
        op = project_pts(obs, K_orbit, w2c_o)
        for i in range(0, len(op), 2):
            u, v_px = int(round(op[i, 0])), int(round(op[i, 1]))
            if 0 <= u < V.W and 0 <= v_px < V.H:
                frame[v_px, u] = (120, 120, 120)
        frame = draw_parts_skeleton(frame, verts_art, joints_final, K_orbit, w2c_o)
        writer.append_data(
            V.label(frame, "articulated fit vs obs PCD (grey)", color=V.WHITE))
    writer.close()

    summary = {
        "rig": {
            "backend": "RigAnything (official ckpt)",
            "n_joints": int(len(joints_c)),
            "weight_transfer_nn_median": round(float(np.median(nn_dist)), 4),
        },
        "timings_s": timings,
        "fit": {
            "root_rotation_deg": round(root_deg, 1),
            "root_translation_mm": [round(float(x) * 1000, 1)
                                    for x in u_root_t.detach().cpu().numpy()],
            "joint_angle_deg_mean": round(float(angle_mag.mean()), 1),
            "joint_angle_deg_max": round(float(angle_mag.max()), 1),
            "joints_over_30deg": int((angle_mag > 30).sum()),
            "bone_length_change_mm_max": round(
                float(np.abs(bone_len_final - bone_len_rest).max()) * 1000, 3),
        },
        "metrics": metrics,
        "references_metrics_json": previous["metrics"],
        "sample_candidates": sample_counts,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
