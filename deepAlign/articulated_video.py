"""Compose the articulated-align diagnostic video (proposal section-10 style).

    CUDA_VISIBLE_DEVICES=1 conda run -n demo_2_max python deepAlign/articulated_video.py

Reads deepAlign/outputs/articulated/fit_states.npz (written by
articulated_fit.py) plus the Task-3 replay rep.pkl, and renders

    deepAlign/outputs/articulated/articulated_alignment.mp4

sections (all two-panel, 1696x540):
  A  live frame + mask        | rigid PnP+scale placement
  B  rigid placement          | RigAnything rig on it (parts + skeleton)
  C  fit rollout: textured overlay | parts+skeleton view, per snapshot
  D  articulated final        | legacy full-ARAP result
  E  silhouette XOR: articulated | legacy   (green=missed, blue=extra)
  F  distance-to-obs heatmap: articulated | legacy  (0..60mm)
  G  orbit: textured render   | part dots + skeleton + obs PCD
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh
from scipy.spatial import KDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "deepAlign"))

import visualize_align as V  # noqa: E402
from articulated_fit import PALETTE, look_at_w2c, project_pts  # noqa: E402

OUT_DIR = V.OUT_DIR / "articulated"
FPS = 24
CANVAS = (V.H + 60, V.W * 2)
ORANGE = (255, 170, 60)


def main() -> int:
    with open(V.OUT_DIR / "replay" / "rep.pkl", "rb") as f:
        rep = pickle.load(f)
    S = np.load(OUT_DIR / "fit_states.npz", allow_pickle=True)
    summary = json.loads((OUT_DIR / "summary.json").read_text())
    met = summary["metrics"]

    raw, mask_img = rep["raw_img"], rep["mask_img"]
    K, w2c, fov = rep["intrinsic"], rep["w2c"], rep["fov"]
    tidx, faces, obs = rep["trimesh_indices"], rep["mesh_faces"], rep["obs_points"]
    template = V.as_mesh(trimesh.load_mesh(rep["mesh_path"], force="mesh"))
    parents, dominant = S["parents"], S["dominant"]

    obs_tree = KDTree(obs)

    def draw_parts(canvas, verts, joints, K_, w2c_, dot=2):
        pts = project_pts(verts, K_, w2c_)
        h_, w_ = canvas.shape[:2]
        for i in np.argsort(pts[:, 2])[::-1]:
            u, v_px = int(round(pts[i, 0])), int(round(pts[i, 1]))
            if 0 <= u < w_ and 0 <= v_px < h_ and pts[i, 2] > 0:
                cv2.circle(canvas, (u, v_px), dot,
                           tuple(int(c) for c in PALETTE[dominant[i] % len(PALETTE)]), -1)
        jp = project_pts(joints, K_, w2c_)
        for j, p in enumerate(parents):
            if j != p:
                cv2.line(canvas, (int(round(jp[j, 0])), int(round(jp[j, 1]))),
                         (int(round(jp[p, 0])), int(round(jp[p, 1]))),
                         (245, 245, 245), 2, cv2.LINE_AA)
        for j in range(len(jp)):
            c = (int(round(jp[j, 0])), int(round(jp[j, 1])))
            cv2.circle(canvas, c, 4, (20, 20, 20), -1)
            cv2.circle(canvas, c, 3, tuple(int(x) for x in PALETTE[j % len(PALETTE)]), -1)
        return canvas

    def overlay(verts, caption, color=V.WHITE):
        colors, depths = V.render_world_mesh(verts, template, tidx, [w2c], fov)
        return V.label(V.overlay_on_real(raw, colors[0], depths[0]), caption,
                       color=color)

    def xor_panel(verts, caption, color):
        sil = V.trusted_silhouette(verts[tidx], faces, K, w2c)
        gt = mask_img > 127
        panel = raw.copy() // 3
        panel[gt & ~sil] = (40, 200, 40)
        panel[sil & ~gt] = (70, 110, 255)
        return V.label(panel, caption, color=color)

    def heat_panel(verts, caption, color):
        dist_mm = obs_tree.query(verts)[0] * 1000.0
        norm = np.clip(dist_mm / 60.0, 0, 1)
        cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        panel = raw.copy() // 3
        pts = project_pts(verts, K, w2c)
        for i in np.argsort(pts[:, 2])[::-1]:
            u, v_px = int(round(pts[i, 0])), int(round(pts[i, 1]))
            if 0 <= u < V.W and 0 <= v_px < V.H and pts[i, 2] > 0:
                cv2.circle(panel, (u, v_px), 2, tuple(int(c) for c in cmap[i, 0]), -1)
        V.label(panel, caption, color=color)
        return V.label(panel, f"median {np.median(dist_mm):.0f}mm  "
                              f"p90 {np.percentile(dist_mm, 90):.0f}mm", y=56)

    writer = imageio.get_writer(OUT_DIR / "articulated_alignment.mp4", fps=FPS)

    def emit(frame, seconds):
        V.hold(writer, frame, seconds, CANVAS)

    # A: live + rigid
    live = raw.copy()
    contour, _ = cv2.findContours((mask_img > 127).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(live, contour, -1, V.GREEN, 2)
    rigid_ov = overlay(S["verts_initial"], "1 rigid placement (legacy PnP+scale)")
    emit(V.two_panel(V.label(live, "live frame-0 + object mask", color=V.GREEN),
                     rigid_ov), 2.5)

    # B: rig on rigid placement
    parts_rest = draw_parts(raw.copy() // 2, S["verts_initial"], S["joints_rest"], K, w2c)
    emit(V.two_panel(rigid_ov,
                     V.label(parts_rest, "RigAnything rig: 28 joints + skinning",
                             color=V.WHITE)), 2.5)

    # C: fit rollout
    tags, losses = S["tags"], S["losses"]
    for i in range(len(tags)):
        verts_s, joints_s = S["snapshots"][i], S["snapshot_joints"][i]
        left = overlay(verts_s, f"articulated ICP  [{tags[i]}]  loss {losses[i]:.4f}",
                       V.RED)
        right = draw_parts(raw.copy() // 2, verts_s, joints_s, K, w2c)
        V.label(right, "root SE(3) + bounded joint rotations", color=V.WHITE)
        emit(V.two_panel(left, right), 2 / FPS if i else 1.0)
    emit(V.two_panel(left, right), 1.5)

    # D: articulated vs legacy
    art = met["articulated_final"]
    leg = met["legacy_arap"]
    art_ov = overlay(S["verts_art"],
                     f"articulated: limb distortion {art['part_rigid_intra_mean_pct']}%",
                     V.RED)
    leg_ov = overlay(S["verts_legacy"],
                     f"legacy ARAP: limb distortion {leg['part_rigid_intra_mean_pct']}%",
                     ORANGE)
    emit(V.two_panel(art_ov, leg_ov), 3.0)

    # E: XOR
    emit(V.two_panel(
        xor_panel(S["verts_art"],
                  f"articulated XOR  IoU {art['silhouette_iou']}  "
                  "(green=missed blue=extra)", V.RED),
        xor_panel(S["verts_legacy"],
                  f"legacy ARAP XOR  IoU {leg['silhouette_iou']}", ORANGE)), 3.0)

    # F: distance heatmap
    emit(V.two_panel(
        heat_panel(S["verts_art"], "dist to obs 0-60mm: articulated (rigid limbs)",
                   V.RED),
        heat_panel(S["verts_legacy"],
                   "legacy ARAP: low residual by CRUSHING the mesh", ORANGE)), 3.5)

    # G: orbit, textured | parts + obs pcd
    center = obs.mean(0)
    radius = 2.5 * np.linalg.norm(obs - center, axis=1).max()
    fov_k = 0.5 * V.W / np.tan(fov / 2)
    K_orbit = np.array([[fov_k, 0, V.W / 2], [0, fov_k, V.H / 2], [0, 0, 1.0]])
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    w2cs_orbit = [look_at_w2c(center + np.array([radius * np.cos(a),
                                                 radius * np.sin(a),
                                                 0.55 * radius]), center)
                  for a in angles]
    for chunk_start in range(0, len(w2cs_orbit), 12):
        chunk = w2cs_orbit[chunk_start:chunk_start + 12]
        colors, depths = V.render_world_mesh(S["verts_art"], template, tidx, chunk, fov)
        for k in range(len(chunk)):
            left = np.full((V.H, V.W, 3), 12, dtype=np.uint8)
            vis = depths[k] > 0
            left[vis] = colors[k][vis]
            V.label(left, "articulated final (textured)", color=V.RED)
            right = np.full((V.H, V.W, 3), 12, dtype=np.uint8)
            op = project_pts(obs, K_orbit, chunk[k])
            for i in range(0, len(op), 2):
                u, v_px = int(round(op[i, 0])), int(round(op[i, 1]))
                if 0 <= u < V.W and 0 <= v_px < V.H:
                    right[v_px, u] = (150, 150, 150)
            right = draw_parts(right, S["verts_art"], S["joints_final"],
                               K_orbit, chunk[k])
            V.label(right, "parts + skeleton vs obs PCD (grey)", color=V.WHITE)
            V.hold(writer, V.two_panel(left, right), 1 / FPS, CANVAS)

    writer.close()
    print(f"[video] {OUT_DIR / 'articulated_alignment.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
