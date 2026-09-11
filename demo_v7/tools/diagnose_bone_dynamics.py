"""Production-faithful bone dynamics diagnostics for the FORMAL gaussian path.

Usage::

    python demo_v7/tools/diagnose_bone_dynamics.py <run_base_path>

Reads ``<run>/capture/prepared_phystwin/*.npz`` plus
``<run>/gaussian/gaussian_anchors.npz`` (CPU only, no GPU, no rendering) and
reports what actually drives the mesh during FORMAL.

It exists because the first version of this analysis (2026-08-14) drew wrong
conclusions from four methodological errors, all fixed here:

1. it used ``x_t - (x_{t-1}+x_{t-2})/2`` as a "jitter" metric — that is NOT a
   second difference and equals ``1.5*v`` for constant-velocity motion, so it
   reported ordinary object speed as tracker noise. The true second
   difference is ``x_t - 2*x_{t-1} + x_{t-2}``, and it is taken here AFTER
   removing the frame-to-frame rigid fit so real limb motion does not leak in;
2. it recomputed object membership every frame from the current object mask.
   Production freezes ``query_is_object`` once at query seeding WITH hand
   priority (tracker_geometry._classify_query_targets_yx:
   ``in_object = object & ~(hand_a | hand_b)``), so the old replay promoted
   hand-overlapping queries to object bones exactly where it matters most;
3. it counted stale queries across all 5000 tracker queries, while the live
   renderer skins only the frozen ``_bone_ids`` subset;
4. bone COUNTS are the wrong unit anyway. What moves the mesh is how much
   SKINNING WEIGHT the held bones carry, so the headline number here is the
   per-vertex held influence mass ``m_v = sum_k w_vk * 1[k held]`` under the
   production K=16 inverse-distance weighting.

A held bone is one the tracker did not see this frame while its last-known
world position keeps driving its splats (gaussian_live only flags a bone
"stale" after 10 missed packets, though the rigidity outlier test can catch
an obviously lagging one earlier).
"""

import numpy as np, trimesh
from pathlib import Path

import sys
RUN = Path(sys.argv[1] if len(sys.argv)>1 else '/home/xinjie/single_proj_qqtt/outputs')
PREP = RUN / 'capture' / 'prepared_phystwin'
frames = sorted(PREP.glob('*.npz'))
f0 = np.load(frames[0])

H, W = f0['mask_object'].shape
qp = np.asarray(f0['query_points_yx'], np.float32).reshape(-1, 2)
yy = np.clip(np.rint(qp[:, 0]).astype(int), 0, H - 1)
xx = np.clip(np.rint(qp[:, 1]).astype(int), 0, W - 1)
ha = np.asarray(f0['mask_hand_a'], bool)[yy, xx]
hb = np.asarray(f0['mask_hand_b'], bool)[yy, xx] & ~ha
is_object0 = np.asarray(f0['mask_object'], bool)[yy, xx] & ~(ha | hb)   # FROZEN
print(f'frame-0 frozen classification: object={is_object0.sum()} '
      f'hand_a={ha.sum()} hand_b={hb.sum()} of {len(qp)}')

def lift(npz):
    t = np.asarray(npz['tracks_yx'], float)
    r = np.clip(np.rint(t[:, 0]).astype(int), 0, H - 1)
    c = np.clip(np.rint(t[:, 1]).astype(int), 0, W - 1)
    g = np.asarray(npz['pcd_points'][0], np.float32)
    w = g[r, c]
    vis = np.asarray(npz['visibility'], bool)
    return w, vis & np.isfinite(w).all(1)

# rest positions exactly like load_formal_frame0_rest_positions (no hand mask)
w0, ok0 = lift(f0)
rest_ok = ok0 & np.asarray(f0['mask_object'], bool)[
    np.clip(np.rint(np.asarray(f0['tracks_yx'], float)[:, 0]).astype(int), 0, H - 1),
    np.clip(np.rint(np.asarray(f0['tracks_yx'], float)[:, 1]).astype(int), 0, W - 1)]
bone_ids = np.flatnonzero(is_object0 & rest_ok)     # frozen bone subset
rest = w0[bone_ids].astype(np.float64)
print(f'frozen BONE subset: {len(bone_ids)} (renderer skins only these)')

# production skinning: K=16 inverse-distance, rest bones -> rest mesh vertices
mesh = trimesh.load(RUN / 'gaussian' / 'gaussian_world.ply', force='mesh', process=False) \
    if False else None
anch = np.load(RUN / 'gaussian' / 'gaussian_anchors.npz')
verts = np.asarray(anch['rest_vertices'], np.float64)
K = 16
d = np.linalg.norm(verts[:, None, :] - rest[None, :, :], axis=2)
idx = np.argpartition(d, K, axis=1)[:, :K]
dk = np.take_along_axis(d, idx, axis=1)
wk = 1.0 / (dk + 1e-6)
wk /= wk.sum(1, keepdims=True)
print(f'skinning built: {verts.shape[0]} mesh verts x K={K}')

last_seen = np.full(len(bone_ids), -10**9)
last_pos = np.full((len(bone_ids), 3), np.nan)
hist = []
rows = []
for i, p in enumerate(frames):
    npz = np.load(p)
    w, ok = lift(npz)
    vis_b = ok[bone_ids]
    cur = np.where(vis_b[:, None], w[bone_ids], last_pos)
    held = (~vis_b) & (last_seen > -10**8)
    age = i - last_seen
    held_short = held & (age >= 1) & (age <= 10)

    # held influence mass per vertex (the meaningful quantity)
    held_w = held_short[idx]
    mass = (wk * held_w).sum(1)

    # local motion AFTER removing the global rigid fit (visible bones only)
    hist.append((vis_b.copy(), cur.copy()))
    acc = np.nan
    if len(hist) >= 3:
        (v2, x2), (v1, x1), (v0_, x0_) = hist[-3], hist[-2], hist[-1]
        both = v2 & v1 & v0_
        if both.sum() > 50:
            def kab(a, b):
                ca, cb = a.mean(0), b.mean(0)
                U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
                D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
                R = Vt.T @ D @ U.T
                return R, cb - R @ ca
            R1, t1 = kab(x2[both], x1[both]); R2, t2 = kab(x2[both], x0_[both])
            a1 = x1[both] - (x2[both] @ R1.T + t1)      # residual, rigid removed
            a2 = x0_[both] - (x2[both] @ R2.T + t2)
            # true 2nd difference of the rigid-free residual
            acc = float(np.median(np.linalg.norm(a2 - 2 * a1, axis=1)))
    rows.append(dict(
        frame=i, bones_vis=int(vis_b.sum()), held_short=int(held_short.sum()),
        mass_p50=float(np.median(mass)), mass_p95=float(np.quantile(mass, .95)),
        mass_max=float(mass.max()),
        verts_over_10pct=int((mass > 0.10).sum()),
        acc_mm=None if np.isnan(acc) else acc * 1000,
    ))
    last_seen[vis_b] = i
    last_pos[vis_b] = w[bone_ids][vis_b]

warm = [r for r in rows[20:] if r['acc_mm'] is not None]
def q(k, p): return np.percentile([r[k] for r in warm], p)
print(f'\nframes analysed: {len(warm)}')
print(f'held(1-10 frames) BONES:  p50={q("held_short",50):.0f} '
      f'p90={q("held_short",90):.0f} max={q("held_short",100):.0f} '
      f'of {len(bone_ids)}')
print(f'per-vertex HELD influence mass: p50={q("mass_p50",50):.4f} '
      f'p95={q("mass_p95",90):.4f} max={q("mass_max",100):.3f}')
print(f'mesh verts with >10% held influence: p50={q("verts_over_10pct",50):.0f} '
      f'p90={q("verts_over_10pct",90):.0f} max={q("verts_over_10pct",100):.0f} '
      f'of {verts.shape[0]}')
print(f'TRUE local accel |x_t-2x_t-1+x_t-2| (rigid removed): '
      f'p50={q("acc_mm",50):.2f}mm p90={q("acc_mm",90):.2f}mm '
      f'max={q("acc_mm",100):.2f}mm')
