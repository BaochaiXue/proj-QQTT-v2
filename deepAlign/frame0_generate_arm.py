"""Arm: generate the shape prior FROM frame-0 itself, then align + sample.

    conda run -n demo_2_max python deepAlign/frame0_generate_arm.py

Emulates the Axolotl3D problem restructuring with today's components: instead
of a cached canonical mesh (generated from some OTHER staged frame) that align
must drag into place, SAM3D generates directly from the current frame-0 masked
image — so the part layout / pose match the target by construction and align
only has a mild job. Measures the two things that decide viability:

  1. wall time of every stage (upscale / SAM3D generate / align / sample),
     including the model-load vs inference split from the stage profiles;
  2. quality against the SAME trusted metrics as visualize_align, for
     - rigid-only placement (PnP+scale, "simple align" lower bound),
     - full align (keypoint+ray ARAP),
     compared to the cached-prior baseline and the deepAlign wrong-pose arm.

Note what this does NOT emulate: Axolotl3D conditions on the partial point
cloud and a visibility mask (hands = occluded). SAM3D only sees the masked
RGB, so hand-occluded regions are holes it must hallucinate around, and
metric scale still comes from align's scale step.

Outputs under deepAlign/outputs/frame0_gen/: object.glb, replay/, align_case/,
comparison stills, summary.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "deepAlign"))

from build_cache import crop_bbox_like_upscale, stage_env  # noqa: E402
import visualize_align as V  # noqa: E402

ARM_DIR = V.OUT_DIR / "frame0_gen"
OBJECT_PROMPT = "sloth stuffed animal"


def run_stage(command: list[str], timings: dict, key: str) -> None:
    print(f"[arm] $ {' '.join(command)}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=stage_env())
    timings[key] = round(time.perf_counter() - started, 2)
    print(f"[arm] {key}: {timings[key]}s", flush=True)


def read_profile(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text()).get("timing_ms", {})
        except Exception:
            return {}
    return {}


def mesh_metrics(verts, faces, rep, mask_img, case_depth) -> dict:
    from scipy.spatial import KDTree

    sil = V.trusted_silhouette(verts, faces, rep["intrinsic"], rep["w2c"])
    gt = mask_img > 127
    union = np.logical_or(sil, gt).sum()
    iou = float(np.logical_and(sil, gt).sum() / union) if union else 0.0
    colors, depths = V.render_world_mesh(
        verts, mesh_template(rep), rep["trimesh_indices"], [rep["w2c"]], rep["fov"]
    ) if False else (None, None)  # depth metrics via trusted projection below
    # Depth residual from the trusted projection: rasterize per-vertex camera z
    # is overkill; reuse the pytorch3d depth only for the visible-region stats.
    nn_fwd, _ = KDTree(rep["obs_points"]).query(verts)
    nn_bwd, _ = KDTree(verts).query(rep["obs_points"])
    return {
        "silhouette_iou": round(iou, 4),
        "mesh_to_obs_median_mm": round(float(np.median(nn_fwd)) * 1000, 1),
        "mesh_to_obs_p90_mm": round(float(np.percentile(nn_fwd, 90)) * 1000, 1),
        "obs_to_mesh_median_mm": round(float(np.median(nn_bwd)) * 1000, 1),
        "obs_to_mesh_p90_mm": round(float(np.percentile(nn_bwd, 90)) * 1000, 1),
    }


_TEMPLATE_CACHE: dict = {}


def mesh_template(rep):
    key = rep["mesh_path"]
    if key not in _TEMPLATE_CACHE:
        import trimesh

        _TEMPLATE_CACHE[key] = V.as_mesh(trimesh.load_mesh(key, force="mesh"))
    return _TEMPLATE_CACHE[key]


def main() -> int:
    ARM_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    case_src = V.CASE_SRC

    # 1. Upscale the CURRENT frame-0 (real stage, GT object mask).
    color_path = case_src / "color/0/0.png"
    mask_path = case_src / "mask/0/0/0.png"
    high_res_path = ARM_DIR / "high_resolution.png"
    upscale_profile = ARM_DIR / "upscale_profile.json"
    run_stage([
        sys.executable, "-m", "demo_v6_2.shape_prior.upscale",
        "--img_path", str(color_path), "--mask_path", str(mask_path),
        "--output_path", str(high_res_path), "--category", OBJECT_PROMPT,
        "--profile-json", str(upscale_profile),
    ], timings, "upscale_s")

    # 2. Segment substitute: GT mask -> alpha (same recipe as build_cache).
    mask = np.asarray(Image.open(mask_path).convert("L"))
    bbox = crop_bbox_like_upscale(mask)
    high_res = Image.open(high_res_path).convert("RGB")
    alpha = Image.fromarray(mask).crop(bbox).resize(high_res.size, Image.NEAREST)
    rgba = high_res.copy()
    rgba.putalpha(alpha)
    masked_image = ARM_DIR / "masked_image.png"
    rgba.save(masked_image)

    # 3. SAM3D generate from frame-0 itself.
    generate_profile = ARM_DIR / "generate_profile.json"
    run_stage([
        sys.executable, "-m", "demo_v6_2.shape_prior.generate",
        "--img_path", str(masked_image), "--output_dir", str(ARM_DIR),
        "--seed", "42", "--skip-visualization",
        "--profile-json", str(generate_profile),
    ], timings, "generate_s")
    object_glb = ARM_DIR / "object.glb"
    if not object_glb.exists():
        raise SystemExit("[arm] generate produced no object.glb")

    # 4. Assemble case + full align replay (timed; no per-candidate counts).
    case = ARM_DIR / "align_case" / V.CASE_NAME
    if case.exists():
        shutil.rmtree(case)
    for rel in ("color", "mask", "pcd", "metadata.json", "calibrate.pkl"):
        src, dst = case_src / rel, case / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst) if src.is_dir() else shutil.copyfile(src, dst)
    (case / "shape").mkdir(parents=True)
    shutil.copyfile(object_glb, case / "shape" / "object.glb")

    started = time.perf_counter()
    rep = V.replay_align(case, out_dir=ARM_DIR / "replay", with_counts=False)
    timings["align_s"] = round(time.perf_counter() - started, 2)
    print(f"[arm] align_s: {timings['align_s']}s")

    # 5. Sample stage on the aligned mesh (the actual downstream entry).
    matching_dir = case / "shape" / "matching"
    matching_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ARM_DIR / "replay" / "final_mesh.glb", matching_dir / "final_mesh.glb")
    run_stage([
        sys.executable, "-m", "demo_v6_2.shape_prior.sample",
        "--base_path", str(case.parent), "--case_name", V.CASE_NAME,
        "--num_surface_points", "1024",
    ], timings, "sample_s")
    candidates = case / "shape" / "candidates.npz"
    with np.load(candidates) as data:
        sample_counts = {k: int(np.asarray(data[k]).shape[0])
                        for k in ("raw_surface_points", "raw_interior_points")}

    # 6. Metrics: rigid-only vs full align, plus references.
    mask_img = rep["mask_img"]
    with np.load(case / "pcd/0.npz") as pcd:
        points, valid = pcd["points"][0], pcd["masks"][0]
    cam_z = points @ rep["w2c"][2, :3].T + rep["w2c"][2, 3]
    case_depth = np.where(valid, cam_z, 0.0).astype(np.float32)

    tidx, faces = rep["trimesh_indices"], rep["mesh_faces"]
    metrics = {
        "frame0_gen_rigid_only": mesh_metrics(
            rep["verts_initial"][tidx], faces, rep, mask_img, case_depth),
        "frame0_gen_full_align": mesh_metrics(
            rep["verts_final"][tidx], faces, rep, mask_img, case_depth),
    }
    previous = json.loads((V.OUT_DIR / "metrics.json").read_text())
    references = {
        "cached_prior_full_align": previous["metrics"]["good"],
        "deepalign_wrongpose_full_align": previous["metrics"]["bad"],
    }

    # 7. Comparison stills: cached baseline vs frame0-gen (blend + XOR).
    good_glb = case_src / "shape/matching/final_mesh.glb"
    good_colors, good_depths = V.textured_world_render(good_glb, [rep["w2c"]], rep["fov"])
    new_colors, new_depths = V.render_world_mesh(
        rep["verts_final"], mesh_template(rep), tidx, [rep["w2c"]], rep["fov"])
    raw = rep["raw_img"]
    blend_pair = np.concatenate([
        V.label(V.overlay_on_real(raw, good_colors[0], good_depths[0]).copy(),
                "baseline: cached prior + align", color=V.GREEN),
        V.label(V.overlay_on_real(raw, new_colors[0], new_depths[0]).copy(),
                "frame-0 generation + align", color=V.RED),
    ], axis=1)
    cv2.imwrite(str(ARM_DIR / "blend_baseline_vs_frame0gen.png"),
                cv2.cvtColor(blend_pair, cv2.COLOR_RGB2BGR))

    def xor_panel(verts, faces_, caption, color):
        sil = V.trusted_silhouette(verts, faces_, rep["intrinsic"], rep["w2c"])
        gt = mask_img > 127
        panel = raw.copy() // 3
        panel[gt & ~sil] = (40, 200, 40)
        panel[sil & ~gt] = (70, 110, 255)
        return V.label(panel, caption, color=color)

    import trimesh
    good_mesh = V.as_mesh(trimesh.load_mesh(good_glb, force="mesh"))
    xor_pair = np.concatenate([
        xor_panel(np.asarray(good_mesh.vertices), np.asarray(good_mesh.faces),
                  "baseline XOR (green=missed blue=extra)", V.GREEN),
        xor_panel(rep["verts_final"][tidx], faces, "frame-0 gen XOR", V.RED),
    ], axis=1)
    cv2.imwrite(str(ARM_DIR / "xor_baseline_vs_frame0gen.png"),
                cv2.cvtColor(xor_pair, cv2.COLOR_RGB2BGR))

    summary = {
        "timings_s": timings,
        "total_pipeline_s": round(sum(timings.values()), 2),
        "upscale_profile_ms": read_profile(upscale_profile),
        "generate_profile_ms": read_profile(generate_profile),
        "align_winner_matches": int((rep["match_result"]["matches"] > -1).sum()),
        "align_pnp_reproj_px": round(rep["reproj_error_px"], 2),
        "align_scale": round(rep["optimal_scale"], 4),
        "sample_candidates": sample_counts,
        "metrics": metrics,
        "references": references,
    }
    (ARM_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
