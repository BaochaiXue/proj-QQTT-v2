from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_DIRS = [
    "gaussian_splatting",
    "configs",
    "taichi_simulator_test",
    "qqtt/data",
    "qqtt/engine",
    "qqtt/model",
    "qqtt/utils",
    "data_process/models",
    "data_process/utils",
    "test",
]

FORBIDDEN_TOP_LEVEL_FILES = [
    "process_data.py",
    "optimize_cma.py",
    "train_warp.py",
    "inference_warp.py",
    "inference_optimization_warp.py",
    "final_pipeline.sh",
    "combine_video.py",
    "optical_frames.py",
    "data_config.csv",
    "physics_dynamics_module.py",
    "prepare_results.py",
    "outdomain_exp.py",
    "env_install/download_pretrained_models.sh",
]

FORBIDDEN_TOP_LEVEL_PREFIXES = [
    "evaluate_",
    "export_",
    "gs_",
    "visualize_",
    "interactive_",
    "script",
]

FORBIDDEN_DATA_PROCESS_FILES = [
    "align.py",
    "data_process_mask.py",
    "data_process_pcd.py",
    "data_process_sample.py",
    "data_process_track.py",
    "dense_track.py",
    "image_upscale.py",
    "match_pairs.py",
    "outdomain_align.py",
    "prepare_gt_track.py",
    "segment.py",
    "segment_util_image.py",
    "segment_util_video.py",
    "shape_prior.py",
]

README_BANNED_FRAGMENTS = [
    "builds digital twins",
    "inverse physics over a differentiable spring-mass model",
    "gaussian splatting for realistic appearance rendering",
    "inverse-physics",
    "gaussian splatting pipeline",
]

ENV_INSTALL_BANNED_FRAGMENTS = [
    "warp-lang",
    "pytorch",
    "pytorch3d",
    "grounded-sam-2",
    "groundingdino",
    "diffusers",
    "accelerate",
    "trellis",
    "gsplat",
    "kornia",
]


def check_absent(path_strings: list[str], errors: list[str]) -> None:
    for relative in path_strings:
        if (ROOT / relative).exists():
            errors.append(f"Forbidden path still present: {relative}")


def check_top_level_patterns(errors: list[str]) -> None:
    for item in ROOT.iterdir():
        if not item.is_file():
            continue
        name = item.name
        if any(name.startswith(prefix) for prefix in FORBIDDEN_TOP_LEVEL_PREFIXES):
            errors.append(f"Forbidden top-level file still present: {name}")


def check_data_process_scope(errors: list[str]) -> None:
    data_process_dir = ROOT / "data_process"
    for name in FORBIDDEN_DATA_PROCESS_FILES:
        if (data_process_dir / name).exists():
            errors.append(f"Forbidden data_process file still present: data_process/{name}")


def check_qqtt_exports(errors: list[str]) -> None:
    text = (ROOT / "qqtt" / "__init__.py").read_text(encoding="utf-8")
    if "CameraSystem" not in text:
        errors.append("qqtt/__init__.py no longer exports CameraSystem")
    banned_terms = ["SpringMassSystemWarp", "InvPhyTrainerWarp", "OptimizerCMA"]
    for term in banned_terms:
        if term in text:
            errors.append(f"qqtt/__init__.py still references banned export: {term}")


def check_readme_scope(errors: list[str]) -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for fragment in README_BANNED_FRAGMENTS:
        if fragment in text:
            errors.append(f"README still contains banned active-scope fragment: {fragment}")


def check_env_install(errors: list[str]) -> None:
    text = (ROOT / "env_install" / "env_install.sh").read_text(encoding="utf-8").lower()
    for fragment in ENV_INSTALL_BANNED_FRAGMENTS:
        if fragment in text:
            errors.append(f"env_install/env_install.sh still references banned dependency: {fragment}")


def main() -> int:
    errors: list[str] = []
    check_absent(FORBIDDEN_DIRS, errors)
    check_absent(FORBIDDEN_TOP_LEVEL_FILES, errors)
    check_top_level_patterns(errors)
    check_data_process_scope(errors)
    check_qqtt_exports(errors)
    check_readme_scope(errors)
    check_env_install(errors)

    if errors:
        print("Scope check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Scope check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
