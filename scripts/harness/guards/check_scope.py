from __future__ import annotations

from pathlib import Path
import sys


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "qqtt").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"failed to locate repo root from {start}")


ROOT = _find_repo_root(Path(__file__).resolve())

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

DEMO32_SHAPE_PRIOR_CARVEOUTS = [
    "demo_v5_1/shape_prior.py",
    "demo_v5_1/shape_prior_worker.py",
    "qqtt/demo/shape_prior_warmup.py",
    "qqtt/demo/single_view_shape_align.py",
]

FORMAL_RECORDING_ALIGNMENT_FILES = [
    "record_data.py",
    "record_data_realtime_align.py",
    "data_process/record_data_align.py",
]

FORMAL_SHAPE_PRIOR_BANNED_FRAGMENTS = [
    "shape_prior_warmup",
    "single_view_shape_align",
    "shape_prior_remote",
    "data_process_sam3d",
    "sam3d",
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

BRANCH_POLICY_REQUIRED_TEXT = {
    "AGENTS.md": [
        "Single-Camera Branch Policy",
        "single-camera-specific modifications must be made, committed, "
        "and pushed on the `single-camera` branch",
        "Do not commit or push single-camera changes directly to `main`",
        "git push origin single-camera",
    ],
    "scripts/harness/README.md": [
        "Single-Camera Branch Safety",
        "Single-camera-specific modifications belong on the `single-camera` branch",
        "Do not commit or push single-camera changes directly to `main`",
        "git push origin single-camera",
    ],
}


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


def check_demo32_shape_prior_carveout(errors: list[str]) -> None:
    for relative in DEMO32_SHAPE_PRIOR_CARVEOUTS:
        if not (ROOT / relative).exists():
            errors.append(f"Missing Demo 3.2 shape-prior carveout path: {relative}")
    for relative in FORMAL_RECORDING_ALIGNMENT_FILES:
        path = ROOT / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for fragment in FORMAL_SHAPE_PRIOR_BANNED_FRAGMENTS:
            if fragment in text:
                errors.append(
                    "Formal recording/alignment file "
                    f"{relative} references shape-prior fragment: {fragment}"
                )


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
    text = (
        (ROOT / "env_install" / "env_install.sh")
        .read_text(encoding="utf-8")
        .lower()
    )
    for fragment in ENV_INSTALL_BANNED_FRAGMENTS:
        if fragment in text:
            errors.append(
                "env_install/env_install.sh still references banned dependency: "
                f"{fragment}"
            )


def check_single_camera_branch_policy(errors: list[str]) -> None:
    for relpath, fragments in BRANCH_POLICY_REQUIRED_TEXT.items():
        path = ROOT / relpath
        if not path.exists():
            errors.append(f"Missing branch-policy file: {relpath}")
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in text:
                errors.append(f"{relpath} missing single-camera branch-policy fragment: {fragment}")


def main() -> int:
    errors: list[str] = []
    check_absent(FORBIDDEN_DIRS, errors)
    check_absent(FORBIDDEN_TOP_LEVEL_FILES, errors)
    check_top_level_patterns(errors)
    check_data_process_scope(errors)
    check_demo32_shape_prior_carveout(errors)
    check_qqtt_exports(errors)
    check_readme_scope(errors)
    check_env_install(errors)
    check_single_camera_branch_policy(errors)

    if errors:
        print("Scope check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Scope check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
