from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a single-frame Native-vs-FFS pointcloud board before and after SAM 3.1 masking."
    )
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--realsense_case", type=str, default=None)
    parser.add_argument("--ffs_case", type=str, default=None)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument(
        "--mask_source_mode",
        choices=("reuse_or_generate", "generate", "require_existing"),
        default="reuse_or_generate",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--source_mode", choices=("auto", "mp4", "frames"), default="auto")
    parser.add_argument("--ann_frame_index", type=int, default=None)
    parser.add_argument("--camera_ids", nargs="+", type=int, default=None)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--max_points_per_camera", type=int, default=None)
    parser.add_argument("--depth_min_m", type=float, default=0.0)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument(
        "--use_float_ffs_depth_when_available",
        dest="use_float_ffs_depth_when_available",
        action="store_true",
    )
    parser.add_argument(
        "--no_use_float_ffs_depth_when_available",
        dest="use_float_ffs_depth_when_available",
        action="store_false",
    )
    parser.set_defaults(use_float_ffs_depth_when_available=True)
    return parser.parse_args()


def _default_output_dir(
    *,
    aligned_root: Path,
    case_name: str | None,
    realsense_case: str | None,
    ffs_case: str | None,
    frame_idx: int,
) -> Path:
    if case_name:
        return aligned_root / case_name / f"masked_pointcloud_compare_frame_{int(frame_idx):04d}"
    return aligned_root / f"masked_pointcloud_compare_{realsense_case}_vs_{ffs_case}_frame_{int(frame_idx):04d}"


def _resolve_or_generate_mask_root(
    *,
    case_dir: Path,
    role_name: str,
    output_dir: Path,
    text_prompt: str,
    camera_ids: list[int],
    frame_idx: int,
    mask_source_mode: str,
    checkpoint: Path | None,
    source_mode: str,
) -> tuple[Path, str]:
    existing_root = case_dir / "sam31_masks"
    if mask_source_mode == "reuse_or_generate" and existing_root.is_dir():
        return existing_root.resolve(), "reused_existing"
    if mask_source_mode == "require_existing":
        if not existing_root.is_dir():
            raise FileNotFoundError(
                f"Missing existing sam31_masks for {role_name}: {existing_root}"
            )
        return existing_root.resolve(), "reused_existing"

    generated_root = output_dir / "_generated_masks" / role_name / "sam31_masks"
    from scripts.harness.sam31_mask_helper import run_case_segmentation

    result = run_case_segmentation(
        case_root=case_dir,
        text_prompt=text_prompt,
        camera_ids=camera_ids,
        output_dir=generated_root,
        source_mode=source_mode,
        checkpoint_path=checkpoint,
        ann_frame_index=int(frame_idx),
        keep_session_frames=False,
        session_root=None,
        overwrite=True,
        async_loading_frames=False,
        compile_model=False,
        max_num_objects=16,
    )
    return Path(result["output_dir"]).resolve(), "generated_now"


def resolve_mask_roots_for_compare(
    *,
    aligned_root: Path,
    case_name: str | None,
    realsense_case: str | None,
    ffs_case: str | None,
    output_dir: Path,
    text_prompt: str,
    camera_ids: list[int],
    frame_idx: int,
    mask_source_mode: str,
    checkpoint: Path | None,
    source_mode: str,
) -> dict[str, dict[str, Any]]:
    from data_process.visualization.io_case import resolve_case_dirs

    native_case_dir, ffs_case_dir, same_case_mode = resolve_case_dirs(
        aligned_root=Path(aligned_root).resolve(),
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
    )
    if same_case_mode:
        mask_root, source_kind = _resolve_or_generate_mask_root(
            case_dir=native_case_dir,
            role_name="shared",
            output_dir=output_dir,
            text_prompt=text_prompt,
            camera_ids=camera_ids,
            frame_idx=frame_idx,
            mask_source_mode=mask_source_mode,
            checkpoint=checkpoint,
            source_mode=source_mode,
        )
        return {
            "same_case_mode": True,
            "native": {"case_dir": native_case_dir, "mask_root": mask_root, "mask_source": source_kind},
            "ffs": {"case_dir": ffs_case_dir, "mask_root": mask_root, "mask_source": source_kind},
        }

    native_mask_root, native_source = _resolve_or_generate_mask_root(
        case_dir=native_case_dir,
        role_name="native",
        output_dir=output_dir,
        text_prompt=text_prompt,
        camera_ids=camera_ids,
        frame_idx=frame_idx,
        mask_source_mode=mask_source_mode,
        checkpoint=checkpoint,
        source_mode=source_mode,
    )
    ffs_mask_root, ffs_source = _resolve_or_generate_mask_root(
        case_dir=ffs_case_dir,
        role_name="ffs",
        output_dir=output_dir,
        text_prompt=text_prompt,
        camera_ids=camera_ids,
        frame_idx=frame_idx,
        mask_source_mode=mask_source_mode,
        checkpoint=checkpoint,
        source_mode=source_mode,
    )
    return {
        "same_case_mode": False,
        "native": {"case_dir": native_case_dir, "mask_root": native_mask_root, "mask_source": native_source},
        "ffs": {"case_dir": ffs_case_dir, "mask_root": ffs_mask_root, "mask_source": ffs_source},
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _default_output_dir(
            aligned_root=Path(args.aligned_root).resolve(),
            case_name=args.case_name,
            realsense_case=args.realsense_case,
            ffs_case=args.ffs_case,
            frame_idx=int(args.frame_idx),
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from data_process.visualization.workflows.masked_pointcloud_compare import (
        run_masked_pointcloud_compare_workflow,
    )

    from data_process.visualization.io_case import resolve_case_dirs

    native_case_dir, _, _ = resolve_case_dirs(
        aligned_root=Path(args.aligned_root).resolve(),
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
    )
    if args.camera_ids is None:
        inferred_camera_ids = sorted(int(path.name) for path in (native_case_dir / "color").iterdir() if path.is_dir())
    else:
        inferred_camera_ids = [int(item) for item in args.camera_ids]

    ann_frame_index = int(args.frame_idx) if args.ann_frame_index is None else int(args.ann_frame_index)
    mask_roots = resolve_mask_roots_for_compare(
        aligned_root=Path(args.aligned_root).resolve(),
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        output_dir=output_dir,
        text_prompt=args.text_prompt,
        camera_ids=inferred_camera_ids,
        frame_idx=ann_frame_index,
        mask_source_mode=str(args.mask_source_mode),
        checkpoint=args.checkpoint,
        source_mode=str(args.source_mode),
    )

    result = run_masked_pointcloud_compare_workflow(
        aligned_root=Path(args.aligned_root).resolve(),
        output_dir=output_dir,
        case_name=args.case_name,
        realsense_case=args.realsense_case,
        ffs_case=args.ffs_case,
        frame_idx=int(args.frame_idx),
        camera_ids=inferred_camera_ids,
        native_mask_root=mask_roots["native"]["mask_root"],
        ffs_mask_root=mask_roots["ffs"]["mask_root"],
        native_mask_source=str(mask_roots["native"]["mask_source"]),
        ffs_mask_source=str(mask_roots["ffs"]["mask_source"]),
        mask_source_mode=str(args.mask_source_mode),
        text_prompt=args.text_prompt,
        voxel_size=args.voxel_size,
        max_points_per_camera=args.max_points_per_camera,
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        use_float_ffs_depth_when_available=bool(args.use_float_ffs_depth_when_available),
    )
    print(f"Masked pointcloud compare outputs written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
