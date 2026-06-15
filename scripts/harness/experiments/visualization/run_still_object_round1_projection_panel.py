from __future__ import annotations

import argparse
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.aligned_case_metadata import write_split_aligned_metadata
from data_process.depth_backends import (
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_MODEL_NAME,
    DEFAULT_FFS_MODEL_PATH,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_SCALE,
    DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL,
    DEFAULT_FFS_TRT_ENGINE_SIZE,
    DEFAULT_FFS_TRT_INPUT_SIZE,
    DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
    DEFAULT_FFS_VALID_ITERS,
    FastFoundationStereoTensorRTRunner,
)


DEFAULT_RAW_CASE_NAME = "both_30_still_object_round1_20260428"
DEFAULT_OUTPUT_ROOT = ROOT / "result" / "still_object_round1_projection_panel_13x3_ffs203048_iter4_trt_level5"


class _DefaultTwoStageTrtRunner:
    def __init__(
        self,
        *,
        ffs_repo: str | Path,
        model_path: str | Path,
        scale: float,
        valid_iters: int,
        max_disp: int,
    ) -> None:
        self.runner = FastFoundationStereoTensorRTRunner(
            ffs_repo=ffs_repo,
            model_dir=DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
        )

    def run_pair(self, *args, **kwargs):
        return self.runner.run_pair(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align still-object round 1 with the default FFS TensorRT setting, "
            "generate/reuse masks, and render the requested 13x3 projection panel."
        )
    )
    parser.add_argument("--base_path", type=Path, default=ROOT / "data_collect")
    parser.add_argument("--raw_case_name", type=str, default=DEFAULT_RAW_CASE_NAME)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", type=int, default=136)
    parser.add_argument("--end", type=int, default=190)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--text_prompt", type=str, default="stuffed animal")
    parser.add_argument("--mask_source_mode", choices=("reuse_or_generate", "generate", "require_existing"), default="reuse_or_generate")
    parser.add_argument("--sam_checkpoint", type=Path, default=None)
    parser.add_argument("--source_mode", choices=("auto", "mp4", "frames"), default="auto")
    parser.add_argument("--force_align", action="store_true")
    parser.add_argument("--force_masks", action="store_true")
    parser.add_argument("--force_panel", action="store_true")
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--tile_width", type=int, default=480)
    parser.add_argument("--tile_height", type=int, default=360)
    parser.add_argument("--row_label_width", type=int, default=390)
    parser.add_argument(
        "--max_points_per_camera",
        type=int,
        default=0,
        help="Deterministic sampling cap before fusion; <=0 keeps all valid masked points.",
    )
    parser.add_argument("--phystwin_radius_m", type=float, default=0.01)
    parser.add_argument("--phystwin_nb_points", type=int, default=40)
    parser.add_argument("--enhanced_component_voxel_size_m", type=float, default=0.01)
    parser.add_argument("--enhanced_keep_near_main_gap_m", type=float, default=0.0)
    parser.add_argument("--highlight_alpha", type=float, default=0.70)
    parser.add_argument("--highlight_radius_px", type=int, default=2)
    parser.add_argument("--projection_point_radius_px", type=int, default=1)
    return parser.parse_args()


def _align_case_with_default_trt(args: argparse.Namespace, *, aligned_root: Path) -> Path:
    from data_process.record_data_align import align_case

    output_case_dir = aligned_root / str(args.raw_case_name)
    if args.force_align and output_case_dir.exists():
        shutil.rmtree(output_case_dir)
    if output_case_dir.exists():
        return output_case_dir.resolve()

    aligned_root.mkdir(parents=True, exist_ok=True)
    align_args = Namespace(
        base_path=Path(args.base_path).resolve(),
        case_name=str(args.raw_case_name),
        output_path=aligned_root.resolve(),
        start=int(args.start),
        end=int(args.end),
        fps=None,
        write_mp4=False,
        depth_backend="both",
        ffs_repo=str(DEFAULT_FFS_REPO),
        ffs_model_path=str(DEFAULT_FFS_MODEL_PATH),
        ffs_scale=float(DEFAULT_FFS_SCALE),
        ffs_valid_iters=int(DEFAULT_FFS_VALID_ITERS),
        ffs_max_disp=int(DEFAULT_FFS_MAX_DISP),
        ffs_radius_outlier_filter=False,
        ffs_radius_outlier_radius_m=0.01,
        ffs_radius_outlier_nb_points=40,
        ffs_native_like_postprocess=False,
        ffs_confidence_mode="none",
        ffs_confidence_threshold=0.0,
        ffs_confidence_depth_min_m=0.2,
        ffs_confidence_depth_max_m=1.5,
        write_ffs_confidence_debug=False,
        write_ffs_valid_mask_debug=False,
        write_ffs_float_m=True,
        fail_if_no_ir_stereo=True,
    )
    metadata = align_case(align_args, runner_factory=_DefaultTwoStageTrtRunner)
    ffs_config = dict(metadata.get("ffs_config", {}))
    ffs_config.update(
        {
            "backend": "tensorrt",
            "trt_mode": "two_stage",
            "model_name": DEFAULT_FFS_MODEL_NAME,
            "model_path": str(DEFAULT_FFS_MODEL_PATH),
            "trt_model_dir": str(DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR),
            "builder_optimization_level": int(DEFAULT_FFS_TRT_BUILDER_OPTIMIZATION_LEVEL),
            "trt_input_size_hw": list(DEFAULT_FFS_TRT_INPUT_SIZE),
            "trt_engine_size_hw": list(DEFAULT_FFS_TRT_ENGINE_SIZE),
            "input_policy": "keep_848x480_and_pad_to_864x480_no_resize_down",
        }
    )
    metadata["ffs_config"] = ffs_config
    write_split_aligned_metadata(output_case_dir, metadata)
    return output_case_dir.resolve()


def _camera_mask_has_frame(mask_root: Path, *, camera_idx: int, frame_idx: int) -> bool:
    camera_root = mask_root / "mask" / str(int(camera_idx))
    info_path = mask_root / "mask" / f"mask_info_{int(camera_idx)}.json"
    if not info_path.is_file() or not camera_root.is_dir():
        return False
    return any(path.is_file() for path in camera_root.glob(f"*/{int(frame_idx)}.png"))


def _write_mask_completion(mask_root: Path, *, camera_ids: list[int], frame_idx: int) -> None:
    completion = {
        "frame_idx": int(frame_idx),
        "camera_ids": [int(item) for item in camera_ids],
        "complete_camera_ids": [
            int(camera_idx)
            for camera_idx in camera_ids
            if _camera_mask_has_frame(mask_root, camera_idx=int(camera_idx), frame_idx=int(frame_idx))
        ],
    }
    completion["missing_camera_ids"] = [
        int(camera_idx)
        for camera_idx in camera_ids
        if camera_idx not in set(completion["complete_camera_ids"])
    ]
    with (mask_root / "mask_completion.json").open("w", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2)


def _resolve_or_generate_masks(args: argparse.Namespace, *, case_dir: Path, output_root: Path) -> Path:
    mask_root = output_root / "masks" / "sam31_masks"
    if args.force_masks and mask_root.exists():
        shutil.rmtree(mask_root)
    camera_ids = [0, 1, 2]
    missing_camera_ids = [
        int(camera_idx)
        for camera_idx in camera_ids
        if not _camera_mask_has_frame(mask_root, camera_idx=int(camera_idx), frame_idx=int(args.frame_idx))
    ]
    if args.mask_source_mode in {"reuse_or_generate", "require_existing"} and not missing_camera_ids:
        _write_mask_completion(mask_root, camera_ids=camera_ids, frame_idx=int(args.frame_idx))
        return mask_root.resolve()
    if args.mask_source_mode == "require_existing":
        raise FileNotFoundError(f"Missing required masks for cameras {missing_camera_ids}: {mask_root}")

    from scripts.harness.sam31_mask_helper import run_case_segmentation

    target_camera_ids = camera_ids if args.mask_source_mode == "generate" else missing_camera_ids
    result = run_case_segmentation(
        case_root=case_dir,
        text_prompt=str(args.text_prompt),
        camera_ids=target_camera_ids,
        output_dir=mask_root,
        source_mode=str(args.source_mode),
        checkpoint_path=args.sam_checkpoint,
        ann_frame_index=int(args.frame_idx),
        keep_session_frames=False,
        session_root=None,
        overwrite=True,
        async_loading_frames=False,
        compile_model=False,
        max_num_objects=16,
    )
    _write_mask_completion(mask_root, camera_ids=camera_ids, frame_idx=int(args.frame_idx))
    return Path(result["output_dir"]).resolve()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    aligned_root = output_root / "aligned"
    panel_root = output_root / "panel"
    if args.force_panel and panel_root.exists():
        shutil.rmtree(panel_root)

    case_dir = _align_case_with_default_trt(args, aligned_root=aligned_root)
    mask_root = _resolve_or_generate_masks(args, case_dir=case_dir, output_root=output_root)

    from data_process.visualization.experiments.still_object_projection_panel import (
        run_still_object_projection_panel_workflow,
    )

    summary = run_still_object_projection_panel_workflow(
        aligned_root=aligned_root,
        case_ref=str(args.raw_case_name),
        mask_root=mask_root,
        output_root=panel_root,
        frame_idx=int(args.frame_idx),
        text_prompt=str(args.text_prompt),
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        tile_width=int(args.tile_width),
        tile_height=int(args.tile_height),
        row_label_width=int(args.row_label_width),
        max_points_per_camera=None if int(args.max_points_per_camera) <= 0 else int(args.max_points_per_camera),
        phystwin_radius_m=float(args.phystwin_radius_m),
        phystwin_nb_points=int(args.phystwin_nb_points),
        enhanced_component_voxel_size_m=float(args.enhanced_component_voxel_size_m),
        enhanced_keep_near_main_gap_m=float(args.enhanced_keep_near_main_gap_m),
        highlight_alpha=float(args.highlight_alpha),
        highlight_radius_px=int(args.highlight_radius_px),
        projection_point_radius_px=int(args.projection_point_radius_px),
    )
    print(f"Aligned case: {case_dir}")
    print(f"Mask root: {mask_root}")
    print(f"13x3 panel: {summary['board_path']}")
    print(f"Summary: {panel_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
