from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.depth_backends.ffs_defaults import (
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_MODEL_PATH,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_SCALE,
    DEFAULT_FFS_VALID_ITERS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FFS normal-vs-swapped left/right ordering on one aligned stereo pair.")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--ffs_case", type=str, default=None)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--ffs_repo", type=Path, default=DEFAULT_FFS_REPO)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_FFS_MODEL_PATH)
    parser.add_argument("--scale", type=float, default=DEFAULT_FFS_SCALE)
    parser.add_argument("--valid_iters", type=int, default=DEFAULT_FFS_VALID_ITERS)
    parser.add_argument("--max_disp", type=int, default=DEFAULT_FFS_MAX_DISP)
    parser.add_argument("--face_patches_json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from data_process.visualization.stereo_audit import run_ffs_left_right_audit_workflow

    output_dir = args.output_dir
    if output_dir is None:
        case_token = args.case_name if args.case_name is not None else args.ffs_case
        output_dir = args.aligned_root / f"left_right_audit_{case_token}_cam{args.camera_idx}_frame_{args.frame_idx:04d}"

    result = run_ffs_left_right_audit_workflow(
        aligned_root=args.aligned_root,
        output_dir=output_dir,
        case_name=args.case_name,
        ffs_case=args.ffs_case,
        frame_idx=args.frame_idx,
        camera_idx=args.camera_idx,
        ffs_repo=args.ffs_repo,
        model_path=args.model_path,
        scale=args.scale,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        face_patches_json=args.face_patches_json,
    )
    print(f"Left/right audit written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
