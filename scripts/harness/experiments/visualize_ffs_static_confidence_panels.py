from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FFS_REPO = Path("/home/zhangxinjie/Fast-FoundationStereo")
DEFAULT_MODEL_PATH = DEFAULT_FFS_REPO / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "static" / "ffs_confidence_panels_frame_0000_stuffed_animal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render masked 3x3 static-round FFS confidence boards for frame 0 using PyTorch Fast-FoundationStereo."
    )
    parser.add_argument("--aligned_root", type=Path, default=ROOT / "data")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffs_repo", type=Path, default=DEFAULT_FFS_REPO)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--depth_min_m", type=float, default=0.0)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--metrics", choices=("margin", "max_softmax", "both"), default="both")
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--text_prompt", type=str, default="stuffed animal")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from data_process.visualization.experiments.ffs_confidence_panels import (
        run_ffs_static_confidence_panels_workflow,
    )

    summary = run_ffs_static_confidence_panels_workflow(
        aligned_root=Path(args.aligned_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        ffs_repo=Path(args.ffs_repo).resolve(),
        model_path=Path(args.model_path).resolve(),
        scale=float(args.scale),
        valid_iters=int(args.valid_iters),
        max_disp=int(args.max_disp),
        depth_min_m=float(args.depth_min_m),
        depth_max_m=float(args.depth_max_m),
        metrics=str(args.metrics),
        frame_idx=int(args.frame_idx),
        text_prompt=str(args.text_prompt),
    )
    print(f"FFS static confidence outputs written to {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
