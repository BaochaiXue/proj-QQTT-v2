from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAM 3.1 object masks for a QQTT case without running out of the local PhysTwin checkout."
    )
    parser.add_argument("--case_root", type=Path, required=True, help="Case directory with a color/ subtree.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write mask/ and summary.json. Defaults to <case_root>/sam31_masks.",
    )
    parser.add_argument("--text_prompt", type=str, required=True, help="Prompt string such as `sloth.hand`.")
    parser.add_argument("--camera_ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--source_mode", choices=("auto", "mp4", "frames"), default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional external SAM 3.1 checkpoint path. Otherwise use QQTT_SAM31_CHECKPOINT or HF cache.",
    )
    parser.add_argument("--ann_frame_index", type=int, default=0)
    parser.add_argument("--session_root", type=Path, default=None)
    parser.add_argument("--keep_session_frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--async_loading_frames", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--max_num_objects", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from scripts.harness.sam31_mask_helper import run_case_segmentation

    result = run_case_segmentation(
        case_root=args.case_root,
        text_prompt=args.text_prompt,
        camera_ids=args.camera_ids,
        output_dir=args.output_dir,
        source_mode=args.source_mode,
        checkpoint_path=args.checkpoint,
        ann_frame_index=args.ann_frame_index,
        keep_session_frames=args.keep_session_frames,
        session_root=args.session_root,
        overwrite=args.overwrite,
        async_loading_frames=args.async_loading_frames,
        compile_model=args.compile_model,
        max_num_objects=args.max_num_objects,
    )
    print(f"SAM 3.1 masks written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
