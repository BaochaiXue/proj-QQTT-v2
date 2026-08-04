#!/usr/bin/env python3
"""Run FuturePhysTwin first-order optimization on the stitched Demo v6.2 case.

Mirrors ``~/FuturePhysTwin/train_warp.py`` (config loading, optimal-params
seeding, calibration/metadata wiring, ``InvPhyTrainerWarp.train()``) for a case
directory exported by ``export_stitched_case_for_futurephystwin.py``, then:

  1. renders the ground-truth tracking video (``gt.mp4``),
  2. runs full-sequence inference with the best checkpoint (``inference.mp4``),
  3. copies the visualization videos into ``demo_v6_2/others``.

Must run with the ``phystwin-max`` environment python. The stock code writes
videos with the ``avc1`` fourcc, which this machine's OpenCV/FFmpeg build cannot
open (the writer fails silently and no video is produced), so ``avc1`` is
remapped to ``mp4v`` before any FuturePhysTwin code renders.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FUTUREPHYSTWIN_ROOT = Path.home() / "FuturePhysTwin"
DEFAULT_CASE_NAME = "demo_v6_1_stitched_805"
DEFAULT_CASES_ROOT = REPO_ROOT / "demo_v6_2/others/futurephystwin_stitched/cases"
DEFAULT_VIDEOS_DIR = REPO_ROOT / "demo_v6_2/others"
SEED = 42


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="First-order optimization + visualization for the stitched case."
    )
    parser.add_argument(
        "--futurephystwin-root", type=Path, default=DEFAULT_FUTUREPHYSTWIN_ROOT
    )
    parser.add_argument("--case-name", type=str, default=DEFAULT_CASE_NAME)
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument(
        "--train-frame",
        type=int,
        default=None,
        help="Defaults to split.json train[1], matching script_train.py.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing checkpoints; only render inference/GT videos.",
    )
    return parser


def set_all_seeds(seed: int) -> None:
    """Set all seeds."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def patch_video_codec() -> None:
    """Patch video codec."""
    import cv2

    original_fourcc = cv2.VideoWriter_fourcc

    def fourcc_with_fallback(*chars: str) -> int:
        """Return the fourcc with fallback."""
        if "".join(chars) == "avc1":
            return original_fourcc(*"mp4v")
        return original_fourcc(*chars)

    cv2.VideoWriter_fourcc = fourcc_with_fallback


def find_best_checkpoint(base_dir: Path) -> Path:
    """Find best checkpoint."""
    candidates = sorted(glob.glob(str(base_dir / "train" / "best_*.pth")))
    matched = [
        path
        for path in candidates
        if re.fullmatch(r"best_(\d+)\.pth", os.path.basename(path))
    ]
    if len(matched) != 1:
        raise FileNotFoundError(
            f"expected exactly one best_<epoch>.pth under {base_dir}/train, "
            f"found {matched or candidates}"
        )
    return Path(matched[0])


def find_last_sim_video(base_dir: Path) -> Path | None:
    """Find last sim video."""
    best_iter = -1
    best_path: Path | None = None
    for path in glob.glob(str(base_dir / "train" / "sim_iter*.mp4")):
        match = re.fullmatch(r"sim_iter(\d+)\.mp4", os.path.basename(path))
        if match and int(match.group(1)) > best_iter:
            best_iter = int(match.group(1))
            best_path = Path(path)
    return best_path


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)

    futurephystwin_root = args.futurephystwin_root.resolve()
    case_dir = (args.cases_root / args.case_name).resolve()
    for required in ("final_data.pkl", "calibrate.pkl", "metadata.json", "split.json"):
        if not (case_dir / required).is_file():
            raise FileNotFoundError(case_dir / required)
    optimal_path = (
        futurephystwin_root
        / "experiments_optimization"
        / args.case_name
        / "optimal_params.pkl"
    )
    if not optimal_path.is_file():
        raise FileNotFoundError(
            f"{optimal_path} not found; run export_stitched_case_for_futurephystwin.py"
        )

    os.environ.setdefault("WANDB_MODE", "offline")
    # train_warp.py runs from the FuturePhysTwin root and uses relative paths
    # (experiments/, wandb/, gaussian_splatting assets), so mirror that here.
    os.chdir(futurephystwin_root)
    sys.path.insert(0, str(futurephystwin_root))

    patch_video_codec()
    set_all_seeds(SEED)

    import numpy as np
    import pickle

    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import cfg, logger

    if "cloth" in args.case_name or "package" in args.case_name:
        cfg.load_from_yaml(str(futurephystwin_root / "configs/cloth.yaml"))
    else:
        cfg.load_from_yaml(str(futurephystwin_root / "configs/real.yaml"))

    base_dir = futurephystwin_root / "experiments" / args.case_name

    with optimal_path.open("rb") as handle:
        optimal_params = pickle.load(handle)
    cfg.set_optimal_params(optimal_params)

    with (case_dir / "calibrate.pkl").open("rb") as handle:
        c2ws = pickle.load(handle)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws])
    metadata = json.loads((case_dir / "metadata.json").read_text())
    cfg.intrinsics = np.array(metadata["intrinsics"])
    cfg.WH = metadata["WH"]
    cfg.apply_case_timing_from_metadata(metadata)
    cfg.overlay_path = str(case_dir / "color")

    split = json.loads((case_dir / "split.json").read_text())
    train_frame = args.train_frame
    if train_frame is None:
        train_frame = int(split["train"][1])
    if not 1 < train_frame <= int(split["frame_len"]):
        raise ValueError(f"invalid train_frame {train_frame}")

    logger.set_log_file(path=str(base_dir), name="inv_phy_log")
    logger.info(
        f"[stitched-first-order] case={args.case_name} train_frame={train_frame} "
        f"FPS={cfg.FPS} num_substeps={cfg.num_substeps} dt={cfg.dt}"
    )

    started_s = time.time()
    trainer = InvPhyTrainerWarp(
        data_path=str(case_dir / "final_data.pkl"),
        base_dir=str(base_dir),
        train_frame=train_frame,
        device=args.device,
    )

    if not args.skip_train:
        trainer.train()
    train_done_s = time.time()

    # Ground-truth tracking video via FuturePhysTwin's own renderer.
    trainer.dataset.visualize_data(visualize=False, save_gt=True)

    # Full-sequence rollout with the best checkpoint -> inference.mp4/.pkl.
    best_checkpoint = find_best_checkpoint(base_dir)
    logger.info(f"[stitched-first-order] inference with {best_checkpoint}")
    trainer.test(model_path=str(best_checkpoint))
    finished_s = time.time()

    args.videos_dir.mkdir(parents=True, exist_ok=True)
    copies: dict[str, str] = {}
    video_sources = {
        f"{args.case_name}_gt.mp4": base_dir / "gt.mp4",
        f"{args.case_name}_train_init.mp4": base_dir / "train" / "init.mp4",
        f"{args.case_name}_first_order_inference.mp4": base_dir / "inference.mp4",
    }
    last_sim = find_last_sim_video(base_dir)
    if last_sim is not None:
        video_sources[f"{args.case_name}_train_{last_sim.stem}.mp4"] = last_sim
    for target_name, source in video_sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"expected video missing: {source}")
        shutil.copy2(source, args.videos_dir / target_name)
        copies[target_name] = str(source)

    summary = {
        "base_dir": str(base_dir),
        "best_checkpoint": str(best_checkpoint),
        "case_dir": str(case_dir),
        "case_name": args.case_name,
        "copied_videos": copies,
        "inference_pkl": str(base_dir / "inference.pkl"),
        "num_substeps": int(cfg.num_substeps),
        "seconds_train": round(train_done_s - started_s, 1),
        "seconds_visualize": round(finished_s - train_done_s, 1),
        "skip_train": bool(args.skip_train),
        "train_frame": train_frame,
        "videos_dir": str(args.videos_dir),
    }
    summary_path = args.videos_dir / f"{args.case_name}_first_order_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
