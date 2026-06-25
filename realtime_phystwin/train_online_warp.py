from qqtt import InvPhyTrainerWarp
from qqtt.data import OnlineChunkReader, OnlineFrameBuffer
from qqtt.utils import logger, cfg

from argparse import ArgumentParser
from pathlib import Path
import json
import os
import pickle
import random
import time
import warnings

import numpy as np
import torch


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_WARP_SET_CTRL_WARN_PATTERN = (
    r".*set_control_points.*configured with the option 'enable_backward=False'.*"
)
try:
    from warp.utils import WarpUserWarning  # type: ignore

    warnings.filterwarnings(
        "ignore",
        message=_WARP_SET_CTRL_WARN_PATTERN,
        category=WarpUserWarning,
    )
except Exception:
    warnings.filterwarnings(
        "ignore",
        message=_WARP_SET_CTRL_WARN_PATTERN,
        category=Warning,
    )


def wait_for_initial_frames(reader, buffer, segment_len, poll_sec):
    reader.wait_for_manifest(poll_sec=poll_sec)
    while buffer.num_frames < segment_len:
        chunks = reader.load_new_chunks()
        if len(chunks) > 0:
            buffer.append_chunks(chunks)
            buffer.sync_to_device(cfg.device)

        if buffer.num_frames >= segment_len:
            break
        if reader.is_finished:
            raise RuntimeError(
                "Online stream finished before enough frames were available "
                f"for segment_len={segment_len}"
            )

        logger.info(
            "[Train-Online]: waiting for initial frames, "
            f"available={buffer.num_frames}, need={segment_len}"
        )
        time.sleep(float(poll_sec))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--online_base_path", type=str, default="online_data")
    parser.add_argument("--online_dir", type=str, default=None)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--experiments_dir", type=str, default="experiments_online")
    parser.add_argument("--static_data_path", type=str, default=None)
    parser.add_argument(
        "--optimal_params_path",
        type=str,
        default=None,
        help="Zero-order optimal_params.pkl loaded before trainer initialization.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train_frame", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--segment_len", type=int, default=32)
    parser.add_argument("--segment_stride", type=int, default=16)
    parser.add_argument("--poll_sec", type=float, default=1.0)
    parser.add_argument("--recent_window_count", type=int, default=8)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--stop_when_finished", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--realtime_vis", action="store_true")
    parser.add_argument("--realtime_vis_dir", type=str, default=None)
    parser.add_argument(
        "--realtime_vis_every",
        type=int,
        default=1,
        help="Export latest realtime predictions every N training iterations.",
    )
    parser.add_argument(
        "--no_realtime_iteration_history",
        action="store_true",
        help="Do not keep per-iteration realtime snapshots for the HTML viewer.",
    )
    parser.add_argument("--no_sample_recent", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    base_path = args.base_path
    case_name = args.case_name
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")
    if args.optimal_params_path is not None:
        if not os.path.exists(args.optimal_params_path):
            raise FileNotFoundError(
                f"Optimal parameter file not found: {args.optimal_params_path}"
            )
        with open(args.optimal_params_path, "rb") as f:
            optimal_params = dict(pickle.load(f))
        cfg.set_optimal_params(optimal_params)
        print(f"[OPTIMAL PARAMS]: {args.optimal_params_path}")
    if args.iterations is not None:
        cfg.iterations = int(args.iterations)

    cfg.device = args.device
    print(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"{args.experiments_dir}/{case_name}"
    os.makedirs(base_dir, exist_ok=True)
    realtime_vis_dir = (
        args.realtime_vis_dir
        if args.realtime_vis_dir is not None
        else f"{base_dir}/realtime"
    )

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        metadata = json.load(f)
    cfg.apply_camera_metadata(metadata)
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    online_dir = (
        Path(args.online_dir)
        if args.online_dir is not None
        else Path(args.online_base_path) / case_name
    )
    static_data_path = (
        args.static_data_path
        if args.static_data_path is not None
        else f"{base_path}/{case_name}/final_data.pkl"
    )

    logger.set_log_file(path=base_dir, name="online_inv_phy_log")
    logger.info(f"[Train-Online]: online_dir={online_dir}")
    logger.info(f"[Train-Online]: static_data_path={static_data_path}")
    if args.optimal_params_path is not None:
        logger.info(
            f"[Train-Online]: optimal_params_path={args.optimal_params_path}"
        )

    reader = OnlineChunkReader(online_dir=online_dir)
    buffer = OnlineFrameBuffer(static_data_path=static_data_path, device=cfg.device)
    wait_for_initial_frames(
        reader=reader,
        buffer=buffer,
        segment_len=int(args.segment_len),
        poll_sec=float(args.poll_sec),
    )

    trainer = InvPhyTrainerWarp(
        data_path=static_data_path,
        base_dir=base_dir,
        train_frame=args.train_frame,
        device=args.device,
        dataset_override=buffer,
        batch_mode=True,
        batch_size=args.batch_size,
        segment_len=args.segment_len,
        segment_stride=args.segment_stride,
    )
    trainer.train_online_batched(
        online_reader=reader,
        online_buffer=buffer,
        poll_sec=args.poll_sec,
        recent_window_count=args.recent_window_count,
        checkpoint_interval=args.checkpoint_interval,
        stop_when_finished=args.stop_when_finished,
        save_video=args.save_video,
        realtime_vis_dir=realtime_vis_dir if args.realtime_vis else None,
        realtime_vis_every=args.realtime_vis_every,
        realtime_keep_iterations=not args.no_realtime_iteration_history,
        sample_recent=not args.no_sample_recent,
    )
