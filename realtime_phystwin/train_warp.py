from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import os
import pickle
import json
import warnings


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--segment_len", type=int, default=10)
    parser.add_argument("--segment_stride", type=int, default=10)
    parser.add_argument("--batch_vis_per_instance", action="store_true")
    parser.add_argument("--batch_vis_interval", type=int, default=50)
    parser.add_argument("--batch_vis_num_instances", type=int, default=1)
    parser.add_argument("--batch_vis_num_groups", type=int, default=1)
    parser.add_argument("--rollout_prefix_switch", action="store_true")
    parser.add_argument("--rollout_switch_start_iter", type=int, default=50)
    parser.add_argument("--rollout_switch_ramp_iters", type=int, default=100)
    parser.add_argument("--rollout_replace_thresh", type=float, default=0.03)
    parser.add_argument("--rollout_baseline_iters", type=int, default=5)
    parser.add_argument("--rollout_baseline_ratio", type=float, default=0.8)
    parser.add_argument("--rollout_check_len", type=int, default=5)
    parser.add_argument("--rollout_switch_log_interval", type=int, default=10)
    parser.add_argument("--batch_loss_weighting", action="store_true")
    parser.add_argument("--batch_loss_weight_min", type=float, default=0.5)
    parser.add_argument("--batch_loss_weight_max", type=float, default=2.0)
    parser.add_argument("--batch_loss_weight_log_interval", type=int, default=10)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame
    batch_mode = args.batch_mode
    batch_size = args.batch_size
    segment_len = args.segment_len
    segment_stride = args.segment_stride
    batch_vis_per_instance = args.batch_vis_per_instance
    batch_vis_interval = args.batch_vis_interval
    batch_vis_num_instances = args.batch_vis_num_instances
    batch_vis_num_groups = args.batch_vis_num_groups
    rollout_prefix_switch = args.rollout_prefix_switch
    rollout_switch_start_iter = args.rollout_switch_start_iter
    rollout_switch_ramp_iters = args.rollout_switch_ramp_iters
    rollout_replace_thresh = args.rollout_replace_thresh
    rollout_baseline_iters = args.rollout_baseline_iters
    rollout_baseline_ratio = args.rollout_baseline_ratio
    rollout_check_len = args.rollout_check_len
    rollout_switch_log_interval = args.rollout_switch_log_interval
    batch_loss_weighting = args.batch_loss_weighting
    batch_loss_weight_min = args.batch_loss_weight_min
    batch_loss_weight_max = args.batch_loss_weight_max
    batch_loss_weight_log_interval = args.batch_loss_weight_log_interval

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    print(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"experiments/{case_name}"
 
    # Read the first-satage optimized parameters
    # optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    # assert os.path.exists(
    #     optimal_path
    # ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    # with open(optimal_path, "rb") as f:
    #     optimal_params = pickle.load(f)
    # cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
        batch_mode=batch_mode,
        batch_size=batch_size,
        segment_len=segment_len,
        segment_stride=segment_stride,
        batch_vis_per_instance=batch_vis_per_instance,
        batch_vis_interval=batch_vis_interval,
        batch_vis_num_instances=batch_vis_num_instances,
        batch_vis_num_groups=batch_vis_num_groups,
        rollout_prefix_switch=rollout_prefix_switch,
        rollout_switch_start_iter=rollout_switch_start_iter,
        rollout_switch_ramp_iters=rollout_switch_ramp_iters,
        rollout_replace_thresh=rollout_replace_thresh,
        rollout_baseline_iters=rollout_baseline_iters,
        rollout_baseline_ratio=rollout_baseline_ratio,
        rollout_check_len=rollout_check_len,
        rollout_switch_log_interval=rollout_switch_log_interval,
        batch_loss_weighting=batch_loss_weighting,
        batch_loss_weight_min=batch_loss_weight_min,
        batch_loss_weight_max=batch_loss_weight_max,
        batch_loss_weight_log_interval=batch_loss_weight_log_interval,
    )
    trainer.train()
