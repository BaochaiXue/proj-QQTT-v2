import glob
import os
import json
import pickle

base_path = "./data/different_types"
dir_names = glob.glob(f"{base_path}/*")
segment_len = 32
segment_stride = 30
batch_vis_interval = 50
batch_vis_num_instances = -1
batch_vis_num_groups = -1
rollout_switch_start_iter = 300
rollout_switch_ramp_iters = 300
rollout_replace_thresh = 0.00
rollout_baseline_iters = 5
rollout_baseline_ratio = 0.1
rollout_check_len = 5
rollout_switch_log_interval = 10
batch_loss_weighting = True
batch_loss_weight_min = 0.5
batch_loss_weight_max = 5.0
batch_loss_weight_log_interval = 10

for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
        data = pickle.load(f)

    T_obj = data["object_points"].shape[0]
    T_ctrl = data["controller_points"].shape[0]
    effective_train_frame = min(train_frame, T_obj, T_ctrl)
    case_segment_len = min(segment_len, effective_train_frame)
    if case_segment_len < 2:
        continue
    num_segments = (effective_train_frame - case_segment_len) // segment_stride + 1
    num_segments = max(1, num_segments)

    os.system(
        f"python train_warp.py --base_path {base_path} "
        f"--case_name {case_name} --train_frame {train_frame} "
        f"--batch_mode --batch_size {num_segments} "
        f"--segment_len {case_segment_len} --segment_stride {segment_stride} "
        f"--batch_vis_per_instance "
        f"--batch_vis_interval {batch_vis_interval} "
        f"--batch_vis_num_instances {batch_vis_num_instances} "
        f"--batch_vis_num_groups {batch_vis_num_groups} "
        f"--rollout_prefix_switch "
        f"--rollout_switch_start_iter {rollout_switch_start_iter} "
        f"--rollout_switch_ramp_iters {rollout_switch_ramp_iters} "
        f"--rollout_replace_thresh {rollout_replace_thresh} "
        f"--rollout_baseline_iters {rollout_baseline_iters} "
        f"--rollout_baseline_ratio {rollout_baseline_ratio} "
        f"--rollout_check_len {rollout_check_len} "
        f"--rollout_switch_log_interval {rollout_switch_log_interval} "
        f"{'--batch_loss_weighting ' if batch_loss_weighting else ''}"
        f"--batch_loss_weight_min {batch_loss_weight_min} "
        f"--batch_loss_weight_max {batch_loss_weight_max} "
        f"--batch_loss_weight_log_interval {batch_loss_weight_log_interval}"
    )
