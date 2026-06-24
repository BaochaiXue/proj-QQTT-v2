import glob
import os
import json
import pickle

base_path = "./data/different_types"
dir_names = glob.glob(f"{base_path}/*")
segment_len = 35
segment_stride = 5
batch_debug_interval = 0

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
        f"python optimize_cma.py --base_path {base_path} "
        f"--case_name {case_name} --train_frame {train_frame} "
        f"--batch_mode --batch_size {num_segments} "
        f"--segment_len {case_segment_len} --segment_stride {segment_stride} "
        f"--batch_debug_interval {batch_debug_interval}"
    )
