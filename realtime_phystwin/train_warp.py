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
import csv
import time


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


def write_train_time_record(
    base_dir,
    case_name,
    start_time,
    end_time,
    elapsed_seconds,
    status,
    args,
    error_message="",
):
    train_dir = os.path.join(base_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    record = {
        "case_name": case_name,
        "status": status,
        "start_time": start_time.isoformat(timespec="seconds"),
        "end_time": end_time.isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "iterations": getattr(cfg, "iterations", None),
        "data_type": getattr(cfg, "data_type", None),
        "device": getattr(cfg, "device", None),
        "train_frame": args.train_frame,
        "error": error_message,
    }

    json_path = os.path.join(train_dir, "train_time.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    fieldnames = list(record.keys())
    case_csv_path = os.path.join(train_dir, "train_time.csv")
    with open(case_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(record)

    summary_path = os.path.join("experiments", "train_time_summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not summary_exists:
            writer.writeheader()
        writer.writerow(record)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame

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
    train_start_time = datetime.now()
    train_start_perf = time.perf_counter()
    train_status = "success"
    train_error = ""
    try:
        trainer = InvPhyTrainerWarp(
            data_path=f"{base_path}/{case_name}/final_data.pkl",
            base_dir=base_dir,
            train_frame=train_frame,
        )
        trainer.train()
    except Exception as exc:
        train_status = "failed"
        train_error = repr(exc)
        raise
    finally:
        train_end_time = datetime.now()
        train_elapsed = time.perf_counter() - train_start_perf
        write_train_time_record(
            base_dir=base_dir,
            case_name=case_name,
            start_time=train_start_time,
            end_time=train_end_time,
            elapsed_seconds=train_elapsed,
            status=train_status,
            args=args,
            error_message=train_error,
        )
        logger.info(
            "[Train-Time]: "
            f"case={case_name}, status={train_status}, "
            f"elapsed_seconds={train_elapsed:.2f}, "
            f"elapsed_minutes={train_elapsed / 60.0:.2f}"
        )
