import argparse
import csv
import glob
import json
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import torch


def checkpoint_sort_key(path):
    name = os.path.basename(path)
    match = re.search(r"_(\d+)\.pkl$", name)
    iter_idx = int(match.group(1)) if match else -1
    is_best = 1 if name.startswith("best_") else 0
    return iter_idx, is_best, name


def parse_iteration(path):
    name = os.path.basename(path)
    match = re.search(r"_(\d+)\.pkl$", name)
    if match:
        return int(match.group(1))
    return -1


def save_gif(image_paths, gif_path, duration_ms):
    if not image_paths:
        return

    try:
        from PIL import Image
    except ImportError:
        print("[WARN] Pillow is not installed; skip GIF output.")
        return

    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            frames.append(image.convert("P", palette=Image.ADAPTIVE))

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def find_cases(experiments_dir, data_dir, case_name):
    if case_name is not None:
        return [case_name]

    cases = []
    for case_dir in sorted(glob.glob(os.path.join(experiments_dir, "*"))):
        if not os.path.isdir(case_dir):
            continue
        case = os.path.basename(case_dir)
        data_path = os.path.join(data_dir, case, "final_data.pkl")
        has_final = os.path.exists(os.path.join(case_dir, "inference.pkl"))
        has_by_iter = bool(
            glob.glob(os.path.join(case_dir, "inference_by_iter", "*.pkl"))
        )
        if os.path.exists(data_path) and (has_final or has_by_iter):
            cases.append(case)
    return cases


def load_gt(data_dir, case_name):
    with open(os.path.join(data_dir, case_name, "final_data.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(data_dir, case_name, "split.json"), "r") as f:
        split = json.load(f)

    object_points = torch.tensor(data["object_points"], dtype=torch.float32)
    object_visibilities = torch.tensor(data["object_visibilities"], dtype=torch.bool)
    num_original_points = object_points.shape[1]
    num_surface_points = num_original_points + data["surface_points"].shape[0]

    return {
        "object_points": object_points,
        "object_visibilities": object_visibilities,
        "num_surface_points": num_surface_points,
        "train_frame": int(split["train"][1]),
        "test_frame": int(split["test"][1]),
    }


def load_vertices(path):
    with open(path, "rb") as f:
        vertices = pickle.load(f)
    if isinstance(vertices, torch.Tensor):
        return vertices.detach().cpu().float()
    return torch.tensor(vertices, dtype=torch.float32)


def single_direction_visible_chamfer_per_frame(vertices, gt):
    object_points = gt["object_points"]
    object_visibilities = gt["object_visibilities"]
    num_surface_points = gt["num_surface_points"]
    test_frame = min(int(vertices.shape[0]), gt["test_frame"], object_points.shape[0])

    rows = []
    with torch.no_grad():
        for frame_idx in range(1, test_frame):
            visible = object_visibilities[frame_idx]
            gt_points = object_points[frame_idx][visible]
            pred_points = vertices[frame_idx][:num_surface_points]

            if gt_points.numel() == 0 or pred_points.numel() == 0:
                chamfer_error = float("nan")
            else:
                # Same metric as evaluate_chamfer.py:
                # GT visible object points -> predicted surface, L1 nearest distance.
                dists = torch.cdist(gt_points.unsqueeze(0), pred_points.unsqueeze(0), p=1)
                chamfer_error = float(dists.min(dim=2).values.mean().item())

            split = "train" if frame_idx < gt["train_frame"] else "test"
            rows.append(
                {
                    "frame": frame_idx,
                    "split": split,
                    "chamfer_error": chamfer_error,
                }
            )
    return rows


def compute_ylim(series_list):
    values = []
    for series in series_list:
        values.extend(row["chamfer_error"] for row in series["rows"])
    values = np.array(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    y_min = float(values.min())
    y_max = float(values.max())
    if y_min == y_max:
        pad = max(abs(y_min) * 0.1, 1e-12)
    else:
        pad = (y_max - y_min) * 0.05
    return y_min - pad, y_max + pad


def plot_series(case_name, series, gt, out_path, y_lim):
    frames = [row["frame"] for row in series["rows"]]
    errors = [row["chamfer_error"] for row in series["rows"]]

    plt.figure(figsize=(12, 5))
    plt.plot(frames, errors, linewidth=1.8, label=series["label"])
    plt.axvline(gt["train_frame"], color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Frame")
    plt.ylabel("Eval Chamfer")
    plt.title(f"{case_name} | {series['label']}")
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_overlay(case_name, series_list, gt, out_path, y_lim):
    plt.figure(figsize=(12, 5))
    for series in series_list:
        frames = [row["frame"] for row in series["rows"]]
        errors = [row["chamfer_error"] for row in series["rows"]]
        alpha = 1.0 if series["source"] == "inference" else 0.35
        linewidth = 2.2 if series["source"] == "inference" else 1.0
        plt.plot(frames, errors, linewidth=linewidth, alpha=alpha, label=series["label"])

    plt.axvline(gt["train_frame"], color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Frame")
    plt.ylabel("Eval Chamfer")
    plt.title(f"{case_name} | eval Chamfer per frame")
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.grid(True, alpha=0.3)
    if len(series_list) <= 12:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_csv(case_name, series_list, out_path):
    with open(out_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "case_name",
                "source",
                "checkpoint",
                "iteration",
                "split",
                "frame",
                "chamfer_error",
            ]
        )
        for series in series_list:
            for row in series["rows"]:
                writer.writerow(
                    [
                        case_name,
                        series["source"],
                        series["checkpoint"],
                        series["iteration"],
                        row["split"],
                        row["frame"],
                        row["chamfer_error"],
                    ]
                )


def collect_series(args, case_name, gt):
    case_dir = os.path.join(args.experiments_dir, case_name)
    series_list = []

    if not args.no_inference:
        inference_path = os.path.join(case_dir, "inference.pkl")
        if os.path.exists(inference_path):
            rows = single_direction_visible_chamfer_per_frame(
                load_vertices(inference_path), gt
            )
            series_list.append(
                {
                    "source": "inference",
                    "checkpoint": "inference.pkl",
                    "iteration": -1,
                    "label": "final inference",
                    "rows": rows,
                }
            )

    if not args.no_by_iter:
        by_iter_paths = sorted(
            glob.glob(
                os.path.join(
                    case_dir,
                    "inference_by_iter",
                    args.checkpoint_pattern,
                )
            ),
            key=checkpoint_sort_key,
        )
        for path in by_iter_paths:
            iteration = parse_iteration(path)
            rows = single_direction_visible_chamfer_per_frame(load_vertices(path), gt)
            series_list.append(
                {
                    "source": "checkpoint",
                    "checkpoint": os.path.basename(path),
                    "iteration": iteration,
                    "label": os.path.splitext(os.path.basename(path))[0],
                    "rows": rows,
                }
            )

    return series_list


def plot_case(args, case_name):
    gt = load_gt(args.data_dir, case_name)
    series_list = collect_series(args, case_name, gt)
    if not series_list:
        print(f"[SKIP] {case_name}: no inference.pkl or inference_by_iter pkl found")
        return

    if args.out_dir is None:
        out_dir = os.path.join(
            args.experiments_dir,
            case_name,
            "eval_chamfer_per_frame",
        )
    else:
        out_dir = os.path.join(args.out_dir, case_name)
    os.makedirs(out_dir, exist_ok=True)

    y_lim = compute_ylim(series_list)
    csv_path = os.path.join(out_dir, "eval_chamfer_per_frame.csv")
    write_csv(case_name, series_list, csv_path)

    overlay_path = os.path.join(out_dir, "eval_chamfer_per_frame_overlay.png")
    plot_overlay(case_name, series_list, gt, overlay_path, y_lim)

    per_plot_paths = []
    for series in series_list:
        safe_name = os.path.splitext(series["checkpoint"])[0]
        plot_path = os.path.join(out_dir, f"{safe_name}_eval_chamfer_per_frame.png")
        plot_series(case_name, series, gt, plot_path, y_lim)
        if series["source"] == "checkpoint":
            per_plot_paths.append(plot_path)

    gif_path = os.path.join(out_dir, "checkpoint_eval_chamfer_per_frame.gif")
    save_gif(per_plot_paths, gif_path, args.gif_duration_ms)

    print(f"[OK] {case_name}: saved CSV to {csv_path}")
    print(f"[OK] {case_name}: saved plots to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--data_dir", type=str, default="data/different_types")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--checkpoint_pattern", type=str, default="*.pkl")
    parser.add_argument("--gif_duration_ms", type=int, default=250)
    parser.add_argument("--by_iter", action="store_true", help="Kept for compatibility.")
    parser.add_argument("--no_by_iter", action="store_true")
    parser.add_argument("--no_inference", action="store_true")
    args = parser.parse_args()

    cases = find_cases(args.experiments_dir, args.data_dir, args.case_name)
    if not cases:
        print(f"[WARN] no cases found under {args.experiments_dir}")
        return

    for case_name in cases:
        plot_case(args, case_name)


if __name__ == "__main__":
    main()
