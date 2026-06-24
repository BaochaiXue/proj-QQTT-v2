import argparse
import csv
import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import torch
from pytorch3d.loss import chamfer_distance


def compute_ylim(rows_list):
    values = []
    for rows in rows_list:
        for row in rows:
            value = row["chamfer_error"]
            if value == value:
                values.append(float(value))

    if not values:
        return None

    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        pad = max(abs(y_min) * 0.1, 1e-12)
    else:
        pad = (y_max - y_min) * 0.05
    return y_min - pad, y_max + pad


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


def load_eval_inputs(case_name, data_dir):
    data_path = os.path.join(data_dir, case_name, "final_data.pkl")
    split_path = os.path.join(data_dir, case_name, "split.json")

    if not os.path.exists(data_path):
        print(f"[SKIP] {case_name}: missing {data_path}")
        return None
    if not os.path.exists(split_path):
        print(f"[SKIP] {case_name}: missing {split_path}")
        return None

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(split_path, "r") as f:
        split = json.load(f)

    return data, split


def compute_rows(case_name, inference_path, data, split):
    with open(inference_path, "rb") as f:
        vertices = pickle.load(f)

    object_points = data["object_points"]
    object_visibilities = data["object_visibilities"]

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)

    num_original_points = object_points.shape[1]
    num_surface_points = num_original_points + data["surface_points"].shape[0]

    train_frame = int(split["train"][1])
    test_frame = int(split["test"][1])
    max_frame = min(test_frame, vertices.shape[0], object_points.shape[0])

    rows = []
    for frame_idx in range(1, max_frame):
        pred_surface = vertices[frame_idx, :num_surface_points]
        gt_visible = object_points[frame_idx][object_visibilities[frame_idx]]

        if gt_visible.shape[0] == 0:
            chamfer_value = float("nan")
        else:
            chamfer = chamfer_distance(
                gt_visible.unsqueeze(0),
                pred_surface.unsqueeze(0),
                single_directional=True,
                norm=1,
            )[0]
            chamfer_value = float(chamfer.item())

        split_name = "train" if frame_idx < train_frame else "test"
        rows.append(
            {
                "case_name": case_name,
                "frame": frame_idx,
                "split": split_name,
                "chamfer_error": chamfer_value,
            }
        )

    return rows, train_frame


def compute_per_frame_eval(case_name, experiments_dir, data_dir, out_dir):
    inference_path = os.path.join(experiments_dir, case_name, "inference.pkl")

    if not os.path.exists(inference_path):
        print(f"[SKIP] {case_name}: missing {inference_path}")
        return None, None

    eval_inputs = load_eval_inputs(case_name, data_dir)
    if eval_inputs is None:
        return None, None
    data, split = eval_inputs

    rows, train_frame = compute_rows(case_name, inference_path, data, split)

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{case_name}_eval_chamfer_per_frame.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_name", "frame", "split", "chamfer_error"]
        )
        writer.writeheader()
        writer.writerows(rows)

    png_path = os.path.join(out_dir, f"{case_name}_eval_chamfer_per_frame.png")
    plot_rows(case_name, rows, train_frame, png_path)

    return csv_path, png_path


def compute_by_iter_eval(case_name, experiments_dir, data_dir, out_dir, gif_duration_ms):
    inference_dir = os.path.join(experiments_dir, case_name, "inference_by_iter")
    inference_paths = sorted(glob.glob(os.path.join(inference_dir, "*.pkl")))
    if not inference_paths:
        print(f"[SKIP] {case_name}: no pkl found under {inference_dir}")
        return

    eval_inputs = load_eval_inputs(case_name, data_dir)
    if eval_inputs is None:
        return
    data, split = eval_inputs

    case_out_dir = os.path.join(out_dir, case_name)
    os.makedirs(case_out_dir, exist_ok=True)

    iter_results = []
    for inference_path in inference_paths:
        iter_name = os.path.splitext(os.path.basename(inference_path))[0]
        rows, train_frame = compute_rows(case_name, inference_path, data, split)
        iter_results.append((iter_name, rows, train_frame))

    y_lim = compute_ylim([rows for _iter_name, rows, _train_frame in iter_results])

    image_paths = []
    for iter_name, rows, train_frame in iter_results:
        csv_path = os.path.join(
            case_out_dir, f"{iter_name}_eval_chamfer_per_frame.csv"
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["case_name", "frame", "split", "chamfer_error"]
            )
            writer.writeheader()
            writer.writerows(rows)

        png_path = os.path.join(
            case_out_dir, f"{iter_name}_eval_chamfer_per_frame.png"
        )
        plot_rows(
            f"{case_name} | {iter_name}",
            rows,
            train_frame,
            png_path,
            y_lim=y_lim,
        )
        image_paths.append(png_path)

        print(f"[OK] {case_name} {iter_name}: saved CSV {csv_path}")
        print(f"[OK] {case_name} {iter_name}: saved plot {png_path}")

    gif_path = os.path.join(case_out_dir, "eval_chamfer_by_iter.gif")
    save_gif(image_paths, gif_path, gif_duration_ms)
    print(f"[OK] {case_name}: saved GIF {gif_path}")


def plot_rows(title, rows, train_frame, png_path, y_lim=None):
    frames = [r["frame"] for r in rows]
    values = [r["chamfer_error"] for r in rows]

    plt.figure(figsize=(12, 5))
    plt.plot(frames, values, linewidth=1.8, label="eval chamfer")
    plt.axvline(
        train_frame,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="train/test split",
    )
    plt.xlabel("Frame")
    plt.ylabel("Single-direction visible Chamfer")
    plt.title(title)
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def find_cases(experiments_dir, case_name=None):
    if case_name is not None:
        return [case_name]

    inference_paths = sorted(glob.glob(os.path.join(experiments_dir, "*", "inference.pkl")))
    by_iter_paths = sorted(
        glob.glob(os.path.join(experiments_dir, "*", "inference_by_iter", "*.pkl"))
    )
    cases = [os.path.basename(os.path.dirname(path)) for path in inference_paths]
    cases += [
        os.path.basename(os.path.dirname(os.path.dirname(path)))
        for path in by_iter_paths
    ]
    return sorted(set(cases))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--data_dir", type=str, default="data/different_types")
    parser.add_argument("--out_dir", type=str, default="results/per_frame_eval_chamfer")
    parser.add_argument("--by_iter", action="store_true")
    parser.add_argument("--gif_duration_ms", type=int, default=250)
    args = parser.parse_args()

    cases = find_cases(args.experiments_dir, args.case_name)
    if not cases:
        print(f"[WARN] no inference data found under {args.experiments_dir}")
        return

    for case_name in cases:
        if args.by_iter:
            compute_by_iter_eval(
                case_name=case_name,
                experiments_dir=args.experiments_dir,
                data_dir=args.data_dir,
                out_dir=args.out_dir,
                gif_duration_ms=args.gif_duration_ms,
            )
            continue

        csv_path, png_path = compute_per_frame_eval(
            case_name=case_name,
            experiments_dir=args.experiments_dir,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
        )
        if csv_path is not None:
            print(f"[OK] {case_name}: saved CSV {csv_path}")
            print(f"[OK] {case_name}: saved plot {png_path}")


if __name__ == "__main__":
    main()
