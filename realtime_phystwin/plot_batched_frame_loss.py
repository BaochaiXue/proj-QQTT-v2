import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def save_gif(image_paths, gif_path, duration_ms):
    if len(image_paths) == 0:
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


def infer_case_name(csv_path):
    parts = csv_path.split(os.sep)
    if len(parts) >= 2 and parts[0] == "experiments":
        return parts[1]
    return "unknown_case"


def infer_mode(csv_path):
    name = os.path.basename(csv_path)
    if name == "batched_frame_loss.csv":
        return "batched"
    return "single"


def plot_case(csv_path, loss_name, gif_duration_ms):
    case_name = infer_case_name(csv_path)
    mode = infer_mode(csv_path)
    out_dir = os.path.join(
        "experiments", case_name, "train", "per_frame_loss", "plots"
    )
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {
        "iteration",
        "window_start",
        "global_frame",
        loss_name,
    }
    missing = required_cols - set(df.columns)
    has_window_col = "global_window_id" in df.columns or "window_id" in df.columns
    if not has_window_col:
        missing.add("global_window_id/window_id")
    if missing:
        print(f"[SKIP] {case_name} ({mode}): missing columns {missing}")
        return

    iter0 = df[df["iteration"] == 0]
    if iter0.empty:
        print(f"[SKIP] {case_name} ({mode}): iteration 0 not found")
        return

    window_col = "global_window_id" if "global_window_id" in df.columns else "window_id"

    # New convention:
    #   window/global_window id -1: full-rollout eval checkpoint
    #   iteration 0: before training
    #   iteration >= 1: training-loop rows recorded while optimizing
    # Use the before-training full rollout for the fixed y-scale when available.
    iter0_full = iter0[iter0[window_col] == -1]
    y_source = iter0_full if not iter0_full.empty else iter0
    y_min = float(y_source[loss_name].min())
    y_max = float(y_source[loss_name].max())
    if y_min == y_max:
        pad = max(abs(y_min) * 0.1, 1e-12)
    else:
        pad = (y_max - y_min) * 0.05
    y_lim = (y_min - pad, y_max + pad)

    saved_paths = []

    for iteration in sorted(df["iteration"].unique()):
        df_iter = df[df["iteration"] == iteration]

        # If an old CSV still has training rows at iteration 0, hide them so
        # iteration 0 consistently means "before-training full rollout".
        if int(iteration) == 0 and (df_iter[window_col] == -1).any():
            df_iter = df_iter[df_iter[window_col] == -1]

        plt.figure(figsize=(12, 6))

        for window_id in sorted(df_iter[window_col].unique()):
            sub = df_iter[df_iter[window_col] == window_id]
            sub = sub.sort_values("global_frame")

            window_start = int(sub["window_start"].iloc[0])
            if int(window_id) == -1:
                label = "full rollout eval"
                linewidth = 2.5
            else:
                label = f"train win {int(window_id)} start {window_start}"
                linewidth = 1.5
            plt.plot(
                sub["global_frame"],
                sub[loss_name],
                label=label,
                linewidth=linewidth,
            )

        plt.xlabel("Global Frame")
        plt.ylabel(loss_name)
        plt.title(f"{case_name} | {mode} | iter {iteration} | {loss_name}")
        plt.ylim(*y_lim)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        save_path = os.path.join(
            out_dir, f"{mode}_iter_{int(iteration):04d}_{loss_name}.png"
        )
        plt.savefig(save_path, dpi=200)
        plt.close()
        saved_paths.append(save_path)

    gif_path = os.path.join(out_dir, f"{mode}_{loss_name}_all_iterations.gif")
    save_gif(saved_paths, gif_path, gif_duration_ms)

    print(f"[OK] {case_name} ({mode}): saved plots to {out_dir}")
    print(f"[OK] {case_name} ({mode}): saved GIF to {gif_path}")


def find_csv_paths(case_name):
    if case_name is not None:
        patterns = [
            f"experiments/{case_name}/train/per_frame_loss/batched_frame_loss.csv",
            f"experiments/{case_name}/train/per_frame_loss/frame_loss.csv",
        ]
    else:
        patterns = [
            "experiments/*/train/per_frame_loss/batched_frame_loss.csv",
            "experiments/*/train/per_frame_loss/frame_loss.csv",
        ]

    csv_paths = []
    for pattern in patterns:
        csv_paths.extend(glob.glob(pattern))
    return sorted(set(csv_paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument(
        "--loss_name",
        type=str,
        default="total_loss",
        choices=["total_loss", "chamfer_loss", "track_loss", "acc_loss"],
    )
    parser.add_argument("--gif_duration_ms", type=int, default=250)
    args = parser.parse_args()

    csv_paths = find_csv_paths(args.case_name)
    if not csv_paths:
        print("[WARN] No per-frame loss CSV found.")
        return

    for csv_path in csv_paths:
        plot_case(csv_path, args.loss_name, args.gif_duration_ms)


if __name__ == "__main__":
    main()
