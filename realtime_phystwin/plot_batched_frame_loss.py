import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_case(csv_path, loss_name, gif_duration_ms):
    case_name = csv_path.split(os.sep)[1]
    out_dir = os.path.join(
        "experiments", case_name, "train", "per_frame_loss", "plots"
    )
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {
        "iteration",
        "global_window_id",
        "window_start",
        "global_frame",
        loss_name,
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[SKIP] {case_name}: missing columns {missing}")
        return

    iter0 = df[df["iteration"] == 0]
    if iter0.empty:
        print(f"[SKIP] {case_name}: iteration 0 not found")
        return

    # New convention:
    #   global_window_id == -1, iteration 0: before-training full rollout eval
    #   global_window_id >= 0, iteration >= 1: window training losses
    # Prefer the before-training full rollout to define the fixed y scale. This
    # avoids old iteration-0 window rows changing the visual scale.
    iter0_full = iter0[iter0["global_window_id"] == -1]
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

        # If this is a CSV generated before the iteration-renumbering change,
        # iteration 0 may still contain window rows. Hide those so iteration 0
        # always means "before training" in the plot.
        if int(iteration) == 0 and (df_iter["global_window_id"] == -1).any():
            df_iter = df_iter[df_iter["global_window_id"] == -1]

        plt.figure(figsize=(12, 6))

        for window_id in sorted(df_iter["global_window_id"].unique()):
            sub = df_iter[df_iter["global_window_id"] == window_id]
            sub = sub.sort_values("global_frame")

            window_start = int(sub["window_start"].iloc[0])
            if int(window_id) == -1:
                label = "full rollout eval"
                linewidth = 2.5
            else:
                label = f"train win {window_id} start {window_start}"
                linewidth = 1.5
            plt.plot(
                sub["global_frame"],
                sub[loss_name],
                label=label,
                linewidth=linewidth,
            )

        plt.xlabel("Global Frame")
        plt.ylabel(loss_name)
        plt.title(f"{case_name} | iter {iteration} | {loss_name}")
        plt.ylim(*y_lim)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        save_path = os.path.join(
            out_dir, f"iter_{int(iteration):04d}_{loss_name}.png"
        )
        plt.savefig(save_path, dpi=200)
        plt.close()
        saved_paths.append(save_path)

    gif_path = os.path.join(out_dir, f"{loss_name}_all_iterations.gif")
    save_gif(saved_paths, gif_path, gif_duration_ms)

    print(f"[OK] {case_name}: saved plots to {out_dir}")
    print(f"[OK] {case_name}: saved GIF to {gif_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loss_name",
        type=str,
        default="total_loss",
        choices=["total_loss", "chamfer_loss", "track_loss", "acc_loss"],
    )
    parser.add_argument("--gif_duration_ms", type=int, default=250)
    args = parser.parse_args()

    csv_paths = sorted(
        glob.glob("experiments/*/train/per_frame_loss/batched_frame_loss.csv")
    )

    if not csv_paths:
        print("[WARN] No batched_frame_loss.csv found.")
        return

    for csv_path in csv_paths:
        plot_case(csv_path, args.loss_name, args.gif_duration_ms)


if __name__ == "__main__":
    main()
