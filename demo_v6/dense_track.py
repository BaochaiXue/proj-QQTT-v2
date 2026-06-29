# Use CoTracker to track the object and controller in the video.
# Pick 5000 pixels from the masked area as query points.

import torch
import imageio.v3 as iio
from utils.visualizer import Visualizer
import cv2
import numpy as np
import os
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
device = "cuda"


def discover_camera_indices(case_dir):
    depth_dir = case_dir / "depth"
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    camera_indices = sorted(
        int(path.name)
        for path in depth_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not camera_indices:
        raise ValueError(f"No camera depth directories found under {depth_dir}")
    return camera_indices


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    mask = mask > 0
    return mask


def read_first_frame_query_mask(mask_dir):
    mask_paths = sorted(mask_dir.glob("*/0.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No first-frame masks found under {mask_dir}")

    combined_mask = None
    for mask_path in mask_paths:
        current_mask = read_mask(mask_path)
        if combined_mask is None:
            combined_mask = current_mask
        else:
            combined_mask = np.logical_or(combined_mask, current_mask)

    if not np.any(combined_mask):
        raise ValueError(f"First-frame mask is empty under {mask_dir}")
    return combined_mask


def exist_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(argv=None):
    args = parser.parse_args(argv)
    base_path = args.base_path
    case_name = args.case_name
    case_dir = Path(base_path) / case_name
    camera_indices = discover_camera_indices(case_dir)

    cotracker_dir = case_dir / "cotracker"
    exist_dir(cotracker_dir)

    for i in camera_indices:
        print(f"Processing {i}th camera")
        # Load the video
        video_path = case_dir / "color" / f"{i}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Color video not found: {video_path}")
        frames = iio.imread(str(video_path), plugin="FFMPEG")
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T C H W
        # Load the first-frame mask to get all query points from all masks
        mask = read_first_frame_query_mask(case_dir / "mask" / str(i))

        # Draw the mask
        query_pixels = np.argwhere(mask)
        # Revert x and y
        query_pixels = query_pixels[:, ::-1]
        query_pixels = np.concatenate(
            [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
        )
        query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(device)
        # Randomly select 5000 query points
        query_pixels = query_pixels[torch.randperm(query_pixels.shape[0])[:5000]]

        # cotracker = torch.hub.load(
        #     "facebookresearch/co-tracker", "cotracker3_offline"
        # ).to(device)
        # pred_tracks, pred_visibility = cotracker(
        #     video, queries=query_pixels[None], backward_tracking=True
        # )
        # pred_tracks, pred_visibility = cotracker(video, grid_query_frame=0)

        # # Run Online CoTracker:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(device)
        cotracker(video_chunk=video, is_first_step=True, queries=query_pixels[None])

        # Process the video
        pred_tracks = None
        pred_visibility = None
        for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2]
            )  # B T N 2,  B T N 1
        if pred_tracks is None or pred_visibility is None:
            raise RuntimeError(
                f"CoTracker produced no tracking chunks for camera {i}."
            )

        vis = Visualizer(
            save_dir=str(cotracker_dir),
            pad_value=0,
            linewidth=3,
        )
        vis.visualize(video, pred_tracks, pred_visibility, filename=f"{i}")
        # Save the tracking data into npz
        track_to_save = pred_tracks[0].cpu().numpy()[:, :, ::-1]
        visibility_to_save = pred_visibility[0].cpu().numpy()
        np.savez(
            cotracker_dir / f"{i}.npz",
            tracks=track_to_save,
            visibility=visibility_to_save,
        )


if __name__ == "__main__":
    main()
