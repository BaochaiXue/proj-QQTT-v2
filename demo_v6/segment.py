# Process to get the masks of the controller and the object
from argparse import ArgumentParser
from pathlib import Path
import shutil
import subprocess
import sys

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--TEXT_PROMPT", type=str, required=True)


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


def run_camera_segmentation(case_dir, case_name, text_prompt, camera_idx):
    video_path = case_dir / "color" / f"{camera_idx}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Color video not found: {video_path}")

    script_path = Path(__file__).resolve().with_name("segment_util_video.py")
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--base_path",
            str(case_dir.parent),
            "--case_name",
            case_name,
            "--TEXT_PROMPT",
            text_prompt,
            "--camera_idx",
            str(camera_idx),
        ],
        check=True,
    )


def main(argv=None):
    args = parser.parse_args(argv)
    base_path = args.base_path
    case_name = args.case_name
    case_dir = Path(base_path) / case_name
    camera_indices = discover_camera_indices(case_dir)

    print(f"Processing {case_name}")

    for camera_idx in camera_indices:
        print(f"Processing {case_name} camera {camera_idx}")
        try:
            run_camera_segmentation(
                case_dir,
                case_name,
                args.TEXT_PROMPT,
                camera_idx,
            )
        finally:
            tmp_data_dir = case_dir / "tmp_data"
            if tmp_data_dir.exists():
                shutil.rmtree(tmp_data_dir)


if __name__ == "__main__":
    main()
