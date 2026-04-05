import os
import glob

from pathlib import Path

_PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)


def _resolve_path(path: str) -> str:
    return str((_PROJECT_ROOT / path).resolve())



base_path = _resolve_path("./data/different_types")
output_path = _resolve_path("./data/different_types_gt_track")

def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

exist_dir(output_path)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}")

    exist_dir(f"{output_path}/{case_name}")
    # Copy the video to the output_path
    os.system(f"cp {dir_name}/color/0.mp4 {output_path}/{case_name}/0.mp4")

    

