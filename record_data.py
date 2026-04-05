from datetime import datetime
from qqtt.env import CameraSystem
import os
from pathlib import Path
from shutil import copy2


_PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)


def _resolve_path(path: str) -> str:
    return str((_PROJECT_ROOT / path).resolve())

def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    camera_system = CameraSystem()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_collect_dir = _resolve_path("./data_collect")
    exist_dir(data_collect_dir)
    output_path = _resolve_path(f"./data_collect/{current_time}")
    camera_system.record(
        output_path=output_path
    )
    # Copy the camera calibration file to the output path
    copy2(_resolve_path("./calibrate.pkl"), output_path)
