import os
import glob

from pathlib import Path

_PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)


def _resolve_path(path: str) -> str:
    return str((_PROJECT_ROOT / path).resolve())



base_path = _resolve_path("./experiments")
output_path = _resolve_path("./experiments_transfer")

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

existDir(output_path)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    existDir(f"{output_path}/{case_name}")
    os.system(
        f"cp {base_path}/{case_name}/inference.pkl {output_path}/{case_name}/inference.pkl"
    )
    os.system(
        f"cp {base_path}/{case_name}/inference.mp4 {output_path}/{case_name}/inference.mp4"
    )

