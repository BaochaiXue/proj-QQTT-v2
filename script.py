import os
import csv

from pathlib import Path

_PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)


def _resolve_path(path: str) -> str:
    return str((_PROJECT_ROOT / path).resolve())



base_path = _resolve_path("./data/more_clothes")

os.system("rm -f timer.log")

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]
        if shape_prior.lower() == "true":
            os.system(
                f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category} --shape_prior"
            )
        else:
            os.system(
                f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category}"
            )
