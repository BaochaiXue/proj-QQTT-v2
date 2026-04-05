import os
import glob
import csv

from pathlib import Path

_PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parent,
)


def _resolve_path(path: str) -> str:
    return str((_PROJECT_ROOT / path).resolve())



base_path = _resolve_path("./past_data_collect/more_clothes")
output_path = _resolve_path("./data/more_clothes")

# Read the csv
with open(f"clip_start_end.csv", "r") as f:
    reader = csv.reader(f)
    data = list(reader)

data = data[1:]
starts = {}
ends = {}
for row in data:
    case_name, start, end = row
    starts[case_name] = int(start)
    ends[case_name] = int(end)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}")

    os.system(f"python data_process/record_data_align.py --base_path {base_path} --case_name {case_name} --output_path {output_path} --start {starts[case_name]} --end {ends[case_name]}")

    