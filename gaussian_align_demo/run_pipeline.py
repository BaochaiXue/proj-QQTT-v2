"""Orchestrate the full offline pipeline (subprocess per stage, two envs).

    python gaussian_align_demo/run_pipeline.py \
        --case-dir outputs/shape_prior_case/shape_prior_frame0 \
        --final-data outputs/data/final_data.pkl \
        --run-dir gaussian_align_demo/runs/myrun

Stages: input -> generate -> gallery -> [human picks a seed] -> align ->
refine -> animate. The pipeline STOPS after `gallery` unless the run already
has selected_seed.json or --seed is given — seed choice is deliberately a
human decision (inspect seed_gallery/seed_comparison_grid.mp4 first).

Generation runs in the TripoSplat conda env; everything else runs in the
current interpreter's env (launch this script from the demo env).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIPOSPLAT_PYTHON = "/home/xinjie/miniforge3/envs/triposplat/bin/python"
ALL_STAGES = ("input", "generate", "gallery", "align", "refine", "animate")


def run(command: list[str]) -> None:
    print(f"[pipeline] $ {' '.join(str(c) for c in command)}", flush=True)
    subprocess.run([str(c) for c in command], check=True, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--final-data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stages", default=",".join(ALL_STAGES),
                        help=f"comma list from {ALL_STAGES}")
    parser.add_argument("--seed", type=int, default=None,
                        help="skip the human stop by selecting this seed")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-gaussians", default="65536,262144")
    parser.add_argument("--triposplat-python", default=DEFAULT_TRIPOSPLAT_PYTHON)
    args = parser.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise SystemExit(f"unknown stages: {sorted(unknown)}")
    run_dir = Path(args.run_dir)
    python = sys.executable

    if "input" in stages:
        run([python, "-m", "gaussian_align_demo.prepare_input",
             "--case-dir", args.case_dir, "--run-dir", run_dir])
    if "generate" in stages:
        run([args.triposplat_python, "gaussian_align_demo/triposplat_driver.py",
             "--rgba", run_dir / "input" / "frame0_rgba.png",
             "--output-dir", run_dir / "seeds",
             "--seeds", args.seeds, "--num-gaussians", args.num_gaussians])
    if "gallery" in stages:
        run([python, "-m", "gaussian_align_demo.seed_gallery", "--run-dir", run_dir])

    selected = run_dir / "selected_seed.json"
    if args.seed is not None:
        run([python, "-m", "gaussian_align_demo.seed_gallery", "--run-dir", run_dir,
             "--select", str(args.seed)])
    remaining = [s for s in stages if s in ("align", "refine", "animate")]
    if remaining and not selected.exists():
        print(f"[pipeline] STOP: pick a seed first — watch "
              f"{run_dir}/seed_gallery/seed_comparison_grid.mp4 then run\n"
              f"  {python} -m gaussian_align_demo.seed_gallery --run-dir {run_dir} --select <N>\n"
              f"and re-run this pipeline with --stages {','.join(remaining)}")
        return 0

    if "align" in stages:
        ply = json.loads(selected.read_text())["selected_ply"]
        run([python, "-m", "gaussian_align_demo.align_gaussian",
             "--case-dir", args.case_dir, "--ply", ply, "--run-dir", run_dir])
    if "refine" in stages:
        run([python, "-m", "gaussian_align_demo.refine_alignment",
             "--case-dir", args.case_dir, "--run-dir", run_dir])
    if "animate" in stages:
        run([python, "-m", "gaussian_align_demo.animate_trajectory",
             "--run-dir", run_dir, "--case-dir", args.case_dir,
             "--final-data", args.final_data])
    print("[pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
