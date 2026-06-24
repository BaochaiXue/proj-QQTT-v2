#!/usr/bin/env python3
import re
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
TRAIN_RE = re.compile(rf"Iteration:\s*(\d+)\s*,\s*Loss:\s*({FLOAT_RE})")
OPT_RE = re.compile(rf"\[\s*INFO\]\s+(\d+)\s+(\d+)\s+({FLOAT_RE})")


def parse_train_log(path: Path):
    # 用 dict 去重：如果同一 iteration 出现多次，保留最后一次
    data = {}
    if not path.exists():
        return [], []
    for line in path.read_text(errors="ignore").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            it = int(m.group(1))
            loss = float(m.group(2))
            data[it] = loss
    xs = sorted(data.keys())
    ys = [data[x] for x in xs]
    return xs, ys


def parse_opt_log(path: Path):
    # CMA 日志格式：iter fevals function_value ...
    data = {}
    if not path.exists():
        return [], []
    for line in path.read_text(errors="ignore").splitlines():
        m = OPT_RE.search(line)
        if m:
            it = int(m.group(1))
            loss = float(m.group(3))
            data[it] = loss
    xs = sorted(data.keys())
    ys = [data[x] for x in xs]
    return xs, ys


def collect_cases(train_root: Path, opt_root: Path):
    cases = set()
    for p in train_root.glob("*/inv_phy_log.log"):
        cases.add(p.parent.name)
    for p in opt_root.glob("*/optimize_cma_log.log"):
        cases.add(p.parent.name)
    return sorted(cases)


def draw_case(case_name: str, train_root: Path, opt_root: Path, out_dir: Path):
    train_log = train_root / case_name / "inv_phy_log.log"
    opt_log = opt_root / case_name / "optimize_cma_log.log"

    tx, ty = parse_train_log(train_log)
    ox, oy = parse_opt_log(opt_log)

    if not tx and not ox:
        return False

    ncols = 2 if tx and ox else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if tx:
        axes[idx].plot(tx, ty, color="tab:blue", linewidth=1.8)
        axes[idx].set_title(f"{case_name} - Train Loss")
        axes[idx].set_xlabel("Iteration")
        axes[idx].set_ylabel("Loss")
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    if ox:
        axes[idx].plot(ox, oy, color="tab:orange", linewidth=1.8)
        axes[idx].set_title(f"{case_name} - CMA Loss")
        axes[idx].set_xlabel("CMA Iteration")
        axes[idx].set_ylabel("Function Value")
        axes[idx].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{case_name}_loss.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="experiments")
    parser.add_argument("--opt_root", type=str, default="experiments_optimization")
    parser.add_argument("--out_dir", type=str, default="loss_curves")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="指定 case 名称列表；不填则自动处理全部 case",
    )
    args = parser.parse_args()

    train_root = Path(args.train_root)
    opt_root = Path(args.opt_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        cases = args.cases
    else:
        cases = collect_cases(train_root, opt_root)

    if not cases:
        print("No cases found.")
        return

    saved = 0
    for case_name in cases:
        ok = draw_case(case_name, train_root, opt_root, out_dir)
        if ok:
            saved += 1

    print(f"Done. {saved}/{len(cases)} case figures saved to: {out_dir}")


if __name__ == "__main__":
    main()

