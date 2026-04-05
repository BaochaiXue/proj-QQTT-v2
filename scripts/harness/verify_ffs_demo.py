from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FFS_REPO = Path(r"C:\Users\zhang\external\Fast-FoundationStereo")
DEFAULT_MODEL_PATH = DEFAULT_FFS_REPO / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
DEFAULT_DOC_PATH = ROOT / "docs" / "generated" / "ffs_demo_validation.md"
DEFAULT_OUT_DIR = ROOT / "data" / "ffs_proof_of_life" / "official_demo"


WRAPPER_SOURCE = r"""
from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path


def _patch_os_system() -> None:
    original_system = os.system

    def patched(command: str) -> int:
        parts = [part.strip() for part in command.split("&&")]
        if len(parts) == 2 and parts[0].startswith("rm -rf ") and parts[1].startswith("mkdir -p "):
            rm_target = Path(parts[0][7:].strip())
            mkdir_target = Path(parts[1][8:].strip())
            shutil.rmtree(rm_target, ignore_errors=True)
            mkdir_target.mkdir(parents=True, exist_ok=True)
            return 0
        return original_system(command)

    os.system = patched


def _patch_cv2() -> None:
    import cv2

    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: 0
    cv2.destroyAllWindows = lambda *args, **kwargs: None


def _patch_torch_compile() -> None:
    import torch

    def identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(inner):
                return inner
            return decorator
        return fn

    torch.compile = identity_compile


def _patch_open3d() -> None:
    try:
        import open3d as o3d
    except Exception:
        return

    class DummyRenderOption:
        pass

    class DummyViewControl:
        def set_front(self, *args, **kwargs) -> None:
            return None

        def set_lookat(self, *args, **kwargs) -> None:
            return None

        def set_up(self, *args, **kwargs) -> None:
            return None

    class DummyVisualizer:
        def __init__(self) -> None:
            self._render_option = DummyRenderOption()
            self._view_control = DummyViewControl()

        def create_window(self, *args, **kwargs) -> bool:
            return True

        def add_geometry(self, *args, **kwargs) -> bool:
            return True

        def get_render_option(self) -> DummyRenderOption:
            return self._render_option

        def get_view_control(self) -> DummyViewControl:
            return self._view_control

        def run(self) -> None:
            return None

        def destroy_window(self) -> None:
            return None

    o3d.visualization.Visualizer = DummyVisualizer


def main() -> int:
    run_demo_path = Path(sys.argv[1]).resolve()
    forwarded_args = sys.argv[2:]
    _patch_os_system()
    _patch_cv2()
    _patch_torch_compile()
    _patch_open3d()
    sys.argv = [str(run_demo_path)] + forwarded_args
    runpy.run_path(str(run_demo_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the official Fast-FoundationStereo demo.")
    parser.add_argument("--ffs_repo", default=str(DEFAULT_FFS_REPO))
    parser.add_argument("--env_name", default="ffs-standalone")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--doc_path", default=str(DEFAULT_DOC_PATH))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--get_pc", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def resolve_env_python(env_name: str) -> Path:
    base = subprocess.run(
        ["conda", "info", "--base"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(base) / "envs" / env_name / "python.exe"


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def write_report(
    *,
    doc_path: Path,
    env_name: str,
    execution_mode: str,
    command: list[str],
    result: subprocess.CompletedProcess[str],
    out_dir: Path,
    verified: list[str],
) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fast-FoundationStereo Demo Validation",
        "",
        f"- Environment: `{env_name}`",
        f"- Execution mode: `{execution_mode}`",
        f"- Exit code: `{result.returncode}`",
        f"- Output directory: `{out_dir}`",
        "",
        "## Command",
        "",
        "```text",
        " ".join(command),
        "```",
        "",
        "## Verified Outputs",
        "",
    ]
    if verified:
        lines.extend([f"- `{item}`" for item in verified])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Stdout",
            "",
            "```text",
            result.stdout.rstrip(),
            "```",
            "",
            "## Stderr",
            "",
            "```text",
            result.stderr.rstrip(),
            "```",
            "",
        ]
    )
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ffs_repo = Path(args.ffs_repo).resolve()
    model_path = Path(args.model_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    doc_path = Path(args.doc_path).resolve()
    run_demo_path = ffs_repo / "scripts" / "run_demo.py"
    left_file = ffs_repo / "demo_data" / "left.png"
    right_file = ffs_repo / "demo_data" / "right.png"
    intrinsic_file = ffs_repo / "demo_data" / "K.txt"

    missing = [
        str(path)
        for path in (run_demo_path, model_path, left_file, right_file, intrinsic_file)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required FFS paths: {missing}")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wrapper_path = Path(tmp_dir) / "run_demo_wrapper.py"
        wrapper_path.write_text(WRAPPER_SOURCE, encoding="utf-8")

        forwarded_args = [
            str(run_demo_path),
            "--model_dir",
            str(model_path),
            "--left_file",
            str(left_file),
            "--right_file",
            str(right_file),
            "--intrinsic_file",
            str(intrinsic_file),
            "--out_dir",
            str(out_dir),
            "--scale",
            str(args.scale),
            "--valid_iters",
            str(args.valid_iters),
            "--max_disp",
            str(args.max_disp),
            "--get_pc",
            str(args.get_pc),
            "--remove_invisible",
            "0",
            "--denoise_cloud",
            "0",
        ]

        conda_command = ["conda", "run", "-n", args.env_name, "python", str(wrapper_path), *forwarded_args]
        result = run_command(conda_command, cwd=ffs_repo)
        execution_mode = "conda-run"
        command_used = conda_command

        if result.returncode != 0:
            env_python = resolve_env_python(args.env_name)
            direct_command = [str(env_python), str(wrapper_path), *forwarded_args]
            direct_result = run_command(direct_command, cwd=ffs_repo)
            if direct_result.returncode == 0:
                result = direct_result
                execution_mode = "direct-env-python-fallback"
                command_used = direct_command
            else:
                write_report(
                    doc_path=doc_path,
                    env_name=args.env_name,
                    execution_mode="conda-run-and-direct-fallback-failed",
                    command=direct_command,
                    result=direct_result,
                    out_dir=out_dir,
                    verified=[],
                )
                raise SystemExit(direct_result.returncode)

    verified = []
    for required_name in ("disp_vis.png", "depth_meter.npy"):
        if not (out_dir / required_name).is_file():
            raise FileNotFoundError(f"Official demo missing expected output: {required_name}")
        verified.append(required_name)
    if args.get_pc:
        if not (out_dir / "cloud.ply").is_file():
            raise FileNotFoundError("Official demo missing expected output: cloud.ply")
        verified.append("cloud.ply")

    write_report(
        doc_path=doc_path,
        env_name=args.env_name,
        execution_mode=execution_mode,
        command=command_used,
        result=result,
        out_dir=out_dir,
        verified=verified,
    )
    print(f"Verified Fast-FoundationStereo demo outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
