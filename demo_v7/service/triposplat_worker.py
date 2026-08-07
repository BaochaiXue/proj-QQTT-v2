"""Persistent TripoSplat generation worker (one load, many seeds).

Spawned by the camera service on the shape-prior GPU (CUDA_VISIBLE_DEVICES
set by the spawner) once the shape-prior chain is READY, and told to exit
before FORMAL starts (PhysTwin children take that GPU). Runs in the same
conda env as the service (TripoSplat is pure torch — no extra env).

Line-JSON protocol on stdin -> stdout (stderr is free-form logging):

  in : {"cmd": "generate", "image": ..., "out_dir": ..., "seed": 42,
        "num_gaussians": 131072, "steps": 20}
  out: {"event": "ready", "load_s": ...}          (once, after model load)
       {"event": "progress", "step": i, "total": n}
       {"event": "done", "seed": ..., "ply": ..., "prepared": ...,
        "contact_sheet": ..., "generation_s": ..., "num_splats": ...}
       {"event": "error", "message": ...}
  in : {"cmd": "exit"}  (or EOF)                  -> clean exit

The worker also renders the 拣选 turntable previews itself (gsplat) so the
GUI never needs CUDA: 8 orbit angles composited into one contact sheet.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_v7.service.gaussian_utils import (  # noqa: E402
    load_gaussian_ply,
    render_gaussians,
)

_TURNTABLE_ANGLES = 8
_PREVIEW_SIZE = 360


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _warm_splats():
    """A handful of dummy splats for the post-load gsplat JIT warmup."""
    import numpy as np

    from demo_v7.service.gaussian_utils import GaussianSplats

    count = 16
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(count, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return GaussianSplats(
        means=rng.normal(size=(count, 3)).astype(np.float32),
        quats=quats,
        scales=np.full((count, 3), 0.05, dtype=np.float32),
        opacities=np.full((count,), 0.8, dtype=np.float32),
        colors=np.full((count, 3), 0.5, dtype=np.float32),
    )


def _log(message: str) -> None:
    print(f"[triposplat-worker] {message}", file=sys.stderr, flush=True)


def _load_pipeline(repo: Path, device: str):
    sys.path.insert(0, str(repo))
    from triposplat import TripoSplatPipeline

    ckpts = repo / "ckpts"
    return TripoSplatPipeline(
        ckpt_path=str(ckpts / "diffusion_models" / "triposplat_fp16.safetensors"),
        decoder_path=str(ckpts / "vae" / "triposplat_vae_decoder_fp16.safetensors"),
        dinov3_path=str(ckpts / "clip_vision" / "dino_v3_vit_h.safetensors"),
        flux2_vae_encoder_path=str(ckpts / "vae" / "flux2-vae.safetensors"),
        rmbg_path=str(ckpts / "background_removal" / "birefnet.safetensors"),
        device=device,
    )


def _render_contact_sheet(ply_path: Path, out_path: Path, device: str) -> None:
    """8-angle orbit contact sheet (2x4) around the canonical gaussian."""
    import numpy as np
    import cv2

    splats = load_gaussian_ply(ply_path)
    center = np.percentile(splats.means, 50, axis=0)
    extent = float(
        np.linalg.norm(
            np.percentile(splats.means, 97, axis=0)
            - np.percentile(splats.means, 3, axis=0)
        )
    )
    radius = 1.6 * extent
    size = _PREVIEW_SIZE
    focal = 1.1 * size
    intrinsics = np.array(
        [[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], dtype=np.float64
    )
    tiles = []
    # TripoSplat canonical keeps the IMAGE row axis: +y points down (an
    # up=(0,1,0) orbit renders the object head-down — verified on the sloth).
    up_world = np.array([0.0, -1.0, 0.0])
    for i in range(_TURNTABLE_ANGLES):
        azimuth = 2 * np.pi * i / _TURNTABLE_ANGLES
        # Camera orbits in the canonical horizontal plane, slightly above.
        eye = center + radius * np.array(
            [np.sin(azimuth), -0.35, np.cos(azimuth)]
        )
        forward = center - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up_world)
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        w2c = np.eye(4)
        w2c[:3, :3] = np.stack([right, down, forward])
        w2c[:3, 3] = -w2c[:3, :3] @ eye
        rgb, _alpha = render_gaussians(
            splats,
            viewmat=w2c,
            intrinsics=intrinsics,
            width=size,
            height=size,
            background=(0.13, 0.13, 0.14),
            device=device,
        )
        tiles.append(rgb[..., ::-1])  # BGR for imwrite
    rows = [np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)]
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="demo_v7 TripoSplat worker")
    parser.add_argument("--triposplat-repo", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args(argv)

    load_start = time.perf_counter()
    try:
        pipe = _load_pipeline(args.triposplat_repo, args.device)
    except Exception as exc:
        _emit({"event": "error", "message": f"pipeline load failed: {exc}"})
        return 1
    _emit({"event": "ready", "load_s": round(time.perf_counter() - load_start, 2)})

    # Warm the gsplat CUDA extension in a BACKGROUND thread: sampling (the
    # first generate) is pure torch and does not need gsplat — only the
    # turntable render at the end does, and the render path join()s this
    # thread first. A warm cache costs 0.1s; a cold rebuild (~137s, seen
    # when a foreign environment poisons the JIT cache) then overlaps
    # sampling + the alignment park window instead of stalling the queue.
    import threading

    def _warm_gsplat() -> None:
        try:
            import numpy as np

            warm_start = time.perf_counter()
            render_gaussians(
                _warm_splats(),
                viewmat=np.eye(4),
                intrinsics=np.array(
                    [[100.0, 0, 16], [0, 100.0, 16], [0, 0, 1.0]]
                ),
                width=32,
                height=32,
            )
            print(
                f"[triposplat-worker] gsplat warmed in "
                f"{time.perf_counter() - warm_start:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(
                f"[triposplat-worker] gsplat warmup failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    warmup_thread = threading.Thread(
        target=_warm_gsplat, name="gsplat-warmup", daemon=True
    )
    warmup_thread.start()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"event": "error", "message": f"bad request json: {exc}"})
            continue
        cmd = request.get("cmd")
        if cmd == "exit":
            break
        if cmd != "generate":
            _emit({"event": "error", "message": f"unknown cmd: {cmd!r}"})
            continue
        try:
            out_dir = Path(request["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            seed = int(request.get("seed", 42))
            steps = int(request.get("steps", 20))
            num_gaussians = int(request.get("num_gaussians", 131072))
            generation_start = time.perf_counter()

            def _progress(step: int, total: int) -> None:
                _emit({"event": "progress", "step": int(step), "total": int(total)})

            gaussian, prepared = pipe.run(
                str(request["image"]),
                seed=seed,
                steps=steps,
                num_gaussians=num_gaussians,
                callback=_progress,
            )
            sampling_s = time.perf_counter() - generation_start
            ply_path = out_dir / "gaussian.ply"
            prepared_path = out_dir / "gaussian_prepared.png"
            sheet_path = out_dir / "gaussian_turntable.png"
            gaussian.save_ply(str(ply_path))
            prepared.save(str(prepared_path))
            # The turntable is the first gsplat consumer: make sure the
            # background warmup (possibly a cold JIT rebuild) is done. The
            # stage event keeps the GUI status honest during that window.
            _emit({"event": "stage", "name": "turntable"})
            turntable_start = time.perf_counter()
            warmup_thread.join()
            _render_contact_sheet(ply_path, sheet_path, args.device)
            turntable_s = time.perf_counter() - turntable_start
            generation_s = time.perf_counter() - generation_start
            num_splats = len(load_gaussian_ply(ply_path))
            provenance = {
                "image": str(request["image"]),
                "seed": seed,
                "steps": steps,
                "num_gaussians": num_gaussians,
                "num_finite_splats": num_splats,
                "generation_s": round(generation_s, 2),
            }
            (out_dir / "gaussian_provenance.json").write_text(
                json.dumps(provenance, indent=1)
            )
            _emit(
                {
                    "event": "done",
                    "seed": seed,
                    "ply": str(ply_path),
                    "prepared": str(prepared_path),
                    "contact_sheet": str(sheet_path),
                    "provenance": str(out_dir / "gaussian_provenance.json"),
                    "generation_s": round(generation_s, 2),
                    "sampling_s": round(sampling_s, 2),
                    "turntable_s": round(turntable_s, 2),
                    "num_splats": num_splats,
                }
            )
        except Exception as exc:  # keep serving after a failed generate
            _log(f"generate failed: {type(exc).__name__}: {exc}")
            _emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
