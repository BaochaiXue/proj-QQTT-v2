#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASE_DIR = ROOT / "data/dynamics/ffs_dynamics_round1_20260414"
DEFAULT_OUTPUT_DIR = ROOT / "result/edgetam_vs_sam21_compile_fairness_ablation_20260501"
DEFAULT_MD = ROOT / "docs/generated/edgetam_vs_sam21_speed_ablation.md"
DEFAULT_JSON = ROOT / "docs/generated/edgetam_vs_sam21_speed_ablation.json"
DEFAULT_EDGETAM_REPO = Path.home() / "EdgeTAM"
DEFAULT_EDGETAM_CHECKPOINT = DEFAULT_EDGETAM_REPO / "checkpoints/edgetam.pt"
DEFAULT_EDGETAM_CONFIG = "configs/edgetam.yaml"
DEFAULT_SAM21_REPO = Path.home() / "external/sam2"
DEFAULT_SAM21_CACHE = Path.home() / ".cache/huggingface/sam2.1"
DEFAULT_TEXT_PROMPT = "sloth"
DEFAULT_CAMERA_IDS = (0, 1, 2)


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _normalize_label(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _parse_prompts(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _sorted_frame_tokens(case_dir: Path, *, camera_idx: int, frames: int | None) -> list[str]:
    paths = sorted((case_dir / "color" / str(int(camera_idx))).glob("*.png"), key=lambda path: int(path.stem))
    if frames is not None:
        paths = paths[: int(frames)]
    if not paths:
        raise FileNotFoundError(f"No RGB frames found: {case_dir}/color/{camera_idx}")
    return [path.stem for path in paths]


def _load_mask_info(mask_root: Path, *, camera_idx: int) -> dict[int, str]:
    path = mask_root / "mask" / f"mask_info_{int(camera_idx)}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Mask info must be a dict: {path}")
    return {int(key): str(value) for key, value in payload.items()}


def _load_union_mask(
    *,
    case_dir: Path,
    camera_idx: int,
    frame_token: str,
    text_prompt: str,
) -> np.ndarray:
    image = cv2.imread(str(case_dir / "color" / str(int(camera_idx)) / f"{frame_token}.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing RGB frame for mask shape: cam{camera_idx} frame {frame_token}")
    height, width = image.shape[:2]
    prompts = {_normalize_label(item) for item in _parse_prompts(text_prompt)}
    mask_root = case_dir / "sam31_masks"
    matched_ids = [
        obj_id
        for obj_id, label in _load_mask_info(mask_root, camera_idx=int(camera_idx)).items()
        if _normalize_label(label) in prompts
    ]
    if not matched_ids:
        raise RuntimeError(f"No SAM3.1 mask labels match {text_prompt!r} for cam{camera_idx}")
    union = np.zeros((height, width), dtype=bool)
    for obj_id in matched_ids:
        path = mask_root / "mask" / str(int(camera_idx)) / str(int(obj_id)) / f"{frame_token}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Missing SAM3.1 mask: {path}")
        if mask.shape[:2] != union.shape:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        union |= mask > 0
    if not np.any(union):
        raise RuntimeError(f"Empty SAM3.1 union mask for cam{camera_idx} frame {frame_token}")
    return union


def _prepare_video_dir(case_dir: Path, *, camera_idx: int, frame_tokens: Sequence[str], work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for idx, token in enumerate(frame_tokens):
        src = case_dir / "color" / str(int(camera_idx)) / f"{token}.png"
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing RGB frame: {src}")
        cv2.imwrite(str(work_dir / f"{idx:05d}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return work_dir


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(list(command), cwd=None if cwd is None else str(cwd), text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _edgetam_config_report(edgetam_repo: Path, checkpoint: Path, config: str) -> dict[str, Any]:
    config_path = edgetam_repo / "sam2" / config
    config_text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    build_text = (edgetam_repo / "sam2/build_sam.py").read_text(encoding="utf-8")
    model_text = (edgetam_repo / "sam2/modeling/sam2_base.py").read_text(encoding="utf-8")
    return {
        "repo": str(edgetam_repo),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"], cwd=edgetam_repo),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": str(config_path),
        "config_has_compile_image_encoder_false": "compile_image_encoder: false" in config_text,
        "backbone_repvit_m1_dist_in1k": "repvit_m1.dist_in1k" in config_text,
        "build_sam2_video_predictor_has_vos_optimized": "vos_optimized" in build_text,
        "supports_compile_image_encoder": "compile_image_encoder" in model_text and "torch.compile" in model_text,
        "supports_full_vos_optimized_compile": "SAM2VideoPredictorVOS" in build_text,
    }


def _sam21_config_report(sam21_repo: Path, checkpoint_cache: Path) -> dict[str, Any]:
    return {
        "repo": str(sam21_repo),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"], cwd=sam21_repo),
        "small_checkpoint": str(checkpoint_cache / "sam2.1_hiera_small.pt"),
        "small_checkpoint_sha256": _sha256(checkpoint_cache / "sam2.1_hiera_small.pt"),
        "tiny_checkpoint": str(checkpoint_cache / "sam2.1_hiera_tiny.pt"),
        "tiny_checkpoint_sha256": _sha256(checkpoint_cache / "sam2.1_hiera_tiny.pt"),
    }


def _cuda_sync(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _time_ms(torch_module: Any, fn: Any) -> tuple[Any, float]:
    _cuda_sync(torch_module)
    start = time.perf_counter()
    value = fn()
    _cuda_sync(torch_module)
    return value, float((time.perf_counter() - start) * 1000.0)


def _mark_compile_step(torch_module: Any) -> None:
    compiler = getattr(torch_module, "compiler", None)
    marker = None if compiler is None else getattr(compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def _configure_torch(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation.")
    props = torch_module.cuda.get_device_properties(0)
    if int(props.major) >= 8:
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True


def _build_predictor(args: argparse.Namespace, torch_module: Any) -> tuple[Any, dict[str, Any]]:
    from sam2.build_sam import build_sam2_video_predictor

    backend = str(args.backend)
    mode = str(args.compile_mode)
    metadata: dict[str, Any] = {
        "backend": backend,
        "compile_mode": mode,
        "manual_compile_applied": False,
        "hydra_overrides_extra": [],
        "position_encoding_cache_clone_patch": False,
    }
    if backend == "sam21":
        predictor = build_sam2_video_predictor(
            str(args.model_cfg),
            str(args.checkpoint),
            device="cuda",
            vos_optimized=True,
        )
        metadata["vos_optimized"] = True
        return predictor, metadata

    if backend != "edgetam":
        raise ValueError(f"Unsupported worker backend: {backend}")

    overrides: list[str] = []
    if mode == "compile_image_encoder_no_pos_cache_patch":
        _patch_position_encoding_no_cache(torch_module)
        metadata["position_encoding_cache_clone_patch"] = False
        metadata["position_encoding_no_cache_patch"] = True
        overrides.append("model.compile_image_encoder=true")
    elif mode == "compile_image_encoder_cache_clone_patch":
        _patch_position_encoding_cache_clone(torch_module)
        metadata["position_encoding_cache_clone_patch"] = True
        metadata["position_encoding_no_cache_patch"] = False
        overrides.append("model.compile_image_encoder=true")
    elif mode == "compile_image_encoder":
        overrides.append("model.compile_image_encoder=true")
    predictor = build_sam2_video_predictor(
        str(args.model_cfg),
        str(args.checkpoint),
        device="cuda",
        hydra_overrides_extra=overrides,
    )
    metadata["hydra_overrides_extra"] = overrides
    metadata["vos_optimized"] = False
    if mode == "manual_image_encoder_reduce_overhead":
        if not hasattr(predictor, "image_encoder"):
            raise RuntimeError("EdgeTAM predictor has no image_encoder attribute for manual compile.")
        predictor.image_encoder.forward = torch_module.compile(
            predictor.image_encoder.forward,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
        metadata["manual_compile_applied"] = True
        metadata["manual_compile_target"] = "predictor.image_encoder.forward"
        metadata["manual_compile_mode"] = "reduce-overhead"
    elif mode not in {
        "eager",
        "compile_image_encoder",
        "compile_image_encoder_cache_clone_patch",
        "compile_image_encoder_no_pos_cache_patch",
    }:
        raise ValueError(f"Unsupported EdgeTAM compile mode: {mode}")
    return predictor, metadata


def _position_encoding_forward_no_cache(torch_module: Any) -> Any:
    def forward(self: Any, x: Any) -> Any:
        y_embed = (
            torch_module.arange(1, x.shape[-2] + 1, dtype=torch_module.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch_module.arange(1, x.shape[-1] + 1, dtype=torch_module.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch_module.arange(self.num_pos_feats, dtype=torch_module.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch_module.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch_module.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch_module.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return forward


def _patch_position_encoding_no_cache(torch_module: Any) -> None:
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    if getattr(PositionEmbeddingSine.forward, "_qqtt_no_cache_patch", False):
        return
    forward = _position_encoding_forward_no_cache(torch_module)
    forward._qqtt_no_cache_patch = True  # type: ignore[attr-defined]
    PositionEmbeddingSine.forward = forward


def _patch_position_encoding_cache_clone(torch_module: Any) -> None:
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    if getattr(PositionEmbeddingSine.forward, "_qqtt_cache_clone_patch", False):
        return

    def forward(self: Any, x: Any) -> Any:
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        pos = _position_encoding_forward_no_cache(torch_module)(self, x)
        self.cache[cache_key] = pos[0].detach().clone()
        return pos

    forward._qqtt_cache_clone_patch = True  # type: ignore[attr-defined]
    PositionEmbeddingSine.forward = forward


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _configure_torch(torch)
    case_dir = Path(args.case_dir).resolve()
    result_json = Path(args.result_json).resolve()
    result_json.parent.mkdir(parents=True, exist_ok=True)
    frame_tokens = _sorted_frame_tokens(case_dir, camera_idx=int(args.camera_idx), frames=args.frames)
    init_mask = _load_union_mask(
        case_dir=case_dir,
        camera_idx=int(args.camera_idx),
        frame_token=frame_tokens[0],
        text_prompt=str(args.text_prompt),
    )

    temp_root = Path(tempfile.mkdtemp(prefix="compile_ablation_", dir=str(result_json.parent)))
    video_dir = temp_root / "video_jpg"
    try:
        _, jpeg_prep_ms = _time_ms(
            torch,
            lambda: _prepare_video_dir(
                case_dir,
                camera_idx=int(args.camera_idx),
                frame_tokens=frame_tokens,
                work_dir=video_dir,
            ),
        )
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor, compile_metadata = _time_ms(torch, lambda: _build_predictor(args, torch))[0]

            def init_state() -> Any:
                if bool(args.use_cudagraph_step_marker):
                    _mark_compile_step(torch)
                kwargs = {}
                if str(args.backend) == "sam21":
                    kwargs = {
                        "offload_video_to_cpu": False,
                        "offload_state_to_cpu": False,
                        "async_loading_frames": False,
                    }
                return predictor.init_state(video_path=str(video_dir), **kwargs)

            state, init_state_ms = _time_ms(torch, init_state)

            def prompt_state() -> Any:
                if bool(args.use_cudagraph_step_marker):
                    _mark_compile_step(torch)
                return predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    mask=np.asarray(init_mask, dtype=bool),
                )

            prompt_response, prompt_ms = _time_ms(torch, prompt_state)
            run_records: list[dict[str, Any]] = []
            total_warmup_ms = 0.0
            total_warmup_frames = 0
            total_measured_ms = 0.0
            total_measured_frames = 0
            for run_idx in range(int(args.runs)):
                def propagate_once() -> int:
                    count = 0
                    iterator = predictor.propagate_in_video(state)
                    while True:
                        if bool(args.use_cudagraph_step_marker):
                            _mark_compile_step(torch)
                        try:
                            _out_frame_idx, _out_obj_ids, _out_mask_logits = next(iterator)
                        except StopIteration:
                            break
                        count += 1
                    return count

                frame_count, elapsed_ms = _time_ms(torch, propagate_once)
                phase = "warmup" if run_idx < int(args.warmup_runs) else "measured"
                if phase == "warmup":
                    total_warmup_ms += float(elapsed_ms)
                    total_warmup_frames += int(frame_count)
                else:
                    total_measured_ms += float(elapsed_ms)
                    total_measured_frames += int(frame_count)
                run_records.append(
                    {
                        "run_idx": int(run_idx),
                        "phase": phase,
                        "frame_count": int(frame_count),
                        "elapsed_ms": float(elapsed_ms),
                        "fps": float(1000.0 * int(frame_count) / max(1e-9, float(elapsed_ms))),
                    }
                )
            try:
                prompt_obj_ids = [int(item) for item in prompt_response[1]]
            except Exception:
                prompt_obj_ids = []

        result = {
            "status": "ok",
            "backend": str(args.backend),
            "backend_key": str(args.backend_key),
            "compile_mode": str(args.compile_mode),
            "case_dir": str(case_dir),
            "camera_idx": int(args.camera_idx),
            "text_prompt": str(args.text_prompt),
            "frame_count": int(len(frame_tokens)),
            "runs": int(args.runs),
            "warmup_runs": int(args.warmup_runs),
            "measured_runs": int(max(0, int(args.runs) - int(args.warmup_runs))),
            "checkpoint": str(args.checkpoint),
            "model_cfg": str(args.model_cfg),
            "compile_metadata": compile_metadata,
            "jpeg_prep_ms": float(jpeg_prep_ms),
            "init_state_ms": float(init_state_ms),
            "prompt_ms": float(prompt_ms),
            "prompt_obj_ids": prompt_obj_ids,
            "run_records": run_records,
            "warmup_total_frames": int(total_warmup_frames),
            "warmup_total_ms": float(total_warmup_ms),
            "warmup_fps": float(1000.0 * total_warmup_frames / max(1e-9, total_warmup_ms)),
            "measured_total_frames": int(total_measured_frames),
            "measured_total_ms": float(total_measured_ms),
            "measured_fps": float(1000.0 * total_measured_frames / max(1e-9, total_measured_ms)),
            "measured_ms_per_frame": float(total_measured_ms / max(1, total_measured_frames)),
            "max_cuda_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
            "max_cuda_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)),
            "process_guard": {
                "pid": int(os.getpid()),
                "python": sys.executable,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "bfloat16_autocast": True,
                "use_cudagraph_step_marker": bool(args.use_cudagraph_step_marker),
                "timed_loop_contract": (
                    "propagate_in_video pass-only; no threshold, clone, CPU copy, save, PCD, or render; "
                    f"cudagraph marker={'used' if bool(args.use_cudagraph_step_marker) else 'not used'}"
                ),
            },
        }
        result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        failure = {
            "status": "failed",
            "backend": str(args.backend),
            "backend_key": str(args.backend_key),
            "compile_mode": str(args.compile_mode),
            "case_dir": str(case_dir),
            "camera_idx": int(args.camera_idx),
            "checkpoint": str(args.checkpoint),
            "model_cfg": str(args.model_cfg),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        result_json.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return failure
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _backend_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    checkpoint_cache = Path(args.sam21_checkpoint_cache).expanduser().resolve()
    edgetam_repo = Path(args.edgetam_repo).expanduser().resolve()
    specs = [
        {
            "key": "sam21_small_vos",
            "env": str(args.sam21_env_name),
            "cwd": str(Path(args.sam21_repo).expanduser().resolve()),
            "backend": "sam21",
            "compile_mode": "vos_optimized",
            "checkpoint": str(checkpoint_cache / "sam2.1_hiera_small.pt"),
            "model_cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "label": "SAM2.1 Small vos_optimized=True",
        },
        {
            "key": "sam21_tiny_vos",
            "env": str(args.sam21_env_name),
            "cwd": str(Path(args.sam21_repo).expanduser().resolve()),
            "backend": "sam21",
            "compile_mode": "vos_optimized",
            "checkpoint": str(checkpoint_cache / "sam2.1_hiera_tiny.pt"),
            "model_cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "label": "SAM2.1 Tiny vos_optimized=True",
        },
        {
            "key": "edgetam_eager",
            "env": str(args.edgetam_env_name),
            "cwd": str(edgetam_repo),
            "backend": "edgetam",
            "compile_mode": "eager",
            "checkpoint": str(args.edgetam_checkpoint),
            "model_cfg": str(args.edgetam_config),
            "label": "EdgeTAM eager",
        },
        {
            "key": "edgetam_compile_image_encoder",
            "env": str(args.edgetam_env_name),
            "cwd": str(edgetam_repo),
            "backend": "edgetam",
            "compile_mode": "compile_image_encoder",
            "checkpoint": str(args.edgetam_checkpoint),
            "model_cfg": str(args.edgetam_config),
            "label": "EdgeTAM compile_image_encoder=true",
        },
        {
            "key": "edgetam_compile_image_encoder_cache_clone_patch",
            "env": str(args.edgetam_env_name),
            "cwd": str(edgetam_repo),
            "backend": "edgetam",
            "compile_mode": "compile_image_encoder_cache_clone_patch",
            "checkpoint": str(args.edgetam_checkpoint),
            "model_cfg": str(args.edgetam_config),
            "label": "EdgeTAM compile_image_encoder=true + cache clone patch",
        },
        {
            "key": "edgetam_compile_image_encoder_no_pos_cache_patch",
            "env": str(args.edgetam_env_name),
            "cwd": str(edgetam_repo),
            "backend": "edgetam",
            "compile_mode": "compile_image_encoder_no_pos_cache_patch",
            "checkpoint": str(args.edgetam_checkpoint),
            "model_cfg": str(args.edgetam_config),
            "label": "EdgeTAM compile_image_encoder=true + no position-cache patch",
        },
    ]
    if not bool(args.skip_manual_compile):
        specs.append(
            {
                "key": "edgetam_manual_image_encoder_reduce_overhead",
                "env": str(args.edgetam_env_name),
                "cwd": str(edgetam_repo),
                "backend": "edgetam",
                "compile_mode": "manual_image_encoder_reduce_overhead",
                "checkpoint": str(args.edgetam_checkpoint),
                "model_cfg": str(args.edgetam_config),
                "label": "EdgeTAM manual torch.compile(image_encoder, reduce-overhead)",
            }
        )
    return specs


def _run_worker_subprocess(
    *,
    script_path: Path,
    spec: dict[str, Any],
    case_dir: Path,
    text_prompt: str,
    camera_idx: int,
    frames: int | None,
    runs: int,
    warmup_runs: int,
    use_cudagraph_step_marker: bool,
    result_json: Path,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(spec["env"]),
        "python",
        str(script_path),
        "--worker",
        "--backend-key",
        str(spec["key"]),
        "--backend",
        str(spec["backend"]),
        "--compile-mode",
        str(spec["compile_mode"]),
        "--case-dir",
        str(case_dir),
        "--text-prompt",
        str(text_prompt),
        "--camera-idx",
        str(int(camera_idx)),
        "--checkpoint",
        str(spec["checkpoint"]),
        "--model-cfg",
        str(spec["model_cfg"]),
        "--runs",
        str(int(runs)),
        "--warmup-runs",
        str(int(warmup_runs)),
        "--result-json",
        str(result_json),
    ]
    if frames is None:
        command.append("--all-frames")
    else:
        command.extend(["--frames", str(int(frames))])
    if use_cudagraph_step_marker:
        command.append("--use-cudagraph-step-marker")
    print(f"[ablation] {spec['key']} cam{camera_idx}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            command,
            check=False,
            cwd=str(spec["cwd"]),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    if not result_json.is_file():
        return {
            "status": "failed",
            "backend_key": str(spec["key"]),
            "camera_idx": int(camera_idx),
            "error": f"Worker did not write result JSON. See {log_path}",
            "worker_log_path": str(log_path),
        }
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    payload["worker_log_path"] = str(log_path)
    result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _aggregate(records: Sequence[dict[str, Any]], specs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_key.setdefault(str(record.get("backend_key")), []).append(dict(record))
    spec_by_key = {str(spec["key"]): spec for spec in specs}
    aggregate: dict[str, Any] = {}
    for key, items in sorted(by_key.items()):
        ok_items = [item for item in items if item.get("status") == "ok"]
        total_frames = int(sum(int(item.get("measured_total_frames", 0)) for item in ok_items))
        total_ms = float(sum(float(item.get("measured_total_ms", 0.0)) for item in ok_items))
        warmup_frames = int(sum(int(item.get("warmup_total_frames", 0)) for item in ok_items))
        warmup_ms = float(sum(float(item.get("warmup_total_ms", 0.0)) for item in ok_items))
        aggregate[key] = {
            "label": spec_by_key.get(key, {}).get("label", key),
            "status": "ok" if len(ok_items) == len(items) and ok_items else ("partial" if ok_items else "failed"),
            "camera_count": int(len(items)),
            "ok_camera_count": int(len(ok_items)),
            "measured_total_frames": total_frames,
            "measured_total_ms": total_ms,
            "measured_fps": float(1000.0 * total_frames / max(1e-9, total_ms)) if ok_items else None,
            "measured_ms_per_frame": float(total_ms / max(1, total_frames)) if ok_items else None,
            "warmup_fps": float(1000.0 * warmup_frames / max(1e-9, warmup_ms)) if ok_items else None,
            "max_cuda_memory_allocated_mb": (
                float(max(float(item.get("max_cuda_memory_allocated_mb", 0.0)) for item in ok_items))
                if ok_items
                else None
            ),
            "failures": [item for item in items if item.get("status") != "ok"],
        }
    return aggregate


def _decision(aggregate: dict[str, Any]) -> dict[str, Any]:
    small = aggregate.get("sam21_small_vos", {}).get("measured_fps")
    tiny = aggregate.get("sam21_tiny_vos", {}).get("measured_fps")
    edge_candidates = {
        key: value.get("measured_fps")
        for key, value in aggregate.items()
        if key.startswith("edgetam") and value.get("measured_fps") is not None
    }
    if not edge_candidates or small is None or tiny is None:
        return {
            "recommendation": "inconclusive",
            "reason": "At least one required backend failed.",
        }
    best_edge_key, best_edge_fps = max(edge_candidates.items(), key=lambda item: float(item[1]))
    baseline = max(float(small), float(tiny))
    if float(best_edge_fps) >= 1.15 * baseline:
        recommendation = "EdgeTAM can be considered a candidate default after quality validation."
    else:
        recommendation = "Keep SAM2.1 Small as default, SAM2.1 Tiny as fast mode, EdgeTAM as experimental/edge backend."
    return {
        "best_edgetam_key": best_edge_key,
        "best_edgetam_fps": float(best_edge_fps),
        "best_sam21_small_tiny_fps": baseline,
        "edgetam_speedup_over_best_small_tiny": float(best_edge_fps) / max(1e-9, baseline),
        "recommendation": recommendation,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# EdgeTAM vs SAM2.1 Compile Fairness Ablation",
        "",
        "## Protocol",
        "",
        f"- case: `{payload['case_dir']}`",
        f"- cameras: `{payload['camera_ids']}`",
        f"- frames per camera: `{payload['frames_per_camera']}`",
        f"- runs: `{payload['runs']}` total, first `{payload['warmup_runs']}` warmup",
        "- timed loop: `for ... in predictor.propagate_in_video(state): pass`",
        "- excluded from FPS: model build, JPEG prep, init_state, prompt, threshold, CPU copy, mask save, PCD, render",
        f"- cudagraph step marker: `{'used' if payload.get('use_cudagraph_step_marker') else 'not used'}`",
        "",
        "## Local EdgeTAM Path",
        "",
    ]
    edge = payload.get("edgetam_config", {})
    for key in (
        "repo",
        "git_commit",
        "checkpoint",
        "checkpoint_sha256",
        "config",
        "backbone_repvit_m1_dist_in1k",
        "config_has_compile_image_encoder_false",
        "supports_compile_image_encoder",
        "build_sam2_video_predictor_has_vos_optimized",
        "supports_full_vos_optimized_compile",
    ):
        lines.append(f"- {key}: `{edge.get(key)}`")
    lines.extend(
        [
            "",
            "## Speed",
            "",
            "| backend | status | ms/frame | FPS | warmup FPS | peak CUDA MB | ok cams |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in (
        "sam21_small_vos",
        "sam21_tiny_vos",
        "edgetam_eager",
        "edgetam_compile_image_encoder",
        "edgetam_compile_image_encoder_cache_clone_patch",
        "edgetam_compile_image_encoder_no_pos_cache_patch",
        "edgetam_manual_image_encoder_reduce_overhead",
    ):
        item = aggregate.get(key)
        if not item:
            continue
        ms = item.get("measured_ms_per_frame")
        fps = item.get("measured_fps")
        warm = item.get("warmup_fps")
        peak = item.get("max_cuda_memory_allocated_mb")
        lines.append(
            f"| {item['label']} | {item['status']} | "
            f"{float(ms):.2f} | {float(fps):.2f} | {float(warm):.2f} | "
            f"{float(peak):.0f} | {int(item['ok_camera_count'])}/{int(item['camera_count'])} |"
            if ms is not None and fps is not None and warm is not None and peak is not None
            else f"| {item['label']} | {item['status']} | n/a | n/a | n/a | n/a | {int(item['ok_camera_count'])}/{int(item['camera_count'])} |"
        )
    failures = [failure for item in aggregate.values() for failure in item.get("failures", [])]
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(
                f"- `{failure.get('backend_key')}` cam{failure.get('camera_idx')}: "
                f"{failure.get('error')} log=`{failure.get('worker_log_path')}`"
            )
    no_cache = aggregate.get("edgetam_compile_image_encoder_no_pos_cache_patch", {})
    if no_cache.get("status") == "ok":
        lines.extend(
            [
                "",
                "## EdgeTAM Compile Note",
                "",
                (
                    "`edgetam_compile_image_encoder_no_pos_cache_patch` is a process-local benchmark patch. "
                    "It does not modify `/home/zhangxinjie/EdgeTAM`; it monkey-patches "
                    "`PositionEmbeddingSine.forward` inside the worker process to avoid writing position "
                    "encoding tensors into `self.cache` from inside the compiled image encoder. "
                    "The unpatched EdgeTAM compile modes remain recorded as failures above."
                ),
            ]
        )
    decision = payload.get("decision", {})
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- best EdgeTAM mode: `{decision.get('best_edgetam_key')}`",
            f"- best EdgeTAM FPS: `{decision.get('best_edgetam_fps')}`",
            f"- best SAM2.1 Small/Tiny FPS: `{decision.get('best_sam21_small_tiny_fps')}`",
            f"- EdgeTAM / best Small-or-Tiny: `{decision.get('edgetam_speedup_over_best_small_tiny')}`",
            f"- recommendation: {decision.get('recommendation')}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_orchestrator(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / "records"
    logs_dir = output_dir / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    specs = _backend_specs(args)
    selected_keys = set(str(item) for item in args.backend_keys) if args.backend_keys else None
    if selected_keys is not None:
        specs = [spec for spec in specs if str(spec["key"]) in selected_keys]
    if not specs:
        raise ValueError("No backend specs selected.")
    camera_ids = _parse_csv_ints(args.camera_ids) or list(DEFAULT_CAMERA_IDS)
    case_dir = Path(args.case_dir).resolve()
    frames = None if args.all_frames else int(args.frames)
    all_records: list[dict[str, Any]] = []
    for spec in specs:
        for camera_idx in camera_ids:
            result_json = records_dir / f"{spec['key']}_cam{int(camera_idx)}.json"
            if result_json.is_file() and not args.overwrite:
                record = json.loads(result_json.read_text(encoding="utf-8"))
            else:
                record = _run_worker_subprocess(
                    script_path=Path(__file__).resolve(),
                    spec=spec,
                    case_dir=case_dir,
                    text_prompt=str(args.text_prompt),
                    camera_idx=int(camera_idx),
                    frames=frames,
                    runs=int(args.runs),
                    warmup_runs=int(args.warmup_runs),
                    use_cudagraph_step_marker=not bool(args.no_cudagraph_step_marker),
                    result_json=result_json,
                    log_path=logs_dir / f"{spec['key']}_cam{int(camera_idx)}.log",
                )
            all_records.append(record)
    first_tokens = _sorted_frame_tokens(case_dir, camera_idx=int(camera_ids[0]), frames=frames)
    payload = {
        "output_dir": str(output_dir),
        "case_dir": str(case_dir),
        "text_prompt": str(args.text_prompt),
        "camera_ids": [int(item) for item in camera_ids],
        "frames": None if frames is None else int(frames),
        "frames_per_camera": int(len(first_tokens)),
        "runs": int(args.runs),
        "warmup_runs": int(args.warmup_runs),
        "use_cudagraph_step_marker": not bool(args.no_cudagraph_step_marker),
        "backend_specs": specs,
        "edgetam_config": _edgetam_config_report(
            Path(args.edgetam_repo).expanduser().resolve(),
            Path(args.edgetam_checkpoint).expanduser().resolve(),
            str(args.edgetam_config),
        ),
        "sam21_config": _sam21_config_report(
            Path(args.sam21_repo).expanduser().resolve(),
            Path(args.sam21_checkpoint_cache).expanduser().resolve(),
        ),
        "records": all_records,
        "aggregate": _aggregate(all_records, specs),
    }
    payload["decision"] = _decision(payload["aggregate"])
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(output_md, payload)
    payload["docs"] = {"json": str(output_json), "markdown": str(output_md)}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official-style pass-only EdgeTAM compile-mode vs SAM2.1 Small/Tiny speed ablation."
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--text-prompt", default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--camera-ids", default="0,1,2")
    parser.add_argument("--frames", type=int, default=71)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument(
        "--no-cudagraph-step-marker",
        action="store_true",
        help=(
            "Disable torch.compiler.cudagraph_mark_step_begin(). On this CUDA 13/PyTorch 2.11 "
            "stack, compiled predictors usually require the marker to avoid CUDA Graph overwrite errors."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend-keys", nargs="*")
    parser.add_argument("--skip-manual-compile", action="store_true")
    parser.add_argument("--edgetam-env-name", default="edgetam-max")
    parser.add_argument("--edgetam-repo", type=Path, default=DEFAULT_EDGETAM_REPO)
    parser.add_argument("--edgetam-checkpoint", type=Path, default=DEFAULT_EDGETAM_CHECKPOINT)
    parser.add_argument("--edgetam-config", default=DEFAULT_EDGETAM_CONFIG)
    parser.add_argument("--sam21-env-name", default="SAM21-max")
    parser.add_argument("--sam21-repo", type=Path, default=DEFAULT_SAM21_REPO)
    parser.add_argument("--sam21-checkpoint-cache", type=Path, default=DEFAULT_SAM21_CACHE)

    worker = parser.add_argument_group("worker mode")
    worker.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    worker.add_argument("--backend-key", help=argparse.SUPPRESS)
    worker.add_argument("--backend", choices=("edgetam", "sam21"), help=argparse.SUPPRESS)
    worker.add_argument("--compile-mode", help=argparse.SUPPRESS)
    worker.add_argument("--camera-idx", type=int, help=argparse.SUPPRESS)
    worker.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--model-cfg", help=argparse.SUPPRESS)
    worker.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--use-cudagraph-step-marker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        missing = [
            name
            for name in ("backend_key", "backend", "compile_mode", "camera_idx", "checkpoint", "model_cfg", "result_json")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"worker mode missing: {', '.join(missing)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        result = run_worker(args)
        if result.get("status") == "ok":
            print(
                f"[worker] {result['backend_key']} cam{result['camera_idx']}: "
                f"{result['measured_ms_per_frame']:.2f} ms/frame {result['measured_fps']:.2f} FPS",
                flush=True,
            )
        else:
            print(
                f"[worker] {result.get('backend_key')} cam{result.get('camera_idx')} failed: {result.get('error')}",
                flush=True,
            )
        return 0
    payload = run_orchestrator(args)
    print(f"[ablation] report: {payload['docs']['markdown']}", flush=True)
    print(f"[ablation] json: {payload['docs']['json']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
