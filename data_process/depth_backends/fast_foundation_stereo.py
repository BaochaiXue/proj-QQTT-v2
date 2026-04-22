from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import disparity_to_metric_depth


def _disable_torch_compile(torch_module) -> None:
    def identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(inner):
                return inner
            return decorator
        return fn

    torch_module.compile = identity_compile


def compute_disparity_audit_stats(disparity_raw: np.ndarray) -> dict[str, float]:
    disparity = np.asarray(disparity_raw, dtype=np.float32)
    finite = np.isfinite(disparity)
    finite_values = disparity[finite]
    total_count = int(disparity.size)
    finite_count = int(np.count_nonzero(finite))
    positive = finite & (disparity > 0)
    nonpositive = finite & (disparity <= 0)
    stats = {
        "pixel_count": float(total_count),
        "finite_ratio": float(finite_count / max(1, total_count)),
        "positive_ratio": float(np.count_nonzero(positive) / max(1, total_count)),
        "nonpositive_ratio": float(np.count_nonzero(nonpositive) / max(1, total_count)),
        "positive_fraction_of_finite": 0.0,
        "nonpositive_fraction_of_finite": 0.0,
        "min_disparity": 0.0,
        "max_disparity": 0.0,
        "mean_disparity": 0.0,
        "mean_abs_disparity": 0.0,
        "p50_abs_disparity": 0.0,
        "p90_abs_disparity": 0.0,
    }
    if finite_count <= 0:
        return stats
    stats["positive_fraction_of_finite"] = float(np.count_nonzero(positive) / finite_count)
    stats["nonpositive_fraction_of_finite"] = float(np.count_nonzero(nonpositive) / finite_count)
    stats["min_disparity"] = float(np.min(finite_values))
    stats["max_disparity"] = float(np.max(finite_values))
    stats["mean_disparity"] = float(np.mean(finite_values))
    abs_values = np.abs(finite_values)
    stats["mean_abs_disparity"] = float(np.mean(abs_values))
    stats["p50_abs_disparity"] = float(np.quantile(abs_values, 0.50))
    stats["p90_abs_disparity"] = float(np.quantile(abs_values, 0.90))
    return stats


def apply_remove_invisible_mask(disparity_raw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    disparity = np.asarray(disparity_raw, dtype=np.float32).clip(0, None)
    if disparity.ndim != 2:
        raise ValueError(f"Expected 2D disparity map, got shape={disparity.shape}.")

    _, width = disparity.shape
    xx = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], disparity.shape)
    remove_mask = np.isfinite(disparity) & (disparity > 0) & ((xx - disparity) < 0)
    masked = disparity.copy()
    masked[remove_mask] = np.inf

    pixel_count = int(disparity.size)
    removed_count = int(np.count_nonzero(remove_mask))
    return masked, {
        "pixel_count": float(pixel_count),
        "remove_invisible_pixel_count": float(removed_count),
        "remove_invisible_ratio": float(removed_count / max(1, pixel_count)),
    }


def build_disparity_products(
    disparity_raw: np.ndarray,
    *,
    K_ir_left: np.ndarray,
    baseline_m: float,
    scale: float,
    scale_x: float | None = None,
    scale_y: float | None = None,
    valid_iters: int,
    max_disp: int,
    audit_mode: bool,
) -> dict[str, np.ndarray | float | list[list[float]]]:
    disparity_raw = np.asarray(disparity_raw, dtype=np.float32)
    disparity = disparity_raw.clip(0, None).astype(np.float32)
    scale_x = float(scale if scale_x is None else scale_x)
    scale_y = float(scale if scale_y is None else scale_y)
    K_used = np.asarray(K_ir_left, dtype=np.float32).copy()
    K_used[0, :] *= scale_x
    K_used[1, :] *= scale_y
    depth_ir_left_m = disparity_to_metric_depth(
        disparity,
        fx_ir=float(K_used[0, 0]),
        baseline_m=float(baseline_m),
    )
    result = {
        "disparity": disparity,
        "depth_ir_left_m": depth_ir_left_m,
        "K_ir_left_used": K_used,
        "baseline_m": float(baseline_m),
        "scale": float(scale),
        "resize_scale_x": float(scale_x),
        "resize_scale_y": float(scale_y),
        "valid_iters": int(valid_iters),
        "max_disp": int(max_disp),
    }
    if audit_mode:
        result["disparity_raw"] = disparity_raw
        result["audit_stats"] = compute_disparity_audit_stats(disparity_raw)
    return result


def _ensure_ffs_repo_on_sys_path(ffs_repo: Path) -> None:
    repo_path = str(ffs_repo)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _configure_tensorrt_runtime_search_paths(trt_root: Path | None) -> list[Any]:
    dll_handles: list[Any] = []
    if trt_root is None:
        return dll_handles
    if not trt_root.exists():
        raise FileNotFoundError(f"TensorRT runtime root not found: {trt_root}")

    search_paths = [trt_root / "lib", trt_root / "bin"]
    existing_search_paths = [path for path in search_paths if path.exists()]
    if existing_search_paths:
        os.environ["PATH"] = os.pathsep.join(
            [*(str(path) for path in existing_search_paths), os.environ.get("PATH", "")]
        )
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for path in existing_search_paths:
            dll_handles.append(os.add_dll_directory(str(path)))
    return dll_handles


def load_tensorrt_model_config(model_dir: str | Path) -> dict[str, Any]:
    import yaml

    model_dir = Path(model_dir).resolve()
    cfg_path = model_dir / "onnx.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"TensorRT metadata not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    image_size = cfg.get("image_size")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError(f"Expected onnx.yaml image_size=[H, W], got {image_size!r} in {cfg_path}")
    cfg["image_size"] = [int(image_size[0]), int(image_size[1])]
    return cfg


def _patch_tensorrt_triton_cost_volume(*, ffs_repo: Path) -> Any:
    _ensure_ffs_repo_on_sys_path(ffs_repo)
    import core.foundation_stereo as foundation_stereo
    import core.submodule as submodule

    replacement = submodule.build_gwc_volume_optimized_pytorch1
    submodule.build_gwc_volume_triton = replacement
    foundation_stereo.build_gwc_volume_triton = replacement
    return foundation_stereo


class FastFoundationStereoRunner:
    def __init__(
        self,
        *,
        ffs_repo: str | Path,
        model_path: str | Path,
        scale: float = 1.0,
        valid_iters: int = 8,
        max_disp: int = 192,
    ) -> None:
        self.ffs_repo = Path(ffs_repo).resolve()
        self.model_path = Path(model_path).resolve()
        self.scale = float(scale)
        self.valid_iters = int(valid_iters)
        self.max_disp = int(max_disp)

        if not self.ffs_repo.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo repo not found: {self.ffs_repo}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo model not found: {self.model_path}")

        import torch
        import yaml
        from omegaconf import OmegaConf

        _disable_torch_compile(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("Fast-FoundationStereoRunner requires CUDA.")

        _ensure_ffs_repo_on_sys_path(self.ffs_repo)
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE, set_logging_format, set_seed

        with open(self.model_path.parent / "cfg.yaml", "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        cfg.update(
            {
                "model_dir": str(self.model_path),
                "scale": self.scale,
                "valid_iters": self.valid_iters,
                "max_disp": self.max_disp,
            }
        )
        self.cfg = OmegaConf.create(cfg)
        self.torch = torch
        self.InputPadder = InputPadder
        self.AMP_DTYPE = AMP_DTYPE
        set_logging_format()
        set_seed(0)
        torch.autograd.set_grad_enabled(False)

        model = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        model.args.valid_iters = self.valid_iters
        model.args.max_disp = self.max_disp
        self.model = model.cuda().eval()

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.tile(image[..., None], (1, 1, 3))
        image = image[..., :3]
        if self.scale != 1.0:
            image = cv2.resize(image, dsize=None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        return image

    def run_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float | list[list[float]]]:
        left = self._prepare_image(left_image)
        right = self._prepare_image(right_image)
        if right.shape[:2] != left.shape[:2]:
            right = cv2.resize(right, dsize=(left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)

        torch = self.torch
        left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = self.InputPadder(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

        with torch.amp.autocast("cuda", enabled=True, dtype=self.AMP_DTYPE):
            # This is the external Fast-FoundationStereo network forward pass.
            # Input: this camera's left/right stereo images. Output: disparity.
            disparity = self.model.forward(
                left_tensor,
                right_tensor,
                iters=self.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )

        disparity = padder.unpad(disparity.float())
        disparity_raw = disparity.data.cpu().numpy().reshape(left.shape[0], left.shape[1]).astype(np.float32)
        return build_disparity_products(
            disparity_raw,
            K_ir_left=K_ir_left,
            baseline_m=baseline_m,
            scale=self.scale,
            valid_iters=self.valid_iters,
            max_disp=self.max_disp,
            audit_mode=audit_mode,
        )


class FastFoundationStereoTensorRTRunner:
    def __init__(
        self,
        *,
        ffs_repo: str | Path,
        model_dir: str | Path,
        trt_root: str | Path | None = None,
    ) -> None:
        self.ffs_repo = Path(ffs_repo).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.trt_root = None if trt_root is None else Path(trt_root).resolve()
        self.feature_engine_path = self.model_dir / "feature_runner.engine"
        self.post_engine_path = self.model_dir / "post_runner.engine"

        if not self.ffs_repo.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo repo not found: {self.ffs_repo}")
        for path in (self.feature_engine_path, self.post_engine_path):
            if not path.exists():
                raise FileNotFoundError(f"TensorRT engine not found: {path}")

        import torch
        from omegaconf import OmegaConf

        _disable_torch_compile(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("FastFoundationStereoTensorRTRunner requires CUDA.")

        self._dll_handles = _configure_tensorrt_runtime_search_paths(self.trt_root)
        foundation_stereo = _patch_tensorrt_triton_cost_volume(ffs_repo=self.ffs_repo)
        from Utils import set_logging_format, set_seed

        cfg_dict = load_tensorrt_model_config(self.model_dir)
        self.cfg = OmegaConf.create(cfg_dict)
        self.engine_height = int(self.cfg.image_size[0])
        self.engine_width = int(self.cfg.image_size[1])
        self.valid_iters = int(self.cfg.valid_iters)
        self.max_disp = int(self.cfg.max_disp)
        self.torch = torch
        set_logging_format()
        set_seed(0)
        torch.autograd.set_grad_enabled(False)
        self.model = foundation_stereo.TrtRunner(
            self.cfg,
            str(self.feature_engine_path),
            str(self.post_engine_path),
        )

    def _prepare_image(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.tile(image[..., None], (1, 1, 3))
        image = image[..., :3]
        scale_x = float(self.engine_width / image.shape[1])
        scale_y = float(self.engine_height / image.shape[0])
        if image.shape[1] != self.engine_width or image.shape[0] != self.engine_height:
            image = cv2.resize(
                image,
                dsize=(self.engine_width, self.engine_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return image, scale_x, scale_y

    def run_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        *,
        K_ir_left: np.ndarray,
        baseline_m: float,
        audit_mode: bool = False,
    ) -> dict[str, np.ndarray | float | list[list[float]]]:
        left, scale_x, scale_y = self._prepare_image(left_image)
        right, _, _ = self._prepare_image(right_image)

        torch = self.torch
        left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        disparity = self.model.forward(left_tensor, right_tensor)
        disparity_raw = disparity.data.cpu().numpy().reshape(self.engine_height, self.engine_width).astype(np.float32)
        uniform_scale = scale_x if abs(scale_x - scale_y) <= 1e-6 else 1.0
        return build_disparity_products(
            disparity_raw,
            K_ir_left=K_ir_left,
            baseline_m=baseline_m,
            scale=uniform_scale,
            scale_x=scale_x,
            scale_y=scale_y,
            valid_iters=self.valid_iters,
            max_disp=self.max_disp,
            audit_mode=audit_mode,
        )
