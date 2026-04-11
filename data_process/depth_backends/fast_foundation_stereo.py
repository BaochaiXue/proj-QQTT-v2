from __future__ import annotations

import sys
from pathlib import Path

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


def build_disparity_products(
    disparity_raw: np.ndarray,
    *,
    K_ir_left: np.ndarray,
    baseline_m: float,
    scale: float,
    valid_iters: int,
    max_disp: int,
    audit_mode: bool,
) -> dict[str, np.ndarray | float | list[list[float]]]:
    disparity_raw = np.asarray(disparity_raw, dtype=np.float32)
    disparity = disparity_raw.clip(0, None).astype(np.float32)
    K_used = np.asarray(K_ir_left, dtype=np.float32).copy()
    K_used[:2] *= float(scale)
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
        "valid_iters": int(valid_iters),
        "max_disp": int(max_disp),
    }
    if audit_mode:
        result["disparity_raw"] = disparity_raw
        result["audit_stats"] = compute_disparity_audit_stats(disparity_raw)
    return result


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

        sys.path.insert(0, str(self.ffs_repo))
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
