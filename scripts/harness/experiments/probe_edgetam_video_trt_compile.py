from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EDGETAM_REPO = Path("/home/zhangxinjie/EdgeTAM")
DEFAULT_EDGETAM_CONFIG = "configs/edgetam.yaml"
DEFAULT_EDGETAM_CHECKPOINT = Path("/home/zhangxinjie/EdgeTAM/checkpoints/edgetam.pt")
DEFAULT_OUTPUT_DIR = ROOT / "result/edgetam_video_trt_compile_probe_20260502"
DEFAULT_DOC_JSON = ROOT / "docs/generated/edgetam_video_trt_compile_probe.json"
DEFAULT_DOC_MD = ROOT / "docs/generated/edgetam_video_trt_compile_probe.md"


class VideoSamHeadsNoPrompt(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        backbone_features: torch.Tensor,
        high_res_feat0: torch.Tensor,
        high_res_feat1: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.model._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=[high_res_feat0, high_res_feat1],
            multimask_output=False,
        )


class VideoSamHeadsMaskInput(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        backbone_features: torch.Tensor,
        high_res_feat0: torch.Tensor,
        high_res_feat1: torch.Tensor,
        mask_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.model._use_mask_as_output(
            backbone_features=backbone_features,
            high_res_features=[high_res_feat0, high_res_feat1],
            mask_inputs=mask_inputs,
        )


class VideoSamHeadsMaskInputNoAntialias(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        backbone_features: torch.Tensor,
        high_res_feat0: torch.Tensor,
        high_res_feat1: torch.Tensor,
        mask_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
        )
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.model.use_obj_ptrs_in_encoder:
            obj_ptr = torch.zeros(mask_inputs.size(0), self.model.hidden_dim, device=mask_inputs.device)
        else:
            _, _, _, _, _, obj_ptr, _ = self.model._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.model.mask_downsample(mask_inputs_float),
                high_res_features=[high_res_feat0, high_res_feat1],
            )
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.model.pred_obj_scores:
            if self.model.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.model.no_obj_ptr
        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )


class MemoryEncoderWithSpatialPerceiver(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pix_feat: torch.Tensor,
        pred_masks_high_res: torch.Tensor,
        object_score_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_vision_feats = [pix_feat.flatten(2).permute(2, 0, 1)]
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=[(pix_feat.shape[-2], pix_feat.shape[-1])],
            pred_masks_high_res=pred_masks_high_res,
            object_score_logits=object_score_logits,
            is_mask_from_pts=False,
        )
        return maskmem_features, maskmem_pos_enc[-1]


class MemoryAttentionOnePrevious(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        curr: torch.Tensor,
        curr_pos: torch.Tensor,
        spatial_memory: torch.Tensor,
        spatial_memory_pos: torch.Tensor,
        obj_ptr: torch.Tensor,
    ) -> torch.Tensor:
        batch = curr.shape[1]
        hidden_dim = self.model.hidden_dim
        mem_dim = self.model.mem_dim
        ptr_tokens = obj_ptr.reshape(1, batch, hidden_dim // mem_dim, mem_dim)
        ptr_tokens = ptr_tokens.permute(0, 2, 1, 3).flatten(0, 1)
        ptr_pos = torch.zeros_like(ptr_tokens)
        memory = torch.cat([spatial_memory, ptr_tokens], dim=0)
        memory_pos = torch.cat([spatial_memory_pos, ptr_pos], dim=0)
        return self.model.memory_attention(
            curr=curr,
            curr_pos=curr_pos,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=4,
            num_spatial_mem=1,
        )


class MemoryAttentionOnePreviousRealRopePatch(MemoryAttentionOnePrevious):
    def __init__(self, model: torch.nn.Module) -> None:
        _install_real_rope_patch(model)
        super().__init__(model)


def _rotate_real(x: torch.Tensor, freqs_pair: torch.Tensor) -> torch.Tensor:
    freqs_pair = freqs_pair.to(device=x.device)
    x_pair = x.float().reshape(*x.shape[:-1], -1, 2)
    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]
    broadcast_prefix = [1] * (x0.ndim - 2)
    cos = freqs_pair[..., 0].float().view(*broadcast_prefix, freqs_pair.shape[0], freqs_pair.shape[1])
    sin = freqs_pair[..., 1].float().view(*broadcast_prefix, freqs_pair.shape[0], freqs_pair.shape[1])
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack((out0, out1), dim=-1).flatten(-2).type_as(x)


def _apply_rotary_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_pair: torch.Tensor,
    repeat_freqs_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = _rotate_real(xq, freqs_pair)
    if xk.shape[-2] == 0:
        return q, xk
    k_freqs = freqs_pair
    if repeat_freqs_k:
        repeat = xk.shape[-2] // xq.shape[-2]
        k_freqs = freqs_pair.repeat(repeat, 1, 1)
    k = _rotate_real(xk, k_freqs)
    return q, k


def _apply_rotary_real_v2(
    x: torch.Tensor,
    freqs_pair: torch.Tensor,
    repeat_freqs: int,
) -> torch.Tensor:
    if x.shape[-2] == 0:
        return x
    batch, heads, tokens, channels = x.shape
    rope_tokens = freqs_pair.shape[0]
    if tokens == rope_tokens * repeat_freqs:
        x_rope = x
        x_no_rope = None
    else:
        tokens_per_repeat = tokens // repeat_freqs
        no_rope_tokens = tokens_per_repeat - rope_tokens
        x_view = x.view(batch, heads, repeat_freqs, tokens_per_repeat, channels)
        x_no_rope = x_view[..., :no_rope_tokens, :].reshape(batch, heads, -1, channels)
        x_rope = x_view[..., no_rope_tokens:, :].reshape(batch, heads, -1, channels)

    k_freqs = freqs_pair.repeat(repeat_freqs, 1, 1) if repeat_freqs > 1 else freqs_pair
    x_out = _rotate_real(x_rope, k_freqs)
    if x_no_rope is None:
        return x_out
    x_out = x_out.view(batch, heads, repeat_freqs, -1, channels)
    x_no_rope = x_no_rope.view(batch, heads, repeat_freqs, -1, channels)
    return torch.cat((x_no_rope, x_out), dim=3).view(batch, heads, tokens, channels)


def _rope_attention_forward_real(
    self: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_k_exclude_rope: int = 0,
) -> torch.Tensor:
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    num_k_rope = k.size(-2) - num_k_exclude_rope
    q, k_rope = _apply_rotary_real(
        q,
        k[:, :, :num_k_rope],
        self.freqs_cis_real,
        repeat_freqs_k=bool(self.rope_k_repeat),
    )
    k = torch.cat((k_rope, k[:, :, num_k_rope:]), dim=2)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
    out = self._recombine_heads(out)
    return self.out_proj(out)


def _rope_attention_v2_forward_real(
    self: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_k_exclude_rope: int = 0,
    rope_k_repeat: int = -1,
) -> torch.Tensor:
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    q = _apply_rotary_real_v2(q, self.freqs_cis_q_real, repeat_freqs=1)
    num_k_rope = k.size(-2) - num_k_exclude_rope
    k_rope = _apply_rotary_real_v2(
        k[:, :, :num_k_rope],
        self.freqs_cis_k_real,
        repeat_freqs=int(rope_k_repeat),
    )
    k = torch.cat((k_rope, k[:, :, num_k_rope:]), dim=2)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
    out = self._recombine_heads(out)
    return self.out_proj(out)


def _install_real_rope_patch(model: torch.nn.Module) -> None:
    from sam2.modeling.sam.transformer import RoPEAttention, RoPEAttentionv2

    for module in model.modules():
        if getattr(module, "_qqtt_real_rope_patch", False):
            continue
        if isinstance(module, RoPEAttentionv2):
            module.register_buffer(
                "freqs_cis_q_real",
                torch.view_as_real(module.freqs_cis_q.detach().cpu()).clone(),
                persistent=False,
            )
            module.register_buffer(
                "freqs_cis_k_real",
                torch.view_as_real(module.freqs_cis_k.detach().cpu()).clone(),
                persistent=False,
            )
            module.forward = types.MethodType(_rope_attention_v2_forward_real, module)
            module._qqtt_real_rope_patch = True
        elif isinstance(module, RoPEAttention):
            module.register_buffer(
                "freqs_cis_real",
                torch.view_as_real(module.freqs_cis.detach().cpu()).clone(),
                persistent=False,
            )
            module.forward = types.MethodType(_rope_attention_forward_real, module)
            module._qqtt_real_rope_patch = True


def _cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: int = 900,
) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "duration_s": float(time.perf_counter() - started),
        "output": proc.stdout,
    }


def _git_rev_parse(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_edgetam_model(repo: Path, config: str, checkpoint: Path) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    from sam2.build_sam import build_sam2

    model = build_sam2(
        config_file=config,
        ckpt_path=str(checkpoint),
        device="cuda",
        mode="eval",
        apply_postprocessing=False,
    )
    model.eval()
    return model


def _shape_dtype(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [_shape_dtype(item) for item in value]
    return repr(value)


def _onnx_check(path: Path) -> dict[str, Any]:
    import onnx

    try:
        onnx.checker.check_model(str(path))
        model = onnx.load(str(path), load_external_data=False)
        return {
            "status": "ok",
            "node_count": int(len(model.graph.node)),
            "inputs": [value.name for value in model.graph.input],
            "outputs": [value.name for value in model.graph.output],
            "op_histogram": _op_histogram(model),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _op_histogram(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return dict(sorted(counts.items()))


def _trt_env() -> dict[str, str]:
    env = os.environ.copy()
    py = Path(sys.executable).resolve()
    trt_lib = py.parents[1] / "lib/python3.12/site-packages/tensorrt_libs"
    if trt_lib.is_dir():
        env["LD_LIBRARY_PATH"] = f"{trt_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def _build_engine(
    *,
    onnx_path: Path,
    engine_path: Path,
    log_path: Path,
    trtexec: Path,
    fp16: bool,
    timeout_s: int,
) -> dict[str, Any]:
    cmd = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--builderOptimizationLevel=5",
        "--skipInference",
        "--profilingVerbosity=detailed",
    ]
    if fp16:
        cmd.append("--fp16")
    result = _cmd(cmd, env=_trt_env(), timeout_s=timeout_s)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result["output"], encoding="utf-8", errors="replace")
    error_lines = [
        line
        for line in result["output"].splitlines()
        if " [E] " in line or "ERROR:" in line or "FAILED" in line
    ]
    return {
        "status": "ok" if result["returncode"] == 0 and engine_path.is_file() else "failed",
        "returncode": result["returncode"],
        "duration_s": result["duration_s"],
        "engine_path": str(engine_path) if engine_path.is_file() else None,
        "engine_size_bytes": engine_path.stat().st_size if engine_path.is_file() else None,
        "log_path": str(log_path),
        "error_lines": error_lines[-12:],
        "tail": "\n".join(result["output"].splitlines()[-40:]),
    }


def _attempt_export(
    *,
    key: str,
    module_factory: Callable[[torch.nn.Module], torch.nn.Module],
    input_factory: Callable[[torch.device], dict[str, torch.Tensor]],
    output_names: list[str],
    model: torch.nn.Module,
    output_dir: Path,
    opset: int,
    trtexec: Path,
    build_trt: bool,
    fp16_trt: bool,
    trt_timeout_s: int,
) -> dict[str, Any]:
    attempt_dir = output_dir / key
    attempt_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = attempt_dir / f"{key}.onnx"
    engine_path = attempt_dir / f"{key}.engine"
    record: dict[str, Any] = {
        "key": key,
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "opset": int(opset),
    }
    module = module_factory(model).eval().to("cuda")
    inputs = input_factory(torch.device("cuda"))
    input_names = list(inputs.keys())
    args = tuple(inputs[name] for name in input_names)

    try:
        with torch.inference_mode():
            eager_outputs = module(*args)
            torch.cuda.synchronize()
        record["eager"] = {"status": "ok", "outputs": _shape_dtype(eager_outputs)}
    except Exception as exc:
        record["eager"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        return record

    export_errors: list[dict[str, Any]] = []
    for export_device in ("cuda", "cpu"):
        try:
            module = module.to(export_device)
            export_inputs = input_factory(torch.device(export_device))
            export_args = tuple(export_inputs[name] for name in input_names)
            started = time.perf_counter()
            with torch.inference_mode():
                torch.onnx.export(
                    module,
                    export_args,
                    str(onnx_path),
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=int(opset),
                    do_constant_folding=True,
                    dynamo=False,
                    external_data=True,
                )
            record["onnx_export"] = {
                "status": "ok",
                "device": export_device,
                "duration_s": float(time.perf_counter() - started),
                "size_bytes": onnx_path.stat().st_size if onnx_path.is_file() else None,
            }
            if export_errors:
                record["onnx_export_fallback_errors"] = export_errors
            break
        except Exception as exc:
            export_errors.append(
                {
                    "device": export_device,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            record["onnx_export"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
    module.to("cuda")
    if record.get("onnx_export", {}).get("status") != "ok":
        if export_errors:
            record["onnx_export_errors"] = export_errors
        return record

    record["onnx_check"] = _onnx_check(onnx_path)
    if not build_trt or record["onnx_check"].get("status") != "ok":
        return record

    record["trt_build"] = _build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        log_path=attempt_dir / f"{key}_trtexec.log",
        trtexec=trtexec,
        fp16=fp16_trt,
        timeout_s=trt_timeout_s,
    )
    return record


def _common_feature_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "backbone_features": torch.randn(1, 256, 64, 64, device=device),
        "high_res_feat0": torch.randn(1, 32, 256, 256, device=device),
        "high_res_feat1": torch.randn(1, 64, 128, 128, device=device),
    }


def _video_no_prompt_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    return _common_feature_inputs(device)


def _video_mask_input_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    values = _common_feature_inputs(device)
    mask = torch.zeros(1, 1, 1024, 1024, device=device)
    mask[:, :, 256:768, 320:704] = 1.0
    values["mask_inputs"] = mask
    return values


def _memory_encoder_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    mask = torch.randn(1, 1, 1024, 1024, device=device)
    return {
        "pix_feat": torch.randn(1, 256, 64, 64, device=device),
        "pred_masks_high_res": mask,
        "object_score_logits": torch.ones(1, 1, device=device),
    }


def _memory_attention_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "curr": torch.randn(4096, 1, 256, device=device),
        "curr_pos": torch.randn(4096, 1, 256, device=device),
        "spatial_memory": torch.randn(512, 1, 64, device=device),
        "spatial_memory_pos": torch.randn(512, 1, 64, device=device),
        "obj_ptr": torch.randn(1, 256, device=device),
    }


def _attempt_specs() -> list[dict[str, Any]]:
    return [
        {
            "key": "video_sam_heads_no_prompt",
            "factory": VideoSamHeadsNoPrompt,
            "inputs": _video_no_prompt_inputs,
            "outputs": [
                "low_res_multimasks",
                "high_res_multimasks",
                "ious",
                "low_res_masks",
                "high_res_masks",
                "obj_ptr",
                "object_score_logits",
            ],
        },
        {
            "key": "video_sam_heads_mask_input",
            "factory": VideoSamHeadsMaskInput,
            "inputs": _video_mask_input_inputs,
            "outputs": [
                "low_res_multimasks",
                "high_res_multimasks",
                "ious",
                "low_res_masks",
                "high_res_masks",
                "obj_ptr",
                "object_score_logits",
            ],
        },
        {
            "key": "video_sam_heads_mask_input_no_antialias_patch",
            "factory": VideoSamHeadsMaskInputNoAntialias,
            "inputs": _video_mask_input_inputs,
            "outputs": [
                "low_res_multimasks",
                "high_res_multimasks",
                "ious",
                "low_res_masks",
                "high_res_masks",
                "obj_ptr",
                "object_score_logits",
            ],
        },
        {
            "key": "memory_encoder_spatial_perceiver",
            "factory": MemoryEncoderWithSpatialPerceiver,
            "inputs": _memory_encoder_inputs,
            "outputs": ["maskmem_features", "maskmem_pos_enc"],
        },
        {
            "key": "memory_attention_one_previous",
            "factory": MemoryAttentionOnePrevious,
            "inputs": _memory_attention_inputs,
            "outputs": ["memory_conditioned_features"],
        },
        {
            "key": "memory_attention_one_previous_real_rope_patch",
            "factory": MemoryAttentionOnePreviousRealRopePatch,
            "inputs": _memory_attention_inputs,
            "outputs": ["memory_conditioned_features"],
        },
    ]


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# EdgeTAM Video TensorRT Compile Probe",
        "",
        "## Scope",
        "",
        "This probe attempted component-level export/build for the EdgeTAM pieces missing from the existing ONNX/TRT sanity path.",
        "It does not implement a full `propagate_in_video(state)` scheduler.",
        "",
        "## Environment",
        "",
        f"- EdgeTAM repo: `{payload.get('edgetam_repo')}`",
        f"- EdgeTAM commit: `{payload.get('edgetam_commit')}`",
        f"- checkpoint: `{payload.get('checkpoint')}`",
        f"- checkpoint sha256: `{payload.get('checkpoint_sha256')}`",
        f"- torch: `{payload.get('torch')}`",
        f"- torch CUDA: `{payload.get('torch_cuda')}`",
        f"- GPU: `{payload.get('gpu')}`",
        f"- TensorRT: `{payload.get('tensorrt')}`",
        f"- output dir: `{payload.get('output_dir')}`",
        "",
        "## Results",
        "",
        "| component | eager | ONNX export | ONNX check | TensorRT build |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in payload.get("attempts", []):
        lines.append(
            f"| {record.get('key')} | "
            f"{record.get('eager', {}).get('status', 'not_run')} | "
            f"{record.get('onnx_export', {}).get('status', 'not_run')} | "
            f"{record.get('onnx_check', {}).get('status', 'not_run')} | "
            f"{record.get('trt_build', {}).get('status', 'not_run')} |"
        )
    successes = [
        record for record in payload.get("attempts", [])
        if record.get("trt_build", {}).get("status") == "ok"
    ]
    if successes:
        lines.extend(["", "## TensorRT Engines", "", "| component | engine | size |", "| --- | --- | ---: |"])
        for record in successes:
            build = record.get("trt_build", {})
            size_mb = float(build.get("engine_size_bytes", 0.0)) / (1024.0 * 1024.0)
            lines.append(f"| {record.get('key')} | `{build.get('engine_path')}` | {size_mb:.2f} MiB |")
    lines.extend(["", "## Failure Details", ""])
    for record in payload.get("attempts", []):
        detail_lines: list[str] = []
        for stage in ("eager", "onnx_export", "onnx_check", "trt_build"):
            stage_record = record.get(stage, {})
            if stage_record.get("status") == "failed":
                error = stage_record.get("error")
                if not error and stage_record.get("error_lines"):
                    error = " / ".join(str(line) for line in stage_record.get("error_lines", []))
                if not error:
                    error = stage_record.get("tail") or ""
                detail_lines.append(f"- {stage}: `{stage_record.get('error_type', 'failed')}` {str(error)[:1600]}")
                if stage_record.get("log_path"):
                    detail_lines.append(f"- log: `{stage_record.get('log_path')}`")
        if detail_lines:
            lines.append(f"### {record.get('key')}")
            lines.extend(detail_lines)
            lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- A TensorRT build success here means a component can become part of a future scheduler.",
            "- A full EdgeTAM video TRT backend still needs host-side state management for `maskmem_features`, `maskmem_pos_enc`, `obj_ptr`, frame selection, and memory ring updates.",
            "- The existing ONNX/TRT component result remains frame-level SAM-style mask sanity until these memory components and scheduler pass correctness checks.",
            "",
            "## Practical Next Step",
            "",
            "The memory side is now compile-feasible with local wrapper patches. The SAM prompt/video head still needs a TensorRT-friendly prompt encoder rewrite, because the current PyTorch-exported ONNX graph fails TensorRT parsing at `sam_prompt_encoder/Where_6` broadcast shape handling.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe EdgeTAM video component ONNX/TensorRT compile feasibility.")
    parser.add_argument("--edgetam-repo", type=Path, default=DEFAULT_EDGETAM_REPO)
    parser.add_argument("--config", default=DEFAULT_EDGETAM_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_EDGETAM_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_DOC_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_DOC_MD)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--trtexec", type=Path, default=Path("/usr/local/bin/trtexec"))
    parser.add_argument("--skip-trt", action="store_true")
    parser.add_argument("--fp32-trt", action="store_true")
    parser.add_argument("--trt-timeout-s", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_grad_enabled(False)

    model = _load_edgetam_model(args.edgetam_repo.resolve(), args.config, args.checkpoint.resolve())
    try:
        import tensorrt as trt

        trt_version = trt.__version__
    except Exception as exc:  # pragma: no cover - depends on local env
        trt_version = f"unavailable: {exc!r}"

    attempts = []
    for spec in _attempt_specs():
        print(f"[edgetam-video-trt] attempt {spec['key']}", flush=True)
        record = _attempt_export(
            key=str(spec["key"]),
            module_factory=spec["factory"],
            input_factory=spec["inputs"],
            output_names=list(spec["outputs"]),
            model=model,
            output_dir=args.output_dir.resolve(),
            opset=int(args.opset),
            trtexec=args.trtexec.resolve(),
            build_trt=not args.skip_trt,
            fp16_trt=not args.fp32_trt,
            trt_timeout_s=int(args.trt_timeout_s),
        )
        attempts.append(record)
        print(
            "[edgetam-video-trt] "
            f"{record['key']}: eager={record.get('eager', {}).get('status')} "
            f"onnx={record.get('onnx_export', {}).get('status')} "
            f"trt={record.get('trt_build', {}).get('status', 'not_run')}",
            flush=True,
        )

    payload: dict[str, Any] = {
        "edgetam_repo": str(args.edgetam_repo.resolve()),
        "edgetam_commit": _git_rev_parse(args.edgetam_repo.resolve()),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tensorrt": trt_version,
        "opset": int(args.opset),
        "attempts": attempts,
        "summary": {
            "eager_ok": sum(1 for item in attempts if item.get("eager", {}).get("status") == "ok"),
            "onnx_export_ok": sum(1 for item in attempts if item.get("onnx_export", {}).get("status") == "ok"),
            "onnx_check_ok": sum(1 for item in attempts if item.get("onnx_check", {}).get("status") == "ok"),
            "trt_build_ok": sum(1 for item in attempts if item.get("trt_build", {}).get("status") == "ok"),
            "attempt_count": len(attempts),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(args.markdown_output, payload)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[edgetam-video-trt] wrote {args.json_output}", flush=True)
    print(f"[edgetam-video-trt] wrote {args.markdown_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
