"""Frozen orchestrator run configuration and the static run contract."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from demo_v6_2.main_options import (
    python_command_prefix,
    resolve_camera_serials,
    resolve_camera_source_replay_fps,
    resolve_downstream_mode,
    resolve_shape_prior_warmup_cuda_visible_devices,
    resolve_visualizer_cuda_visible_devices,
    resolve_visualizer_layout,
    resolve_write_input_rgb_timeline,
)
from demo_v6_2.main_subprocess import build_visualizer_command, default_capture_dir
from demo_v6_2.orchestration.main_config import (
    DEFAULT_PHYSTWIN_SHEN_CUDA_VISIBLE_DEVICES,
    DEFAULT_PHYSTWIN_SHEN_RUNTIME_CONFIG,
    DOWNSTREAM_MODE_DEMO_VISUALIZER,
    DOWNSTREAM_MODE_PHYSTWIN_SHEN,
    EDGE_TAM_TRACKING_IDENTITIES,
    VISUALIZER_LAYOUT_SIDE_BY_SIDE,
)
from demo_v6_2.orchestration.main_layout import (
    resolve_online_dir,
    resolve_shape_prior_case_root,
    resolve_shape_prior_points_npz,
    resolve_static_data_path,
)
from demo_v6_2.phystwin_shen_launch import (
    PhystwinShenSettings,
    build_full_pipeline_command,
    validate_phystwin_shen_repo,
    validate_phystwin_shen_settings,
)


@dataclass(frozen=True)
class OrchestratorRunConfig:
    """Immutable orchestrator option snapshot derived once from parsed args.

    Built right after CLI parsing (mirror of ``mdp.cli.RunMode``) so every
    cross-option constraint is validated and every derived option is resolved
    exactly once per run instead of being re-derived from the argparse
    Namespace at each call site.
    """

    chunk_frame_count: int
    camera_serials: tuple[str, ...]
    camera_source_replay_fps: float
    camera_source_replay_fps_override: float | None
    downstream_mode: str
    demo_visualizer_enabled: bool
    phystwin_shen_enabled: bool
    side_by_side: bool
    write_input_rgb_timeline: bool
    main_data_processing_cuda_visible_devices: str
    shape_prior_warmup_cuda_visible_devices: str
    visualizer_cuda_visible_devices: str
    visualizer_layout: str
    visualizer_start_policy: str
    visualizer_fps: float
    visualizer_object_color_mode: str
    phystwin_shen_cuda_visible_devices: str
    phystwin_shen_settings: PhystwinShenSettings | None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> OrchestratorRunConfig:
        """Validate cross-option constraints, then build the frozen snapshot."""
        if args.chunk_frame_count is not None:
            chunk_frame_count = int(args.chunk_frame_count)
        else:
            # Chunks are sized on the output replay timeline (--replay-fps),
            # not the camera capture FPS, so each chunk spans chunk_seconds of
            # output.
            chunk_frame_count = int(
                round(float(args.replay_fps) * float(args.chunk_seconds))
            )
        if chunk_frame_count <= 0:
            raise ValueError("chunk frame count must be positive")
        if float(args.chunk_poll_interval_s) <= 0.0:
            raise ValueError("--chunk-poll-interval-s must be positive")
        if not np.isfinite(float(args.visualizer_playback_fps)):
            raise ValueError("--visualizer-playback-fps must be finite")
        if float(args.visualizer_playback_fps) <= 0.0:
            raise ValueError("--visualizer-playback-fps must be positive")
        camera_serials = tuple(resolve_camera_serials(args))
        camera_source_replay_fps = resolve_camera_source_replay_fps(args)
        if (
            bool(getattr(args, "fake_live_case_cli_override", False))
            and str(args.input_source) != "fake-live"
        ):
            raise ValueError("--fake-live-case requires --input-source fake-live")
        if args.max_chunks is not None and int(args.max_chunks) <= 0:
            raise ValueError("--max-chunks must be positive when provided")
        if not np.isfinite(float(args.edgetam_mask_logit_threshold)):
            raise ValueError("--edgetam-mask-logit-threshold must be finite")
        main_data_processing_cuda_visible_devices = str(
            args.main_data_processing_cuda_visible_devices
        ).strip()
        if not main_data_processing_cuda_visible_devices:
            raise ValueError(
                "--main-data-processing-cuda-visible-devices must be non-empty"
            )
        downstream_mode = resolve_downstream_mode(args)
        demo_visualizer_enabled = downstream_mode == DOWNSTREAM_MODE_DEMO_VISUALIZER
        phystwin_shen_enabled = downstream_mode == DOWNSTREAM_MODE_PHYSTWIN_SHEN
        visualizer_layout = resolve_visualizer_layout(args)
        side_by_side = visualizer_layout == VISUALIZER_LAYOUT_SIDE_BY_SIDE
        phystwin_shen_cuda_visible_devices = str(
            getattr(
                args,
                "phystwin_shen_cuda_visible_devices",
                DEFAULT_PHYSTWIN_SHEN_CUDA_VISIBLE_DEVICES,
            )
        ).strip()
        if not phystwin_shen_cuda_visible_devices:
            raise ValueError(
                "--phystwin-shen-cuda-visible-devices must be non-empty when "
                "downstream mode is phystwin_shen"
            )
        phystwin_shen_settings: PhystwinShenSettings | None = None
        if phystwin_shen_enabled:
            # Fail fast before launching subprocesses: a bad checkout/port/GPU
            # config should not surface only at shape-prior-ready time.
            if not bool(args.asap_augment):
                raise ValueError(
                    "--downstream-mode phystwin_shen requires --asap-augment"
                )
            phystwin_shen_settings = PhystwinShenSettings(
                repo_path=Path(args.phystwin_shen_repo).expanduser().resolve(),
                pipeline_config=Path(args.phystwin_shen_pipeline_config).expanduser(),
                conda_env=str(args.phystwin_shen_conda_env),
                base_path=Path(args.base_path).expanduser().resolve(),
                cuda_visible_devices=phystwin_shen_cuda_visible_devices,
                runtime_config=copy.deepcopy(DEFAULT_PHYSTWIN_SHEN_RUNTIME_CONFIG),
            )
            validate_phystwin_shen_repo(
                phystwin_shen_settings.repo_path,
                phystwin_shen_settings.pipeline_config,
            )
            validate_phystwin_shen_settings(
                phystwin_shen_settings,
                python_prefix=python_command_prefix(args.phystwin_shen_conda_env),
            )
            if not str(args.phystwin_shen_conda_env).strip():
                raise ValueError("--phystwin-shen-conda-env must be non-empty")
        if demo_visualizer_enabled:
            if int(args.visualizer_cam_idx) < 0:
                raise ValueError("--visualizer-cam-idx must be non-negative")
            if float(args.visualizer_poll_sec) <= 0.0:
                raise ValueError("--visualizer-poll-sec must be positive")
            if int(args.visualizer_object_stride) <= 0:
                raise ValueError("--visualizer-object-stride must be positive")
            if int(args.visualizer_object_radius) <= 0:
                raise ValueError("--visualizer-object-radius must be positive")
            if int(args.visualizer_controller_radius) <= 0:
                raise ValueError("--visualizer-controller-radius must be positive")
        if not demo_visualizer_enabled:
            visualizer_start_policy = "disabled"
        elif side_by_side:
            visualizer_start_policy = "immediate_after_camera_start"
        else:
            visualizer_start_policy = "after_first_committed_online_chunk"
        return cls(
            chunk_frame_count=chunk_frame_count,
            camera_serials=camera_serials,
            camera_source_replay_fps=camera_source_replay_fps,
            camera_source_replay_fps_override=(
                None
                if args.camera_source_replay_fps is None
                else float(args.camera_source_replay_fps)
            ),
            downstream_mode=downstream_mode,
            demo_visualizer_enabled=demo_visualizer_enabled,
            phystwin_shen_enabled=phystwin_shen_enabled,
            side_by_side=side_by_side,
            write_input_rgb_timeline=resolve_write_input_rgb_timeline(args),
            main_data_processing_cuda_visible_devices=(
                main_data_processing_cuda_visible_devices
            ),
            shape_prior_warmup_cuda_visible_devices=(
                resolve_shape_prior_warmup_cuda_visible_devices(args)
            ),
            visualizer_cuda_visible_devices=resolve_visualizer_cuda_visible_devices(
                args
            ),
            visualizer_layout=visualizer_layout,
            visualizer_start_policy=visualizer_start_policy,
            visualizer_fps=float(args.visualizer_playback_fps),
            visualizer_object_color_mode=str(args.visualizer_object_color_mode),
            phystwin_shen_cuda_visible_devices=phystwin_shen_cuda_visible_devices,
            phystwin_shen_settings=phystwin_shen_settings,
        )


def static_run_contract(
    args: argparse.Namespace, config: OrchestratorRunConfig
) -> dict[str, object]:
    """Static option keys shared by the dry-run contract and run_summary.json.

    ``visualizer_command`` and ``visualizer_capture_dir`` are deliberately not
    here: the dry-run contract always reports capture_dir=None while the run
    summary reports the side-by-side-conditional values.
    """
    return {
        "demo_version": "demo_v6_1",
        "camera_serials": list(config.camera_serials),
        "camera_source_replay_fps": config.camera_source_replay_fps,
        "camera_source_replay_fps_override": config.camera_source_replay_fps_override,
        "camera_lossless_max_backlog_seconds": args.camera_lossless_max_backlog_seconds,
        "camera_headless_prepared_only": bool(args.camera_headless_prepared_only),
        "write_input_rgb_timeline": config.write_input_rgb_timeline,
        "chunk_frame_count": config.chunk_frame_count,
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "max_chunks": args.max_chunks,
        "base_path": str(args.base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "shape_prior_case_root": str(resolve_shape_prior_case_root(args)),
        "shape_prior_points_npz": str(resolve_shape_prior_points_npz(args)),
        "shape_prior_warmup_cuda_visible_devices": (
            config.shape_prior_warmup_cuda_visible_devices
        ),
        "shape_prior_controller_name": str(args.shape_prior_controller_name),
        "shape_prior_sam3d_root": (
            None
            if args.shape_prior_sam3d_root is None
            else str(args.shape_prior_sam3d_root)
        ),
        "shape_prior_config": (
            None if args.shape_prior_config is None else str(args.shape_prior_config)
        ),
        "main_data_processing_capture_dir": str(
            default_capture_dir(args, Path(args.base_path))
        ),
        "main_data_processing_cuda_visible_devices": (
            config.main_data_processing_cuda_visible_devices
        ),
        "downstream_mode": config.downstream_mode,
        "visualizer_layout": config.visualizer_layout,
        "visualizer_cuda_visible_devices": config.visualizer_cuda_visible_devices,
        "visualizer_start_policy": config.visualizer_start_policy,
        "visualizer_fps": config.visualizer_fps,
        "visualizer_object_color_mode": config.visualizer_object_color_mode,
    }


def dry_run_contract(
    args: argparse.Namespace, config: OrchestratorRunConfig
) -> dict[str, object]:
    """Return the dry-run contract without launching subprocesses."""
    settings = config.phystwin_shen_settings
    return {
        **static_run_contract(args, config),
        "input_source": str(args.input_source),
        "replay_fps": float(args.replay_fps),
        "chunk_seconds": float(args.chunk_seconds),
        "depth_backend": str(args.depth_backend),
        "edgetam_tracking_identities": list(EDGE_TAM_TRACKING_IDENTITIES),
        "perception_device": str(args.perception_device),
        "tracker_device": str(args.tracker_device),
        "inference_dtype": str(args.inference_dtype),
        "shape_prior_warmup": bool(args.shape_prior_warmup),
        "shape_prior_prewarm_stage_workers": bool(
            args.shape_prior_prewarm_stage_workers
        ),
        "shape_prior_chunk_wait_timeout_s": float(
            args.shape_prior_chunk_wait_timeout_s
        ),
        "visualizer_command": build_visualizer_command(args),
        "visualizer_capture_dir": None,
        "phystwin_shen_repo_path": str(args.phystwin_shen_repo),
        "phystwin_shen_conda_env": str(args.phystwin_shen_conda_env),
        "phystwin_shen_pipeline_config": str(args.phystwin_shen_pipeline_config),
        "phystwin_shen_cuda_visible_devices": (
            config.phystwin_shen_cuda_visible_devices
        ),
        "phystwin_shen_viewer_urls": (
            {} if settings is None else settings.viewer_urls
        ),
        "phystwin_shen_pipeline_command": (
            None
            if settings is None
            else build_full_pipeline_command(
                settings,
                python_prefix=python_command_prefix(args.phystwin_shen_conda_env),
            )
        ),
    }
