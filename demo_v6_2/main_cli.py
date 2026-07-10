"""Demo v6.1 realtime orchestration CLI parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from demo_v6_2.main_config import (
    CAMERA_FPS_CHOICES,
    CONFIG_SHAPE_PRIOR_CONTROLLER_NAME,
    DEFAULT_CAMERA_COLOR_EXPOSURE,
    DEFAULT_CAMERA_COLOR_GAIN,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CASE_PREFIX,
    DEFAULT_CHUNK_POLL_INTERVAL_S,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_DATA_PROCESS_BASE_PATH,
    DEFAULT_DEPTH_BACKEND,
    DEFAULT_DOWNSTREAM_MODE,
    DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
    DEFAULT_FAKE_LIVE_CASE,
    DEFAULT_INFERENCE_DTYPE,
    DEFAULT_INPUT_SOURCE,
    DEFAULT_MAIN_DATA_PROCESSING_CUDA_VISIBLE_DEVICES,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_PERCEPTION_DEVICE,
    DEFAULT_PHYSTWIN_SHEN_CONDA_ENV,
    DEFAULT_PHYSTWIN_SHEN_CUDA_VISIBLE_DEVICES,
    DEFAULT_PHYSTWIN_SHEN_PIPELINE_CONFIG_PATH,
    DEFAULT_PHYSTWIN_SHEN_REPO_PATH,
    DEFAULT_REPLAY_FPS,
    DEFAULT_SHAPE_PRIOR_CHUNK_WAIT_TIMEOUT_S,
    DEFAULT_SHAPE_PRIOR_CONFIG,
    DEFAULT_SHAPE_PRIOR_SAM3D_ROOT,
    DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
    DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
    DEFAULT_TRACKER_DEVICE,
    DEFAULT_VISUALIZER_CAM_IDX,
    DEFAULT_VISUALIZER_CONDA_ENV,
    DEFAULT_VISUALIZER_CONTROLLER_RADIUS,
    DEFAULT_VISUALIZER_CUDA_VISIBLE_DEVICES,
    DEFAULT_VISUALIZER_LAYOUT,
    DEFAULT_VISUALIZER_OBJECT_COLOR_MODE,
    DEFAULT_VISUALIZER_OBJECT_RADIUS,
    DEFAULT_VISUALIZER_OBJECT_STRIDE,
    DEFAULT_VISUALIZER_PLAYBACK_FPS,
    DEFAULT_VISUALIZER_POLL_SEC,
    DEFAULT_VISUALIZER_RENDER_MODE,
    DOWNSTREAM_MODES,
    VISUALIZER_LAYOUTS,
)


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


class _StoreFakeLiveCase(argparse.Action):
    """Track whether --fake-live-case was explicitly provided."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        setattr(namespace, "fake_live_case_cli_override", True)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Demo v6.1 realtime orchestration."""
    parser = argparse.ArgumentParser(
        description=(
            "Demo v6.1 realtime data_process_sam3d runner. It turns Demo v6.1 "
            "single-camera fake/live capture into one online data_process_sam3d "
            "case and can launch an online visualizer."
        )
    )
    # Input/chunking options define the online case cadence. The camera can run
    # longer than the requested chunk count; the chunk writer is what stops
    # publishing after max_chunks.
    parser.add_argument(
        "--input-source",
        choices=("fake-live", "live"),
        default=DEFAULT_INPUT_SOURCE,
        help="Camera source mode used when Demo v6.1 launches its own capture.",
    )
    parser.add_argument("--replay-fps", type=float, default=DEFAULT_REPLAY_FPS)
    parser.add_argument(
        "--camera-source-replay-fps",
        type=float,
        default=None,
        help=(
            "Optional Demo v6.1 fake-live pacing FPS. When omitted, Demo v6.1 uses "
            "--replay-fps; Demo v6.1 output metadata/window math still use --replay-fps."
        ),
    )
    parser.add_argument(
        "--fake-live-case",
        action=_StoreFakeLiveCase,
        type=Path,
        default=DEFAULT_FAKE_LIVE_CASE,
        help="Raw data_collect case folder passed to Demo v6.1 fake-live replay.",
    )
    parser.set_defaults(fake_live_case_cli_override=False)
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument(
        "--chunk-poll-interval-s",
        type=float,
        default=DEFAULT_CHUNK_POLL_INTERVAL_S,
        help="Polling interval for realtime frames.jsonl chunk tailing.",
    )
    parser.add_argument(
        "--depth-backend",
        choices=("ir-ffs", "native-realsense"),
        default=DEFAULT_DEPTH_BACKEND,
    )
    parser.add_argument(
        "--chunk-frame-count",
        type=int,
        default=None,
        help="Override chunk length in frames. Defaults to round(replay_fps * chunk_seconds).",
    )
    parser.add_argument(
        "--base-path", type=Path, default=DEFAULT_DATA_PROCESS_BASE_PATH
    )
    parser.add_argument("--case-prefix", default=DEFAULT_CASE_PREFIX)
    parser.add_argument(
        "--main-data-processing-cuda-visible-devices",
        default=DEFAULT_MAIN_DATA_PROCESSING_CUDA_VISIBLE_DEVICES,
        help=(
            "CUDA_VISIBLE_DEVICES for main warmup and the realtime "
            "data_process subprocess."
        ),
    )
    parser.add_argument(
        "--shape-prior-warmup-cuda-visible-devices",
        default=DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
        help="CUDA_VISIBLE_DEVICES for the SAM3D shape-prior warmup stages.",
    )
    parser.add_argument(
        "--perception-device",
        default=DEFAULT_PERCEPTION_DEVICE,
        help="Segmentation/depth/perception device passed to the camera runtime.",
    )
    parser.add_argument(
        "--tracker-device",
        default=DEFAULT_TRACKER_DEVICE,
        help="Point-tracker device passed to the camera runtime.",
    )
    parser.add_argument(
        "--inference-dtype",
        choices=("bfloat16", "float16", "float32"),
        default=DEFAULT_INFERENCE_DTYPE,
        help="Torch autocast dtype passed to the camera runtime.",
    )
    parser.add_argument(
        "--edgetam-mask-logit-threshold",
        type=float,
        default=DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
        help=(
            "Logit threshold passed to EdgeTAM mask binarization. "
            "Lower values make masks more permissive."
        ),
    )
    parser.add_argument(
        "--camera-lossless-max-backlog-seconds",
        type=float,
        default=None,
        help=(
            "Optional strict lossless replay backlog window passed to Demo v6.1. "
            "Omit it to keep Demo v6.1 defaults."
        ),
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        choices=CAMERA_FPS_CHOICES,
        default=DEFAULT_CAMERA_FPS,
        help=(
            "RealSense capture FPS passed to Demo v6.1 live camera. "
            "The default 30 FPS input is sampled at replay FPS for output."
        ),
    )
    parser.add_argument(
        "--camera-serial",
        dest="camera_serials",
        action="append",
        default=None,
        help=(
            "RealSense serial for the live camera subprocess; repeatable so the "
            "schema extends to future multi-camera runtimes. Defaults to "
            "camera.camera_serials from config/default.yaml (the "
            "table_calibrate.pkl camera). The current single-camera runtime "
            "requires exactly one serial and fails fast otherwise."
        ),
    )
    parser.add_argument(
        "--camera-color-exposure",
        type=float,
        default=DEFAULT_CAMERA_COLOR_EXPOSURE,
        help="Manual RealSense RGB exposure passed to Demo v6.1 live camera.",
    )
    parser.add_argument(
        "--camera-color-gain",
        type=float,
        default=DEFAULT_CAMERA_COLOR_GAIN,
        help="Manual RealSense RGB gain passed to Demo v6.1 live camera.",
    )
    # The current chunker consumes prepared frames directly. Keeping only
    # prepared artifacts keeps live runs small and avoids old per-frame outputs
    # becoming part of the v6.1 contract.
    parser.add_argument(
        "--camera-headless-prepared-only",
        dest="camera_headless_prepared_only",
        action="store_true",
        help="Ask Demo v6.1 to write only prepared PhysTwin frames needed by Demo v6.1 chunking.",
    )
    parser.add_argument(
        "--camera-legacy-headless-artifacts",
        dest="camera_headless_prepared_only",
        action="store_false",
        help=(
            "Keep Demo v6.1 legacy per-frame headless artifacts in addition to "
            "prepared realtime frames."
        ),
    )
    parser.set_defaults(camera_headless_prepared_only=True)
    parser.add_argument(
        "--write-input-rgb-timeline",
        dest="write_input_rgb_timeline",
        action="store_true",
        default=None,
        help=(
            "Write input_rgb/*.png and input_frames.jsonl for the Demo v6.1 "
            "side-by-side visualizer."
        ),
    )
    parser.add_argument(
        "--no-write-input-rgb-timeline",
        dest="write_input_rgb_timeline",
        action="store_false",
        help=(
            "Disable the side-by-side input RGB timeline even when the "
            "visualizer layout is side-by-side."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help=(
            "Optional realtime chunk cap for debug/short validation runs. "
            "Omit it to stream until the fake-live recording or live capture ends."
        ),
    )
    parser.add_argument(
        "--camera-capture-dir",
        type=Path,
        default=None,
        help="Headless capture directory for the Demo v6.1 realtime subprocess.",
    )
    parser.add_argument(
        "--source-headless-capture",
        type=Path,
        default=None,
        help="Existing Demo v6.1 headless capture directory to chunk without launching capture.",
    )
    parser.add_argument("--surface-points-npy", type=Path, default=None)
    parser.add_argument("--interior-points-npy", type=Path, default=None)
    parser.add_argument(
        "--shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_true",
        help="Keep SAM3D shape-prior warmup enabled for Demo v6.1 capture.",
    )
    parser.add_argument(
        "--no-shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_false",
        help="Disable SAM3D shape-prior warmup.",
    )
    parser.set_defaults(shape_prior_warmup=True)
    parser.add_argument(
        "--shape-prior-prewarm-stage-workers",
        dest="shape_prior_prewarm_stage_workers",
        action="store_true",
        help=(
            "Spawn pre-warmed one-shot upscale/generate/align workers at app "
            "boot so model loading happens before frame 0 arrives."
        ),
    )
    parser.add_argument(
        "--no-shape-prior-prewarm-stage-workers",
        dest="shape_prior_prewarm_stage_workers",
        action="store_false",
        help="Load shape-prior stage models only when the frame-0 request runs.",
    )
    parser.set_defaults(shape_prior_prewarm_stage_workers=True)
    parser.add_argument(
        "--shape-prior-timeout-ms",
        type=int,
        default=DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
    )
    parser.add_argument(
        "--shape-prior-chunk-wait-timeout-s",
        type=float,
        default=DEFAULT_SHAPE_PRIOR_CHUNK_WAIT_TIMEOUT_S,
        help=(
            "How long Demo v6.1 waits for required shape-prior structure points "
            "before writing final_data chunks."
        ),
    )
    parser.add_argument(
        "--shape-prior-controller-name",
        default=CONFIG_SHAPE_PRIOR_CONTROLLER_NAME,
        help="Controller label used when writing the one-camera shape-prior case.",
    )
    parser.add_argument(
        "--shape-prior-sam3d-root",
        type=Path,
        default=DEFAULT_SHAPE_PRIOR_SAM3D_ROOT,
        help="Optional SAM3D checkout override for shape-prior generation.",
    )
    parser.add_argument(
        "--shape-prior-config",
        type=Path,
        default=DEFAULT_SHAPE_PRIOR_CONFIG,
        help="Optional SAM3D pipeline config override.",
    )
    parser.add_argument("--shape-prior-profile-json", type=Path, default=None)
    parser.add_argument(
        "--asap-augment",
        dest="asap_augment",
        action="store_true",
        help=(
            "Fill invalid object_points in place and publish deformed "
            "shape-prior trajectories as asap_surface_points/"
            "asap_interior_points via live ASAP mesh deformation "
            "(design_spec_v6_1.md)."
        ),
    )
    parser.add_argument(
        "--no-asap-augment",
        dest="asap_augment",
        action="store_false",
        help="Disable ASAP augmentation of published chunks.",
    )
    parser.set_defaults(asap_augment=True)
    parser.add_argument(
        "--asap-mesh-path",
        type=Path,
        default=None,
        help=(
            "Explicit final_mesh.glb override for ASAP augmentation. Defaults "
            "to <shape_prior_case_dir>/shape/matching/final_mesh.glb from the "
            "capture metadata; ASAP fails fast when the mesh is missing."
        ),
    )
    # Exactly one downstream consumer runs per session. demo_visualizer keeps
    # the historical viewer policies (side-by-side starts immediately,
    # output-only waits for the first committed chunk); phystwin_shen starts
    # the external CMA/train/two-viewer full pipeline when shape prior is ready.
    parser.add_argument(
        "--downstream-mode",
        choices=DOWNSTREAM_MODES,
        default=DEFAULT_DOWNSTREAM_MODE,
        help=(
            "Downstream consumer of the online stream: disabled, the Demo "
            "v6.1 viewer window, or the Phystwin_shen full online pipeline."
        ),
    )
    parser.add_argument(
        "--phystwin-shen-repo",
        type=Path,
        default=DEFAULT_PHYSTWIN_SHEN_REPO_PATH,
        help="Phystwin_shen checkout used by --downstream-mode phystwin_shen.",
    )
    parser.add_argument(
        "--phystwin-shen-conda-env",
        default=DEFAULT_PHYSTWIN_SHEN_CONDA_ENV,
    )
    parser.add_argument(
        "--phystwin-shen-pipeline-config",
        type=Path,
        default=DEFAULT_PHYSTWIN_SHEN_PIPELINE_CONFIG_PATH,
        help=(
            "Full-pipeline YAML relative to --phystwin-shen-repo or an "
            "absolute path. Demo config values override its runtime options."
        ),
    )
    parser.add_argument(
        "--phystwin-shen-cuda-visible-devices",
        default=DEFAULT_PHYSTWIN_SHEN_CUDA_VISIBLE_DEVICES,
    )
    parser.add_argument(
        "--visualizer-layout",
        choices=VISUALIZER_LAYOUTS,
        default=DEFAULT_VISUALIZER_LAYOUT,
        help="Viewer layout. side-by-side shows live RGB input next to final_data output chunks.",
    )
    parser.add_argument("--visualizer-conda-env", default=DEFAULT_VISUALIZER_CONDA_ENV)
    parser.add_argument(
        "--visualizer-cuda-visible-devices",
        default=DEFAULT_VISUALIZER_CUDA_VISIBLE_DEVICES,
    )
    parser.add_argument(
        "--visualizer-cam-idx", type=int, default=DEFAULT_VISUALIZER_CAM_IDX
    )
    parser.add_argument(
        "--visualizer-poll-sec", type=float, default=DEFAULT_VISUALIZER_POLL_SEC
    )
    parser.add_argument(
        "--visualizer-playback-fps",
        type=float,
        default=DEFAULT_VISUALIZER_PLAYBACK_FPS,
        help="Playback FPS for the visualizer final_data timeline.",
    )
    parser.add_argument(
        "--visualizer-object-stride",
        type=int,
        default=DEFAULT_VISUALIZER_OBJECT_STRIDE,
    )
    parser.add_argument(
        "--visualizer-object-radius",
        type=int,
        default=DEFAULT_VISUALIZER_OBJECT_RADIUS,
    )
    parser.add_argument(
        "--visualizer-controller-radius",
        type=int,
        default=DEFAULT_VISUALIZER_CONTROLLER_RADIUS,
    )
    parser.add_argument(
        "--visualizer-render-mode",
        choices=("rgb-overlay", "sam3d-final-data"),
        default=DEFAULT_VISUALIZER_RENDER_MODE,
    )
    parser.add_argument(
        "--visualizer-object-color-mode",
        choices=("rainbow", "green", "object-colors"),
        default=DEFAULT_VISUALIZER_OBJECT_COLOR_MODE,
    )
    parser.add_argument(
        "--shape-prior-skip-route-visualizations",
        dest="shape_prior_skip_route_visualizations",
        action="store_true",
    )
    parser.add_argument(
        "--shape-prior-render-route-visualizations",
        dest="shape_prior_skip_route_visualizations",
        action="store_false",
    )
    parser.set_defaults(shape_prior_skip_route_visualizations=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved Demo v6.1 contract and exit.",
    )
    return parser
