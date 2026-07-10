"""CLI: build_parser, apply_demo_preset, derived-mode accessors, validate_args."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403

def _parse_rgb_triplet(value: str) -> tuple[int, int, int]:
    """Parse RGB triplet."""
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected R,G,B")
    try:
        rgb = tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer R,G,B") from exc
    if any(item < 0 or item > 255 for item in rgb):
        raise argparse.ArgumentTypeError("R,G,B values must be in [0, 255]")
    return rgb  # type: ignore[return-value]


def _is_replay_input_source(input_source: str) -> bool:
    """Return whether replay input source."""
    return str(input_source) in {INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_RECORDING}


def depth_backend_label(args: argparse.Namespace) -> str:
    """Return the depth backend label."""
    label = getattr(args, "depth_backend_label", None)
    if label is not None and str(label):
        return str(label)
    return str(args.depth_source)


def runtime_metadata_identity(args: argparse.Namespace) -> dict[str, str]:
    """Return the runtime metadata identity."""
    payload: dict[str, str] = {}
    product_name = getattr(args, "runtime_product_name", None)
    if product_name is not None and str(product_name).strip():
        payload["runtime_product_name"] = str(product_name).strip()
    demo_version = getattr(args, "metadata_demo_version", None)
    if demo_version is not None and str(demo_version).strip():
        payload["demo_version"] = str(demo_version).strip()
    reference_pipeline = getattr(args, "metadata_reference_pipeline", None)
    if reference_pipeline is not None and str(reference_pipeline).strip():
        payload["reference_pipeline"] = str(reference_pipeline).strip()
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Single-D455 realtime HF EdgeTAM masked point-cloud demo. Captures live "
            "RealSense color plus FFS stereo depth by default, tracks controller/object "
            "or object-only with one HF EdgeTAM streaming session, and writes headless "
            "masked-PCD/tracking capture products."
        )
    )
    parser.add_argument("--fps", choices=SUPPORTED_CAPTURE_FPS, type=int, default=DEFAULT_FPS, help="Capture FPS.")
    parser.add_argument(
        "--serial",
        default=None,
        help=(
            "RealSense serial for live capture. The default picks the first "
            "sorted D400-series device, which with several cameras connected "
            "can select one without the requested RGB profile (e.g. a D405) "
            "or one that does not match table_calibrate.pkl."
        ),
    )
    parser.add_argument(
        "--warmup-rgb-preview",
        dest="warmup_rgb_preview",
        action="store_true",
        help=(
            "Show the live RGB camera-input window during warm-up (every "
            "downstream mode); it closes when warm-up finishes and "
            "immediately on failure/cancel/early exit."
        ),
    )
    parser.add_argument(
        "--no-warmup-rgb-preview",
        dest="warmup_rgb_preview",
        action="store_false",
        help="Disable the warm-up live RGB input preview window.",
    )
    parser.set_defaults(warmup_rgb_preview=True)
    parser.add_argument(
        "--input-source",
        choices=INPUT_SOURCES,
        default=INPUT_SOURCE_LIVE,
        help=(
            "Frame source. fake-live replays a raw single-camera data_collect case at camera cadence, "
            "dropping source frames to preserve recording time when replay FPS is lower; recording is kept "
            "as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--recording-case",
        type=Path,
        default=None,
        help="Raw data_collect case folder for --input-source recording or fake-live.",
    )
    parser.add_argument(
        "--fake-live-case",
        dest="recording_case",
        type=Path,
        default=None,
        help=f"Alias for --recording-case. fake-live defaults to {DEFAULT_FAKE_LIVE_CASE}.",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=0.0,
        help=(
            "Replay FPS for --input-source recording or fake-live. For fake-live this is the emitted "
            "sample cadence; lower values drop source frames rather than slow motion. Use 0 to read metadata fps."
        ),
    )
    parser.add_argument(
        "--lossless-max-backlog-seconds",
        type=float,
        default=DEFAULT_LOSSLESS_MAX_BACKLOG_SECONDS,
        help=(
            "Maximum strict lossless input-FPS backlog window before treating "
            "the run as stalled."
        ),
    )
    parser.add_argument(
        "--lossless-input-fps",
        type=float,
        default=DEFAULT_LOSSLESS_INPUT_FPS,
        help="Strict lossless camera/fake-live cadence used by tracker-synchronized masked PCD replay.",
    )
    parser.add_argument(
        "--table-calibrate",
        type=Path,
        default=None,
        help=(
            "Optional single-camera table Z=0 calibration pickle. When provided, Demo 3.x PCD "
            "and 3D tracker markers are transformed from camera_color_frame into table_world_z0."
        ),
    )
    parser.add_argument(
        "--depth-source",
        choices=DEPTH_SOURCES,
        default=DEFAULT_DEPTH_SOURCE,
        help="Depth source. ffs streams color+IR stereo and runs local TensorRT FFS.",
    )
    parser.add_argument(
        "--depth-backend-label",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ffs-repo",
        type=Path,
        default=DEFAULT_FFS_REPO,
        help="Fast-FoundationStereo repo path. Used when --depth-source ffs.",
    )
    parser.add_argument(
        "--ffs-trt-model-dir",
        type=Path,
        default=DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR,
        help=(
            "Two-stage TensorRT FFS engine directory. Default is the 20-30-48 / "
            "valid_iters=4 / 848x480->864x480 / builderOptimizationLevel=5 artifact."
        ),
    )
    parser.add_argument(
        "--ffs-trt-root",
        type=Path,
        default=None,
        help="Optional TensorRT Python package/root override forwarded to the FFS runner.",
    )
    parser.add_argument(
        "--emitter",
        choices=("auto", "on", "off"),
        default="auto",
        help="RealSense emitter policy. Defaults to leaving the current device setting unchanged.",
    )
    parser.add_argument(
        "--color-exposure",
        type=float,
        default=None,
        help="Optional manual RealSense RGB exposure. When set, RGB auto exposure is disabled.",
    )
    parser.add_argument(
        "--color-gain",
        type=float,
        default=None,
        help="Optional manual RealSense RGB gain. When set, RGB auto exposure is disabled.",
    )
    parser.add_argument(
        "--track-mode",
        choices=TRACK_MODES,
        default=DEFAULT_TRACK_MODE,
        help="Objects tracked by EdgeTAM. Use none for capture/depth isolation profiling.",
    )
    parser.add_argument(
        "--pcd-mode",
        choices=PCD_MODES,
        default=DEFAULT_PCD_MODE,
        help="Point-cloud stage mode. Use none for EdgeTAM/depth isolation profiling.",
    )
    parser.add_argument(
        "--demo-visual-mode",
        choices=DEMO_VISUAL_MODES,
        default=DEFAULT_DEMO_VISUAL_MODE,
        help="Visual presentation hint forwarded from single-camera wrappers.",
    )
    parser.add_argument("--runtime-product-name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--metadata-demo-version", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--metadata-reference-pipeline", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--tracker-backend",
        choices=TRACKER_BACKENDS,
        default=DEFAULT_TRACKER_BACKEND,
        help="Optional point-tracker overlay backend. tapnextpp adds 3D query/track markers.",
    )
    parser.add_argument(
        "--tracker-device",
        default="cuda:1",
        help="Device for the point-tracker backend. Use cuda:1 on the dual-4090 demo machine.",
    )
    parser.add_argument(
        "--tracker-overlay-max-points",
        type=int,
        default=512,
        help="Maximum visible tracker markers rendered per frame. 0 renders all visible selected points.",
    )
    parser.add_argument(
        "--tracking-product-backend",
        choices=TRACKING_PRODUCT_BACKENDS,
        default=DEFAULT_TRACKING_PRODUCT_BACKEND,
        help=(
            "Final tracking product backend. realtime-overlay keeps the live marker product; "
            "phystwin-strict-tracking writes PhysTwin-compatible headless artifacts using TAPNext++ tracks."
        ),
    )
    parser.add_argument(
        "--phystwin-strict-output-dir",
        type=Path,
        default=None,
        help="Output directory for --tracking-product-backend phystwin-strict-tracking. Defaults to <headless-capture-dir>/phystwin_like.",
    )
    parser.add_argument(
        "--shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_true",
        help="Enable the optional SAM3D shape-prior warmup request path.",
    )
    parser.add_argument(
        "--no-shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_false",
        help="Disable the optional SAM3D shape-prior warmup request path.",
    )
    parser.set_defaults(shape_prior_warmup=False)
    parser.add_argument(
        "--shape-prior-prewarm-stage-workers",
        dest="shape_prior_prewarm_stage_workers",
        action="store_true",
        help=(
            "Spawn pre-warmed one-shot upscale/generate/align workers at boot "
            "so shape-prior model loading happens before frame 0."
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
        default=shape_prior_warmup.DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
    )
    parser.add_argument("--shape-prior-profile-json", type=Path, default=None)
    parser.add_argument(
        "--shape-prior-case-root",
        type=Path,
        default=Path("outputs_v6_1") / "shape_prior_case",
    )
    parser.add_argument(
        "--shape-prior-points-npz",
        type=Path,
        default=shape_prior_warmup.POINTS_NPZ,
    )
    parser.add_argument(
        "--shape-prior-warmup-cuda-visible-devices",
        default=shape_prior_warmup.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
    )
    parser.add_argument(
        "--shape-prior-controller-name",
        default=None,
    )
    parser.add_argument("--shape-prior-sam3d-root", type=Path, default=None)
    parser.add_argument("--shape-prior-config", type=Path, default=None)
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
        "--tracker-retire-filtered-markers",
        dest="tracker_retire_filtered_markers",
        action="store_true",
        help="Opt in to permanently hiding any query marker after it fails the active PCD residual/table-Z gate.",
    )
    parser.add_argument(
        "--no-tracker-retire-filtered-markers",
        dest="tracker_retire_filtered_markers",
        action="store_false",
        help="Use the default per-frame marker gate; filtered markers may reappear later.",
    )
    parser.set_defaults(tracker_retire_filtered_markers=False)
    parser.set_defaults(tapnextpp_fast_postprocess=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device, usually cuda.")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=DEFAULT_DTYPE, help="Inference dtype.")
    parser.add_argument(
        "--edgetam-mask-logit-threshold",
        type=float,
        default=DEFAULT_EDGETAM_MASK_LOGIT_THRESHOLD,
        help=(
            "Logit threshold used to binarize EdgeTAM masks. "
            "Lower values make masks more permissive."
        ),
    )
    parser.add_argument(
        "--pcd-color-mode",
        choices=("rgb", "class"),
        default="rgb",
        help="Point-cloud colors. rgb uses the live color frame; class uses fixed controller/object colors.",
    )
    parser.add_argument(
        "--object-pcd-mask-erode-pixels",
        type=int,
        default=DEFAULT_OBJECT_PCD_MASK_ERODE_PIXELS,
        help="Object-only mask erosion before RGB-D point-cloud backprojection. Defaults to --pcd-mask-erode-pixels.",
    )
    parser.add_argument(
        "--controller-pcd-mask-erode-pixels",
        type=int,
        default=DEFAULT_CONTROLLER_PCD_MASK_ERODE_PIXELS,
        help="Controller-only mask erosion before RGB-D point-cloud backprojection. Defaults to --pcd-mask-erode-pixels.",
    )
    parser.add_argument(
        "--enable-pcd-filter",
        action="store_true",
        help="Enable capped point-cloud filtering. Async mode never blocks capture, EdgeTAM, FFS, or render.",
    )
    parser.add_argument(
        "--pcd-filter-mode",
        choices=PCD_FILTER_MODES,
        default="async",
        help="Point-cloud filter scheduling mode. Requires --enable-pcd-filter unless set to none.",
    )
    parser.add_argument(
        "--pcd-filter-preset",
        choices=PCD_FILTER_PRESETS,
        default=None,
        help=(
            "High-level PCD surface preset. When set, the same preset controls object/controller PCD "
            "and TAPNext++ initial query sampling from filtered residual pixels."
        ),
    )
    parser.add_argument("--object-filter", choices=PCD_FILTERS, default=DEFAULT_OBJECT_FILTER)
    parser.add_argument("--controller-filter", choices=PCD_FILTERS, default=DEFAULT_CONTROLLER_FILTER)
    parser.add_argument("--object-filter-cap", type=int, default=DEFAULT_OBJECT_FILTER_CAP)
    parser.add_argument("--controller-filter-cap", type=int, default=DEFAULT_CONTROLLER_FILTER_CAP)
    parser.add_argument(
        "--object-filter-keep-components",
        type=int,
        default=DEFAULT_OBJECT_FILTER_KEEP_COMPONENTS,
        help="Connected components to keep when --object-filter enhanced-pt is used.",
    )
    parser.add_argument(
        "--controller-filter-keep-components",
        type=int,
        default=DEFAULT_CONTROLLER_FILTER_KEEP_COMPONENTS,
        help="Connected components to keep when --controller-filter enhanced-pt is used.",
    )
    parser.add_argument("--object-filter-voxel-m", type=float, default=0.004)
    parser.add_argument("--controller-filter-voxel-m", type=float, default=0.003)
    parser.add_argument(
        "--filter-every-n",
        type=int,
        default=3,
        help="Submit capped PCD filtering every N PCD packets. Async mode renders the latest available filtered output.",
    )
    parser.add_argument(
        "--filter-max-age-frames",
        type=int,
        default=DEFAULT_FILTER_MAX_AGE_FRAMES,
        help="Maximum async filtered-output age in frames before rendering raw current PCD instead.",
    )
    parser.add_argument(
        "--voxel-density-min-points",
        type=int,
        default=2,
        help="Minimum points per voxel for the realtime voxel-density approximate filter.",
    )
    parser.add_argument("--filter-radius-m", type=float, default=DEFAULT_FILTER_RADIUS_M)
    parser.add_argument("--filter-nb-points", type=int, default=DEFAULT_FILTER_NB_POINTS)
    parser.add_argument("--enhanced-component-voxel-size-m", type=float, default=DEFAULT_ENHANCED_COMPONENT_VOXEL_SIZE_M)
    parser.add_argument("--enhanced-keep-near-main-gap-m", type=float, default=DEFAULT_ENHANCED_KEEP_NEAR_MAIN_GAP_M)
    parser.add_argument(
        "--enable-table-z-filter",
        action="store_true",
        help=(
            "Enable table-world Z filter. Removes target PCD points whose "
            "signed table clearance is <= threshold after PT filtering."
        ),
    )
    parser.add_argument(
        "--disable-table-z-filter",
        action="store_true",
        help="Disable the table-world Z filter when a demo visual preset would enable it by default.",
    )
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional auto-stop duration. Use 0 to run until closed.")
    parser.add_argument(
        "--headless-capture-dir",
        type=Path,
        default=None,
        help=(
            "Save the selected sync PCD preset, color-aligned depth, and TAPNext++ query "
            "trajectory artifacts here. "
            "With --table-calibrate, the default demo preset uses filter none plus the 0 mm table-Z filter."
        ),
    )
    parser.add_argument(
        "--headless-prepared-only",
        action="store_true",
        help="For strict PhysTwin chunk preprocessing, save prepared_phystwin frames and frames.jsonl without legacy per-frame artifacts.",
    )
    parser.add_argument(
        "--write-input-rgb-timeline",
        action="store_true",
        help=(
            "Write input_rgb/*.png and input_frames.jsonl for Demo v6.1 "
            "realtime side-by-side viewing."
        ),
    )
    parser.add_argument("--controller-color", type=_parse_rgb_triplet, default=CONTROLLER_COLOR_RGB, help="Controller RGB color.")
    parser.add_argument("--object-color", type=_parse_rgb_triplet, default=OBJECT_COLOR_RGB, help="Object RGB color.")
    return parser


def apply_demo_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply demo preset."""
    if (
        not bool(getattr(args, "disable_table_z_filter", False))
        and not bool(getattr(args, "enable_table_z_filter", False))
        and getattr(args, "table_calibrate", None) is not None
        and str(getattr(args, "demo_visual_mode", DEFAULT_DEMO_VISUAL_MODE)) in DEMO_VISUAL_MODES
        and headless_capture_enabled(args)
    ):
        args.enable_table_z_filter = True
    return args


def pcd_filter_enabled(args: argparse.Namespace) -> bool:
    """Return whether PCD filter is enabled."""
    return bool(args.enable_pcd_filter) and str(args.pcd_filter_mode) != "none"


def pcd_filter_preset_to_filter(preset: str | None) -> str | None:
    """Return the PCD filter preset to filter."""
    if preset is None:
        return None
    normalized = str(preset).strip().lower()
    if not normalized:
        return None
    if normalized == PCD_FILTER_PRESET_ORIGINAL:
        return PCD_FILTER_NONE
    if normalized == PCD_FILTER_PRESET_PT:
        return PCD_FILTER_PT_FILTER
    if normalized == PCD_FILTER_PRESET_ENHANCED_PT:
        return PCD_FILTER_ENHANCED_PT
    raise ValueError(f"--pcd-filter-preset must be one of {', '.join(PCD_FILTER_PRESETS)}")


def tracker_query_source(args: argparse.Namespace) -> str:
    """Return the tracker query source (phystwin-strict is the only backend)."""
    return TRACKER_QUERY_SOURCE_UNION_MASK


def tracker_marker_gate(args: argparse.Namespace) -> str:
    """Return the tracker marker gate."""
    return (
        TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z
        if tracker_query_source(args) == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL
        else TRACKER_MARKER_GATE_TARGET_MASK_DEPTH
    )


def tracker_retire_filtered_markers(args: argparse.Namespace) -> bool:
    """Return the tracker retire filtered markers."""
    return bool(getattr(args, "tracker_retire_filtered_markers", False))


def tracker_marker_retirement_policy(args: argparse.Namespace) -> str:
    """Return the tracker marker retirement policy."""
    if (
        tracker_retire_filtered_markers(args)
        and tracker_marker_gate(args) == TRACKER_MARKER_GATE_PCD_FILTER_RESIDUAL_TABLE_Z
    ):
        return TRACKER_MARKER_RETIREMENT_POLICY_PCD_FILTER_RESIDUAL_TABLE_Z_ONCE_FALSE
    return TRACKER_MARKER_RETIREMENT_POLICY_DISABLED


def headless_capture_enabled(args: argparse.Namespace) -> bool:
    """Return whether headless capture is enabled."""
    return args.headless_capture_dir is not None


def headless_capture_saved_pcd_source(args: argparse.Namespace) -> str:
    """Return the headless capture saved PCD source."""
    object_filter = str(getattr(args, "object_filter", DEFAULT_OBJECT_FILTER)).replace("-", "_")
    controller_filter = str(getattr(args, "controller_filter", DEFAULT_CONTROLLER_FILTER)).replace("-", "_")
    if object_filter == controller_filter:
        return f"{object_filter}_filtered"
    return f"object_{object_filter}_controller_{controller_filter}_filtered"


def validate_args(args: argparse.Namespace) -> None:
    """Validate args."""
    parse_profile(DEFAULT_PROFILE)
    if args.input_source not in INPUT_SOURCES:
        raise ValueError(f"--input-source must be one of {', '.join(INPUT_SOURCES)}")
    if args.depth_source not in DEPTH_SOURCES:
        raise ValueError(f"--depth-source must be one of {', '.join(DEPTH_SOURCES)}")
    if float(args.replay_fps) < 0.0:
        raise ValueError("--replay-fps must be >= 0")
    if float(args.lossless_max_backlog_seconds) <= 0.0:
        raise ValueError("--lossless-max-backlog-seconds must be positive")
    if float(args.lossless_input_fps) <= 0.0:
        raise ValueError("--lossless-input-fps must be positive")
    if args.table_calibrate is not None:
        table_path = Path(args.table_calibrate).expanduser()
        if not table_path.is_absolute():
            table_path = REPO_ROOT / table_path
        table_path = table_path.resolve(strict=False)
        try:
            load_table_calibration_transforms(table_path)
        except TableCalibrationLoadError as exc:
            message = str(exc)
            if "Missing table calibration file" in message:
                raise ValueError(message) from exc
            raise ValueError(f"Invalid table calibration file: {message}") from exc
        args.table_calibrate = table_path
    if args.input_source == INPUT_SOURCE_FAKE_LIVE and args.recording_case is None:
        args.recording_case = DEFAULT_FAKE_LIVE_CASE
    if _is_replay_input_source(str(args.input_source)):
        if args.recording_case is None:
            raise ValueError(f"--input-source {args.input_source} requires --recording-case or --fake-live-case")
    elif args.recording_case is not None:
        raise ValueError("--recording-case/--fake-live-case requires --input-source recording or fake-live")
    if bool(args.shape_prior_warmup) and not str(args.shape_prior_controller_name or "").strip():
        raise ValueError(
            "--shape-prior-controller-name is required when --shape-prior-warmup "
            "is enabled"
        )
    if 0.2 < 0:
        raise ValueError("--depth-min-m must be >= 0")
    if 1.5 > 0 and 1.5 <= 0.2:
        raise ValueError("--depth-max-m must be <=0 or greater than --depth-min-m")
    if 60000 < 0:
        raise ValueError("--pcd-max-points must be >= 0")
    if 1 < 1:
        raise ValueError("--pcd-stride must be >= 1")
    if int(DEFAULT_PCD_MASK_ERODE_PIXELS) < 0:
        raise ValueError("--pcd-mask-erode-pixels must be >= 0")
    if args.object_pcd_mask_erode_pixels is not None and int(args.object_pcd_mask_erode_pixels) < 0:
        raise ValueError("--object-pcd-mask-erode-pixels must be >= 0")
    if args.controller_pcd_mask_erode_pixels is not None and int(args.controller_pcd_mask_erode_pixels) < 0:
        raise ValueError("--controller-pcd-mask-erode-pixels must be >= 0")
    if int(DEFAULT_EDGETAM_LIVE_SESSION_KEEP_FRAMES) < 0:
        raise ValueError("--edgetam-live-session-keep-frames must be >= 0")
    if not np.isfinite(float(args.edgetam_mask_logit_threshold)):
        raise ValueError("--edgetam-mask-logit-threshold must be finite")
    if float(DEFAULT_TABLE_Z_FILTER_THRESHOLD_M) < 0:
        raise ValueError("--table-z-filter-threshold-m must be >= 0")
    if (
        int(
            getattr(
                args,
                "shape_prior_timeout_ms",
                shape_prior_warmup.DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
            )
        )
        <= 0
    ):
        raise ValueError("--shape-prior-timeout-ms must be positive")
    if bool(getattr(args, "shape_prior_warmup", False)) and not getattr(
        args, "table_calibrate", None
    ):
        # Without the table world frame the frame-0 shape-prior request can
        # never be built, so the prior would sit in 'pending' forever and the
        # formal chunk timeline would never start.
        raise ValueError("--shape-prior-warmup requires --table-calibrate")
    if str(TABLE_Z_FILTER_CLASS_BOTH) not in TABLE_Z_FILTER_CLASSES:
        raise ValueError(
            f"--table-z-filter-classes must be one of {', '.join(TABLE_Z_FILTER_CLASSES)}"
        )
    if args.pcd_filter_mode not in PCD_FILTER_MODES:
        raise ValueError(f"--pcd-filter-mode must be one of {', '.join(PCD_FILTER_MODES)}")
    preset_filter = pcd_filter_preset_to_filter(getattr(args, "pcd_filter_preset", None))
    if preset_filter is not None:
        args.enable_pcd_filter = True
        args.pcd_filter_mode = "sync"
        args.object_filter = preset_filter
        args.controller_filter = preset_filter
        if str(getattr(args, "pcd_filter_preset", "")) == PCD_FILTER_PRESET_ORIGINAL:
            args.object_filter_cap = 0
            args.controller_filter_cap = 0
    for flag in (
        "object_filter_cap",
        "controller_filter_cap",
        "object_filter_keep_components",
        "controller_filter_keep_components",
        "filter_max_age_frames",
    ):
        if int(getattr(args, flag)) < 0:
            raise ValueError(f"--{flag.replace('_', '-')} must be >= 0")
    if args.depth_source == "none" and args.pcd_mode == "masked":
        raise ValueError("--depth-source none requires --pcd-mode none")
    if headless_capture_enabled(args):
        if args.input_source not in {INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_LIVE}:
            raise ValueError("--headless-capture-dir requires --input-source live or fake-live")
        if args.depth_source not in {"ffs", "realsense"}:
            raise ValueError("--headless-capture-dir requires --depth-source ffs or realsense")
        if args.object_filter not in HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS:
            allowed = ", ".join(HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS)
            raise ValueError(f"--headless-capture-dir requires --object-filter one of {allowed}")
        if args.controller_filter not in HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS:
            allowed = ", ".join(HEADLESS_CAPTURE_ALLOWED_PCD_FILTERS)
            raise ValueError(f"--headless-capture-dir requires --controller-filter one of {allowed}")
    args.tracker_backend = normalize_tracker_backend(str(args.tracker_backend))
    args.tracking_product_backend = normalize_tracking_product_backend(
        getattr(args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND)
    )
    if str(args.input_source) not in {INPUT_SOURCE_FAKE_LIVE, INPUT_SOURCE_LIVE}:
        raise ValueError("phystwin-strict-tracking requires --input-source live or fake-live")
    if args.headless_capture_dir is None:
        raise ValueError("phystwin-strict-tracking requires --headless-capture-dir")
    if args.phystwin_strict_output_dir is None:
        args.phystwin_strict_output_dir = Path(args.headless_capture_dir) / "phystwin_like"
    if getattr(args, "color_exposure", None) is not None and float(args.color_exposure) <= 0.0:
        raise ValueError("--color-exposure must be positive")
    if getattr(args, "color_gain", None) is not None and float(args.color_gain) < 0.0:
        raise ValueError("--color-gain must be >= 0")
    if tracker_enabled(args):
        if args.depth_source == "none":
            raise ValueError("--tracker-backend tapnextpp requires RGB-D depth for 3D marker lift")
    if args.depth_source == "ffs":
        validate_ffs_paths(ffs_repo=Path(args.ffs_repo), model_dir=Path(args.ffs_trt_model_dir))


def controller_tracking_enabled(args_or_track_mode: argparse.Namespace | str) -> bool:
    """Return whether controller tracking is enabled."""
    track_mode = args_or_track_mode if isinstance(args_or_track_mode, str) else args_or_track_mode.track_mode
    return str(track_mode) in {TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_CONTROLLER_ONLY}


def object_tracking_enabled(args_or_track_mode: argparse.Namespace | str) -> bool:
    """Return whether object tracking is enabled."""
    track_mode = args_or_track_mode if isinstance(args_or_track_mode, str) else args_or_track_mode.track_mode
    return str(track_mode) in {TRACK_MODE_CONTROLLER_OBJECT, TRACK_MODE_OBJECT_ONLY}


def object_id_labels(track_mode: str = DEFAULT_TRACK_MODE) -> dict[int, str]:
    """Return the object id labels."""
    if track_mode == TRACK_MODE_NONE:
        return {}
    if track_mode == TRACK_MODE_OBJECT_ONLY:
        return {OBJECT_ID: EDGE_TAM_OBJECT_LABELS[OBJECT_ID]}
    if track_mode == TRACK_MODE_CONTROLLER_ONLY:
        return {
            HAND_A_ID: EDGE_TAM_OBJECT_LABELS[HAND_A_ID],
            HAND_B_ID: EDGE_TAM_OBJECT_LABELS[HAND_B_ID],
        }
    if track_mode == TRACK_MODE_CONTROLLER_OBJECT:
        return dict(EDGE_TAM_OBJECT_LABELS)
    raise ValueError(f"unsupported track mode: {track_mode}")


def active_object_id_labels(args: argparse.Namespace) -> dict[int, str]:
    """Return the active object id labels."""
    return object_id_labels(str(args.track_mode))


def active_object_ids(args: argparse.Namespace) -> list[int]:
    """Return the active object ids."""
    return list(active_object_id_labels(args).keys())


def tracker_enabled(args: argparse.Namespace) -> bool:
    """Return whether tracker is enabled."""
    return normalize_tracker_backend(str(getattr(args, "tracker_backend", TRACKER_BACKEND_NONE))) != TRACKER_BACKEND_NONE


def object_pcd_mask_erode_pixels(args: argparse.Namespace) -> int:
    """Return the object PCD mask erode pixels."""
    value = getattr(args, "object_pcd_mask_erode_pixels", None)
    if value is None:
        value = getattr(args, "pcd_mask_erode_pixels", DEFAULT_PCD_MASK_ERODE_PIXELS)
    return int(value)


def controller_pcd_mask_erode_pixels(args: argparse.Namespace) -> int:
    """Return the controller PCD mask erode pixels."""
    value = getattr(args, "controller_pcd_mask_erode_pixels", None)
    if value is None:
        value = getattr(args, "pcd_mask_erode_pixels", DEFAULT_PCD_MASK_ERODE_PIXELS)
    return int(value)


__all__ = [
    "_parse_rgb_triplet",
    "_is_replay_input_source",
    "depth_backend_label",
    "runtime_metadata_identity",
    "build_parser",
    "apply_demo_preset",
    "pcd_filter_enabled",
    "pcd_filter_preset_to_filter",
    "tracker_query_source",
    "tracker_marker_gate",
    "tracker_retire_filtered_markers",
    "tracker_marker_retirement_policy",
    "headless_capture_enabled",
    "headless_capture_saved_pcd_source",
    "validate_args",
    "controller_tracking_enabled",
    "object_tracking_enabled",
    "object_id_labels",
    "active_object_id_labels",
    "active_object_ids",
    "tracker_enabled",
    "object_pcd_mask_erode_pixels",
    "controller_pcd_mask_erode_pixels",
]
