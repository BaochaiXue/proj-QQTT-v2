"""Shared final_data keys that vary along the online frame axis."""

REQUIRED_TIME_KEYS = (
    # Selected object 3D point positions for each frame, shaped T,N,3.
    "object_points",
    # RGB colors aligned one-to-one with object_points, shaped T,N,3.
    "object_colors",
    # Boolean visibility mask for each selected object point, shaped T,N.
    "object_visibilities",
    # Boolean motion/track validity mask for each object point, shaped T,N.
    "object_motions_valid",
    # Selected controller 3D anchor positions for each frame, shaped T,M,3.
    "controller_points",
)

OPTIONAL_TIME_KEYS = (
    # Optional keys are appended to online chunks only when the producer emits
    # the corresponding diagnostics for every frame.
    # Optional ASAP diagnostic for object points filled by recovery logic.
    "asap_object_points_filled",
    # Optional ASAP surface shape-prior points carried with each frame.
    "asap_surface_points",
    # Optional ASAP interior shape-prior points carried with each frame.
    "asap_interior_points",
    # Flag or mask indicating object data recovered instead of observed.
    "object_recovered",
    # Confidence scores associated with recovered object data.
    "object_recovery_confidence",
    # Flag or mask indicating controller data was directly observed.
    "controller_observed",
    # Flag or mask indicating controller data recovered instead of observed.
    "controller_recovered",
    # Confidence scores associated with recovered controller data.
    "controller_recovery_confidence",
    # Source query id used for each emitted controller point.
    "controller_source_query_ids",
    # String mode label describing how each controller point was tracked.
    "controller_track_mode",
    # Numeric confidence for each emitted controller track.
    "controller_track_confidence",
    # String reason attached to controller filtering or recovery decisions.
    "controller_filter_reason",
    # Neighbor count supporting each controller recovery estimate.
    "controller_neighbor_support_count",
    # Count of raw visible neighbors before depth and mask filtering.
    "controller_neighbor_raw_visible_count",
    # Count of neighboring controller candidates with valid depth.
    "controller_neighbor_depth_valid_count",
    # Count of neighbors inside the processed controller mask.
    "controller_neighbor_processed_mask_valid_count",
    # Count of neighbors passing controller motion-validity checks.
    "controller_neighbor_motion_valid_count",
    # Residual error from the neighbor-based controller fit.
    "controller_neighbor_fit_residual",
)

# All online frame-axis keys accepted by chunk writing and aggregation.
TIME_KEYS = (*REQUIRED_TIME_KEYS, *OPTIONAL_TIME_KEYS)
