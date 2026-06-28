"""Shared final_data keys that vary along the online frame axis."""

REQUIRED_TIME_KEYS = (
    "object_points",
    "object_colors",
    "object_visibilities",
    "object_motions_valid",
    "controller_points",
)

OPTIONAL_TIME_KEYS = (
    # Optional keys are appended to online chunks only when the producer emits
    # the corresponding diagnostics for every frame.
    "asap_object_points_filled",
    "asap_surface_points",
    "asap_interior_points",
    "object_recovered",
    "object_recovery_confidence",
    "controller_observed",
    "controller_recovered",
    "controller_recovery_confidence",
    "controller_source_query_ids",
    "controller_track_mode",
    "controller_track_confidence",
    "controller_filter_reason",
    "controller_neighbor_support_count",
    "controller_neighbor_raw_visible_count",
    "controller_neighbor_depth_valid_count",
    "controller_neighbor_processed_mask_valid_count",
    "controller_neighbor_motion_valid_count",
    "controller_neighbor_fit_residual",
)

TIME_KEYS = (*REQUIRED_TIME_KEYS, *OPTIONAL_TIME_KEYS)
