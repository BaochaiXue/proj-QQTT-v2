"""Shared final_data keys that vary along the online frame axis."""

REQUIRED_TIME_KEYS = (
    "object_points",
    "object_colors",
    "object_visibilities",
    "object_motions_valid",
    "controller_points",
)

OPTIONAL_TIME_KEYS = (
    "asap_object_points_filled",
    "asap_surface_points",
    "asap_interior_points",
    "object_recovered",
    "object_recovery_confidence",
    "controller_observed",
    "controller_recovered",
    "controller_recovery_confidence",
    "controller_anchor_source_query_id",
    "controller_anchor_observation_mode",
    "controller_anchor_confidence",
    "controller_anchor_failure_reason",
    "controller_anchor_bundle_support_count",
    "controller_anchor_recovery_residual",
)

TIME_KEYS = (*REQUIRED_TIME_KEYS, *OPTIONAL_TIME_KEYS)
