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
    # Flag or mask indicating controller data was directly observed.
    "controller_observed",
    # Flag or mask indicating controller data recovered instead of observed.
    "controller_recovered",
    # Boolean mask marking controller anchor frames whose value came from
    # local rigid-registration recovery instead of a direct measurement
    # (design_spec.md temporary_invalid handling).
    "controller_proxied",
)

# All online frame-axis keys accepted by chunk writing and aggregation.
TIME_KEYS = (*REQUIRED_TIME_KEYS, *OPTIONAL_TIME_KEYS)
