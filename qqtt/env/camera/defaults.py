"""Shared camera workflow defaults for the camera-only repo."""

DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_NUM_CAM = 1
DEFAULT_EXPOSURE = 70.0
DEFAULT_GAIN = 60.0
DEFAULT_WHITE_BALANCE = 3800.0

# Current lab-rig color controls. The single-camera top-down demo camera was
# rechecked on 2026-06-13 against the sloth/cloth tabletop scene.
DEFAULT_COLOR_EXPOSURE_OVERRIDES = {
    "239222300412": 156.0,
    "239222300781": 70.0,
    "239222303506": 180.0,
}
DEFAULT_COLOR_GAIN_OVERRIDES = {}


def resolve_per_camera_control_values(
    value,
    *,
    overrides: dict[str, float] | None,
    serial_numbers: list[str],
    label: str,
):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        overrides = overrides or {}
        return [
            float(overrides.get(serial, value))
            for serial in serial_numbers
        ]
    values = list(value)
    if len(values) != len(serial_numbers):
        raise ValueError(
            f"{label} list length must match selected cameras. "
            f"got={len(values)}, expected={len(serial_numbers)}"
        )
    return [float(item) for item in values]
