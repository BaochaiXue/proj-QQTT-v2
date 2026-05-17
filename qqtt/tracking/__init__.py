"""Demo 3 tracking benchmark and 3D anchor overlay utilities."""

from .base import BackendAvailability, TrackingResult

__all__ = [
    "BackendAvailability",
    "TrackingResult",
    "available_backend_names",
    "check_backend_availability",
    "create_backend",
]


def __getattr__(name: str):
    if name in {"available_backend_names", "check_backend_availability", "create_backend"}:
        from .registry import available_backend_names, check_backend_availability, create_backend

        return {
            "available_backend_names": available_backend_names,
            "check_backend_availability": check_backend_availability,
            "create_backend": create_backend,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
