from importlib import import_module
from typing import Any

__all__ = ["CameraSystem"]


def __getattr__(name: str) -> Any:
    if name == "CameraSystem":
        module = import_module("qqtt.env.camera")
        value = getattr(module, "CameraSystem")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
