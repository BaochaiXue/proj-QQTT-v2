"""Top-level qqtt exports.

Keep heavy training/simulation modules lazily imported so camera-only workflows
can run with a minimal environment.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "CameraSystem",
    "SpringMassSystemWarp",
    "InvPhyTrainerWarp",
    "OptimizerCMA",
]

_LAZY_IMPORTS = {
    "CameraSystem": ("qqtt.env", "CameraSystem"),
    "SpringMassSystemWarp": ("qqtt.model", "SpringMassSystemWarp"),
    "InvPhyTrainerWarp": ("qqtt.engine", "InvPhyTrainerWarp"),
    "OptimizerCMA": ("qqtt.engine", "OptimizerCMA"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
