"""Tracking backends: the model interface, the registry and the SAM2 adapter.

The backend classes import torch and the vendored SAM 2 (which initializes Hydra on import),
so ``SAM2Tracker``, the registry and the factory are loaded on first use: the result
containers (``masks``, ``tracking_result``), the analysis and the exports of a saved result
work in a process that has no model at all.
"""

from typing import Any

from .base_model import BaseVideoTracker, ModelCapabilities, ModelMetadata

__all__ = [
    "BaseVideoTracker",
    "ModelCapabilities",
    "ModelFactory",
    "ModelMetadata",
    "ModelRegistry",
    "SAM2Tracker",
    "get_model_registry",
]

_LAZY = {
    "ModelFactory": "model_registry",
    "ModelRegistry": "model_registry",
    "get_model_registry": "model_registry",
    "SAM2Tracker": "sam2_tracker",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module_name}"), name)
