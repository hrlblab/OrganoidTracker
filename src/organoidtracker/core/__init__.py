"""Tracking backends: the model interface, the registry and the SAM2 adapter."""

from .base_model import BaseVideoTracker, ModelCapabilities, ModelMetadata
from .model_registry import ModelFactory, ModelRegistry, get_model_registry
from .sam2_tracker import SAM2Tracker

__all__ = [
    "BaseVideoTracker",
    "ModelCapabilities",
    "ModelFactory",
    "ModelMetadata",
    "ModelRegistry",
    "SAM2Tracker",
    "get_model_registry",
]
