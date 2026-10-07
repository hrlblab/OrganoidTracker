"""Core inference functionality: backend interface, model registry, and the SAM2 tracker"""

from .base_model import BaseVideoTracker, ModelMetadata, ModelCapabilities
from .model_registry import ModelRegistry, get_model_registry, ModelFactory
from .sam2_tracker import SAM2Tracker

__all__ = [
    'BaseVideoTracker',
    'ModelMetadata',
    'ModelCapabilities',
    'ModelRegistry',
    'get_model_registry',
    'ModelFactory',
    'SAM2Tracker'
]