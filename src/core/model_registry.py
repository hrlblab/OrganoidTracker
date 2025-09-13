#!/usr/bin/env python3
"""
Model Registry for Video Object Tracking
Centralized system for registering and managing multiple tracking models
"""

from typing import Dict, List, Type, Optional, Any
from .base_model import BaseVideoTracker, ModelMetadata
import importlib


class ModelRegistry:
    """
    Registry for video tracking models
    Provides centralized model management and discovery
    """

    def __init__(self):
        self._models: Dict[str, Type[BaseVideoTracker]] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._instances: Dict[str, BaseVideoTracker] = {}

        # Auto-register available models
        self._auto_register_models()

    def _auto_register_models(self):
        """Automatically register available models"""
        # Register SAM2 first (as default)
        try:
            from .sam2_tracker import SAM2Tracker
            if SAM2Tracker.is_available():
                self.register_model(SAM2Tracker)
            else:
                print("⚠️  SAM2 not available - package not installed")
        except ImportError as e:
            print(f"⚠️  SAM2 not available: {e}")

        # Register Medical-SAM2
        try:
            from .inference import MedicalSAM2Tracker
            self.register_model(MedicalSAM2Tracker)
        except ImportError as e:
            print(f"⚠️  Medical-SAM2 not available: {e}")

    def register_model(self, model_class: Type[BaseVideoTracker]) -> bool:
        """
        Register a new model class

        Args:
            model_class: Model class that inherits from BaseVideoTracker

        Returns:
            bool: True if registered successfully
        """
        try:
            # Get metadata from the model
            metadata = model_class.get_metadata()

            # Store the model class and metadata
            self._models[metadata.name] = model_class
            self._metadata[metadata.name] = metadata

            print(f"✅ Registered model: {metadata.display_name}")
            return True

        except Exception as e:
            print(f"❌ Error registering model {model_class.__name__}: {str(e)}")
            return False

    def get_available_models(self) -> List[ModelMetadata]:
        """
        Get list of all available models

        Returns:
            list: List of ModelMetadata objects
        """
        return list(self._metadata.values())

    def get_model_names(self) -> List[str]:
        """
        Get list of model names

        Returns:
            list: List of model names
        """
        return list(self._models.keys())

    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a specific model

        Args:
            model_name: Name of the model

        Returns:
            ModelMetadata or None if not found
        """
        return self._metadata.get(model_name)

    def create_model_instance(self, model_name: str, **kwargs) -> Optional[BaseVideoTracker]:
        """
        Create an instance of a model

        Args:
            model_name: Name of the model
            **kwargs: Model-specific configuration

        Returns:
            BaseVideoTracker instance or None if failed
        """
        if model_name not in self._models:
            print(f"❌ Model '{model_name}' not found in registry")
            return None

        try:
            model_class = self._models[model_name]
            instance = model_class(**kwargs)

            # Cache the instance
            self._instances[model_name] = instance

            print(f"✅ Created {model_name} instance")
            return instance

        except Exception as e:
            print(f"❌ Error creating {model_name} instance: {str(e)}")
            return None

    def get_model_instance(self, model_name: str) -> Optional[BaseVideoTracker]:
        """
        Get cached model instance

        Args:
            model_name: Name of the model

        Returns:
            BaseVideoTracker instance or None if not found
        """
        return self._instances.get(model_name)

    def has_model(self, model_name: str) -> bool:
        """
        Check if model is registered

        Args:
            model_name: Name of the model

        Returns:
            bool: True if model is available
        """
        return model_name in self._models

    def get_models_by_capability(self, capability: str) -> List[ModelMetadata]:
        """
        Get models that have a specific capability

        Args:
            capability: Capability to filter by

        Returns:
            list: List of models with the capability
        """
        return [
            metadata for metadata in self._metadata.values()
            if metadata.has_capability(capability)
        ]

    def clear_instances(self):
        """Clear all cached model instances"""
        self._instances.clear()
        print("🧹 Cleared all model instances")

    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get registry information

        Returns:
            dict: Registry information
        """
        return {
            'total_models': len(self._models),
            'available_models': [m.display_name for m in self._metadata.values()],
            'cached_instances': list(self._instances.keys()),
            'models_by_capability': {
                'click_prompts': len(self.get_models_by_capability('click_prompts')),
                'bbox_prompts': len(self.get_models_by_capability('bbox_prompts')),
                'multi_object': len(self.get_models_by_capability('multi_object')),
                'medical_optimized': len(self.get_models_by_capability('medical_optimized'))
            }
        }


# Global registry instance
_global_registry = None


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance

    Returns:
        ModelRegistry: The global registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def register_model(model_class: Type[BaseVideoTracker]) -> bool:
    """
    Register a model with the global registry

    Args:
        model_class: Model class to register

    Returns:
        bool: True if successful
    """
    return get_model_registry().register_model(model_class)


def get_available_models() -> List[ModelMetadata]:
    """
    Get all available models from global registry

    Returns:
        list: List of available models
    """
    return get_model_registry().get_available_models()


def create_model(model_name: str, **kwargs) -> Optional[BaseVideoTracker]:
    """
    Create a model instance from global registry

    Args:
        model_name: Name of the model
        **kwargs: Model configuration

    Returns:
        BaseVideoTracker instance or None
    """
    return get_model_registry().create_model_instance(model_name, **kwargs)


class ModelFactory:
    """
    Factory class for creating models with common configurations
    """

    @staticmethod
    def create_medical_sam2(device='cuda', **kwargs) -> Optional[BaseVideoTracker]:
        """Create Medical-SAM2 instance with default config"""
        return create_model('medical_sam2', device=device, **kwargs)

    @staticmethod
    def create_sam2(device='cuda', **kwargs) -> Optional[BaseVideoTracker]:
        """Create SAM2 instance with default config"""
        return create_model('sam2', device=device, **kwargs)

    @staticmethod
    def create_best_available_model(prefer_medical=True, device='cuda', **kwargs) -> Optional[BaseVideoTracker]:
        """
        Create the best available model based on preferences

        Args:
            prefer_medical: Whether to prefer medical-optimized models
            device: Device to use
            **kwargs: Additional model config

        Returns:
            BaseVideoTracker instance or None
        """
        registry = get_model_registry()
        available_models = registry.get_available_models()

        if not available_models:
            print("❌ No models available")
            return None

        # Prioritize models based on preferences
        if prefer_medical:
            medical_models = registry.get_models_by_capability('medical_optimized')
            if medical_models:
                model_name = medical_models[0].name
                return create_model(model_name, device=device, **kwargs)

        # Fallback to first available model
        model_name = available_models[0].name
        return create_model(model_name, device=device, **kwargs)

    @staticmethod
    def get_recommended_model_for_task(task_type: str = 'general') -> Optional[str]:
        """
        Get recommended model name for a specific task

        Args:
            task_type: Type of task ('medical', 'general', 'research')

        Returns:
            str: Recommended model name or None
        """
        registry = get_model_registry()

        if task_type == 'medical':
            medical_models = registry.get_models_by_capability('medical_optimized')
            return medical_models[0].name if medical_models else None

        elif task_type == 'general':
            available = registry.get_available_models()
            # Prefer SAM2 for general use
            for model in available:
                if model.name == 'sam2':
                    return model.name
            return available[0].name if available else None

        elif task_type == 'research':
            # Return all available for research
            available = registry.get_available_models()
            return available[0].name if available else None

        return None