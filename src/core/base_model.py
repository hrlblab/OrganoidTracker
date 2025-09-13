#!/usr/bin/env python3
"""
Base Model Interface for Video Object Tracking
Provides a common interface for all video tracking models (Medical-SAM2, SAM2, etc.)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


class BaseVideoTracker(ABC):
    """
    Abstract base class for video object tracking models

    All tracking models (Medical-SAM2, SAM2, etc.) must implement this interface
    to ensure compatibility with the GUI and provide a consistent API.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base tracker

        Args:
            model_name: Name of the model (e.g., "Medical-SAM2", "SAM2")
            **kwargs: Model-specific configuration
        """
        self.model_name = model_name
        self.video_frames = None
        self.video_path = None
        self.fps = None
        self.prompts = {}
        self.is_loaded = False
        self.is_initialized = False

    @abstractmethod
    def load_model(self, **kwargs) -> bool:
        """
        Load and initialize the model

        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def load_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Load video for tracking

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load

        Returns:
            dict: Video information (num_frames, fps, dimensions, etc.)
        """
        pass

    @abstractmethod
    def get_first_frame(self) -> np.ndarray:
        """
        Get the first frame for GUI display and clicking

        Returns:
            numpy.ndarray: First frame as RGB image
        """
        pass

    @abstractmethod
    def add_click_prompt(self, x: int, y: int, label: int = 1, obj_id: int = 1, frame_idx: int = 0) -> bool:
        """
        Add a click prompt for object tracking

        Args:
            x, y: Click coordinates
            label: 1 for foreground, 0 for background
            obj_id: Object ID to track
            frame_idx: Frame index to add prompt

        Returns:
            bool: True if prompt added successfully
        """
        pass

    @abstractmethod
    def add_bbox_prompt(self, x1: int, y1: int, x2: int, y2: int, obj_id: int = 1, frame_idx: int = 0) -> bool:
        """
        Add a bounding box prompt for object tracking

        Args:
            x1, y1, x2, y2: Bounding box coordinates
            obj_id: Object ID to track
            frame_idx: Frame index to add prompt

        Returns:
            bool: True if prompt added successfully
        """
        pass

    @abstractmethod
    def run_tracking(self, progress_callback: Optional[callable] = None) -> Dict[int, Dict[int, Any]]:
        """
        Run object tracking across all video frames

        Args:
            progress_callback: Optional callback function for progress updates
            Signature: callback(current_frame, total_frames, message)

        Returns:
            dict: Frame-wise segmentation results
        """
        pass

    @abstractmethod
    def get_frame_mask(self, frame_idx: int, obj_id: int = 1, video_segments: Optional[Dict] = None) -> np.ndarray:
        """
        Get binary mask for a specific frame and object

        Args:
            frame_idx: Frame index
            obj_id: Object ID
            video_segments: Tracking results

        Returns:
            numpy.ndarray: Binary mask
        """
        pass

    @abstractmethod
    def get_frame_overlay(self, frame_idx: int, obj_id: int = 1, video_segments: Optional[Dict] = None,
                         color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.3) -> np.ndarray:
        """
        Get frame with segmentation overlay

        Args:
            frame_idx: Frame index
            obj_id: Object ID
            video_segments: Tracking results
            color: Overlay color (RGB)
            alpha: Overlay transparency

        Returns:
            numpy.ndarray: Frame with overlay
        """
        pass

    @abstractmethod
    def clear_prompts(self, obj_id: Optional[int] = None) -> bool:
        """Clear all prompts or prompts for a specific object"""
        pass

    def get_active_objects(self) -> List[int]:
        """Get list of object IDs that have prompts"""
        if not hasattr(self, 'prompts') or not self.prompts:
            return []
        return list(self.prompts.keys())

    def get_prompt_count(self, obj_id: Optional[int] = None) -> int:
        """Get total prompt count or count for specific object"""
        if not hasattr(self, 'prompts') or not self.prompts:
            return 0

        if obj_id is not None:
            return len(self.prompts.get(obj_id, []))
        else:
            return sum(len(obj_prompts) for obj_prompts in self.prompts.values())

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            dict: Model information
        """
        return {
            'name': self.model_name,
            'is_loaded': self.is_loaded,
            'is_initialized': self.is_initialized,
            'video_loaded': self.video_frames is not None,
            'num_prompts': sum(len(prompts) for prompts in self.prompts.values()) if self.prompts else 0
        }

    def get_video_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current video information

        Returns:
            dict: Video information or None if no video loaded
        """
        if self.video_frames is None:
            return None

        return {
            'num_frames': len(self.video_frames),
            'fps': self.fps,
            'dimensions': self.video_frames[0].shape[:2],
            'video_path': self.video_path,
            'prompts': self.prompts
        }

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported video formats

        Returns:
            list: Supported file extensions
        """
        return ['.mp4', '.avi', '.mov', '.mkv']

    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate if video file is supported

        Args:
            video_path: Path to video file

        Returns:
            bool: True if supported
        """
        from pathlib import Path
        return Path(video_path).suffix.lower() in self.get_supported_formats()


class ModelCapabilities:
    """
    Enum-like class defining model capabilities
    """
    CLICK_PROMPTS = "click_prompts"
    BBOX_PROMPTS = "bbox_prompts"
    MULTI_OBJECT = "multi_object"
    REAL_TIME = "real_time"
    GPU_REQUIRED = "gpu_required"
    MEDICAL_OPTIMIZED = "medical_optimized"


class ModelMetadata:
    """
    Metadata container for model information
    """

    def __init__(self, name: str, display_name: str, description: str,
                 capabilities: List[str], requirements: List[str] = None):
        """
        Initialize model metadata

        Args:
            name: Internal model name
            display_name: User-friendly display name
            description: Model description
            capabilities: List of model capabilities
            requirements: List of requirements (e.g., "GPU", "CUDA")
        """
        self.name = name
        self.display_name = display_name
        self.description = description
        self.capabilities = capabilities
        self.requirements = requirements or []

    def has_capability(self, capability: str) -> bool:
        """Check if model has specific capability"""
        return capability in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'capabilities': self.capabilities,
            'requirements': self.requirements
        }