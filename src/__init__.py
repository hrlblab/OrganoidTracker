#!/usr/bin/env python3
"""
Unified Video Tracker Application
Support for multiple models: Medical-SAM2, SAM2, and more
"""

__version__ = "2.0.0"
__author__ = "Video Tracker Team"

# Core components
from .core import BaseVideoTracker, ModelRegistry, ModelFactory
from .gui import VideoTrackerApp
from .utils import VideoOutputGenerator

__all__ = [
    'BaseVideoTracker',
    'ModelRegistry',
    'ModelFactory',
    'VideoTrackerApp',
    'VideoOutputGenerator'
]