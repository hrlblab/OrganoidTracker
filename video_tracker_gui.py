#!/usr/bin/env python3
"""
Multi-Model Video Object Tracker GUI
Main launcher for the graphical user interface

Features:
- Support for multiple models (Medical-SAM2, SAM2, future models)
- Click-based object prompting
- Real-time progress tracking
- Multiple video output formats
- Beginner-friendly interface

Usage:
    python video_tracker_gui.py
"""

import sys
import os
from pathlib import Path

# Fix OpenMP duplicate library issue on Windows (PyTorch + NumPy/SciPy conflict)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("❌ Error: Tkinter not available!")
    print("Please install tkinter:")
    print("  - Ubuntu/Debian: sudo apt-get install python3-tk")
    print("  - macOS: Should be included with Python")
    print("  - Windows: Should be included with Python")
    sys.exit(1)

try:
    from src.gui.main_window import VideoTrackerApp
    from src.core.model_registry import get_model_registry
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure you're running from the OrganoidTracker directory")
    print("and that all dependencies are installed:")
    print("  pip install torch torchvision hydra-core omegaconf")
    print("  pip install opencv-python pillow numpy scipy tqdm")
    sys.exit(1)


def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []

    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        missing_deps.append("PyTorch")

    # Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        missing_deps.append("OpenCV (cv2)")

    # Check PIL
    try:
        from PIL import Image
        print(f"✅ Pillow: {Image.__version__}")
    except ImportError:
        missing_deps.append("Pillow (PIL)")

    # Check NumPy
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        missing_deps.append("NumPy")

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies:")
        print("  pip install torch torchvision opencv-python pillow numpy")
        return False

    return True


def check_models():
    """Check available models"""
    print("\n🔍 Checking available models...")

    registry = get_model_registry()
    available_models = registry.get_available_models()

    if not available_models:
        print("❌ No models available!")
        print("This might indicate:")
        print("  1. Model checkpoints are missing")
        print("  2. SAM2 packages not installed")
        print("  3. Environment not activated")
        return False

    print(f"✅ Found {len(available_models)} model(s):")
    for model in available_models:
        print(f"   • {model.display_name}: {model.description}")

    return True


def main():
    """Main entry point"""
    print("🏥 Multi-Model Video Object Tracker")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check models
    if not check_models():
        print("\n⚠️  Continuing anyway - you can still explore the interface")

    print("\n🚀 Starting GUI...")

    try:
        # Create and run the main application
        app = VideoTrackerApp()

        print("✅ GUI started successfully!")
        print("💡 Tips:")
        print("   • Select a model and click 'Load Model'")
        print("   • Load a video file")
        print("   • Left click on objects to track")
        print("   • Right click for background areas")
        print("   • Start tracking and generate videos!")

        app.run()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"\n❌ Error running GUI: {e}")
        print("Please check your environment and dependencies")
        return 1

    return 0


if __name__ == "__main__":
    # Set up environment
    os.environ['TK_SILENCE_DEPRECATION'] = '1'  # Suppress Tkinter warnings on macOS

    exit_code = main()
    sys.exit(exit_code)