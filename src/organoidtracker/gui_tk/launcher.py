#!/usr/bin/env python3
"""
Launcher for the Tk desktop application (console script ``organoidtracker-tk``).

Checks the environment, lists the available models and starts ``VideoTrackerApp``.
``python video_tracker_gui.py`` at the repository root calls the same ``main``.
"""

import os
import sys


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
        print("  uv sync --extra tk   (or: pip install -e .)")
        return False

    return True


def check_models():
    """Check available models"""
    print("\n🔍 Checking available models...")

    from ..core.model_registry import get_model_registry

    registry = get_model_registry()
    available_models = registry.get_available_models()

    if not available_models:
        print("❌ No models available!")
        print("This might indicate:")
        print("  1. Model checkpoints are missing")
        print("  2. The sam2 package could not be imported")
        print("  3. Environment not activated")
        return False

    print(f"✅ Found {len(available_models)} model(s):")
    for model in available_models:
        print(f"   • {model.display_name}: {model.description}")

    return True


def main():
    """Main entry point"""
    # Fix OpenMP duplicate library issue on Windows (PyTorch + NumPy/SciPy conflict)
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('TK_SILENCE_DEPRECATION', '1')  # Suppress Tkinter warnings on macOS

    print("🏥 Multi-Model Video Object Tracker")
    print("=" * 50)

    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("❌ Error: Tkinter not available!")
        print("Please install tkinter:")
        print("  - Ubuntu/Debian: sudo apt-get install python3-tk")
        print("  - Windows: included with the python.org installer (select tcl/tk)")
        return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    try:
        from .main_window import VideoTrackerApp
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("Make sure the package and its dependencies are installed:")
        print("  uv sync --extra tk   (or: pip install -e .)")
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
        print("   • Left click on an organoid, then drag boxes around its cysts")
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
    sys.exit(main())
