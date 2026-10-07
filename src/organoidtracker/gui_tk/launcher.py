#!/usr/bin/env python3
"""
Launcher for the Tk desktop application (console script ``organoidtracker-tk``).

Checks the environment, lists the available models and starts ``VideoTrackerApp``.
``python video_tracker_gui.py`` at the repository root calls the same ``main``.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []

    # Check PyTorch
    try:
        import torch

        logger.info(f"PyTorch: {torch.__version__}")
    except ImportError:
        missing_deps.append("PyTorch")

    # Check OpenCV
    try:
        import cv2

        logger.info(f"OpenCV: {cv2.__version__}")
    except ImportError:
        missing_deps.append("OpenCV (cv2)")

    # Check PIL
    try:
        from PIL import Image

        logger.info(f"Pillow: {Image.__version__}")
    except ImportError:
        missing_deps.append("Pillow (PIL)")

    # Check NumPy
    try:
        import numpy as np

        logger.info(f"NumPy: {np.__version__}")
    except ImportError:
        missing_deps.append("NumPy")

    if missing_deps:
        logger.error(
            f"Missing dependencies: {', '.join(missing_deps)}. "
            "Install them with `uv sync --extra tk` (or `pip install -e .`)."
        )
        return False

    return True


def check_models():
    """Check available models"""
    logger.info("Checking available models...")

    from ..core.model_registry import get_model_registry

    registry = get_model_registry()
    available_models = registry.get_available_models()

    if not available_models:
        logger.error(
            "No models available. Check that the sam2 package imports (`uv sync`) and that the "
            "environment is activated; checkpoints are only needed when a model is loaded."
        )
        return False

    logger.info(f"Found {len(available_models)} model(s):")
    for model in available_models:
        logger.debug(f"• {model.display_name}: {model.description}")

    return True


def _show_startup_error(message: str) -> None:
    """Show the message in a dialog when Tk can open one; a desktop launch has no visible console."""
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Organoid Tracker", message)
        root.destroy()
    except Exception:
        pass


def main():
    """Main entry point"""
    # Fix OpenMP duplicate library issue on Windows (PyTorch + NumPy/SciPy conflict)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")  # Suppress Tkinter warnings on macOS

    from ..logging_config import configure_logging
    from ..settings import SettingsError

    try:
        from .. import config
    except SettingsError as error:
        # An explicitly supplied but invalid configuration must not be replaced by defaults.
        logging.basicConfig(level=logging.ERROR, format="%(levelname)-8s %(message)s")
        logger.error(f"Invalid settings, not starting: {error}")
        _show_startup_error(f"Invalid settings; the application did not start.\n\n{error}")
        return 2

    log_file = configure_logging()
    logger.info("Multi-Model Video Object Tracker")
    if log_file is not None:
        logger.info(f"Log file: {log_file}")
    if config.LOADED_SETTINGS_FILE is not None:
        logger.info(f"Settings file: {config.LOADED_SETTINGS_FILE}")
    if config.LOADED_USER_CONFIG is not None:
        logger.info(f"User configuration: {config.LOADED_USER_CONFIG}")

    try:
        import tkinter  # noqa: F401
    except ImportError:
        logger.error(
            "Tkinter is not available. Ubuntu/Debian: `sudo apt-get install python3-tk`; "
            "Windows: select tcl/tk in the python.org installer."
        )
        return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    try:
        from .main_window import VideoTrackerApp
    except ImportError as e:
        logger.error(
            f"Error importing modules: {e}. "
            "Make sure the package and its dependencies are installed (`uv sync --extra tk`)."
        )
        return 1

    # Check models
    if not check_models():
        logger.warning("Continuing anyway - you can still explore the interface")

    logger.info("Starting GUI...")

    try:
        # Create and run the main application
        app = VideoTrackerApp()

        logger.info("GUI started successfully!")
        logger.debug("Tips:")
        logger.debug("• Select a model and click 'Load Model'")
        logger.debug("• Load a video file")
        logger.debug("• Left click on an organoid, then drag boxes around its cysts")
        logger.debug("• Start tracking and generate videos!")

        app.run()

    except KeyboardInterrupt:
        logger.info("Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Error running GUI: {e}")
        logger.info("Please check your environment and dependencies")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
