"""Tk desktop application.

``VideoTrackerApp`` is imported lazily so that importing this package (as the console script
does to reach ``launcher.main``) does not load the application, its settings and torch before
the launcher has checked the settings file and configured logging.
"""

__all__ = ["VideoTrackerApp"]


def __getattr__(name: str):
    if name == "VideoTrackerApp":
        from .main_window import VideoTrackerApp

        return VideoTrackerApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
