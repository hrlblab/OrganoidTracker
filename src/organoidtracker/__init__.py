"""Organoid Tracker: SAM2-powered tracking and analysis of cysts in kidney organoid videos."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("organoidtracker")
except PackageNotFoundError:  # imported from a source tree that was not installed
    __version__ = "0.0.0+unknown"
