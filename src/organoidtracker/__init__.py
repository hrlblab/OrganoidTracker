"""Organoid Tracker: SAM2-powered tracking and analysis of cysts in kidney organoid videos."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("organoidtracker")
except PackageNotFoundError:  # imported from a source tree that was not installed
    __version__ = "0.0.0+unknown"

# Version of the scientific results (masks, measurements, time axes) that this code produces.
# It is written into every export and bumped whenever a change alters results; the changelog
# records each bump under "Scientific behavior". 0 is the untouched upstream code (tag
# legacy-baseline); 1 is the corrected frame contract of WP1.
RESULTS_VERSION = 1
