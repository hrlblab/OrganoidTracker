"""GUI-free application services.

The services wrap the existing implementation (the ``BaseVideoTracker`` backend contract, the
analysis engine and report generator, the video writer) behind small, typed entry points that
the command line, the Tk application and the coming Qt application share. Nothing in this
package imports Tk or Qt.

- :mod:`annotations`: organoid points and cyst boxes in source pixels (``AnnotationSet``).
- :mod:`session`: the validated input document of a run (``Session``), loaded from JSON.
- :mod:`video_source`: what is known about the loaded video (``VideoSource``).
- :mod:`tracking_service`: one backend instance, its lifecycle and provenance.
- :mod:`analysis_service`: measurements from a ``TrackingResult`` (``Analysis``).
- :mod:`export_service`: videos, CSV, figures, PDF, prompt record and run manifest.
- :mod:`pipeline`: ``run_session``, the complete headless path.
"""

from .analysis_service import AnalysisService
from .annotations import AnnotationError, AnnotationSet, CystAnnotation, OrganoidAnnotation
from .export_service import ExportError, ExportService
from .pipeline import RunOutcome, run_session
from .session import (
    Calibration,
    Session,
    SessionError,
    Timing,
    TrackingSpec,
    VideoReference,
    load_session,
    session_from_document,
)
from .tracking_service import TrackingError, TrackingService
from .video_source import VideoSource

__all__ = [
    "AnalysisService",
    "AnnotationError",
    "AnnotationSet",
    "Calibration",
    "CystAnnotation",
    "ExportError",
    "ExportService",
    "OrganoidAnnotation",
    "RunOutcome",
    "Session",
    "SessionError",
    "Timing",
    "TrackingError",
    "TrackingService",
    "TrackingSpec",
    "VideoReference",
    "VideoSource",
    "load_session",
    "run_session",
    "session_from_document",
]
