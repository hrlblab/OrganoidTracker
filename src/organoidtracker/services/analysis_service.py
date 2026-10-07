"""Measurements from a tracking result: the analysis half of the report, without files."""

from __future__ import annotations

from ..analysis.organoid_report_generator import Analysis, OrganoidAnalysisReportGenerator
from ..core.tracking_result import TrackingResult
from .annotations import AnnotationSet
from .session import Calibration, Timing


class AnalysisService:
    """``ExperimentData`` plus run facts from a ``TrackingResult``, the annotations, calibration and timing.

    The computation is the report generator's own ``analyze`` step, so the GUI's report and a
    headless run measure identically: chronological frame ids, the frame count of the video
    (not of the frames that kept a mask), untracked frames of a partial run as gaps, organoids
    without cysts kept in the population, growth rates per day on the resolved time axis.
    """

    def __init__(self, generator: OrganoidAnalysisReportGenerator | None = None) -> None:
        self.generator = generator or OrganoidAnalysisReportGenerator()

    def analyze(
        self,
        result: TrackingResult,
        annotations: AnnotationSet,
        calibration: Calibration,
        timing: Timing,
        *,
        debug_mode: bool = False,
    ) -> Analysis:
        resolved = timing.resolve(result.frames_total)
        return self.generator.analyze(
            result,
            annotations.organoid_data(),
            resolved.time_lapse_days,
            calibration.um_per_pixel,
            frame_timestamps=resolved.frame_timestamps,
            debug_mode=debug_mode,
        )
