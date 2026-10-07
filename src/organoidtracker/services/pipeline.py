"""The complete headless path: session -> backend -> tracking -> analysis -> exports -> manifest."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..analysis.organoid_report_generator import Analysis
from ..core.tracking_result import TrackingResult
from .analysis_service import AnalysisService
from .annotations import AnnotationError
from .export_service import ExportService
from .prompt_record import build_prompt_record
from .run_manifest import build_manifest
from .session import Session, SessionError, TrackingSpec
from .tracking_service import TrackingService

logger = logging.getLogger(__name__)

# (phase, current, total, message); phases: "tracking", "videos"
PhaseProgress = Callable[[str, int, int, str], None]
TrackingServiceFactory = Callable[[TrackingSpec], TrackingService]

EXIT_COMPLETED = 0
EXIT_PARTIAL = 3


@dataclass
class RunOutcome:
    """What a run produced. ``status`` is the tracker's: ``completed`` or ``partial`` (never silently the former)."""

    status: str
    output_dir: Path
    manifest_path: Path
    summary: dict[str, Any]
    result: TrackingResult
    analysis: Analysis
    video_paths: dict[str, str | None] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.status == TrackingResult.COMPLETED

    @property
    def exit_code(self) -> int:
        return EXIT_COMPLETED if self.complete else EXIT_PARTIAL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_video(session: Session) -> str:
    """The video must exist and, when the session records a hash, have that content. Returns the sha256."""
    path = Path(session.video.path)
    if not path.is_file():
        raise SessionError(f"video not found: {path} (relocate it with --video)")
    digest = sha256_file(path)
    if session.video.sha256 and digest != session.video.sha256:
        raise SessionError(
            f"video {path} does not match the session's recorded content (sha256 {digest[:12]}... vs "
            f"{session.video.sha256[:12]}...); this is a different video"
        )
    return digest


def run_session(
    session: Session,
    output_dir: Path | str,
    *,
    videos: bool = True,
    video_quality: str = "original",
    overwrite: bool = False,
    debug: bool = False,
    progress: PhaseProgress | None = None,
    tracking_service_factory: TrackingServiceFactory | None = None,
) -> RunOutcome:
    """Track, analyze and export one session into ``output_dir``.

    Raises ``SessionError`` for an unusable input, ``ExportError`` for an unusable output
    directory or a failed export, ``TrackingError`` when the backend fails before producing
    any mask. A run that stopped early returns normally with ``status == "partial"``: its
    exports are written and every one of them says so.
    """
    timings: dict[str, float] = {}
    exporter = ExportService(output_dir)
    exporter.video_export_parameters(video_quality)  # fail early on a bad quality name
    if not session.annotations.cysts:  # before anything is written: nothing to track
        raise AnnotationError("at least one cyst box is required to track")
    video_sha256 = check_video(session)
    exporter.prepare(overwrite=overwrite)
    exporter.write_session(session, video_sha256)

    service = (
        tracking_service_factory(session.tracking)
        if tracking_service_factory
        else TrackingService.create(session.tracking)
    )
    started = time.perf_counter()
    service.load_model()
    timings["load_model_s"] = round(time.perf_counter() - started, 3)

    started = time.perf_counter()
    source = service.open_video(session.video.path)
    timings["load_video_s"] = round(time.perf_counter() - started, 3)
    if source.sha256 and source.sha256 != video_sha256:  # pragma: no cover - the file changed under us
        raise SessionError(f"video {session.video.path} changed while it was being read")
    resolved_timing = session.timing.resolve(source.n_frames)

    service.annotate(session.annotations)
    exporter.write_prompt_record(
        build_prompt_record(
            service.tracker,
            session.video.path,
            session.annotations.organoid_data(),
            resolved_timing.time_lapse_days,
            session.calibration.um_per_pixel,
            frame_times_days=session.timing.frame_times_days,
        )
    )

    started = time.perf_counter()
    result = service.run(_phase("tracking", progress))
    timings["tracking_s"] = round(time.perf_counter() - started, 3)
    if result.is_partial:
        logger.warning(f"Tracking stopped early: {result.summary()}; the exports cover the tracked frames only")

    analysis = AnalysisService(exporter.generator).analyze(
        result, session.annotations, session.calibration, session.timing, debug_mode=debug
    )

    video_export: dict[str, Any] | None = None
    video_paths: dict[str, str | None] = {}
    if videos:
        started = time.perf_counter()
        video_paths = exporter.write_videos(
            service.frames, result, quality=video_quality, progress=_phase("videos", progress), debug=debug
        )
        timings["videos_s"] = round(time.perf_counter() - started, 3)
        video_export = exporter.video_export_parameters(video_quality)

    started = time.perf_counter()
    summary = exporter.write_report(analysis, debug_mode=debug, original_frames=service.frames)
    timings["report_s"] = round(time.perf_counter() - started, 3)

    manifest = build_manifest(
        run_id=service.run_id,
        session=session,
        video=source,
        provenance=service.provenance(),
        result=result,
        analysis=analysis,
        video_export=video_export,
        output_dir=exporter.output_dir,
        timings_s=timings,
    )
    manifest_path = exporter.write_manifest(manifest)
    logger.info(f"Run {service.run_id} {result.status}: outputs in {exporter.output_dir}")
    return RunOutcome(
        status=result.status,
        output_dir=exporter.output_dir,
        manifest_path=manifest_path,
        summary=summary,
        result=result,
        analysis=analysis,
        video_paths=video_paths,
        timings_s=timings,
    )


def _phase(name: str, progress: PhaseProgress | None) -> Callable[[int, int, str], None] | None:
    if progress is None:
        return None

    def report(current: int, total: int, message: str) -> None:
        progress(name, current, total, message)

    return report
