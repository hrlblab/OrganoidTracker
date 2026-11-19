"""The complete headless paths.

``run_session``: session -> backend -> tracking -> saved result -> analysis -> exports -> manifest.
``export_saved_result``: saved result -> analysis -> exports -> manifest, without a model.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .. import RESULTS_VERSION
from ..analysis.organoid_report_generator import Analysis
from ..core.tracking_result import TrackingResult
from .analysis_service import AnalysisService
from .annotations import AnnotationError
from .export_service import PROMPT_RECORD_NAME, ExportService, report_source
from .prompt_record import build_prompt_record
from .run_manifest import build_manifest
from .saved_results import SavedResult
from .session import Calibration, Session, SessionError, Timing, TrackingSpec
from .tracking_service import TrackingService
from .video_frames import load_video_frames, sha256_file

__all__ = ["RunCancelled", "RunOutcome", "check_video", "export_saved_result", "run_session", "sha256_file"]

logger = logging.getLogger(__name__)

# (phase, current, total, message); phases: "tracking", "videos"
PhaseProgress = Callable[[str, int, int, str], None]
TrackingServiceFactory = Callable[[TrackingSpec], TrackingService]

EXIT_COMPLETED = 0
EXIT_PARTIAL = 3
EXIT_CANCELLED = 4


class RunCancelled(RuntimeError):
    """The run was cancelled before any mask was kept: nothing was saved, analyzed or exported."""

    def __init__(self, result: TrackingResult, output_dir: Path) -> None:
        super().__init__(f"tracking cancelled after {result.frames_done} of {result.frames_total} frames, no mask kept")
        self.result = result
        self.output_dir = output_dir


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
    saved_result: SavedResult | None = None  # the results.json and mask file the directory holds

    @property
    def complete(self) -> bool:
        return self.status == TrackingResult.COMPLETED

    @property
    def exit_code(self) -> int:
        if self.complete:
            return EXIT_COMPLETED
        return EXIT_CANCELLED if self.status == TrackingResult.CANCELLED else EXIT_PARTIAL


def check_video(session: Session) -> str:
    """The video must exist, have the recorded content and hold the confirmed frames. Returns the sha256."""
    path = Path(session.video.path)
    if not path.is_file():
        raise SessionError(f"video not found: {path} (relocate it with --video)")
    digest = sha256_file(path)
    if session.video.sha256 and digest != session.video.sha256:
        raise SessionError(
            f"video {path} does not match the session's recorded content (sha256 {digest[:12]}... vs "
            f"{session.video.sha256[:12]}...); this is a different video"
        )
    if session.video.frame_indices is not None:
        # The confirmed selection against the frame count in the file's index, before any model is
        # loaded. Decoding checks it again; an index without a count (0) leaves it to the decoding.
        from ..core.video_decode import indexed_frame_count
        from ..core.video_frames import selected_frame_indices

        try:
            count = indexed_frame_count(path)
        except (OSError, ValueError):
            count = 0
        if count > 0:
            try:
                selected_frame_indices(count, session.video.frame_indices, session.video.decoded_frames)
            except ValueError as error:
                raise SessionError(f"video {path}: {error}") from error
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
    cancel: threading.Event | None = None,
) -> RunOutcome:
    """Track, analyze and export one session into ``output_dir``.

    Raises ``SessionError`` for an unusable input, ``ExportError`` for an unusable output
    directory or a failed export, ``TrackingError`` when the backend fails before producing
    any mask. A run that stopped early returns normally with ``status == "partial"``: its
    exports are written and every one of them says so. ``cancel`` (set from any thread, for
    example by a signal handler) stops the propagation after the frame in progress: with masks
    kept the run returns with ``status == "cancelled"`` and its exports, otherwise it raises
    ``RunCancelled`` and nothing beyond the session and the prompt record is written.
    """
    timings: dict[str, float] = {}
    exporter = ExportService(output_dir)
    exporter.video_export_parameters(video_quality)  # fail early on a bad quality name
    if not session.annotations.cysts:  # before anything is written: nothing to track
        raise AnnotationError("at least one cyst box is required to track")
    if session.video.frame_indices is None:
        raise SessionError(
            "Confirm source frame selection before tracking: set video.frame_indices and video.decoded_frames, or use --frames and --decoded-frames"
        )
    session.timing.resolve(len(session.video.frame_indices))
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
    source = service.open_video(
        session.video.path,
        frame_indices=session.video.frame_indices,
        expected_decoded_frames=session.video.decoded_frames,
    )
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
    result = service.run(_phase("tracking", progress), should_stop=cancel.is_set if cancel is not None else None)
    timings["tracking_s"] = round(time.perf_counter() - started, 3)
    if result.is_cancelled and not result:
        logger.warning(f"Tracking cancelled before any mask was kept ({result.summary()}); nothing to export")
        raise RunCancelled(result, exporter.output_dir)
    if result.is_partial:
        logger.warning(f"Tracking stopped early: {result.summary()}; the exports cover the tracked frames only")
    elif result.is_cancelled:
        logger.warning(f"Tracking cancelled: {result.summary()}; the exports cover the tracked frames only")

    # The complete experiment goes to disk before any export, so that a failed export loses nothing
    if progress:
        progress("saving", 0, 1, "Saving retained masks")
    started = time.perf_counter()
    saved = exporter.write_results(
        run_id=service.run_id, session=session, video=source, provenance=service.provenance(), result=result
    )
    timings["save_results_s"] = round(time.perf_counter() - started, 3)
    logger.info(f"Run {service.run_id}: results saved ({saved.path.name}, {saved.masks_path.name})")

    if progress:
        progress("analysis", 0, 1, "Measuring tracked cysts")
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
    if progress:
        progress("report", 0, 1, "Writing CSV tables, figures and PDF")
    summary = exporter.write_report(
        analysis,
        debug_mode=debug,
        original_frames=service.frames,
        result=result,
        source=report_source(service.run_id, source, service.provenance()),
    )
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
        artifacts=exporter.report_artifacts(summary, video_paths, debug=debug),
        timings_s=timings,
        results=saved.to_manifest_block(),
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
        saved_result=saved,
    )


def export_saved_result(
    saved: SavedResult,
    output_dir: Path | str,
    *,
    videos: bool = True,
    video_quality: str = "original",
    overwrite: bool = False,
    debug: bool = False,
    video_path: Path | str | None = None,
    progress: PhaseProgress | None = None,
    calibration: Calibration | None = None,
    timing: Timing | None = None,
) -> RunOutcome:
    """Analyze and export a saved result again into ``output_dir``, without a model.

    ``output_dir`` may be the run directory itself (``overwrite`` is then required; the saved
    result, the session, the prompt record and the log survive, the exports and the manifest are
    replaced) or another directory, which receives a copy of the session, the prompt record and
    the saved result, so that it is a complete run directory too. ``video_path`` relocates the
    video (same content) for the videos; without videos the video file is not needed. Raises
    ``SessionError`` for a missing or different video, ``ExportError`` for an unusable output
    directory or a failed export. The status, and so the exit code, is the saved run's.
    Optional ``calibration`` and ``timing`` re-analyze the masks in a new output directory;
    every copied input document records those values and the original saved run remains intact.
    """
    timings: dict[str, float] = {}
    exporter = ExportService(output_dir)
    exporter.video_export_parameters(video_quality)  # fail early on a bad quality name
    session = saved.session if video_path is None else saved.session.with_video(video_path)
    in_place = exporter.output_dir.resolve() == saved.directory.resolve()
    reanalysis = calibration is not None or timing is not None
    if reanalysis:
        if in_place:
            raise SessionError("changed analysis inputs require a new export directory")
        session = replace(session, calibration=calibration or session.calibration, timing=timing or session.timing)
        session.timing.resolve(saved.video.n_frames)
    if saved.results_version != RESULTS_VERSION:
        logger.warning(
            f"The saved result holds results version {saved.results_version}; this software exports "
            f"results version {RESULTS_VERSION}, so the measurements may differ from the original report"
        )

    frames = None
    if videos or video_path is not None:
        started = time.perf_counter()
        frames = load_video_frames(session.video.path, saved.video)
        timings["load_video_s"] = round(time.perf_counter() - started, 3)
    key_frames = frames
    if key_frames is None:
        # The report's cyst key shows the masks on the picture when the video can still be read;
        # an export without videos must not need it, so the key falls back to the masks alone.
        try:
            key_frames = load_video_frames(session.video.path, saved.video)
        except (ValueError, OSError):
            key_frames = None

    exporter.prepare(overwrite=overwrite, keep_results=in_place)
    result = saved.result
    # A relocated video becomes the run's video locator in every file that names one (results.json,
    # session.json, the prompt record), so that the exported run reopens without another relocation;
    # the provenance keeps the path the tracker read.
    relocated = Path(session.video.path) != Path(saved.session.video.path)
    new_video = session.video.path if relocated else None
    if in_place:
        copied = exporter.relocate_saved_run(saved, session.video.path) if relocated else saved
    else:
        exporter.write_session(session, saved.video.sha256)
        record = saved.directory / PROMPT_RECORD_NAME
        if record.is_file():
            exporter.copy_prompt_record(record, video_path=new_video, analysis_session=session if reanalysis else None)
        copied = exporter.copy_saved_result(
            saved, video_path=new_video, analysis_session=session if reanalysis else None
        )

    analysis = AnalysisService(exporter.generator).analyze(
        result, session.annotations, session.calibration, session.timing, debug_mode=debug
    )

    video_export: dict[str, Any] | None = None
    video_paths: dict[str, str | None] = {}
    if videos:
        assert frames is not None
        started = time.perf_counter()
        video_paths = exporter.write_videos(
            frames, result, quality=video_quality, progress=_phase("videos", progress), debug=debug
        )
        timings["videos_s"] = round(time.perf_counter() - started, 3)
        video_export = exporter.video_export_parameters(video_quality)

    started = time.perf_counter()
    video = saved.video if video_path is None else replace(saved.video, path=Path(session.video.path))
    summary = exporter.write_report(
        analysis,
        debug_mode=debug,
        original_frames=key_frames,
        result=result,
        source=report_source(saved.run_id, video, saved.provenance),
    )
    timings["report_s"] = round(time.perf_counter() - started, 3)

    manifest = build_manifest(
        run_id=saved.run_id,
        session=session,
        video=video,
        provenance=saved.provenance,
        result=result,
        analysis=analysis,
        video_export=video_export,
        output_dir=exporter.output_dir,
        artifacts=exporter.report_artifacts(summary, video_paths, debug=debug),
        timings_s=timings,
        produced_by="export",
        source={
            "run_directory": str(saved.directory),
            "results_file": saved.path.name,
            "run_created": saved.created,
            "results_version": saved.results_version,
            "software": saved.software,
            "environment": saved.environment,  # where the masks were produced (this process may lack torch)
            "settings": saved.settings,  # the settings the masks were produced under
        },
        results=copied.to_manifest_block(),
    )
    manifest_path = exporter.write_manifest(manifest)
    logger.info(f"Run {saved.run_id} exported again ({result.status}): outputs in {exporter.output_dir}")
    return RunOutcome(
        status=result.status,
        output_dir=exporter.output_dir,
        manifest_path=manifest_path,
        summary=summary,
        result=result,
        analysis=analysis,
        video_paths=video_paths,
        timings_s=timings,
        saved_result=copied,
    )


def _phase(name: str, progress: PhaseProgress | None) -> Callable[[int, int, str], None] | None:
    if progress is None:
        return None

    def report(current: int, total: int, message: str) -> None:
        progress(name, current, total, message)

    return report
