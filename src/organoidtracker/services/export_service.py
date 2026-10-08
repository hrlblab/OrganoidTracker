"""Exports of a run: videos, CSV tables, figures, PDF, prompt record, session copy and manifest.

Everything lands in one output directory: the report files at its root (so the directory can
be re-plotted with ``scripts/csv_visualizer.py``), the videos under ``videos/``, plus
``session.json``, ``prompts.json`` and ``run_manifest.json``. The video and report parameters
are the GUI's (5 frames per second, overlay alpha 0.4, quality presets original/mid/low).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.organoid_report_generator import Analysis, OrganoidAnalysisReportGenerator
from ..core.tracking_result import TrackingResult
from ..io.video_output import VideoOutputGenerator
from .run_manifest import RUN_MANIFEST_NAME
from .saved_results import (
    MASKS_GLOB,
    RESULTS_NAME,
    SavedResult,
    discard_staged_result,
    publish_staged_result,
    stage_saved_result,
    write_saved_result,
)
from .session import Session
from .video_source import VideoSource

logger = logging.getLogger(__name__)

VIDEO_FPS = 5.0
VIDEO_ALPHA = 0.4
VIDEO_QUALITY_SCALES = {"original": 1.0, "mid": 0.5, "low": 0.25}
SESSION_NAME = "session.json"
PROMPT_RECORD_NAME = "prompts.json"
VIDEO_DIR_NAME = "videos"
# Everything a run writes into its directory (the log is kept across runs; unknown files are never touched)
RUN_ARTIFACT_FILES = (
    RUN_MANIFEST_NAME,
    SESSION_NAME,
    PROMPT_RECORD_NAME,
    RESULTS_NAME,
    "raw_cyst_data.csv",
    "cyst_summary.csv",
    "organoid_summary.csv",
    "analysis_summary.json",
    "organoid_analysis_report.pdf",
    "experiment_data_debug.json",
)
RUN_ARTIFACT_DIRS = ("visualizations", VIDEO_DIR_NAME)
# What survives when a run is exported again in place (with the mask file results.json names)
SAVED_RESULT_FILES = (SESSION_NAME, PROMPT_RECORD_NAME, RESULTS_NAME)
# The saved-run files (manifest first: it must never describe files of another run)
SAVED_RUN_FILES = (RUN_MANIFEST_NAME, SESSION_NAME, PROMPT_RECORD_NAME, RESULTS_NAME)
# What a report must contain; a missing item fails the run instead of being logged and forgotten
REQUIRED_CSV = (
    ("raw_data", "raw_cyst_data.csv"),
    ("cyst_summary", "cyst_summary.csv"),
    ("organoid_summary", "organoid_summary.csv"),
)
REQUIRED_FIGURES = (
    "organoids_with_cysts",
    "cyst_organoid_ratio",
    "cyst_areas_multiline",
    "cyst_circularity_multiline",
    "circularity_scatter",
    "lasagna_plot",
)

ProgressCallback = Callable[[int, int, str], None]


class ExportError(RuntimeError):
    """The output directory cannot be used or an export step failed."""


class ExportService:
    def __init__(
        self,
        output_dir: Path | str,
        *,
        generator: OrganoidAnalysisReportGenerator | None = None,
        video_generator: VideoOutputGenerator | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.generator = generator or OrganoidAnalysisReportGenerator()
        self.video_generator = video_generator or VideoOutputGenerator()

    # ------------------------------------------------------------------ directory
    def previous_run_artifacts(self) -> list[Path]:
        """Files and directories of an earlier run in the output directory, finished or not."""
        found = [self.output_dir / name for name in RUN_ARTIFACT_FILES if (self.output_dir / name).is_file()]
        if self.output_dir.is_dir():
            found += sorted(path for path in self.output_dir.glob(MASKS_GLOB) if path.is_file())
        found += [self.output_dir / name for name in RUN_ARTIFACT_DIRS if (self.output_dir / name).is_dir()]
        return found

    def previous_saved_run(self) -> list[Path]:
        """The saved-run files the directory holds: manifest, session, prompt record, result and mask files."""
        found = [self.output_dir / name for name in SAVED_RUN_FILES if (self.output_dir / name).is_file()]
        if self.output_dir.is_dir():
            found += sorted(path for path in self.output_dir.glob(MASKS_GLOB) if path.is_file())
        return found

    def prepare(self, overwrite: bool = False, keep_results: bool = False) -> Path:
        """Create the output directory; a directory holding an earlier run is refused unless ``overwrite``.

        With ``overwrite`` every artifact of the earlier run is removed before anything is written, the
        manifest first: the directory never holds a completed manifest next to files of another run,
        and a run that fails midway leaves no manifest at all. With ``keep_results`` (a run exported
        again in place) the saved result, the session and the prompt record survive; the exports go.
        """
        existing = self.previous_run_artifacts()
        if existing and not overwrite:
            names = ", ".join(sorted(path.name for path in existing))
            raise ExportError(
                f"{self.output_dir} already holds a run ({names}); choose another directory or allow overwriting"
            )
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ExportError(f"cannot create the output directory {self.output_dir}: {error}") from error
        if existing:
            removed = self.clear_previous_run(keep_results=keep_results)
            logger.info(f"Replaced the previous run in {self.output_dir}: removed {len(removed)} artifact(s)")
        return self.output_dir

    def clear_previous_run(self, keep_results: bool = False) -> list[str]:
        """Remove the earlier run's artifacts: the manifest, the files it lists, the known output names.

        With ``keep_results`` the saved result (``results.json`` and its mask file), the session and
        the prompt record stay, so that a run can be exported again in place.
        """
        removed: list[str] = []
        root = self.output_dir.resolve()

        def kept(relative: str) -> bool:
            path = Path(relative)
            return keep_results and (
                relative in SAVED_RESULT_FILES or (path.parent == Path() and path.match(MASKS_GLOB))
            )

        manifest = self.output_dir / RUN_MANIFEST_NAME
        inventory: list[str] = []
        if manifest.is_file():
            try:
                inventory = list(json.loads(manifest.read_text(encoding="utf-8")).get("files", {}))
            except (OSError, ValueError):
                inventory = []
            manifest.unlink()
            removed.append(RUN_MANIFEST_NAME)
        for relative in inventory:
            if kept(relative):
                continue
            path = self.output_dir / relative
            try:
                inside = path.resolve().is_relative_to(root)
            except OSError:
                inside = False
            if inside and path.is_file():
                path.unlink()
                removed.append(relative)
        for name in RUN_ARTIFACT_FILES:
            path = self.output_dir / name
            if path.is_file() and not kept(name):
                path.unlink()
                removed.append(name)
        if not keep_results:
            for path in sorted(self.output_dir.glob(MASKS_GLOB)):
                if path.is_file():
                    path.unlink()
                    removed.append(path.name)
        for name in RUN_ARTIFACT_DIRS:
            path = self.output_dir / name
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(name + "/")
        for relative in inventory:  # directories the inventory entries left empty
            parent = (self.output_dir / relative).parent
            while parent != self.output_dir and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
        return removed

    # ------------------------------------------------------------------ small files
    @staticmethod
    def session_document(session: Session, video_sha256: str | None = None) -> dict[str, Any]:
        """The session as saved next to a run: absolute file references, the video hash filled in."""
        document = session.to_document()
        document["video"]["path"] = os.path.abspath(session.video.path)  # holds from any directory
        if session.tracking.checkpoint_path is not None:
            document["tracking"]["checkpoint_path"] = os.path.abspath(session.tracking.checkpoint_path)
        if video_sha256 and not document["video"].get("sha256"):
            document["video"]["sha256"] = video_sha256
        return document

    def write_session(self, session: Session, video_sha256: str | None = None) -> Path:
        return self._write_json(self.output_dir / SESSION_NAME, self.session_document(session, video_sha256))

    def write_prompt_record(self, record: dict[str, Any]) -> Path:
        return self._write_json(self.output_dir / PROMPT_RECORD_NAME, record)

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        return self._write_json(self.output_dir / RUN_MANIFEST_NAME, manifest)

    def _write_json(self, path: Path, data: dict[str, Any]) -> Path:
        """Write atomically: the file either holds the complete document or is unchanged; no temporary is left."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        return path

    def copy_prompt_record(self, record_path: Path) -> Path:
        """The prompt record of the run being exported again, copied next to the new exports."""
        target = self.output_dir / PROMPT_RECORD_NAME
        try:
            shutil.copyfile(record_path, target)
        except OSError as error:
            raise ExportError(f"cannot copy the prompt record to {target}: {error}") from error
        return target

    # ------------------------------------------------------------------ saved result
    def write_results(
        self,
        *,
        run_id: str,
        session: Session,
        video: VideoSource,
        provenance: Mapping[str, Any],
        result: TrackingResult,
    ) -> SavedResult:
        """``results.json`` and the mask file: the run's complete experiment, reloadable without a model."""
        return write_saved_result(
            self.output_dir, run_id=run_id, session=session, video=video, provenance=provenance, result=result
        )

    def save_run(
        self,
        *,
        run_id: str,
        session: Session,
        video: VideoSource,
        provenance: Mapping[str, Any],
        result: TrackingResult,
        prompt_record: Mapping[str, Any],
        replace_existing: bool = False,
    ) -> SavedResult:
        """Save a run's result, session and prompt record together (the window's Save Results).

        A directory that already holds a saved run is refused unless ``replace_existing``. Everything
        is staged under temporary names first; the previous saved run, with its manifest (which must not
        describe the new files), stays in place until the replacement is complete, so a save that fails
        leaves the previous run usable. Exported videos, tables and figures are never touched.
        """
        existing = self.previous_saved_run()
        if existing and not replace_existing:
            names = ", ".join(path.name for path in existing)
            raise ExportError(f"{self.output_dir} already holds a saved run ({names}); confirm replacing it first")
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ExportError(f"cannot create the output directory {self.output_dir}: {error}") from error
        staged = stage_saved_result(
            self.output_dir, run_id=run_id, session=session, video=video, provenance=provenance, result=result
        )
        token = uuid.uuid4().hex
        session_temp = self.output_dir / f".{SESSION_NAME}.{token}.tmp"
        record_temp = self.output_dir / f".{PROMPT_RECORD_NAME}.{token}.tmp"
        try:
            session_temp.write_text(
                json.dumps(self.session_document(session, video.sha256), indent=2, default=str), encoding="utf-8"
            )
            record_temp.write_text(json.dumps(dict(prompt_record), indent=2, default=str), encoding="utf-8")
        except OSError as error:
            discard_staged_result(staged)
            session_temp.unlink(missing_ok=True)
            record_temp.unlink(missing_ok=True)
            raise ExportError(f"cannot write the session or the prompt record in {self.output_dir}: {error}") from error
        manifest = self.output_dir / RUN_MANIFEST_NAME
        try:
            if manifest.is_file():
                manifest.unlink()  # it described the previous run's files
            saved = publish_staged_result(staged)
            os.replace(session_temp, self.output_dir / SESSION_NAME)
            os.replace(record_temp, self.output_dir / PROMPT_RECORD_NAME)
        except OSError as error:
            session_temp.unlink(missing_ok=True)
            record_temp.unlink(missing_ok=True)
            raise ExportError(f"cannot publish the saved run in {self.output_dir}: {error}") from error
        except Exception:
            session_temp.unlink(missing_ok=True)
            record_temp.unlink(missing_ok=True)
            raise
        return saved

    def copy_saved_result(self, saved: SavedResult) -> SavedResult:
        """A byte-identical copy of a saved result (its two files) into the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        masks_target = self.output_dir / saved.masks_path.name
        results_target = self.output_dir / saved.path.name
        temporary = results_target.with_name(results_target.name + ".tmp")
        try:
            shutil.copyfile(saved.masks_path, masks_target)
            shutil.copyfile(saved.path, temporary)
            os.replace(temporary, results_target)  # results.json appears complete or not at all
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ExportError(f"cannot copy the saved result to {self.output_dir}: {error}") from error
        return replace(saved, path=results_target, masks_path=masks_target)

    # ------------------------------------------------------------------ videos
    @staticmethod
    def video_export_parameters(quality: str) -> dict[str, Any]:
        if quality not in VIDEO_QUALITY_SCALES:
            raise ExportError(f"unknown video quality {quality!r}; expected one of {list(VIDEO_QUALITY_SCALES)}")
        return {
            "fps": VIDEO_FPS,
            "alpha": VIDEO_ALPHA,
            "quality": quality,
            "quality_scale": VIDEO_QUALITY_SCALES[quality],
        }

    def write_videos(
        self,
        frames: Sequence[np.ndarray],
        result: TrackingResult,
        *,
        quality: str = "original",
        progress: ProgressCallback | None = None,
        debug: bool = False,
        directory: Path | None = None,
    ) -> dict[str, str | None]:
        """Overlay, mask and side-by-side videos (chronological frames, as the GUI exports them).

        They go to ``videos/`` under the output directory unless ``directory`` names another place
        (the window writes them straight into the directory the user picked).
        """
        parameters = self.video_export_parameters(quality)
        setattr(self.video_generator, "debug_mode", debug)  # noqa: B010  # dynamic flag, as the GUI sets it
        created = self.video_generator.create_optimized_multi_object_videos(
            frames=list(frames),
            video_segments=result,
            output_dir=str(directory if directory is not None else self.output_dir / VIDEO_DIR_NAME),
            fps=parameters["fps"],
            alpha=parameters["alpha"],
            progress_callback=progress,
            quality_scale=parameters["quality_scale"],
        )
        failed = sorted(name for name, path in created.items() if not path)
        if not created or failed:
            raise ExportError(f"video export failed for {failed or 'all types'}; see the log")
        return created

    # ------------------------------------------------------------------ report
    def write_report(
        self, analysis: Analysis, *, debug_mode: bool = False, original_frames: Sequence[np.ndarray] | None = None
    ) -> dict[str, Any]:
        """CSV tables, figures, PDF and ``analysis_summary.json`` at the output directory's root."""
        self.generator._original_frames = list(original_frames) if original_frames is not None else None
        summary = self.generator.write_report(analysis, str(self.output_dir), debug_mode=debug_mode)
        missing = self.missing_report_artifacts(summary)
        if missing:
            # The generator logs and swallows export errors; the directory must not look successful.
            summary["success"] = False
            summary["error"] = f"incomplete report, missing: {', '.join(missing)}"
            self._write_json(self.output_dir / "analysis_summary.json", summary)
            raise ExportError(f"the report is incomplete, missing: {', '.join(missing)}; the log names the error")
        return summary

    def missing_report_artifacts(self, summary: dict[str, Any]) -> list[str]:
        """Required report files that the summary does not name or that do not exist."""
        outputs = summary.get("output_files", {}) or {}
        missing = []
        csv_files = outputs.get("csv_files", {}) or {}
        for key, name in REQUIRED_CSV:
            if not csv_files.get(key) or not Path(csv_files[key]).is_file():
                missing.append(name)
        pdf = outputs.get("pdf_report")
        if not pdf or not Path(pdf).is_file():
            missing.append("organoid_analysis_report.pdf")
        figures = outputs.get("visualizations", {}) or {}
        for key in REQUIRED_FIGURES:
            if not figures.get(key) or not Path(figures[key]).is_file():
                missing.append(f"figure {key}")
        if not (self.output_dir / "analysis_summary.json").is_file():
            missing.append("analysis_summary.json")
        return missing
