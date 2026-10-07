"""Exports of a run: videos, CSV tables, figures, PDF, prompt record, session copy and manifest.

Everything lands in one output directory: the report files at its root (so the directory can
be re-plotted with ``scripts/csv_visualizer.py``), the videos under ``videos/``, plus
``session.json``, ``prompts.json`` and ``run_manifest.json``. The video and report parameters
are the GUI's (5 frames per second, overlay alpha 0.4, quality presets original/mid/low).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.organoid_report_generator import Analysis, OrganoidAnalysisReportGenerator
from ..core.tracking_result import TrackingResult
from ..io.video_output import VideoOutputGenerator
from .run_manifest import RUN_MANIFEST_NAME
from .session import Session

logger = logging.getLogger(__name__)

VIDEO_FPS = 5.0
VIDEO_ALPHA = 0.4
VIDEO_QUALITY_SCALES = {"original": 1.0, "mid": 0.5, "low": 0.25}
SESSION_NAME = "session.json"
PROMPT_RECORD_NAME = "prompts.json"
VIDEO_DIR_NAME = "videos"

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
    def prepare(self, overwrite: bool = False) -> Path:
        """Create the output directory; refuse to overwrite a previous run unless asked."""
        manifest = self.output_dir / RUN_MANIFEST_NAME
        if manifest.exists() and not overwrite:
            raise ExportError(
                f"{self.output_dir} already holds a run ({RUN_MANIFEST_NAME}); choose another directory or allow overwriting"
            )
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ExportError(f"cannot create the output directory {self.output_dir}: {error}") from error
        return self.output_dir

    # ------------------------------------------------------------------ small files
    def write_session(self, session: Session, video_sha256: str | None = None) -> Path:
        document = session.to_document()
        document["video"]["path"] = str(Path(session.video.path).resolve())
        if video_sha256 and not document["video"].get("sha256"):
            document["video"]["sha256"] = video_sha256
        return self._write_json(self.output_dir / SESSION_NAME, document)

    def write_prompt_record(self, record: dict[str, Any]) -> Path:
        return self._write_json(self.output_dir / PROMPT_RECORD_NAME, record)

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        return self._write_json(self.output_dir / RUN_MANIFEST_NAME, manifest)

    def _write_json(self, path: Path, data: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

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
    ) -> dict[str, str | None]:
        """Overlay, mask and side-by-side videos (chronological frames, as the GUI exports them)."""
        parameters = self.video_export_parameters(quality)
        setattr(self.video_generator, "debug_mode", debug)  # noqa: B010  # dynamic flag, as the GUI sets it
        created = self.video_generator.create_optimized_multi_object_videos(
            frames=list(frames),
            video_segments=result,
            output_dir=str(self.output_dir / VIDEO_DIR_NAME),
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
        csv_files = summary.get("output_files", {}).get("csv_files", {})
        missing = [name for name in ("raw_data", "cyst_summary", "organoid_summary") if not csv_files.get(name)]
        if missing:
            raise ExportError(f"the report did not produce the CSV tables {missing}; see the log")
        return summary
