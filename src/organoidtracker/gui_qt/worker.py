"""Serialized background operations. No worker reads or calls a widget."""

import logging
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..services.pipeline import RunCancelled, check_video, export_saved_result, run_session
from ..services.saved_results import SavedResult, load_saved_result
from ..services.session import Session
from ..services.video_frames import decode_video, load_video_frames
from ..services.video_source import VideoSource


@dataclass
class Opened:
    session: Session
    source: VideoSource
    frames: list[np.ndarray]
    saved: SavedResult | None = None
    warning: str | None = None
    source_frames: list[np.ndarray] | None = None


def preview_session(session):
    from ..core.video_frames import selected_frame_indices

    digest = check_video(session)
    frames, fps = decode_video(session.video.path)
    indices = selected_frame_indices(len(frames), session.video.frame_indices, session.video.decoded_frames)
    selected = [frames[index] for index in indices]
    height, width = selected[0].shape[:2]
    session.annotations.check_inside(width, height)
    if session.video.frame_indices is not None:
        session.timing.resolve(len(selected))
    source = VideoSource(
        session.video.path,
        digest,
        len(frames),
        tuple(indices),
        height,
        width,
        fps,
        session.tracking.direction,
        len(selected) - 1 if session.tracking.reverse else 0,
    )
    return Opened(session, source, selected, source_frames=frames)


def open_saved(path):
    saved = load_saved_result(path)
    warning = None
    try:
        frames = load_video_frames(saved.session.video.path, saved.video)
    except (ValueError, OSError) as error:
        frames = []
        warning = f"Source video unavailable: {error}. Masks and report export remain available."
    return Opened(saved.session, saved.video, frames, saved, warning)


class Operation(QThread):
    progress = Signal(str, str, int, int, str)

    def __init__(self, token, action, inputs, parent=None, tracking_factory=None):
        super().__init__(parent)
        self.token, self.action, self.inputs = token, action, inputs
        self.cancel = threading.Event()
        self.tracking_factory = tracking_factory
        self.value: Any = None
        self.error = None
        self.recovery = None
        self.cancelled_without_masks = False

    def run(self):
        try:
            if self.action == "preview":
                self.value = preview_session(self.inputs["session"])
            elif self.action == "open":
                self.value = open_saved(self.inputs["path"])
            elif self.action == "relocate":
                saved, path = self.inputs["saved"], self.inputs["path"]
                frames = load_video_frames(path, saved.video)
                self.value = Opened(saved.session.with_video(path), replace(saved.video, path=path), frames, saved)
            else:

                def progress(phase, current, total, message):
                    self.progress.emit(self.token, phase, current, total, message)

                if self.action == "track":
                    self.value = run_session(
                        **self.inputs,
                        progress=progress,
                        cancel=self.cancel,
                        tracking_service_factory=self.tracking_factory,
                    )
                elif self.action == "export":
                    self.value = export_saved_result(**self.inputs, progress=progress)
                else:
                    raise ValueError(f"Unknown operation: {self.action}")
        except RunCancelled:
            self.cancelled_without_masks = True
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            logging.getLogger(__name__).exception("%s failed", self.action)
            directory = self.inputs.get("output_dir")
            if directory and (Path(directory) / "results.json").is_file():
                try:
                    self.recovery = load_saved_result(directory)
                except (ValueError, OSError):
                    logging.getLogger(__name__).exception("Could not reopen saved results after failure")
