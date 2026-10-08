"""The unique chronological frames of a saved result, decoded without a model.

``SAM2Tracker.load_video`` decodes with OpenCV in chronological order, converts to RGB and
collapses near-identical consecutive frames; the decoded index of every unique frame is the
``frame_map`` recorded with each result. Re-exporting videos from a saved result needs the same
unique frames, so this module decodes the same way and then selects the recorded indices
instead of collapsing again: the frames are the ones the tracker saw, whatever the
duplicate-frame setting is at export time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .session import SessionError
from .video_source import VideoSource


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_video(path: Path | str) -> tuple[list[np.ndarray], float | None]:
    """Every frame of the file as an RGB array in decoded order, and the nominal frame rate."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SessionError(f"cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SessionError(f"no frames could be decoded from: {path}")
    return frames, float(fps) if fps else None


def load_video_frames(path: Path | str, video: VideoSource) -> list[np.ndarray]:
    """The unique frames of ``video`` decoded from ``path`` (the recorded file or a relocated copy of it).

    The file must exist and have the recorded content; it must decode to the recorded number of
    frames at the recorded size, so that the selected frames are the tracker's.
    """
    path = Path(path)
    if not path.is_file():
        raise SessionError(f"video not found: {path} (relocate it with --video, or export without videos)")
    digest = sha256_file(path)
    if video.sha256 and digest != video.sha256:
        raise SessionError(
            f"video {path} does not match the saved result's recorded content (sha256 {digest[:12]}... vs "
            f"{video.sha256[:12]}...); this is a different video"
        )
    frames, _fps = decode_video(path)
    if len(frames) != video.decoded_frames:
        raise SessionError(
            f"video {path} decodes to {len(frames)} frames but the result was tracked on {video.decoded_frames} "
            "decoded frames (a different OpenCV build?)"
        )
    if any(index < 0 or index >= len(frames) for index in video.frame_map):
        raise SessionError(f"the recorded frame map {list(video.frame_map)} exceeds the {len(frames)} decoded frames")
    selected = [frames[index] for index in video.frame_map]
    height, width = selected[0].shape[:2]
    if (height, width) != (video.height, video.width):
        raise SessionError(f"video {path} is {width}x{height}, the result expects {video.width}x{video.height}")
    return selected
