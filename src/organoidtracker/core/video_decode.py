"""Shared OpenCV pixels with a container-index completeness check.

OpenCV may estimate CAP_PROP_FRAME_COUNT from duration and frame rate. PyAV exposes
AVStream.nb_frames instead: zero means unknown, not a count to compare against.
"""

from pathlib import Path

import av
import cv2
import numpy as np


def indexed_frame_count(path: Path | str) -> int:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in: {path}")
        return int(container.streams.video[0].frames)


def decode_frames(path: Path | str, max_frames: int | None = None) -> tuple[list[np.ndarray], float, int]:
    """Decode chronological RGB frames; reject a short read against an explicit index.

    A requested prefix is intentional. Unknown index counts are not replaced by FPS
    estimates. No frame collapse or timing calculation happens until this check passes.
    """
    try:
        expected = indexed_frame_count(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read video index: {path}: {error}") from error
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        while max_frames is None or len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    required = min(expected, max_frames) if max_frames is not None else expected
    if expected > 0 and len(frames) != required:
        raise ValueError(
            f"Incomplete video: index declares {expected} frames, decoded {len(frames)} "
            f"(expected {required} for this read). Restore an intact source; the experiment time axis was not changed."
        )
    if not frames:
        raise ValueError(f"No frames could be decoded from: {path}")
    return frames, fps, expected
