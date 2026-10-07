"""What is known about the loaded video: chronological frame ids, dimensions, timing facts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoSource:
    """The unique chronological frames of a video as the tracker holds them.

    ``frame_map[i]`` is the decoded index of unique frame ``i``; consecutive near-identical
    frames were collapsed. ``annotation_frame`` is the chronological index the prompts go to
    (the last frame in reverse mode).
    """

    path: Path
    sha256: str | None
    decoded_frames: int
    frame_map: tuple[int, ...]
    height: int
    width: int
    nominal_fps: float | None
    direction: str
    annotation_frame: int

    @property
    def n_frames(self) -> int:
        return len(self.frame_map)

    @property
    def frame_ids(self) -> range:
        return range(self.n_frames)

    @property
    def duplicate_frames_removed(self) -> int:
        return self.decoded_frames - self.n_frames

    def to_document(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "decoded_frames": self.decoded_frames,
            "unique_frames": self.n_frames,
            "frame_map": list(self.frame_map),
            "dimensions": [self.height, self.width],
            "nominal_fps": self.nominal_fps,
            "direction": self.direction,
            "annotation_frame": self.annotation_frame,
        }
