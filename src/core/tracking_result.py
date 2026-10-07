"""Tracking results with an explicit outcome.

The GUI, the video writer and the analysis engine all consume tracking results as a plain
mapping ``{chronological_frame_index: {object_id: mask}}``. ``TrackingResult`` keeps that
interface (it is a ``dict``) and adds what a plain dict could not say: whether the run
completed, stopped early or failed, how many frames were produced, which direction was
tracked, which frame carried the prompts, and the per-object presence scores SAM2 reports.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class TrackingResult(dict):
    """``{frame_idx: {obj_id: mask}}`` plus run metadata."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"

    def __init__(
        self,
        *args,
        status: str = COMPLETED,
        frames_total: int = 0,
        frames_done: int = 0,
        error: Optional[str] = None,
        direction: str = "reverse",
        annotation_frame: Optional[int] = None,
        frame_map: Optional[List[int]] = None,
        presence: Optional[Dict[int, Dict[int, float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.status = status
        self.frames_total = int(frames_total)
        self.frames_done = int(frames_done)
        self.error = error
        self.direction = direction
        self.annotation_frame = annotation_frame
        self.frame_map = list(frame_map) if frame_map is not None else []
        self.presence = presence if presence is not None else {}

    @property
    def is_complete(self) -> bool:
        return self.status == self.COMPLETED

    @property
    def is_partial(self) -> bool:
        return self.status == self.PARTIAL

    def object_ids(self) -> List[int]:
        ids = set()
        for frame_masks in self.values():
            ids.update(frame_masks.keys())
        return sorted(ids)

    def summary(self) -> str:
        base = f"{self.status}: {self.frames_done}/{self.frames_total} frames, {len(self.object_ids())} objects, {self.direction}"
        if self.error:
            base += f" ({self.error})"
        return base
