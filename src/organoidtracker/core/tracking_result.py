"""Tracking results with an explicit outcome.

The GUI, the video writer and the analysis engine all consume tracking results as a plain
mapping ``{chronological_frame_index: {object_id: mask}}``. ``TrackingResult`` keeps that
interface (it is a ``dict``) and adds what a plain dict could not say: whether the run
completed, stopped early or failed, how many frames were produced, which direction was
tracked, which frame carried the prompts, and the per-object presence scores SAM2 reports.
"""

from __future__ import annotations


class TrackingResult(dict):
    """``{frame_idx: {obj_id: mask}}`` plus run metadata."""

    COMPLETED = "completed"
    PARTIAL = "partial"  # an error stopped the propagation
    FAILED = "failed"
    CANCELLED = "cancelled"  # a cancel request stopped it between frames

    def __init__(
        self,
        *args,
        status: str = COMPLETED,
        frames_total: int = 0,
        frames_done: int = 0,
        error: str | None = None,
        direction: str = "reverse",
        annotation_frame: int | None = None,
        frame_map: list[int] | None = None,
        presence: dict[int, dict[int, float]] | None = None,
        tracked_frames: list[int] | None = None,
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
        # Frames the propagation visited, in order; a visited frame may have no accepted mask.
        self.tracked_frames = list(tracked_frames) if tracked_frames is not None else []

    @property
    def is_complete(self) -> bool:
        return self.status == self.COMPLETED

    @property
    def is_partial(self) -> bool:
        return self.status == self.PARTIAL

    @property
    def is_cancelled(self) -> bool:
        return self.status == self.CANCELLED

    def object_ids(self) -> list[int]:
        ids = set()
        for frame_masks in self.values():
            ids.update(frame_masks.keys())
        return sorted(ids)

    def summary(self) -> str:
        base = f"{self.status}: {self.frames_done}/{self.frames_total} frames, {len(self.object_ids())} objects, {self.direction}"
        if self.error:
            base += f" ({self.error})"
        return base
