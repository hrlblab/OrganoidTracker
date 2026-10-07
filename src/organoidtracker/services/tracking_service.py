"""One tracking backend instance, its lifecycle and its provenance.

``TrackingService`` wraps the existing ``BaseVideoTracker`` contract (``load_model``,
``load_video``, ``add_bbox_prompt``, ``run_tracking``, ``provenance``) so that callers never
touch the backend directly. One service owns one backend and runs one operation at a time; the
vendored SAM 2 keeps global Hydra and Torch state, so two model operations must never overlap.
Frame ids are chronological throughout (see ``core.sam2_tracker``); prompts go to the
annotation frame, which is display index 0 of the backend.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..core.base_model import BaseVideoTracker
from ..core.tracking_result import TrackingResult
from .annotations import AnnotationError, AnnotationSet
from .session import TrackingSpec
from .video_source import VideoSource

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

BACKEND_NAME = "sam2"


class TrackingError(RuntimeError):
    """The backend could not be created or loaded, refused a prompt, or produced no masks."""


class TrackingService:
    """Lifecycle: idle -> preparing (model) -> ready (video, prompts) -> running -> completed | partial | failed."""

    def __init__(self, tracker: BaseVideoTracker, run_id: str | None = None) -> None:
        self.tracker = tracker
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.state = "idle"
        self.video: VideoSource | None = None
        self.annotations: AnnotationSet | None = None
        self.result: TrackingResult | None = None

    @classmethod
    def create(cls, spec: TrackingSpec, registry: Any | None = None) -> TrackingService:
        """A service around a new backend instance built from the registry, as the GUI does."""
        if registry is None:
            from ..core.model_registry import get_model_registry

            registry = get_model_registry()
        kwargs: dict[str, Any] = {
            "device": spec.device,
            "model_config": spec.model_config,
            "checkpoint_path": None if spec.checkpoint_path is None else str(spec.checkpoint_path),
            "enable_reverse_tracking": spec.reverse,
        }
        if spec.checkpoint_family is not None:
            kwargs["checkpoint_family"] = spec.checkpoint_family
        tracker = registry.create_model_instance(BACKEND_NAME, **kwargs)
        if tracker is None:
            raise TrackingError(
                f"the {BACKEND_NAME!r} backend is not available or could not be created with {kwargs}; see the log"
            )
        return cls(tracker)

    # ------------------------------------------------------------------ lifecycle
    def load_model(self) -> None:
        self.state = "preparing"
        if not self.tracker.load_model():
            self.state = "failed"
            checkpoint = getattr(self.tracker, "checkpoint_path", None)
            raise TrackingError(
                f"the model did not load (checkpoint {checkpoint}); see the log for the reason, "
                "for example a missing checkpoint file"
            )
        logger.info(f"Run {self.run_id}: model loaded")

    def open_video(self, path: Path | str) -> VideoSource:
        """Decode the video through the backend and describe what it holds."""
        path = Path(path)
        info = self.tracker.load_video(str(path))
        dimensions = info.get("dimensions") or (0, 0)
        self.video = VideoSource(
            path=path,
            sha256=getattr(self.tracker, "video_sha256", None),
            decoded_frames=int(info.get("decoded_frames", info["num_frames"])),
            frame_map=tuple(int(i) for i in info.get("frame_map", range(info["num_frames"]))),
            height=int(dimensions[0]),
            width=int(dimensions[1]),
            nominal_fps=info.get("fps"),
            direction=str(info.get("direction", "reverse")),
            annotation_frame=int(info.get("annotation_frame_index", 0)),
        )
        self.state = "ready"
        logger.info(
            f"Run {self.run_id}: {self.video.n_frames} unique frames ({self.video.decoded_frames} decoded), "
            f"{self.video.width}x{self.video.height}, annotation frame {self.video.annotation_frame} ({self.video.direction})"
        )
        return self.video

    def annotate(self, annotations: AnnotationSet) -> None:
        """Send every cyst box to the backend as a prompt on the annotation frame (object id = cyst id)."""
        if self.video is None:
            raise TrackingError("open a video before adding annotations")
        if not annotations.cysts:
            raise AnnotationError("at least one cyst box is required to track")
        annotations.check_inside(self.video.width, self.video.height)
        for cyst in annotations.cysts:
            x1, y1, x2, y2 = cyst.bbox
            # frame_idx is a display index for the backend; 0 is the annotation frame
            if not self.tracker.add_bbox_prompt(x1, y1, x2, y2, obj_id=cyst.cyst_id, frame_idx=0):
                raise TrackingError(f"the backend rejected the box of cyst {cyst.cyst_id}: {list(cyst.bbox)}")
        self.annotations = annotations
        logger.info(
            f"Run {self.run_id}: {len(annotations.cysts)} cyst prompts for {len(annotations.organoids)} organoids"
        )

    def run(self, progress: ProgressCallback | None = None) -> TrackingResult:
        """Propagate the prompts; the result says whether the run completed or stopped early."""
        if self.annotations is None:
            raise TrackingError("add annotations before tracking")
        self.state = "running"
        try:
            result = self.tracker.run_tracking(progress)
        except Exception as error:
            self.state = "failed"
            raise TrackingError(f"tracking failed: {error}") from error
        if not isinstance(result, TrackingResult):
            # another backend returned the plain legacy mapping; wrap it with what is known
            result = TrackingResult(
                result,
                status=TrackingResult.COMPLETED,
                frames_total=self.video.n_frames if self.video else len(result),
                frames_done=len(result),
                direction=self.video.direction if self.video else "reverse",
                annotation_frame=self.video.annotation_frame if self.video else None,
                frame_map=list(self.video.frame_map) if self.video else None,
                tracked_frames=sorted(result),
            )
        self.result = result
        self.state = result.status
        logger.info(f"Run {self.run_id}: tracking {result.summary()}")
        return result

    # ------------------------------------------------------------------ facts
    @property
    def frames(self) -> list[np.ndarray]:
        frames = self.tracker.video_frames
        if frames is None:
            raise TrackingError("no video is loaded")
        return frames

    def provenance(self) -> dict[str, Any]:
        """The backend's reproduction facts (software, model, video, frame handling, mask settings)."""
        describe = getattr(self.tracker, "provenance", None)
        if callable(describe):
            facts = dict(describe())
        else:
            facts = {"backend": self.tracker.model_name}
        facts.setdefault("backend", BACKEND_NAME)
        return facts
