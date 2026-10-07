"""Synthetic fixtures shared by the tests. No lab data is used anywhere in the suite."""

from __future__ import annotations

import pathlib

import numpy as np


class DiscVideo:
    """A bright disc moving along the diagonal, one position per frame, on a dark background."""

    def __init__(self, n_frames=8, size=512, radius=30, start=80, step=50):
        self.n_frames, self.size, self.radius = n_frames, size, radius
        self.centers = [(start + step * k, start + step * k) for k in range(n_frames)]

    def frame(self, k: int) -> np.ndarray:
        import cv2

        img = np.full((self.size, self.size, 3), 40, np.uint8)
        cv2.circle(img, self.centers[k], self.radius, (235, 235, 235), -1)
        return img

    def frames(self):
        return [self.frame(k) for k in range(self.n_frames)]

    def disc_mask(self, k: int) -> np.ndarray:
        yy, xx = np.mgrid[0 : self.size, 0 : self.size]
        cx, cy = self.centers[k]
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= self.radius**2

    def box(self, k: int, pad: int = 8):
        cx, cy = self.centers[k]
        r = self.radius + pad
        return (cx - r, cy - r, cx + r, cy + r)

    def closest_frame(self, image: np.ndarray) -> int:
        """Which synthetic frame an (RGB) image matches best, by mean absolute difference."""
        diffs = [np.abs(image.astype(np.int16) - self.frame(k).astype(np.int16)).mean() for k in range(self.n_frames)]
        return int(np.argmin(diffs))

    def write(self, path: pathlib.Path, repeat_pattern=None) -> pathlib.Path:
        """Write frames as mp4; ``repeat_pattern`` lists frame indices to emit (duplicates allowed)."""
        import cv2

        order = list(range(self.n_frames)) if repeat_pattern is None else list(repeat_pattern)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (self.size, self.size))
        assert writer.isOpened(), "OpenCV cannot write mp4v video"
        for k in order:
            writer.write(self.frame(k))  # grayscale content, so BGR/RGB order does not matter
        writer.release()
        return path


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def read_video(path) -> list:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


class FakeTracker:
    """A backend that returns the synthetic disc's masks without a model.

    Implements the ``BaseVideoTracker`` surface the services use (``load_model``, ``load_video``,
    ``add_bbox_prompt``, ``run_tracking``, ``provenance``) and reproduces the situations the
    analysis must survive: an interrupted run (``partial_after``), tracked frames without any
    accepted mask (``frames_without_masks``), missing observations (``drop``), annotated cysts
    the tracker never produces (``never_track``). Object ``o`` is a disc shifted by 5 px per
    object whose radius grows by ``grow`` px per chronological frame.
    """

    def __init__(
        self,
        disc: DiscVideo,
        *,
        enable_reverse_tracking=True,
        partial_after=None,
        frames_without_masks=(),
        drop=(),
        never_track=(),
        grow=2,
        fail_load=False,
        reject_prompts=False,
        checkpoint_path="/nonexistent/fake.pt",
        **kwargs,
    ):
        import hashlib

        self.disc = disc
        self.model_name = "fake"
        self.enable_reverse_tracking = bool(enable_reverse_tracking)
        self.partial_after = partial_after
        self.frames_without_masks = set(frames_without_masks)
        self.drop = set(drop)
        self.never_track = set(never_track)
        self.grow = grow
        self.fail_load = fail_load
        self.reject_prompts = reject_prompts
        self.kwargs = kwargs
        self.video_frames = None
        self.video_path = None
        self.fps = None
        self.prompts = {}
        self.is_loaded = False
        self.frame_map = []
        self.decoded_frame_count = 0
        self.annotation_frame_index = 0
        self.video_sha256 = None
        self.checkpoint_path = checkpoint_path
        self._hashlib = hashlib

    def load_model(self, **kwargs):
        self.is_loaded = not self.fail_load
        return self.is_loaded

    def load_video(self, video_path, max_frames=None):
        frames = read_video(video_path)
        self.video_frames = frames
        self.video_path = video_path
        self.fps = 2.0
        self.frame_map = list(range(len(frames)))
        self.decoded_frame_count = len(frames)
        self.video_sha256 = self._hashlib.sha256(pathlib.Path(video_path).read_bytes()).hexdigest()
        self.annotation_frame_index = len(frames) - 1 if self.enable_reverse_tracking else 0
        self.prompts = {}
        return {
            "num_frames": len(frames),
            "decoded_frames": len(frames),
            "duplicate_frames_removed": 0,
            "frame_map": list(self.frame_map),
            "fps": self.fps,
            "dimensions": frames[0].shape[:2],
            "direction": "reverse" if self.enable_reverse_tracking else "forward",
            "annotation_frame_index": self.annotation_frame_index,
        }

    def get_first_frame(self):
        return self.video_frames[self.annotation_frame_index]

    def add_bbox_prompt(self, x1, y1, x2, y2, obj_id=1, frame_idx=0):
        if self.reject_prompts:
            return False
        n = len(self.video_frames)
        chronological = n - 1 - frame_idx if self.enable_reverse_tracking else frame_idx
        self.prompts.setdefault(obj_id, []).append(
            {
                "frame_idx": chronological,
                "display_frame_idx": frame_idx,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "type": "bbox",
            }
        )
        return True

    def mask(self, k: int, obj_id: int) -> np.ndarray:
        yy, xx = np.mgrid[0 : self.disc.size, 0 : self.disc.size]
        cx, cy = self.disc.centers[k]
        r = self.disc.radius + self.grow * k
        return (xx - (cx + 5 * (obj_id - 1))) ** 2 + (yy - cy) ** 2 <= r * r

    def run_tracking(self, progress_callback=None):
        from organoidtracker.core.masks import PackedMask
        from organoidtracker.core.tracking_result import TrackingResult

        n = len(self.video_frames)
        order = range(n - 1, -1, -1) if self.enable_reverse_tracking else range(n)
        result = TrackingResult(
            status=TrackingResult.COMPLETED,
            frames_total=n,
            direction="reverse" if self.enable_reverse_tracking else "forward",
            annotation_frame=self.annotation_frame_index,
            frame_map=self.frame_map,
        )
        for frame_idx in order:
            if self.partial_after is not None and result.frames_done >= self.partial_after:
                result.status = TrackingResult.PARTIAL
                result.error = "RuntimeError: synthetic interruption"
                break
            frame_masks = {}
            for obj_id in sorted(self.prompts):
                if (
                    obj_id in self.never_track
                    or frame_idx in self.frames_without_masks
                    or (frame_idx, obj_id) in self.drop
                ):
                    continue
                frame_masks[obj_id] = PackedMask(self.mask(frame_idx, obj_id))
            if frame_masks:
                result[frame_idx] = frame_masks
            result.presence[frame_idx] = {
                obj_id: 10.0 for obj_id in sorted(self.prompts) if obj_id not in self.never_track
            }
            result.tracked_frames.append(frame_idx)
            result.frames_done += 1
            if progress_callback:
                progress_callback(result.frames_done, n, f"Processing frame {result.frames_done}/{n}")
        if not result:
            raise RuntimeError("No tracking results obtained. Check model compatibility.")
        return result

    def provenance(self):
        return {
            "backend": "fake",
            "model_config": "sam2_hiera_t",
            "checkpoint_family": "2.1",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": None,
            "config_file": None,
            "device": "cpu",
            "video_path": self.video_path,
            "video_sha256": self.video_sha256,
            "direction": "reverse" if self.enable_reverse_tracking else "forward",
            "annotation_frame_index": self.annotation_frame_index,
        }


def expected_growth_per_day(areas_by_frame: dict, timestamps: list, conversion: float) -> float:
    """The paper's overall growth rate from per-frame areas (pixels): mean of dA/dt in um^2 per day."""
    frames = sorted(areas_by_frame)
    rates = []
    for a, b in zip(frames, frames[1:]):
        dt = timestamps[b] - timestamps[a]
        rates.append((areas_by_frame[b] - areas_by_frame[a]) * conversion**2 / dt)
    return float(np.mean(rates)) if rates else 0.0
