"""The session document: the validated inputs of a tracking and analysis run.

Schema ``organoidtracker.session/1`` (JSON)::

    {
      "schema": "organoidtracker.session/1",
      "video": {"path": "well.mp4", "sha256": "<optional content hash>"},
      "tracking": {"direction": "reverse", "model_config": "sam2_hiera_b",
                   "checkpoint_family": "2.1", "checkpoint_path": null, "device": "cuda"},
      "calibration": {"um_per_pixel": 1.6934},
      "timing": {"time_lapse_days": 6.0},
      "organoids": [{"organoid_id": 1, "point": [x, y],
                     "cysts": [{"cyst_id": 1, "bbox": [x1, y1, x2, y2]}]}]
    }

Every key of ``tracking`` is optional (the defaults are the application's). ``timing`` holds
either ``time_lapse_days`` (the span from the first to the last unique frame, spread uniformly
as in the paper, days numbered from 1) or ``frame_times_days`` (one explicit time per unique
frame). A relative video path is resolved against the directory of the session file. The
prompt record the application writes when tracking starts (schema ``organoidtracker.prompts/1``)
is accepted as well, so a run can be repeated from its record.

Values are type-checked and unknown keys are rejected, in the style of the settings file: a typo
cannot silently fall back to a default.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .annotations import AnnotationError, AnnotationSet

SCHEMA = "organoidtracker.session/1"
PROMPT_RECORD_SCHEMA = "organoidtracker.prompts/1"

# Kept in step with ``core.sam2_tracker`` (tested); listed here so that validating a session
# does not import torch.
MODEL_CONFIGS = ("sam2_hiera_t", "sam2_hiera_s", "sam2_hiera_b", "sam2_hiera_l")
CHECKPOINT_FAMILIES = ("2", "2.1")
DIRECTIONS = ("reverse", "forward")
DEFAULT_MODEL_CONFIG = "sam2_hiera_b"  # the GUI's default size (base-plus)
DEFAULT_DEVICE = "cuda"


class SessionError(ValueError):
    """An invalid session document: unreadable, unknown key, wrong type or inconsistent values."""


def _absolute(path: Path, base_dir: Path | None) -> Path:
    """``path`` made absolute: relative to ``base_dir`` (the session file's directory) or else to the cwd.

    Saved sessions and prompt records then carry references that hold from any directory.
    """
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return Path(os.path.abspath(path))


def _positive_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SessionError(f"{where}: expected a positive number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class VideoReference:
    """The input video and, optionally, the sha256 of its content."""

    path: Path
    sha256: str | None = None

    def to_document(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class TrackingSpec:
    """Backend choices for a run; ``checkpoint_family=None`` means the configured default."""

    direction: str = "reverse"
    model_config: str = DEFAULT_MODEL_CONFIG
    checkpoint_family: str | None = None
    checkpoint_path: Path | None = None
    device: str = DEFAULT_DEVICE

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise SessionError(f"tracking.direction: expected one of {list(DIRECTIONS)}, got {self.direction!r}")
        if self.model_config not in MODEL_CONFIGS:
            raise SessionError(
                f"tracking.model_config: expected one of {list(MODEL_CONFIGS)}, got {self.model_config!r}"
            )
        if self.checkpoint_family is not None and self.checkpoint_family not in CHECKPOINT_FAMILIES:
            raise SessionError(
                f"tracking.checkpoint_family: expected one of {list(CHECKPOINT_FAMILIES)}, got {self.checkpoint_family!r}"
            )
        device = str(self.device)
        if not (device == "cpu" or device == "cuda" or device.startswith("cuda:")):
            raise SessionError(f"tracking.device: expected 'cpu', 'cuda' or 'cuda:<index>', got {self.device!r}")

    @property
    def reverse(self) -> bool:
        return self.direction == "reverse"

    def to_document(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "model_config": self.model_config,
            "checkpoint_family": self.checkpoint_family,
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "device": self.device,
        }


@dataclass(frozen=True)
class Calibration:
    """Micrometers per pixel of the source video."""

    um_per_pixel: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "um_per_pixel", _positive_number(self.um_per_pixel, "calibration.um_per_pixel"))

    def to_document(self) -> dict[str, Any]:
        return {"um_per_pixel": self.um_per_pixel}


@dataclass(frozen=True)
class ResolvedTiming:
    """The time axis for a known number of unique frames."""

    time_lapse_days: float
    frame_timestamps: list[float] | None  # None: uniform spacing from day 1 (the paper's axis)


@dataclass(frozen=True)
class Timing:
    """Either the span from the first to the last frame or one explicit time per frame."""

    time_lapse_days: float | None = None
    frame_times_days: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (self.time_lapse_days is None) == (self.frame_times_days is None):
            raise SessionError("timing: give exactly one of time_lapse_days or frame_times_days")
        if self.time_lapse_days is not None:
            object.__setattr__(
                self, "time_lapse_days", _positive_number(self.time_lapse_days, "timing.time_lapse_days")
            )
        else:
            times = tuple(self.frame_times_days or ())
            if not times:
                raise SessionError("timing.frame_times_days: expected at least one value")
            values = []
            for index, value in enumerate(times):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise SessionError(f"timing.frame_times_days[{index}]: expected a number, got {value!r}")
                values.append(float(value))
            if any(b <= a for a, b in zip(values, values[1:])):
                raise SessionError("timing.frame_times_days: values must increase strictly")
            object.__setattr__(self, "frame_times_days", tuple(values))

    def resolve(self, n_frames: int) -> ResolvedTiming:
        """The time axis for ``n_frames`` unique frames; explicit times must match that count."""
        if n_frames < 1:
            raise SessionError("timing: the video has no frames")
        if self.frame_times_days is None:
            assert self.time_lapse_days is not None
            return ResolvedTiming(self.time_lapse_days, None)
        if len(self.frame_times_days) != n_frames:
            raise SessionError(
                f"timing.frame_times_days has {len(self.frame_times_days)} values but the video has "
                f"{n_frames} unique frames"
            )
        span = self.frame_times_days[-1] - self.frame_times_days[0] if n_frames > 1 else 0.0
        return ResolvedTiming(span, list(self.frame_times_days))

    def to_document(self) -> dict[str, Any]:
        if self.frame_times_days is not None:
            return {"frame_times_days": list(self.frame_times_days)}
        return {"time_lapse_days": self.time_lapse_days}


@dataclass(frozen=True)
class Session:
    """Validated inputs of a run. ``source`` is the file it was loaded from, if any."""

    video: VideoReference
    tracking: TrackingSpec
    calibration: Calibration
    timing: Timing
    annotations: AnnotationSet
    source: Path | None = None

    def with_video(self, path: Path | str) -> Session:
        """The same session for a relocated video file (the recorded hash still applies)."""
        return replace(self, video=replace(self.video, path=_absolute(Path(path).expanduser(), None)))

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "video": self.video.to_document(),
            "tracking": self.tracking.to_document(),
            "calibration": self.calibration.to_document(),
            "timing": self.timing.to_document(),
            "organoids": self.annotations.to_documents(),
        }


def load_session(path: str | Path) -> Session:
    """Read and validate a session file (or a prompt record)."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SessionError(f"cannot read session file {path}: {error}") from error
    return session_from_document(data, base_dir=path.parent, source=path)


def session_from_document(data: Any, *, base_dir: Path | None = None, source: Path | None = None) -> Session:
    """Validate a parsed session document; relative video paths resolve against ``base_dir``."""
    where = str(source) if source is not None else "session"
    if not isinstance(data, Mapping):
        raise SessionError(f"{where}: expected a JSON object")
    schema = data.get("schema")
    if schema == PROMPT_RECORD_SCHEMA:
        return _from_prompt_record(data, base_dir, source)
    if schema != SCHEMA:
        raise SessionError(
            f"{where}: schema: expected {SCHEMA!r} (or a prompt record {PROMPT_RECORD_SCHEMA!r}), got {schema!r}"
        )
    known = {"schema", "video", "tracking", "calibration", "timing", "organoids"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise SessionError(f"{where}: unknown key(s) {unknown}; valid keys are {sorted(known)}")
    try:
        video = _video(_section(data, "video", where), base_dir, where)
        tracking = _tracking(data.get("tracking", {}), base_dir, where)
        calibration_block = _section(data, "calibration", where)
        _check_keys(calibration_block, ("um_per_pixel",), f"{where}: calibration", required=("um_per_pixel",))
        calibration = Calibration(calibration_block["um_per_pixel"])
        timing = _timing(_section(data, "timing", where), where)
        organoids = data.get("organoids")
        if not isinstance(organoids, list):
            raise SessionError(f"{where}: organoids: expected a list")
        annotations = AnnotationSet.from_documents(organoids)
    except AnnotationError as error:
        raise SessionError(f"{where}: organoids: {error}") from error
    return Session(video, tracking, calibration, timing, annotations, source=source)


def _section(data: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    section = data.get(key)
    if not isinstance(section, Mapping):
        raise SessionError(f"{where}: {key}: expected an object")
    return section


def _check_keys(section: Mapping[str, Any], allowed: Sequence[str], where: str, required: Sequence[str] = ()) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise SessionError(f"{where}: unknown key(s) {unknown}; valid keys are {list(allowed)}")
    missing = [key for key in required if key not in section]
    if missing:
        raise SessionError(f"{where}: missing key(s) {missing}")


def _video(section: Mapping[str, Any], base_dir: Path | None, where: str) -> VideoReference:
    _check_keys(section, ("path", "sha256"), f"{where}: video")
    raw = section.get("path")
    if not isinstance(raw, str) or not raw:
        raise SessionError(f"{where}: video.path: expected a non-empty string")
    sha256 = section.get("sha256")
    if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
        raise SessionError(f"{where}: video.sha256: expected a 64-character hex digest or null")
    return VideoReference(_absolute(Path(raw).expanduser(), base_dir), sha256.lower() if sha256 else None)


def _tracking(section: Any, base_dir: Path | None, where: str) -> TrackingSpec:
    if not isinstance(section, Mapping):
        raise SessionError(f"{where}: tracking: expected an object")
    _check_keys(
        section, ("direction", "model_config", "checkpoint_family", "checkpoint_path", "device"), f"{where}: tracking"
    )
    checkpoint = section.get("checkpoint_path")
    checkpoint_path = None
    if checkpoint is not None:
        if not isinstance(checkpoint, str):
            raise SessionError(f"{where}: tracking.checkpoint_path: expected a string or null")
        checkpoint_path = _absolute(Path(checkpoint).expanduser(), base_dir)
    for key in ("direction", "model_config", "checkpoint_family", "device"):
        value = section.get(key)
        if value is not None and not isinstance(value, str):
            raise SessionError(f"{where}: tracking.{key}: expected a string")
    try:
        return TrackingSpec(
            direction=section.get("direction", "reverse"),
            model_config=section.get("model_config", DEFAULT_MODEL_CONFIG),
            checkpoint_family=section.get("checkpoint_family"),
            checkpoint_path=checkpoint_path,
            device=section.get("device", DEFAULT_DEVICE),
        )
    except SessionError as error:
        raise SessionError(f"{where}: {error}") from error


def _timing(section: Mapping[str, Any], where: str) -> Timing:
    _check_keys(section, ("time_lapse_days", "frame_times_days"), f"{where}: timing")
    times = section.get("frame_times_days")
    if times is not None and not isinstance(times, (list, tuple)):
        raise SessionError(f"{where}: timing.frame_times_days: expected a list of numbers")
    try:
        return Timing(section.get("time_lapse_days"), tuple(times) if times is not None else None)
    except SessionError as error:
        raise SessionError(f"{where}: {error}") from error


def _from_prompt_record(data: Mapping[str, Any], base_dir: Path | None, source: Path | None) -> Session:
    """A session from the prompt record the application writes when tracking starts."""
    where = str(source) if source is not None else "prompt record"
    video = data.get("video")
    if not isinstance(video, Mapping) or not video.get("path"):
        raise SessionError(f"{where}: video.path: the prompt record names no video")
    model_block, tracking_raw, inputs_raw = data.get("model"), data.get("tracking"), data.get("analysis_inputs")
    model: Mapping[str, Any] = model_block if isinstance(model_block, Mapping) else {}
    tracking_block: Mapping[str, Any] = tracking_raw if isinstance(tracking_raw, Mapping) else {}
    inputs: Mapping[str, Any] = inputs_raw if isinstance(inputs_raw, Mapping) else {}
    document = {
        "schema": SCHEMA,
        "video": {"path": video["path"], "sha256": video.get("sha256")},
        "tracking": {
            "direction": tracking_block.get("direction", "reverse"),
            "model_config": model.get("model_config", DEFAULT_MODEL_CONFIG),
            "checkpoint_family": model.get("checkpoint_family"),
            "checkpoint_path": model.get("checkpoint_path"),
            "device": str(model.get("device", DEFAULT_DEVICE)).split(":")[0],
        },
        "calibration": {"um_per_pixel": inputs.get("conversion_factor_um_per_pixel")},
        "timing": (
            {"frame_times_days": inputs["frame_times_days"]}
            if inputs.get("frame_times_days") is not None
            else {"time_lapse_days": inputs.get("time_lapse_days")}
        ),
        "organoids": data.get("organoids", []),
    }
    try:
        return session_from_document(document, base_dir=base_dir, source=source)
    except SessionError as error:
        raise SessionError(f"{error} (from the prompt record's analysis_inputs, model and organoids)") from error
