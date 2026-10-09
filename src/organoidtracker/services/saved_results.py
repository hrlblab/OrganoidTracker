"""Saved tracking results: the output half of the session format.

A run's complete experiment is stored as two files in its directory:

- ``results.json`` (schema ``organoidtracker.results/1``): the run id and time, the results
  version and software, the environment and the effective settings, the validated session
  (video, backend choices, calibration, timing, every organoid including those without cysts,
  every cyst box), the video facts (frame map, decoded and unique counts, dimensions,
  direction, annotation frame), the backend provenance, the tracking facts (status, frame
  coverage, error, tracked frames, object ids, presence scores) and an index of the masks
  (shape, area and sha256 of each);
- ``masks-<digest>.npz``: one array per (chronological frame, object id) holding the mask as
  row-major packed bits, exactly the tracker's ``PackedMask`` representation. The file is
  named after the sha256 of its content, which ``results.json`` records with the size.

Reloading rebuilds the ``TrackingResult`` the tracker returned, bit for bit, with the
``Session`` and the ``VideoSource`` of the run, without a model. Everything is checked: the
schema and keys, the types, the mask file's size and hash, every mask's length, hash and area,
and the consistency of the bookkeeping (frames with masks are tracked frames, tracked frames
lie in the video, a completed run tracked every frame). A damaged, truncated, edited or
inconsistent file is refused with the reason; nothing is read partially.

Writing is safe against interruption: the mask file is written under a temporary name and
renamed, ``results.json`` is replaced atomically last, and because mask files are named after
their content a new save never overwrites the mask file of the previous result. A save that
fails leaves the previous pair intact; a successful one removes the mask files it no longer
references.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import RESULTS_VERSION, __version__
from ..core.masks import PackedMask
from ..core.tracking_result import TrackingResult
from ..paths import source_revision
from .session import DIRECTIONS, Session, SessionError, session_from_document
from .video_source import VideoSource

RESULTS_SCHEMA = "organoidtracker.results/1"
RESULTS_NAME = "results.json"
MASKS_PREFIX = "masks-"
MASKS_SUFFIX = ".npz"
MASKS_GLOB = f"{MASKS_PREFIX}*{MASKS_SUFFIX}"
MASKS_FORMAT = "packed-bits"  # numpy.packbits of the boolean mask, row-major, one array per (frame, object)
STATUSES = (TrackingResult.COMPLETED, TrackingResult.PARTIAL)
TOP_LEVEL_KEYS = (
    "schema",
    "run_id",
    "created",
    "results_version",
    "software",
    "environment",
    "settings",
    "session",
    "video",
    "provenance",
    "tracking",
    "masks",
)
VIDEO_KEYS = (
    "path",
    "sha256",
    "decoded_frames",
    "unique_frames",
    "frame_map",
    "dimensions",
    "nominal_fps",
    "direction",
    "annotation_frame",
)
TRACKING_KEYS = (
    "status",
    "frames_total",
    "frames_done",
    "error",
    "direction",
    "annotation_frame",
    "frame_map",
    "tracked_frames",
    "object_ids",
    "presence",
)
MASKS_KEYS = ("file", "sha256", "bytes", "format", "index")
MASK_ENTRY_KEYS = ("key", "shape", "area", "sha256")


class SavedResultError(ValueError):
    """A saved result that cannot be used: unreadable, unknown key, wrong type, damaged or inconsistent."""


@dataclass(frozen=True)
class SavedResult:
    """A reloaded run: its result, session and video facts, and where it came from."""

    path: Path  # results.json
    masks_path: Path
    run_id: str
    created: str
    results_version: int
    software: dict[str, Any]
    environment: dict[str, Any]
    settings: dict[str, Any]
    session: Session
    video: VideoSource
    provenance: dict[str, Any]
    result: TrackingResult

    @property
    def directory(self) -> Path:
        return self.path.parent

    def to_manifest_block(self) -> dict[str, Any]:
        """What a run manifest records about the saved result it sits next to."""
        return {
            "file": self.path.name,
            "masks_file": self.masks_path.name,
            "run_id": self.run_id,
            "created": self.created,
            "results_version": self.results_version,
        }


# ---------------------------------------------------------------------------------- writing
def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _packed(mask: Any) -> PackedMask:
    return mask if isinstance(mask, PackedMask) else PackedMask(np.asarray(mask))


def tracking_document(result: TrackingResult) -> dict[str, Any]:
    """The tracking facts of a result at full precision (the manifest rounds the presence scores)."""
    return {
        "status": result.status,
        "frames_total": int(result.frames_total),
        "frames_done": int(result.frames_done),
        "error": result.error,
        "direction": result.direction,
        "annotation_frame": None if result.annotation_frame is None else int(result.annotation_frame),
        "frame_map": [int(index) for index in result.frame_map],
        "tracked_frames": [int(index) for index in result.tracked_frames],
        "object_ids": [int(obj) for obj in result.object_ids()],
        "presence": {
            str(int(frame)): {str(int(obj)): float(score) for obj, score in sorted(scores.items())}
            for frame, scores in sorted(result.presence.items())
        },
    }


@dataclass(frozen=True)
class StagedResult:
    """A saved result written under temporary names, invisible to readers until it is published."""

    directory: Path
    masks_temp: Path
    masks_path: Path  # the content-named final file
    results_temp: Path
    document: dict[str, Any]
    run_id: str
    session: Session
    video: VideoSource
    result: TrackingResult


def stage_saved_result(
    directory: Path | str,
    *,
    run_id: str,
    session: Session,
    video: VideoSource,
    provenance: Mapping[str, Any],
    result: TrackingResult,
    settings: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> StagedResult:
    """Write the mask file and ``results.json`` under temporary names; nothing readable changes yet."""
    directory = Path(directory)
    if result.status not in STATUSES:
        raise SavedResultError(f"a result with status {result.status!r} cannot be saved")
    if not result:
        raise SavedResultError("a result without any mask cannot be saved")
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SavedResultError(f"cannot create {directory}: {error}") from error

    arrays: dict[str, np.ndarray] = {}
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for frame in sorted(result):
        for obj, mask in sorted(result[frame].items()):
            packed = _packed(mask)
            key = f"f{int(frame)}_o{int(obj)}"
            arrays[key] = packed.packed_bits
            index.setdefault(str(int(frame)), {})[str(int(obj))] = {
                "key": key,
                "shape": [packed.shape[0], packed.shape[1]],
                "area": packed.area,
                "sha256": hashlib.sha256(packed.packed_bits.tobytes()).hexdigest(),
            }

    if settings is None or environment is None:
        from .run_manifest import environment_facts, settings_snapshot

        settings = settings_snapshot() if settings is None else settings
        environment = environment_facts() if environment is None else environment

    token = uuid.uuid4().hex
    masks_temp = directory / f".{MASKS_PREFIX}{token}.tmp"
    results_temp = directory / f".{RESULTS_NAME}.{token}.tmp"
    try:
        with open(masks_temp, "wb") as handle:
            _write_npz(handle, arrays)
        digest = sha256_file(masks_temp)
        masks_path = directory / f"{MASKS_PREFIX}{digest[:16]}{MASKS_SUFFIX}"
        document: dict[str, Any] = {
            "schema": RESULTS_SCHEMA,
            "run_id": run_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results_version": RESULTS_VERSION,
            "software": {"organoidtracker": __version__, "source_revision": source_revision()},
            "environment": dict(environment),
            "settings": dict(settings),
            "session": _session_document(session, video),
            "video": video.to_document(),
            "provenance": dict(provenance),
            "tracking": tracking_document(result),
            "masks": {
                "file": masks_path.name,
                "sha256": digest,
                "bytes": masks_temp.stat().st_size,
                "format": MASKS_FORMAT,
                "index": index,
            },
        }
        _write_text(results_temp, json.dumps(document, indent=2, default=str))
    except OSError as error:
        masks_temp.unlink(missing_ok=True)
        results_temp.unlink(missing_ok=True)
        raise SavedResultError(f"cannot write the saved result in {directory}: {error}") from error
    return StagedResult(directory, masks_temp, masks_path, results_temp, document, run_id, session, video, result)


def publish_files(directory: Path, replacements: Mapping[str, Path | None]) -> None:
    """Put staged files in place as one step: every listed name gets its new content, or every one keeps its old.

    ``replacements`` maps a file name to its staged file, or to None for a file that must go (a previous
    manifest, which must never describe the new files). The previous files are set aside first, in the
    given order, then the staged files are renamed in. When any rename fails, the files already placed
    are removed and the previous ones put back, so the directory never holds a mix of two runs. Renames
    within one directory are atomic; should a restore fail as well, the previous file stays beside under
    its ``.previous`` name, which the error says.
    """
    token = uuid.uuid4().hex
    set_aside: list[tuple[Path, Path]] = []
    placed: list[Path] = []
    try:
        for name, source in replacements.items():
            target = directory / name
            if target.is_file():
                backup = directory / f".{name}.{token}.previous"
                os.replace(target, backup)
                set_aside.append((target, backup))
            if source is not None:
                os.replace(source, target)
                placed.append(target)
    except OSError as error:
        left_aside = []
        for target in reversed(placed):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        for target, backup in reversed(set_aside):
            try:
                os.replace(backup, target)
            except OSError:
                left_aside.append(backup.name)
        if left_aside:
            raise OSError(f"{error} (the previous files are kept beside as {', '.join(left_aside)})") from error
        raise
    for _target, backup in set_aside:
        backup.unlink(missing_ok=True)


def remove_stale_mask_files(directory: Path, keep: str) -> None:
    """Drop the mask files a directory's ``results.json`` no longer names (best effort)."""
    for stale in directory.glob(MASKS_GLOB):
        if stale.name != keep:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass


def saved_result_from_staged(staged: StagedResult) -> SavedResult:
    """The ``SavedResult`` a published stage describes."""
    document = staged.document
    return SavedResult(
        path=staged.directory / RESULTS_NAME,
        masks_path=staged.masks_path,
        run_id=staged.run_id,
        created=document["created"],
        results_version=RESULTS_VERSION,
        software=document["software"],
        environment=document["environment"],
        settings=document["settings"],
        session=staged.session,
        video=staged.video,
        provenance=document["provenance"],
        result=staged.result,
    )


def publish_staged_result(staged: StagedResult) -> SavedResult:
    """Make a staged result the directory's saved result: both files in one step, the previous pair restored
    on failure (see :func:`publish_files`); mask files the new ``results.json`` does not name go afterwards."""
    try:
        publish_files(staged.directory, {staged.masks_path.name: staged.masks_temp, RESULTS_NAME: staged.results_temp})
    except OSError as error:
        discard_staged_result(staged)
        raise SavedResultError(f"cannot publish the saved result in {staged.directory}: {error}") from error
    remove_stale_mask_files(staged.directory, keep=staged.masks_path.name)
    return saved_result_from_staged(staged)


def discard_staged_result(staged: StagedResult) -> None:
    """Remove a staged result that will not be published."""
    staged.masks_temp.unlink(missing_ok=True)
    staged.results_temp.unlink(missing_ok=True)


def write_saved_result(
    directory: Path | str,
    *,
    run_id: str,
    session: Session,
    video: VideoSource,
    provenance: Mapping[str, Any],
    result: TrackingResult,
    settings: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> SavedResult:
    """Write ``results.json`` and the mask file into ``directory`` (staged, then published)."""
    return publish_staged_result(
        stage_saved_result(
            directory,
            run_id=run_id,
            session=session,
            video=video,
            provenance=provenance,
            result=result,
            settings=settings,
            environment=environment,
        )
    )


def relocated_document(results_path: Path | str, video_path: Path | str) -> dict[str, Any]:
    """The results document with its video locators pointing at ``video_path``.

    The session's and the video block's paths say where the file is; the provenance keeps the path the
    tracker read, so a relocated copy still records where the masks came from.
    """
    try:
        data = json.loads(Path(results_path).read_text(encoding="utf-8"))
        absolute = os.path.abspath(video_path)
        data["session"]["video"]["path"] = absolute
        data["video"]["path"] = absolute
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SavedResultError(f"cannot relocate the video of {results_path}: {error}") from error
    return data


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_npz(handle: Any, arrays: Mapping[str, np.ndarray]) -> None:
    """A compressed NPZ archive (what ``numpy.savez_compressed`` writes) with fixed entry timestamps.

    ``numpy.load`` reads it like any NPZ; the fixed timestamps make the bytes, and so the file
    name derived from them, depend on the masks alone.
    """
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, array in arrays.items():
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as entry:
                np.lib.format.write_array(entry, np.ascontiguousarray(array), allow_pickle=False)


def _session_document(session: Session, video: VideoSource) -> dict[str, Any]:
    document = session.to_document()
    document["video"]["path"] = os.path.abspath(session.video.path)  # holds from any directory
    if session.tracking.checkpoint_path is not None:
        document["tracking"]["checkpoint_path"] = os.path.abspath(session.tracking.checkpoint_path)
    if not document["video"].get("sha256") and video.sha256:
        document["video"]["sha256"] = video.sha256
    return document


def _referenced_mask_files(directory: Path) -> set[str]:
    """The mask file the directory's ``results.json`` names, if it can be read."""
    try:
        data = json.loads((directory / RESULTS_NAME).read_text(encoding="utf-8"))
        name = data["masks"]["file"]
        return {name} if isinstance(name, str) else set()
    except (OSError, ValueError, KeyError, TypeError):
        return set()


# ---------------------------------------------------------------------------------- reading
def load_saved_result(path: Path | str) -> SavedResult:
    """Read and check a saved result (``results.json`` or the directory holding it)."""
    path = Path(path)
    if path.is_dir():
        path = path / RESULTS_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SavedResultError(f"cannot read saved result {path}: {error}") from error
    where = str(path)
    if not isinstance(data, Mapping):
        raise SavedResultError(f"{where}: expected a JSON object")
    schema = data.get("schema")
    if schema != RESULTS_SCHEMA:
        raise SavedResultError(f"{where}: schema: expected {RESULTS_SCHEMA!r}, got {schema!r}")
    _check_keys(data, TOP_LEVEL_KEYS, where, required=TOP_LEVEL_KEYS)

    run_id = _string(data["run_id"], f"{where}: run_id")
    created = _string(data["created"], f"{where}: created")
    results_version = _int(data["results_version"], f"{where}: results_version", minimum=0)
    blocks = {
        name: _mapping(data[name], f"{where}: {name}") for name in ("software", "environment", "settings", "provenance")
    }
    try:
        session = session_from_document(data["session"], base_dir=path.parent)
    except SessionError as error:
        raise SavedResultError(f"{where}: session: {error}") from error
    video = _video_source(data["video"], f"{where}: video")
    tracking = _tracking(data["tracking"], f"{where}: tracking", video)
    masks_block = _mapping(data["masks"], f"{where}: masks")
    _check_keys(masks_block, MASKS_KEYS, f"{where}: masks", required=MASKS_KEYS)
    if masks_block["format"] != MASKS_FORMAT:
        raise SavedResultError(f"{where}: masks.format: expected {MASKS_FORMAT!r}, got {masks_block['format']!r}")
    masks_path, masks = _load_masks(path.parent, masks_block, tracking, video, f"{where}: masks")

    try:
        session.timing.resolve(video.n_frames)
    except SessionError as error:
        raise SavedResultError(f"{where}: session: {error}") from error
    if session.video.sha256 and video.sha256 and session.video.sha256 != video.sha256:
        raise SavedResultError(f"{where}: the session and the video block record different video hashes")

    result = TrackingResult(
        masks,
        status=tracking["status"],
        frames_total=tracking["frames_total"],
        frames_done=tracking["frames_done"],
        error=tracking["error"],
        direction=tracking["direction"],
        annotation_frame=tracking["annotation_frame"],
        frame_map=tracking["frame_map"],
        presence=tracking["presence"],
        tracked_frames=tracking["tracked_frames"],
    )
    return SavedResult(
        path=path,
        masks_path=masks_path,
        run_id=run_id,
        created=created,
        results_version=results_version,
        software=blocks["software"],
        environment=blocks["environment"],
        settings=blocks["settings"],
        session=session,
        video=video,
        provenance=blocks["provenance"],
        result=result,
    )


def _check_keys(
    section: Mapping[str, Any], allowed: tuple[str, ...], where: str, required: tuple[str, ...] = ()
) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise SavedResultError(f"{where}: unknown key(s) {unknown}; valid keys are {list(allowed)}")
    missing = [key for key in required if key not in section]
    if missing:
        raise SavedResultError(f"{where}: missing key(s) {missing}")


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SavedResultError(f"{where}: expected an object")
    return dict(value)


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise SavedResultError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _int(value: Any, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SavedResultError(f"{where}: expected an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise SavedResultError(f"{where}: expected an integer >= {minimum}, got {value}")
    return value


def _int_list(value: Any, where: str, minimum: int = 0) -> list[int]:
    if not isinstance(value, list):
        raise SavedResultError(f"{where}: expected a list of integers")
    return [_int(item, f"{where}[{index}]", minimum) for index, item in enumerate(value)]


def _video_source(value: Any, where: str) -> VideoSource:
    block = _mapping(value, where)
    _check_keys(block, VIDEO_KEYS, where, required=VIDEO_KEYS)
    path = _string(block["path"], f"{where}.path")
    sha256 = block["sha256"]
    if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
        raise SavedResultError(f"{where}.sha256: expected a 64-character hex digest or null")
    decoded = _int(block["decoded_frames"], f"{where}.decoded_frames", minimum=1)
    unique = _int(block["unique_frames"], f"{where}.unique_frames", minimum=1)
    frame_map = _int_list(block["frame_map"], f"{where}.frame_map")
    if len(frame_map) != unique:
        raise SavedResultError(f"{where}.frame_map: {len(frame_map)} entries for {unique} unique frames")
    if any(b <= a for a, b in zip(frame_map, frame_map[1:])) or (frame_map and frame_map[-1] >= decoded):
        raise SavedResultError(f"{where}.frame_map: must increase strictly within {decoded} decoded frames")
    dimensions = _int_list(block["dimensions"], f"{where}.dimensions", minimum=1)
    if len(dimensions) != 2:
        raise SavedResultError(f"{where}.dimensions: expected [height, width]")
    fps = block["nominal_fps"]
    if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps)):
        raise SavedResultError(f"{where}.nominal_fps: expected a number or null, got {fps!r}")
    direction = block["direction"]
    if direction not in DIRECTIONS:
        raise SavedResultError(f"{where}.direction: expected one of {list(DIRECTIONS)}, got {direction!r}")
    annotation_frame = _int(block["annotation_frame"], f"{where}.annotation_frame", minimum=0)
    if annotation_frame >= unique:
        raise SavedResultError(f"{where}.annotation_frame: {annotation_frame} is not one of {unique} unique frames")
    return VideoSource(
        path=Path(path),
        sha256=sha256.lower() if sha256 else None,
        decoded_frames=decoded,
        frame_map=tuple(frame_map),
        height=dimensions[0],
        width=dimensions[1],
        nominal_fps=None if fps is None else float(fps),
        direction=direction,
        annotation_frame=annotation_frame,
    )


def _tracking(value: Any, where: str, video: VideoSource) -> dict[str, Any]:
    block = _mapping(value, where)
    _check_keys(block, TRACKING_KEYS, where, required=TRACKING_KEYS)
    status = block["status"]
    if status not in STATUSES:
        raise SavedResultError(f"{where}.status: expected one of {list(STATUSES)}, got {status!r}")
    frames_total = _int(block["frames_total"], f"{where}.frames_total", minimum=1)
    frames_done = _int(block["frames_done"], f"{where}.frames_done", minimum=0)
    error = block["error"]
    if error is not None and not isinstance(error, str):
        raise SavedResultError(f"{where}.error: expected a string or null")
    direction = block["direction"]
    if direction not in DIRECTIONS:
        raise SavedResultError(f"{where}.direction: expected one of {list(DIRECTIONS)}, got {direction!r}")
    annotation_frame = block["annotation_frame"]
    if annotation_frame is not None:
        annotation_frame = _int(annotation_frame, f"{where}.annotation_frame", minimum=0)
    frame_map = _int_list(block["frame_map"], f"{where}.frame_map")
    tracked_frames = _int_list(block["tracked_frames"], f"{where}.tracked_frames")
    object_ids = _int_list(block["object_ids"], f"{where}.object_ids", minimum=1)
    presence_block = _mapping(block["presence"], f"{where}.presence")

    if frames_total != video.n_frames or frame_map != list(video.frame_map):
        raise SavedResultError(f"{where}: the frame count or frame map differs from the video block")
    if direction != video.direction or (annotation_frame is not None and annotation_frame != video.annotation_frame):
        raise SavedResultError(f"{where}: the direction or annotation frame differs from the video block")
    if len(set(tracked_frames)) != len(tracked_frames) or any(f >= frames_total for f in tracked_frames):
        raise SavedResultError(f"{where}.tracked_frames: expected distinct frames below {frames_total}")
    if frames_done != len(tracked_frames):
        raise SavedResultError(
            f"{where}: frames_done is {frames_done} but {len(tracked_frames)} frames are listed as tracked"
        )
    if status == TrackingResult.COMPLETED and frames_done != frames_total:
        raise SavedResultError(
            f"{where}: a completed run must have tracked every frame ({frames_done} of {frames_total})"
        )
    if status == TrackingResult.PARTIAL and frames_done >= frames_total:
        raise SavedResultError(
            f"{where}: a partial run cannot have tracked every frame ({frames_done} of {frames_total})"
        )
    presence: dict[int, dict[int, float]] = {}
    for frame_key, scores in presence_block.items():
        frame = _key(frame_key, f"{where}.presence")
        if frame >= frames_total:
            raise SavedResultError(f"{where}.presence: frame {frame} is not one of {frames_total} frames")
        scores_block = _mapping(scores, f"{where}.presence[{frame_key}]")
        presence[frame] = {}
        for obj_key, score in scores_block.items():
            obj = _key(obj_key, f"{where}.presence[{frame_key}]")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise SavedResultError(f"{where}.presence[{frame_key}][{obj_key}]: expected a number, got {score!r}")
            presence[frame][obj] = float(score)
    return {
        "status": status,
        "frames_total": frames_total,
        "frames_done": frames_done,
        "error": error,
        "direction": direction,
        "annotation_frame": annotation_frame,
        "frame_map": frame_map,
        "tracked_frames": tracked_frames,
        "object_ids": object_ids,
        "presence": presence,
    }


def _key(value: Any, where: str) -> int:
    if not isinstance(value, str) or not value.isdigit():
        raise SavedResultError(f"{where}: expected an integer key, got {value!r}")
    return int(value)


def _load_masks(
    directory: Path, block: Mapping[str, Any], tracking: Mapping[str, Any], video: VideoSource, where: str
) -> tuple[Path, dict[int, dict[int, PackedMask]]]:
    file_name = _string(block["file"], f"{where}.file")
    if Path(file_name).name != file_name:
        raise SavedResultError(f"{where}.file: expected a plain file name, got {file_name!r}")
    masks_path = directory / file_name
    if not masks_path.is_file():
        raise SavedResultError(f"{where}: the mask file {file_name} is missing from {directory}")
    expected_bytes = _int(block["bytes"], f"{where}.bytes", minimum=0)
    size = masks_path.stat().st_size
    if size != expected_bytes:
        raise SavedResultError(
            f"{where}: {file_name} has {size} bytes, {expected_bytes} were recorded; the file is damaged or truncated"
        )
    expected_digest = _string(block["sha256"], f"{where}.sha256")
    digest = sha256_file(masks_path)
    if digest != expected_digest:
        raise SavedResultError(f"{where}: {file_name} does not match its recorded sha256; the file is damaged")
    index = _mapping(block["index"], f"{where}.index")

    masks: dict[int, dict[int, PackedMask]] = {}
    used: set[str] = set()
    tracked = set(tracking["tracked_frames"])
    try:
        archive = np.load(masks_path)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SavedResultError(f"{where}: cannot read {file_name}: {error}") from error
    with archive:
        listed = set(archive.files)
        for frame_key, objects in index.items():
            frame = _key(frame_key, f"{where}.index")
            if frame >= tracking["frames_total"]:
                raise SavedResultError(f"{where}.index: frame {frame} is not one of {tracking['frames_total']} frames")
            if frame not in tracked:
                raise SavedResultError(f"{where}.index: frame {frame} holds masks but is not a tracked frame")
            entries = _mapping(objects, f"{where}.index[{frame_key}]")
            if not entries:
                raise SavedResultError(f"{where}.index[{frame_key}]: a frame with masks lists at least one object")
            for obj_key, raw_entry in entries.items():
                obj = _key(obj_key, f"{where}.index[{frame_key}]")
                here = f"{where}.index[{frame_key}][{obj_key}]"
                entry = _mapping(raw_entry, here)
                _check_keys(entry, MASK_ENTRY_KEYS, here, required=MASK_ENTRY_KEYS)
                key = _string(entry["key"], f"{here}.key")
                shape = _int_list(entry["shape"], f"{here}.shape", minimum=1)
                if len(shape) != 2 or (shape[0], shape[1]) != (video.height, video.width):
                    raise SavedResultError(f"{here}.shape: expected [{video.height}, {video.width}], got {shape}")
                area = _int(entry["area"], f"{here}.area", minimum=0)
                mask_digest = _string(entry["sha256"], f"{here}.sha256")
                if key not in listed:
                    raise SavedResultError(
                        f"{where}: mask {key} (frame {frame}, object {obj}) is missing from {file_name}"
                    )
                bits = archive[key]
                if bits.dtype != np.uint8 or bits.ndim != 1:
                    raise SavedResultError(f"{where}: mask {key} is not a packed-bit array")
                if hashlib.sha256(bits.tobytes()).hexdigest() != mask_digest:
                    raise SavedResultError(
                        f"{where}: mask {key} (frame {frame}, object {obj}) does not match its recorded hash"
                    )
                try:
                    mask = PackedMask.from_packed_bits(bits, shape)
                except ValueError as error:
                    raise SavedResultError(f"{where}: mask {key}: {error}") from error
                if mask.area != area:
                    raise SavedResultError(f"{where}: mask {key} has area {mask.area}, {area} was recorded")
                masks.setdefault(frame, {})[obj] = mask
                used.add(key)
        extra = sorted(listed - used)
        if extra:
            raise SavedResultError(
                f"{where}: {file_name} holds {len(extra)} mask(s) the index does not list: {extra[:5]}"
            )
    found_objects = sorted({obj for objects in masks.values() for obj in objects})
    if found_objects != tracking["object_ids"]:
        raise SavedResultError(
            f"{where}: tracking.object_ids {tracking['object_ids']} differs from the masks' objects {found_objects}"
        )
    if not masks:
        raise SavedResultError(f"{where}: the result holds no mask")
    return masks_path, masks
