"""The run manifest (schema ``organoidtracker.run/1``): what a headless run produced and from what.

It records the software, environment and settings, the validated session, the video facts,
the backend provenance, the tracking facts (status, frame coverage, errors), a digest of every
mask (area and sha256 of the packed bits, so a later run can be checked for identity without
storing masks), the analysis facts, and the sha256 of every file written.
"""

from __future__ import annotations

import hashlib
import platform
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .. import RESULTS_VERSION, __version__
from ..analysis.organoid_report_generator import Analysis
from ..core.tracking_result import TrackingResult
from ..paths import source_revision
from .session import Session
from .video_source import VideoSource

RUN_MANIFEST_SCHEMA = "organoidtracker.run/1"
RUN_MANIFEST_NAME = "run_manifest.json"


def environment_facts() -> dict[str, Any]:
    """Interpreter, platform and library versions, and the GPU when one is visible."""
    facts: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda"] = torch.version.cuda
        facts["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            facts["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        else:
            facts["gpu"] = None
    except Exception:  # pragma: no cover - torch is a hard dependency, but keep the manifest writable
        facts["torch"] = None
    for module_name, attribute in (
        ("cv2", "__version__"),
        ("numpy", "__version__"),
        ("matplotlib", "__version__"),
        ("pandas", "__version__"),
        ("reportlab", "Version"),
    ):
        try:
            module = __import__(module_name)
            facts["opencv" if module_name == "cv2" else module_name] = getattr(module, attribute, None)
        except Exception:
            facts["opencv" if module_name == "cv2" else module_name] = None
    return facts


def settings_snapshot() -> dict[str, Any]:
    """The effective settings (defaults plus the loaded file) under their UPPERCASE names."""
    from .. import config

    snapshot = {}
    for name, value in config.SETTINGS.as_constants().items():
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, dict):
            value = {k: list(v) if isinstance(v, tuple) else v for k, v in value.items()}
        snapshot[name] = value
    return snapshot


def mask_digest(mask: Any) -> dict[str, Any]:
    """Area, shape and sha256 of the packed bits (``numpy.packbits`` of the boolean mask, row-major)."""
    arr = mask.numpy() if hasattr(mask, "numpy") and not isinstance(mask, np.ndarray) else np.asarray(mask)
    if arr.ndim > 2:
        arr = arr.squeeze()
    mask_bool = arr.astype(bool, copy=False)
    return {
        "area": int(np.count_nonzero(mask_bool)),
        "shape": [int(mask_bool.shape[0]), int(mask_bool.shape[1])],
        "sha256": hashlib.sha256(np.packbits(mask_bool.ravel()).tobytes()).hexdigest(),
    }


def mask_digests(result: TrackingResult) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        str(frame): {str(obj): mask_digest(mask) for obj, mask in sorted(result[frame].items())}
        for frame in sorted(result)
    }


def tracking_facts(result: TrackingResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "frames_total": result.frames_total,
        "frames_done": result.frames_done,
        "frames_with_masks": len(result),
        "error": result.error,
        "direction": result.direction,
        "annotation_frame": result.annotation_frame,
        "frame_map": list(result.frame_map),
        "tracked_frames": list(result.tracked_frames),
        "object_ids": result.object_ids(),
        "presence": {
            str(frame): {str(obj): round(float(score), 4) for obj, score in sorted(scores.items())}
            for frame, scores in sorted(result.presence.items())
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root: Path, exclude: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    """``{relative posix path: {"sha256", "bytes"}}`` for every file under ``root``."""
    skipped = set(exclude)
    inventory = {}
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in skipped or any(relative.startswith(prefix) for prefix in skipped if prefix.endswith("/")):
            continue
        if relative.startswith("organoidtracker.log"):
            continue  # still being written
        inventory[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return inventory


def build_manifest(
    *,
    run_id: str,
    session: Session,
    video: VideoSource,
    provenance: dict[str, Any],
    result: TrackingResult,
    analysis: Analysis,
    video_export: dict[str, Any] | None,
    output_dir: Path,
    timings_s: dict[str, float],
) -> dict[str, Any]:
    experiment = analysis.experiment
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": result.status,
        "complete": analysis.complete,
        "results_version": RESULTS_VERSION,
        "software": {"organoidtracker": __version__, "source_revision": source_revision()},
        "environment": environment_facts(),
        "settings": settings_snapshot(),
        "session": session.to_document(),
        "video": video.to_document(),
        "provenance": provenance,
        "tracking": tracking_facts(result),
        "masks": mask_digests(result),
        "video_export": video_export,
        "analysis": {
            "complete": analysis.complete,
            "total_organoids": len(experiment.organoids),
            "organoids_without_cysts": [oid for oid, organoid in experiment.organoids.items() if not organoid.cysts],
            "total_cysts": len(experiment.get_all_cysts()),
            "untracked_cysts": list(analysis.untracked_cysts),
            "unannotated_objects": list(analysis.unannotated_objects),
            "time_lapse_days": experiment.time_lapse_days,
            "frame_timestamps": list(experiment.frame_timestamps),
            "observed_frames": experiment.frames_observed(),
            "conversion_um_per_pixel": experiment.conversion_factor_um_per_pixel,
            "warnings": list(analysis.validation.get("warnings", [])),
        },
        "files": file_inventory(output_dir, exclude=(RUN_MANIFEST_NAME,)),
        "timings_s": timings_s,
    }
