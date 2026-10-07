"""The prompt record written when tracking starts (schema ``organoidtracker.prompts/1``).

It holds the prompts, the organoid associations and the backend's provenance, so that a run
can be reproduced or repeated headlessly (``organoidtracker run --session <record>``). The Tk
application and the command line write the same record through this module.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import RESULTS_VERSION

PROMPT_RECORD_SCHEMA = "organoidtracker.prompts/1"
DEFAULT_PROMPT_RECORD_DIR = Path("data") / "output_videos" / "prompts"


def build_prompt_record(
    model: Any,
    video_path: str | Path,
    organoid_data: dict[int, dict[str, Any]],
    time_lapse_days: float | None,
    conversion_factor: float | None,
    frame_times_days: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The record for a loaded backend (``model``) with prompts and the GUI-shaped ``organoid_data``.

    ``frame_times_days`` carries explicit per-frame times when the run used them; replaying the
    record then reproduces the same time axis, not a uniform one over the same span.
    """
    frames = getattr(model, "video_frames", None)
    describe = getattr(model, "provenance", None)
    return {
        "results_version": RESULTS_VERSION,
        "schema": PROMPT_RECORD_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video": {
            "path": str(video_path),
            "sha256": getattr(model, "video_sha256", None),
            "decoded_frames": getattr(model, "decoded_frame_count", None),
            "unique_frames": len(frames) if frames else 0,
            "frame_map": list(getattr(model, "frame_map", [])),
        },
        "tracking": {
            "direction": "reverse" if getattr(model, "enable_reverse_tracking", True) else "forward",
            "annotation_frame_index": getattr(model, "annotation_frame_index", 0),
        },
        "model": describe() if callable(describe) else {"name": getattr(model, "model_name", "?")},
        "organoids": [
            {
                "organoid_id": organoid_id,
                "point": list(info["point"]),
                "cysts": [{"cyst_id": c["cyst_id"], "bbox": list(c["bbox"])} for c in info["cysts"]],
            }
            for organoid_id, info in organoid_data.items()
        ],
        "prompts": {str(obj_id): prompts for obj_id, prompts in getattr(model, "prompts", {}).items()},
        "analysis_inputs": {
            "time_lapse_days": time_lapse_days,
            "conversion_factor_um_per_pixel": conversion_factor,
            "frame_times_days": [float(t) for t in frame_times_days] if frame_times_days is not None else None,
        },
    }


def default_prompt_record_path(video_path: str | Path, directory: Path = DEFAULT_PROMPT_RECORD_DIR) -> Path:
    """``data/output_videos/prompts/<video stem>_<timestamp>.json``, the application's convention."""
    return Path(directory) / f"{Path(str(video_path)).stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"


def write_prompt_record(record: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
