"""Rebuild an ``ExperimentData`` from the exported CSV and summary files.

The raw data table carries each observation's frame index and timestamp; ``analysis_summary.json``
carries the experiment parameters and, since results version 2, the tracked frames. Reloading
keeps the exported time axis (``Time_Days``) and the set of tracked frames, so re-plotting from
CSV gives the same growth rates and the same gaps as the original report.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .organoid_cyst_data import CystFrameData, CystTrajectory, ExperimentData, OrganoidData


def experiment_from_csv(raw_csv_path: str | Path, summary_json_path: str | Path | None = None) -> ExperimentData:
    """``ExperimentData`` from ``raw_cyst_data.csv`` and, when given, ``analysis_summary.json``."""
    with open(raw_csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{raw_csv_path}: no measurements")

    info: dict = {}
    tracking: dict = {}
    if summary_json_path is not None and Path(summary_json_path).is_file():
        summary = json.loads(Path(summary_json_path).read_text(encoding="utf-8"))
        info = summary.get("experiment_info", {}) or {}
        tracking = summary.get("tracking", {}) or {}

    frames = [int(row["Frame"]) for row in rows]
    times_by_frame = {int(row["Frame"]): float(row["Time_Days"]) for row in rows}
    total_frames = int(info.get("total_frames") or (max(frames) + 1))
    conversion = float(info.get("conversion_factor_um_per_pixel", 1.0))
    if "time_lapse_days" in info:
        time_lapse_days = float(info["time_lapse_days"])
    else:
        time_lapse_days = max(times_by_frame.values()) - min(times_by_frame.values())

    # Uniform axis as the report built it, then the exported timestamps where a frame was observed
    # (they coincide for reports written by this code; the CSV wins if they do not).
    experiment = ExperimentData(
        total_frames=total_frames, time_lapse_days=time_lapse_days, conversion_factor_um_per_pixel=conversion
    )
    for frame_idx, time_days in times_by_frame.items():
        if 0 <= frame_idx < total_frames:
            experiment.frame_timestamps[frame_idx] = time_days
    tracked = tracking.get("tracked_frames")
    experiment.observed_frames = sorted(int(f) for f in tracked) if tracked else None

    organoids: dict[int, OrganoidData] = {}
    cysts: dict[tuple[int, int], CystTrajectory] = {}
    for row in rows:
        organoid_id, cyst_id = int(row["Organoid_ID"]), int(row["Cyst_ID"])
        centroid = (float(row["Centroid_X"]), float(row["Centroid_Y"]))
        if organoid_id not in organoids:
            organoids[organoid_id] = OrganoidData(organoid_id=organoid_id, marker_point=centroid)
        key = (organoid_id, cyst_id)
        if key not in cysts:
            cysts[key] = CystTrajectory(cyst_id=cyst_id, organoid_id=organoid_id)
        cysts[key].add_frame_data(
            CystFrameData(
                frame_index=int(row["Frame"]),
                area_pixels=float(row["Area_um2"]) / (conversion**2),
                circularity=float(row["Circularity"]),
                centroid=centroid,
            )
        )
    for (organoid_id, _), trajectory in cysts.items():
        organoids[organoid_id].add_cyst(trajectory)
    for organoid in organoids.values():
        experiment.add_organoid(organoid)
    return experiment
