"""The analysis layer reports what was tracked: no invented cysts, no borrowed masks, the full time axis,
partial runs labelled as such, and growth rates per day (regression tests for findings B13 to B16)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from organoidtracker.analysis.organoid_analysis_engine import OrganoidAnalysisEngine
from organoidtracker.analysis.organoid_csv_exporter import OrganoidCSVExporter
from organoidtracker.analysis.organoid_cyst_data import CystFrameData, CystTrajectory, ExperimentData, OrganoidData
from organoidtracker.analysis.organoid_report_generator import OrganoidAnalysisReportGenerator
from organoidtracker.core.masks import PackedMask
from organoidtracker.core.tracking_result import TrackingResult

SIZE = 48


def disc(cx: int, r: int = 5) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    return (xx - cx) ** 2 + (yy - 24) ** 2 <= r * r


def results_for(frames, object_ids=(1,), status="completed", frames_total=None, **extra) -> TrackingResult:
    """One disc per object per frame; frame k puts object o at a distinct position."""
    data = {k: {o: PackedMask(disc(8 + 4 * k + 2 * o, r=4 + k % 3)) for o in object_ids} for k in frames}
    total = frames_total if frames_total is not None else max(frames) + 1
    return TrackingResult(data, status=status, frames_total=total, frames_done=len(frames), **extra)


def organoids(*cyst_ids):
    return {1: {"point": (5.0, 5.0), "cysts": [{"cyst_id": c, "bbox": (0, 0, 10, 10)} for c in cyst_ids]}}


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def run_report(tmp_path, results, organoid_data, time_lapse_days=6.0):
    return OrganoidAnalysisReportGenerator().generate_complete_analysis_report(
        tracking_results=results,
        organoid_data=organoid_data,
        time_lapse_days=time_lapse_days,
        conversion_factor=1.0,
        output_dir=str(tmp_path),
        debug_mode=False,
        original_frames=None,
    )


def test_one_annotated_cyst_gives_one_cyst(tmp_path):
    summary = run_report(tmp_path, results_for(range(8)), organoids(1))
    assert summary["success"] and summary["complete"]
    assert summary["experiment_info"]["total_cysts"] == 1
    rows = read_csv(summary["output_files"]["csv_files"]["cyst_summary"])
    assert [(r["Organoid_ID"], r["Cyst_ID"], r["Frames_Tracked"]) for r in rows] == [("1", "1", "8")]
    assert not any("missing" in w.lower() for w in summary["validation_results"]["warnings"])
    assert summary["tracking"]["object_ids"] == [1] and summary["tracking"]["status"] == "completed"
    assert json.loads(Path(tmp_path, "analysis_summary.json").read_text())["results_version"] >= 2


def test_a_missing_mask_is_not_borrowed_from_another_object_or_frame():
    results = results_for(range(4))  # object 1 only
    engine = OrganoidAnalysisEngine()
    assert engine._get_mask_for_cyst_frame(results, cyst_id=2, frame_idx=1) is None
    # object 1 present at frames 0..3 except frame 2: the trajectory simply lacks frame 2
    del results[2][1]
    trajectory = engine._extract_cyst_trajectory(1, 1, results, total_frames=4)
    assert sorted(trajectory.frame_data) == [0, 1, 3]


def test_unannotated_tracked_objects_and_untracked_cysts_are_reported_not_fixed(tmp_path):
    results = results_for(range(5), object_ids=(1, 2))  # two tracked objects
    summary = run_report(tmp_path, results, organoids(1, 3))  # cyst 3 annotated but never tracked
    assert summary["experiment_info"]["total_cysts"] == 1  # cyst 1 only; object 2 is ignored, cyst 3 has no data
    assert any("Annotated cysts without tracked masks: [3]" in w for w in summary["validation_results"]["warnings"])


def test_a_frame_without_masks_keeps_the_time_axis(tmp_path):
    results = results_for(range(1, 7), frames_total=7)  # frame 0 produced no accepted mask
    summary = run_report(tmp_path, results, organoids(1), time_lapse_days=6.0)
    assert summary["experiment_info"]["total_frames"] == 7
    assert summary["tracking"]["frames_with_masks"] == 6
    rows = read_csv(summary["output_files"]["csv_files"]["raw_data"])
    times = {int(r["Frame"]): float(r["Time_Days"]) for r in rows}
    assert times == {k: float(1 + k) for k in range(1, 7)}  # days 2..7 with one frame per day


def test_a_partial_run_is_labelled_partial(tmp_path):
    results = results_for(
        (4, 5, 6), frames_total=7, status="partial", error="RuntimeError: boom", direction="reverse", annotation_frame=6
    )
    summary = run_report(tmp_path, results, organoids(1))
    assert summary["success"] and not summary["complete"]
    assert summary["tracking"]["status"] == "partial" and summary["tracking"]["frames_done"] == 3
    assert summary["tracking"]["frames_total"] == 7 and summary["tracking"]["error"] == "RuntimeError: boom"
    assert summary["experiment_info"]["total_cysts"] == 1 and summary["experiment_info"]["total_frames"] == 7
    assert any(w.startswith("Tracking partial: 3 of 7 frames") for w in summary["validation_results"]["warnings"])
    rows = read_csv(summary["output_files"]["csv_files"]["raw_data"])
    assert sorted(int(r["Frame"]) for r in rows) == [4, 5, 6]


def test_plain_dict_results_count_frames_up_to_the_highest_index():
    generator = OrganoidAnalysisReportGenerator()
    plain = {k: {1: PackedMask(disc(10 + k))} for k in (1, 2, 5)}
    assert generator._determine_total_frames(plain) == 6
    with pytest.raises(ValueError, match="empty"):
        generator._determine_total_frames({})


def growth_experiment(time_lapse_days: float):
    experiment = ExperimentData(total_frames=2, time_lapse_days=time_lapse_days, conversion_factor_um_per_pixel=1.0)
    trajectory = CystTrajectory(cyst_id=1, organoid_id=1)
    for frame, area in ((0, 100.0), (1, 110.0)):
        trajectory.add_frame_data(CystFrameData(frame_index=frame, area_pixels=area, circularity=0.9, centroid=(0, 0)))
    organoid = OrganoidData(organoid_id=1, marker_point=(0.0, 0.0))
    organoid.add_cyst(trajectory)
    experiment.add_organoid(organoid)
    return experiment, trajectory


@pytest.mark.parametrize(("time_lapse_days", "expected_per_day"), [(1.0, 10.0), (2.0, 5.0), (0.5, 20.0)])
def test_growth_rate_is_area_change_per_day(tmp_path, time_lapse_days, expected_per_day):
    experiment, trajectory = growth_experiment(time_lapse_days)
    assert experiment.growth_rate_per_day(trajectory) == pytest.approx(expected_per_day)
    assert trajectory.get_mean_area_growth_rate(1.0) == pytest.approx(10.0)  # per frame, without a time axis
    path = OrganoidCSVExporter().export_summary_table(experiment, str(tmp_path / "cyst_summary.csv"))
    assert float(read_csv(path)[0]["Growth_Rate_um2_per_day"]) == pytest.approx(expected_per_day)
    summary = OrganoidAnalysisReportGenerator()._create_analysis_summary(experiment, {}, {}, {}, None)
    assert summary["growth_statistics"]["mean_growth_rate_um2_per_day"] == pytest.approx(expected_per_day)
