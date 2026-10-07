"""Re-plotting from the exported CSV keeps the time axis, the tracked frames and the growth rates."""

import json

import pytest

from organoidtracker.analysis.csv_import import experiment_from_csv
from organoidtracker.analysis.organoid_csv_exporter import OrganoidCSVExporter
from test_analysis_bookkeeping import growth_experiment, organoids, results_for, run_report


def test_growth_rate_survives_the_csv_round_trip(tmp_path):
    experiment, trajectory = growth_experiment(time_lapse_days=2.0)
    assert experiment.growth_rate_per_day(trajectory) == pytest.approx(5.0)
    raw = OrganoidCSVExporter().export_raw_data_table(experiment, str(tmp_path / "raw_cyst_data.csv"))
    summary = tmp_path / "analysis_summary.json"
    summary.write_text(
        json.dumps(
            {"experiment_info": {"total_frames": 2, "time_lapse_days": 2.0, "conversion_factor_um_per_pixel": 1.0}}
        )
    )
    reloaded = experiment_from_csv(raw, summary)
    assert reloaded.frame_timestamps == [1.0, 3.0]
    cyst = reloaded.get_all_cysts()[0]
    assert reloaded.growth_rate_per_day(cyst) == pytest.approx(5.0)
    assert cyst.get_area_at_frame(1, 1.0) == pytest.approx(110.0)


def test_partial_run_report_round_trips_with_its_gaps(tmp_path):
    results = results_for((4, 5, 6), frames_total=7, status="partial", tracked_frames=[4, 5, 6])
    summary = run_report(tmp_path, results, organoids(1), time_lapse_days=6.0)
    reloaded = experiment_from_csv(summary["output_files"]["csv_files"]["raw_data"], tmp_path / "analysis_summary.json")
    assert reloaded.total_frames == 7 and reloaded.observed_frames == [4, 5, 6]
    assert reloaded.frame_timestamps == [float(1 + k) for k in range(7)]
    assert sorted(reloaded.get_all_cysts()[0].frame_data) == [4, 5, 6]
