"""Independent numerical examples for the paper's measurements and their exported meaning."""

import csv
import json
import math

import numpy as np
import pytest
from matplotlib.axes import Axes

from organoidtracker.analysis.csv_import import experiment_from_csv
from organoidtracker.analysis.organoid_analysis_engine import OrganoidAnalysisEngine
from organoidtracker.analysis.organoid_csv_exporter import OrganoidCSVExporter
from organoidtracker.analysis.organoid_cyst_data import CystFrameData, CystTrajectory, ExperimentData, OrganoidData
from organoidtracker.analysis.organoid_report_generator import OrganoidAnalysisReportGenerator
from organoidtracker.analysis.organoid_visualizations import OrganoidVisualizationSuite, observed_series
from organoidtracker.core.masks import PackedMask
from organoidtracker.core.tracking_result import TrackingResult


def rectangle(width, height=10):
    mask = np.zeros((48, 48), dtype=bool)
    mask[5 : 5 + height, 5 : 5 + width] = True
    return mask


def measurement_experiment():
    # Unequal intervals distinguish the mean of interval slopes from the endpoint slope.
    result = TrackingResult(
        {2: {1: PackedMask(rectangle(18))}, 1: {1: PackedMask(rectangle(12))}, 0: {1: PackedMask(rectangle(10))}},
        status="completed",
        frames_total=3,
        frames_done=3,
    )
    annotations = {1: {"point": (1, 1), "cysts": [{"cyst_id": 1, "bbox": (5, 5, 23, 15)}]}}
    return (
        OrganoidAnalysisReportGenerator()
        .analyze(result, annotations, time_lapse_days=4, conversion_factor=1.5, frame_timestamps=[1, 2, 5])
        .experiment
    )


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_measurements(experiment, directory):
    exporter = OrganoidCSVExporter()
    raw = directory / "raw_cyst_data.csv"
    exporter.export_raw_data_table(experiment, str(raw))
    exporter.export_organoid_summary(experiment, str(directory / "organoid_summary.csv"))
    summary = OrganoidAnalysisReportGenerator()._create_analysis_summary(experiment, {}, {}, {}, None)
    (directory / "analysis_summary.json").write_text(json.dumps(summary))
    return raw


def test_area_and_interval_growth_have_independent_expected_values(tmp_path):
    experiment = measurement_experiment()
    cyst = experiment.get_all_cysts()[0]
    assert [cyst.get_area_at_frame(i, 1.5) for i in range(3)] == [225, 270, 405]
    # (45/1 + 135/3)/2 = 45 square micrometers per day.
    assert experiment.growth_rate_per_day(cyst) == pytest.approx(45)
    raw = write_measurements(experiment, tmp_path)
    assert [float(row["Area_um2"]) for row in read_rows(raw)] == [225, 270, 405]
    reloaded = experiment_from_csv(raw)
    assert reloaded.growth_rate_per_day(reloaded.get_all_cysts()[0]) == pytest.approx(45)


def test_irregular_growth_is_the_mean_of_interval_slopes():
    experiment = measurement_experiment()
    cyst = experiment.get_all_cysts()[0]
    cyst.frame_data[1].area_pixels = 140
    # Areas 225, 315, 405; slopes 90 and 30. The endpoint slope would be 45.
    assert experiment.growth_rate_per_day(cyst) == pytest.approx(60)
    assert experiment.growth_rate_per_day(cyst) != pytest.approx(45)


def test_circularity_keeps_the_documented_contour_estimator():
    engine = OrganoidAnalysisEngine(conversion_factor_um_per_pixel=2)
    measurement = engine._calculate_cyst_metrics_from_mask(rectangle(10, 6), 0)
    assert measurement.area_pixels == 60
    assert measurement.get_area_um2(2) == 240
    # Pixel-center contour: 9 by 5 rectangle, area 45 and perimeter 28.
    assert measurement.circularity == pytest.approx(4 * math.pi * 45 / 28**2)
    assert measurement.circularity != pytest.approx(4 * math.pi * 60 / 28**2)
    # The retained implementation uses the largest external contour for circularity,
    # while area counts all foreground pixels. This is an explicit limitation.
    disconnected = rectangle(10, 6)
    disconnected[30:34, 30:34] = True
    other = engine._calculate_cyst_metrics_from_mask(disconnected, 0)
    assert other.area_pixels == 76
    assert other.circularity == pytest.approx(measurement.circularity)


def test_population_denominator_includes_organoids_without_cysts():
    masks = {
        0: {1: PackedMask(rectangle(10))},
        1: {i: PackedMask(rectangle(10)) for i in (1, 2, 3)},
        2: {},
    }
    result = TrackingResult(masks, status="completed", frames_total=3, frames_done=3)
    annotations = {
        1: {"point": (1, 1), "cysts": [{"cyst_id": i, "bbox": (5, 5, 15, 15)} for i in (1, 2)]},
        2: {"point": (2, 2), "cysts": [{"cyst_id": 3, "bbox": (5, 5, 15, 15)}]},
        3: {"point": (3, 3), "cysts": []},
    }
    experiment = OrganoidAnalysisReportGenerator().analyze(result, annotations, 6, 1).experiment
    assert experiment.frame_timestamps == [1, 4, 7]
    _, formation = observed_series(experiment, experiment.get_percentage_organoids_with_cysts_at_frame)
    _, density = observed_series(experiment, experiment.get_cyst_to_organoid_ratio_at_frame)
    assert formation.tolist() == pytest.approx([100 / 3, 200 / 3, 0])
    assert density.tolist() == pytest.approx([1 / 3, 1, 0])
    experiment.observed_frames = [0, 1]
    _, density = observed_series(experiment, experiment.get_cyst_to_organoid_ratio_at_frame)
    assert math.isnan(density[2])


def test_raw_csv_preserves_small_intervals_and_calibrated_measurements(tmp_path):
    experiment = measurement_experiment()
    experiment.frame_timestamps = [1.0001234, 1.0002234, 1.0005234]
    experiment.time_lapse_days = 0.0004
    experiment.conversion_factor_um_per_pixel = 0.123456789
    expected = experiment.growth_rate_per_day(experiment.get_all_cysts()[0])
    raw = write_measurements(experiment, tmp_path)
    reloaded = experiment_from_csv(raw)
    assert reloaded.frame_timestamps == experiment.frame_timestamps
    assert reloaded.growth_rate_per_day(reloaded.get_all_cysts()[0]) == pytest.approx(expected, rel=1e-12)
    for row in read_rows(raw):
        index = int(row["Frame"])
        assert float(row["Area_um2"]) == pytest.approx([100, 120, 180][index] * 0.123456789**2, rel=1e-14)


def test_csv_row_order_does_not_reverse_initial_and_final_area(tmp_path):
    raw = write_measurements(measurement_experiment(), tmp_path)
    rows = read_rows(raw)
    with open(raw, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(reversed(rows))
    reloaded = experiment_from_csv(raw)
    out = tmp_path / "cyst_summary.csv"
    OrganoidCSVExporter().export_summary_table(reloaded, str(out))
    row = read_rows(out)[0]
    assert (int(row["First_Frame"]), int(row["Last_Frame"])) == (0, 2)
    assert (float(row["Initial_Area_um2"]), float(row["Final_Area_um2"])) == (225, 405)


def test_heatmap_order_and_percentile_classes_use_cyst_rates(tmp_path, monkeypatch):
    experiment = ExperimentData(2, 2, 1)
    organoid = OrganoidData(1, (0, 0))
    for index, rate in enumerate([-2, 0, 4, 8, 20], 1):
        cyst = CystTrajectory(index, 1)
        for frame, area in [(0, 100), (1, 100 + rate * 2)]:
            cyst.add_frame_data(CystFrameData(frame, area, 0.5, (0, 0)))
        organoid.add_cyst(cyst)
    experiment.add_organoid(organoid)
    calls = []
    original = Axes.barh

    def capture(axis, y, width, *args, **kwargs):
        calls.append((list(y), list(width), list(kwargs["color"])))
        return original(axis, y, width, *args, **kwargs)

    monkeypatch.setattr(Axes, "barh", capture)
    visualizer = OrganoidVisualizationSuite()
    path = tmp_path / "growth.png"
    assert visualizer.plot_lasagna_heatmap(experiment, str(path)) == str(path)
    # The 33rd and 67th percentile thresholds are 1.28 and 6.72; ties use <=.
    groups = visualizer.color_schemes["organoid_groups"]
    assert calls == [([4, 3, 2, 1, 0], [20, 8, 4, 0, -2], [groups[i] for i in (0, 0, 1, 2, 2)])]
    assert path.is_file()
