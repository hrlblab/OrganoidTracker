"""Report readers can identify missing observations, units and the cyst represented by a bar."""

import numpy as np
import pytest

from organoidtracker.analysis import organoid_visualizations as plots
from test_analysis_bookkeeping import organoids, results_for, run_report
from test_analysis_fidelity import measurement_experiment


def capture_figures(monkeypatch):
    figures = []

    def capture(self, path, **kwargs):
        figure = plots.plt.gcf()
        figure.canvas.draw()
        figures.append(figure)

    monkeypatch.setattr(plots.OrganoidVisualizationSuite, "_save_figure", capture)
    return figures


def test_heatmap_and_growth_bar_rows_align_in_rendered_coordinates(monkeypatch, tmp_path):
    from copy import deepcopy

    experiment = measurement_experiment()
    cyst = experiment.get_all_cysts()[0]
    for identifier in range(2, 6):
        other = deepcopy(cyst)
        other.cyst_id = identifier
        for frame in other.frame_data.values():
            frame.area_pixels *= identifier
        experiment.organoids[1].add_cyst(other)
    figures = capture_figures(monkeypatch)
    assert plots.OrganoidVisualizationSuite(dpi=60).plot_lasagna_heatmap(experiment, str(tmp_path / "heatmap.png"))
    main, bars = figures[0].axes[:2]
    assert len(bars.patches) == 5
    for row, bar in enumerate(bars.patches):
        pixel_y = main.transData.transform((0, row))[1]
        bar_y = bars.transData.transform((0, bar.get_y() + bar.get_height() / 2))[1]
        assert pixel_y == pytest.approx(bar_y, abs=0.01)
    assert main.get_xlabel() == "Acquisition day"
    assert "µm²/day" in bars.get_xlabel()


def test_single_observation_and_missing_middle_frame_are_visible(monkeypatch, tmp_path):
    experiment = measurement_experiment()
    cyst = experiment.get_all_cysts()[0]
    del cyst.frame_data[1]
    figures = capture_figures(monkeypatch)
    suite = plots.OrganoidVisualizationSuite(dpi=60)
    assert suite.plot_cyst_areas_multiline(experiment, str(tmp_path / "areas.png"))
    line = figures[-1].axes[0].lines[0]
    assert np.isnan(line.get_ydata()[1]) and line.get_marker() == "o"
    del cyst.frame_data[2]
    assert suite.plot_cyst_areas_multiline(experiment, str(tmp_path / "single.png"))
    assert np.isfinite(figures[-1].axes[0].lines[0].get_ydata()).sum() == 1
    assert "µm²" in figures[-1].axes[0].get_ylabel()


def test_svg_report_embeds_raster_companions_and_measurement_warnings(monkeypatch, tmp_path):
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate

    captured = []
    original = SimpleDocTemplate.build

    def build(self, story, **kwargs):
        captured.extend(story)
        return original(self, story, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", build)

    monkeypatch.setattr(plots, "VISUALIZATION_FORMAT", "svg")
    monkeypatch.setattr(plots, "MATPLOTLIB_DPI", 60)
    summary = run_report(tmp_path, results_for(range(3)), organoids(1, 2))
    assert summary["success"]
    assert len(list((tmp_path / "visualizations").glob("*.svg"))) == 6
    assert len(list((tmp_path / "visualizations").glob("*.png"))) == 6
    assert (tmp_path / "organoid_analysis_report.pdf").stat().st_size > 1000
    text = " ".join(item.getPlainText() for item in captured if isinstance(item, Paragraph))
    assert "Annotated cysts without tracked masks" in text
    assert "does not establish biological absence" in text
    assert "Growth definitions" in text
    assert sum(isinstance(item, Image) for item in captured) == 6


def test_replot_refuses_to_invent_missing_time_axis(tmp_path):
    from organoidtracker.analysis.csv_import import experiment_from_csv

    raw = tmp_path / "raw_cyst_data.csv"
    raw.write_text("Frame,Time_Days\n1,2\n3,9\n")
    with pytest.raises(ValueError, match="analysis_summary.json"):
        experiment_from_csv(raw)
