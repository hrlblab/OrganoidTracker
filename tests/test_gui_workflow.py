"""The Tk window drives the user's workflow through the services and produces exactly what a headless run of the
same session produces (marker: gui; a display is needed, no model: the fake backend stands in).

Covered: load model, load video, organoid clicks and cyst boxes, a reverted cyst that must not be tracked, an
organoid without cysts that must stay in the population, tracking, videos, the report; a partial run that the
log reports as partial; an analysis export that cannot be completed, reported as an error rather than success.
Windows runs one Tk interpreter per process (the smoke test holds it), so these tests run on Linux only.
"""

import csv
import hashlib
import json
import sys
import tkinter

import pytest

from helpers import DiscVideo, FakeTracker, TkDriver
from organoidtracker.services.pipeline import run_session
from organoidtracker.services.run_manifest import mask_digests
from organoidtracker.services.session import session_from_document
from organoidtracker.services.tracking_service import TrackingService
from test_gui_smoke import environment_skip_reason

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(
        sys.platform == "win32", reason="one Tk interpreter per process on Windows; the smoke test holds it"
    ),
]

N = 8


@pytest.fixture(scope="module")
def small_disc():
    return DiscVideo(n_frames=N, size=128, radius=10, start=20, step=12)


@pytest.fixture(scope="module")
def small_video(small_disc, tmp_path_factory):
    return small_disc.write(tmp_path_factory.mktemp("video") / "disc.mp4")


@pytest.fixture
def app(monkeypatch, small_disc, request):
    """A window whose backend is the fake tracker; ``request.param`` holds the fake's keyword arguments."""
    from organoidtracker.gui_tk.main_window import VideoTrackerApp

    fake_kwargs = getattr(request, "param", {})
    monkeypatch.setattr(
        TrackingService,
        "create",
        classmethod(
            lambda cls, spec, registry=None: cls(
                FakeTracker(small_disc, enable_reverse_tracking=spec.reverse, **fake_kwargs)
            )
        ),
    )
    try:
        window = VideoTrackerApp()
    except tkinter.TclError as error:
        reason = environment_skip_reason(error)
        if reason is None:
            raise
        pytest.skip(reason)
    yield window
    window.root.destroy()


def log_of(app) -> str:
    return app.output_info_text.get("1.0", "end")


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def workflow(
    app,
    monkeypatch,
    video,
    out,
    *,
    boxes,
    time_lapse=7.0,
    conversion=1.0,
    revert_last_cyst=False,
    extra_empty_point=None,
):
    """Drive: load model, load video, annotate (optionally revert the last cyst, add an empty organoid), track, videos, report."""
    from organoidtracker.gui_tk import main_window

    videos_dir, report_dir = out / "videos", out / "report"
    videos_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(video))
        app.load_video()

    def annotate_and_track():
        for point, cysts in boxes:
            app.on_canvas_click(*point)
            for box in cysts:
                app.on_canvas_bbox(*box)
        if revert_last_cyst:
            app.revert_last_action()
        if extra_empty_point is not None:
            app.on_canvas_click(*extra_empty_point)
        app.time_lapse_var.set(time_lapse)
        app.conversion_factor_var.set(conversion)
        app.start_tracking()

    def generate_videos():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(videos_dir))
        app.generate_videos()

    def generate_report():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(report_dir))
        app.generate_analysis_report()

    def tracked():
        return (
            not app.tracking_in_progress and app.video_segments is not None and str(app.track_btn["state"]) == "normal"
        )

    def videos_written():
        return str(app.generate_btn["state"]) == "normal" and "Video generation completed" in log_of(app)

    def report_done():
        return str(app.analysis_btn["state"]) == "normal" and (
            "analysis report generated successfully" in log_of(app) or "ANALYSIS FAILED" in log_of(app)
        )

    (
        TkDriver(app.root)
        .step(
            "model loaded",
            load_model,
            lambda: app.current_model is not None and str(app.load_model_btn["state"]) == "normal",
            30,
        )
        .step(
            "video loaded",
            load_video,
            lambda: app.current_video_path is not None and str(app.track_btn["state"]) == "normal",
            30,
        )
        .step("tracked", annotate_and_track, tracked, 60)
        .step("videos written", generate_videos, videos_written, 60)
        .step("report done", generate_report, report_done, 120)
        .run()
    )
    return videos_dir, report_dir


def equivalent_session(video, organoids, time_lapse=7.0, conversion=1.0):
    return session_from_document(
        {
            "schema": "organoidtracker.session/1",
            "video": {"path": str(video), "sha256": hashlib.sha256(video.read_bytes()).hexdigest()},
            "tracking": {"model_config": "sam2_hiera_s", "device": "cpu"},
            "calibration": {"um_per_pixel": conversion},
            "timing": {"time_lapse_days": time_lapse},
            "organoids": organoids,
        }
    )


def test_window_matches_the_headless_run_and_honours_reverts_and_empty_organoids(
    app, monkeypatch, small_disc, small_video, tmp_path
):
    cx, cy = small_disc.centers[N - 1]
    box1 = tuple(int(v) for v in small_disc.box(N - 1))
    box2 = (cx - 15, cy - 15, cx + 20, cy + 15)
    videos_dir, report_dir = workflow(
        app,
        monkeypatch,
        small_video,
        tmp_path / "gui",
        boxes=[((cx - 30, cy - 30), [box1]), ((40, 40), [box2])],
        revert_last_cyst=True,  # cyst 2 removed; its organoid 2 is auto-removed with it
        extra_empty_point=(5, 5),  # a new organoid 2 without cysts
        time_lapse=14.0,
        conversion=1.6934,
    )
    assert app.video_segments.object_ids() == [1], "the reverted cyst must not be tracked"
    assert sorted(app.organoid_data) == [1, 2] and app.organoid_data[2]["cysts"] == []
    assert "✅ Tracking completed" in log_of(app) and "❌" not in log_of(app)
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (videos_dir / name).is_file()
    assert (report_dir / "organoid_analysis_report.pdf").is_file()

    # the same session headlessly: byte-identical tables, identical masks
    session = equivalent_session(
        small_video,
        [
            {"organoid_id": 1, "point": [cx - 30, cy - 30], "cysts": [{"cyst_id": 1, "bbox": list(box1)}]},
            {"organoid_id": 2, "point": [5, 5], "cysts": []},
        ],
        time_lapse=14.0,
        conversion=1.6934,
    )
    outcome = run_session(
        session,
        tmp_path / "cli",
        videos=True,
        tracking_service_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse)
        ),
    )
    for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
        assert (report_dir / name).read_bytes() == (outcome.output_dir / name).read_bytes(), name
    assert mask_digests(app.video_segments) == mask_digests(outcome.result)
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (videos_dir / name).stat().st_size == (outcome.output_dir / "videos" / name).stat().st_size, name
    gui_summary = json.loads((report_dir / "analysis_summary.json").read_text())
    cli_summary = json.loads((outcome.output_dir / "analysis_summary.json").read_text())
    for key in ("experiment_info", "growth_statistics", "quality_metrics", "tracking", "complete"):
        assert gui_summary[key] == cli_summary[key], key
    assert gui_summary["experiment_info"]["total_organoids"] == 2
    rows = read_csv(report_dir / "organoid_summary.csv")
    assert [(r["Organoid_ID"], r["Total_Cysts"]) for r in rows] == [("1", "1"), ("2", "0")]


@pytest.mark.parametrize("app", [{"partial_after": 3}], indirect=True)
def test_window_reports_a_partial_run_as_partial(app, monkeypatch, small_disc, small_video, tmp_path):
    cx, cy = small_disc.centers[N - 1]
    _videos_dir, report_dir = workflow(
        app,
        monkeypatch,
        small_video,
        tmp_path / "gui",
        boxes=[((cx - 30, cy - 30), [tuple(int(v) for v in small_disc.box(N - 1))])],
    )
    assert app.video_segments.status == "partial"
    log = log_of(app)
    assert "⚠️ Tracking stopped early" in log and "treat exports as partial" in log
    assert "PARTIAL TRACKING RUN: 3 of 8 frames" in log
    summary = json.loads((report_dir / "analysis_summary.json").read_text())
    assert summary["success"] and not summary["complete"] and summary["tracking"]["status"] == "partial"


def test_window_reports_an_incomplete_report_as_a_failure(app, monkeypatch, small_disc, small_video, tmp_path):
    from organoidtracker.analysis.organoid_report_generator import OrganoidAnalysisReportGenerator

    monkeypatch.setattr(OrganoidAnalysisReportGenerator, "_generate_enhanced_pdf_report", lambda self, *a, **k: None)
    cx, cy = small_disc.centers[N - 1]
    _videos_dir, report_dir = workflow(
        app,
        monkeypatch,
        small_video,
        tmp_path / "gui",
        boxes=[((cx - 30, cy - 30), [tuple(int(v) for v in small_disc.box(N - 1))])],
    )
    log = log_of(app)
    assert "ORGANOID ANALYSIS FAILED" in log and "organoid_analysis_report.pdf" in log
    assert "report generated successfully" not in log
    assert json.loads((report_dir / "analysis_summary.json").read_text())["success"] is False


def test_reloading_the_model_discards_stale_results_and_recovers(app, monkeypatch, small_disc, small_video, tmp_path):
    """Load, annotate, track; load a model again: results and export controls of the old backend go away, a stale
    export request is refused without getting stuck, and the workflow runs again on the new backend."""
    from organoidtracker.gui_tk import main_window

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    state: dict = {}

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def model_loaded():
        return app.current_model is not None and str(app.load_model_btn["state"]) == "normal"

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(small_video))
        app.load_video()

    def video_loaded():
        return app.current_video_path is not None and str(app.track_btn["state"]) == "normal"

    def annotate_and_track():
        app.on_canvas_click(cx - 30, cy - 30)
        app.on_canvas_bbox(*box)
        app.start_tracking()

    def tracked():
        return (
            not app.tracking_in_progress and app.video_segments is not None and str(app.track_btn["state"]) == "normal"
        )

    def reload_model():
        state["first_service"] = app.tracking
        state["first_results"] = app.video_segments
        assert str(app.generate_btn["state"]) == "normal" and str(app.analysis_btn["state"]) == "normal"
        app.load_selected_model()

    def reloaded():
        return (
            app.tracking is not None
            and app.tracking is not state["first_service"]
            and str(app.load_model_btn["state"]) == "normal"
        )

    def stale_export_requests():
        # everything downstream of the old backend is gone and disabled
        assert app.video_segments is None and app.current_video_path is None and not app.organoid_data
        assert not app.active_object_ids and app.current_organoid_id is None
        for button in (app.track_btn, app.generate_btn, app.analysis_btn, app.clear_prompts_btn, app.revert_btn):
            assert str(button["state"]) == "disabled", button
        assert "discarded" in log_of(app)
        # a stale request (as if the buttons had still been enabled) is refused and nothing gets stuck
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(videos_dir))
        app.generate_videos()
        app.generate_analysis_report()
        assert "No tracking results available" in app.status_label["text"]
        assert "Generating videos" not in app.status_label["text"]
        # the race between the swap and a click: old results still present, new backend without a video
        app.video_segments = state["first_results"]
        app.generate_videos()
        assert "Load a video and run tracking first" in app.status_label["text"]
        assert str(app.generate_btn["state"]) == "disabled" and not list(videos_dir.iterdir())
        app.video_segments = None

    def second_run():
        load_video()

    def second_tracked():
        return tracked() and app.video_segments is not state["first_results"]

    def generate_videos():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(videos_dir))
        app.generate_videos()

    (
        TkDriver(app.root)
        .step("model loaded", load_model, model_loaded, 30)
        .step("video loaded", load_video, video_loaded, 30)
        .step("tracked", annotate_and_track, tracked, 60)
        .step("model reloaded", reload_model, reloaded, 30)
        .step("stale requests refused", stale_export_requests, None, 5)
        .step("video loaded again", second_run, video_loaded, 30)
        .step("tracked again", annotate_and_track, second_tracked, 60)
        .step("videos written", generate_videos, lambda: "Video generation completed" in log_of(app), 60)
        .run()
    )
    assert app.video_segments.object_ids() == [1]
    assert all((videos_dir / f"multi_object_{t}.mp4").is_file() for t in ("overlay", "mask", "side_by_side"))
    assert str(app.generate_btn["state"]) == "normal" and "❌ Video generation failed" not in log_of(app)
