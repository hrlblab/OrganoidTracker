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


def test_reload_completing_inside_the_folder_chooser_is_refused(app, monkeypatch, small_disc, small_video, tmp_path):
    """The folder chooser is modal and runs the event loop, so a model reload can complete while it is open and
    discard the results the readiness check had just approved. Both export paths must notice when the chooser
    returns, before changing any control, and refuse the request without getting stuck."""
    from organoidtracker.gui_tk import main_window

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    out = tmp_path / "out"
    out.mkdir()

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

    def chooser_during_which_a_reload_completes(*args, **kwargs):
        # what the main loop does while the real chooser is open: it drains the queue and installs the new backend
        app.on_model_loaded_success(TrackingService(FakeTracker(small_disc)), 0.0)
        return str(out)

    def interleaved(request):
        def run():
            assert str(app.generate_btn["state"]) == "normal" and str(app.analysis_btn["state"]) == "normal"
            monkeypatch.setattr(main_window.filedialog, "askdirectory", chooser_during_which_a_reload_completes)
            request()
            # refused after the chooser: no control changed by the request itself, nothing written
            assert "Generating" not in app.status_label["text"]
            assert "No tracking results available" in app.status_label["text"]
            assert app.video_segments is None and not list(out.iterdir())

        return run

    (
        TkDriver(app.root)
        .step("model loaded", load_model, model_loaded, 30)
        .step("video loaded", load_video, video_loaded, 30)
        .step("tracked", annotate_and_track, tracked, 60)
        .step("videos refused after the chooser", interleaved(app.generate_videos), None, 5)
        .step("video loaded again", load_video, video_loaded, 30)
        .step("tracked again", annotate_and_track, tracked, 60)
        .step("report refused after the chooser", interleaved(app.generate_analysis_report), None, 5)
        .step("video loaded once more", load_video, video_loaded, 30)
        .step("tracked once more", annotate_and_track, tracked, 60)
        .step("save refused after the chooser", interleaved(app.save_results), None, 5)
        .step("video loaded a fourth time", load_video, video_loaded, 30)
        .step("tracked a fourth time", annotate_and_track, tracked, 60)
        .run()
    )
    # the window is still fully usable after both refusals
    assert str(app.generate_btn["state"]) == "normal" and str(app.analysis_btn["state"]) == "normal"
    assert app.video_segments.object_ids() == [1]


def test_window_saves_results_that_export_identically_without_the_tracker(
    app, monkeypatch, small_disc, small_video, tmp_path
):
    """Track in the window, save the results, export them again headlessly: the same masks, tables and videos."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.pipeline import export_saved_result
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box1 = tuple(int(v) for v in small_disc.box(N - 1))
    videos_dir, report_dir = workflow(
        app,
        monkeypatch,
        small_video,
        tmp_path / "gui",
        boxes=[((cx - 30, cy - 30), [box1])],
        extra_empty_point=(5, 5),
        time_lapse=14.0,
        conversion=1.6934,
    )
    save_dir = tmp_path / "saved"
    answers = {"replace": False}

    def save():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(save_dir))
        monkeypatch.setattr(main_window.messagebox, "askyesno", lambda *a, **k: answers["replace"])
        app.save_results()

    def saved():
        return str(app.save_results_btn["state"]) == "normal" and "Results saved" in log_of(app)

    def save_again_cancelled():
        save()  # the directory now holds a saved run and the question is answered "no"
        assert "Saving cancelled" in app.status_label["text"]

    def save_again_replacing():
        answers["replace"] = True
        save()

    (
        TkDriver(app.root)
        .step("results saved", save, saved, 60)
        .step("second save cancelled", save_again_cancelled, None, 5)
        .step("second save replaces", save_again_replacing, lambda: log_of(app).count("Results saved") == 2, 60)
        .run()
    )
    assert sorted(p.name for p in save_dir.iterdir() if not p.name.startswith("masks-")) == [
        "prompts.json",
        "results.json",
        "session.json",
    ]
    assert len(list(save_dir.glob("masks-*.npz"))) == 1
    reloaded = load_saved_result(save_dir)
    assert mask_digests(reloaded.result) == mask_digests(app.video_segments)
    assert reloaded.session.annotations.organoid_data() == {
        1: {"point": (cx - 30, cy - 30), "cysts": [{"cyst_id": 1, "bbox": box1}]},
        2: {"point": (5, 5), "cysts": []},
    }
    assert reloaded.session.timing.time_lapse_days == 14.0 and reloaded.session.calibration.um_per_pixel == 1.6934
    assert reloaded.run_id == app.tracking_run_id and reloaded.provenance["backend"] == "fake"

    again = export_saved_result(reloaded, tmp_path / "export", videos=True)
    assert again.exit_code == 0
    for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
        assert (again.output_dir / name).read_bytes() == (report_dir / name).read_bytes(), name
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (again.output_dir / "videos" / name).stat().st_size == (videos_dir / name).stat().st_size, name
    assert "💾 Results saved" in log_of(app) and "❌" not in log_of(app)


def test_window_opens_a_saved_run_without_a_model_and_exports_it(app, monkeypatch, small_disc, small_video, tmp_path):
    """A run saved by the command line opens in a window without any model; its videos and report come out identical,
    with the run's explicit irregular time axis; loading a video through a model then sets the reopened run aside."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.session import session_from_document

    cx, cy = small_disc.centers[N - 1]
    box1 = tuple(int(v) for v in small_disc.box(N - 1))
    times = [0, 1, 3, 6, 10, 15, 21, 28]
    session = session_from_document(
        {
            "schema": "organoidtracker.session/1",
            "video": {"path": str(small_video), "sha256": hashlib.sha256(small_video.read_bytes()).hexdigest()},
            "tracking": {"model_config": "sam2_hiera_s", "device": "cpu"},
            "calibration": {"um_per_pixel": 1.6934},
            "timing": {"frame_times_days": times},
            "organoids": [
                {"organoid_id": 1, "point": [cx - 30, cy - 30], "cysts": [{"cyst_id": 1, "bbox": list(box1)}]},
                {"organoid_id": 2, "point": [5, 5], "cysts": []},
            ],
        }
    )
    run = run_session(
        session,
        tmp_path / "run",
        videos=True,
        tracking_service_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse, partial_after=5, grow=3)
        ),
    )
    assert run.status == "partial" and app.current_model is None  # no model in this window

    videos_dir, report_dir = tmp_path / "videos", tmp_path / "report"
    videos_dir.mkdir()
    report_dir.mkdir()

    def open_results():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(run.output_dir))
        app.open_results()

    def opened():
        return app.opened is not None and str(app.open_results_btn["state"]) == "normal"

    def check_state():
        assert app.video_segments is run.saved_result.result or mask_digests(app.video_segments) == mask_digests(
            run.result
        )
        assert app.organoid_data == session.annotations.organoid_data()
        assert app.time_lapse_var.get() == 28.0 and app.conversion_factor_var.get() == 1.6934
        assert str(app.track_btn["state"]) == "disabled" and str(app.save_results_btn["state"]) == "disabled"
        assert str(app.generate_btn["state"]) == "normal" and str(app.analysis_btn["state"]) == "normal"
        assert "reopened run is partial" in log_of(app) and "explicit frame times" in log_of(app)
        app.on_canvas_click(5, 5)  # inert: the annotations shown are the run's
        app.on_canvas_bbox(1, 1, 10, 10)
        assert app.organoid_data == session.annotations.organoid_data()
        assert "not editable" in app.status_label["text"]

    def generate_videos():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(videos_dir))
        app.generate_videos()

    def generate_report():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(report_dir))
        app.generate_analysis_report()

    def report_done():
        return str(app.analysis_btn["state"]) == "normal" and (
            "analysis report generated successfully" in log_of(app) or "ANALYSIS FAILED" in log_of(app)
        )

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(small_video))
        app.load_video()

    (
        TkDriver(app.root)
        .step("results opened", open_results, opened, 30)
        .step("reopened state", check_state, None, 5)
        .step("videos written", generate_videos, lambda: "Video generation completed" in log_of(app), 60)
        .step("report done", generate_report, report_done, 120)
        .step("model loaded", load_model, lambda: app.current_model is not None, 30)
        .step("set aside", lambda: None, lambda: app.opened is not None, 5)  # a first model keeps the reopened run
        .step("video loaded", load_video, lambda: app.current_video_path == str(small_video), 30)
        .run()
    )
    for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
        assert (report_dir / name).read_bytes() == (run.output_dir / name).read_bytes(), name
    rows = read_csv(report_dir / "raw_cyst_data.csv")
    assert sorted({float(r["Time_Days"]) for r in rows}) == [6.0, 10.0, 15.0, 21.0, 28.0]  # the explicit axis
    summary = json.loads((report_dir / "analysis_summary.json").read_text())
    assert summary["tracking"]["status"] == "partial" and summary["experiment_info"]["total_organoids"] == 2
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (videos_dir / name).stat().st_size == (run.output_dir / "videos" / name).stat().st_size, name
    # the video loaded through the model replaced the reopened run
    assert app.opened is None and app.video_segments is None and not app.organoid_data
    assert "set aside" in log_of(app) and str(app.generate_btn["state"]) == "disabled"
    assert str(app.track_btn["state"]) == "normal"


def _track_in_window(app, monkeypatch, video, box, point=None):
    """Driver steps that load the fake model, the video, one organoid with one cyst box, and track."""
    from organoidtracker.gui_tk import main_window

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(video))
        app.load_video()

    def annotate_and_track():
        app.on_canvas_click(*(point or (5, 5)))
        app.on_canvas_bbox(*box)
        app.start_tracking()

    def tracked():
        return (
            not app.tracking_in_progress and app.video_segments is not None and str(app.track_btn["state"]) == "normal"
        )

    return (
        TkDriver(app.root)
        .step("model loaded", load_model, lambda: app.current_model is not None, 30)
        .step("video loaded", load_video, lambda: app.current_video_path == str(video), 30)
        .step("tracked", annotate_and_track, tracked, 60)
    )


def test_a_failed_save_over_a_saved_run_keeps_it(app, monkeypatch, small_disc, small_video, tmp_path):
    """Review finding: Save Results cleared the previous saved run before writing; a failed write left nothing."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services import saved_results
    from organoidtracker.services.saved_results import load_saved_result

    box = tuple(int(v) for v in small_disc.box(N - 1))
    cx, cy = small_disc.centers[N - 1]
    previous = run_session(
        equivalent_session(
            small_video, [{"organoid_id": 1, "point": [1, 1], "cysts": [{"cyst_id": 1, "bbox": list(box)}]}]
        ),
        tmp_path / "dest",
        videos=False,
        tracking_service_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse, grow=5)
        ),
    )
    dest = previous.output_dir
    before = {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()}

    def save():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(dest))
        monkeypatch.setattr(main_window.messagebox, "askyesno", lambda *a, **k: True)
        app.save_results()

    def failing_save():
        monkeypatch.setattr(
            saved_results,
            "_write_npz",
            lambda handle, arrays: (_ for _ in ()).throw(OSError("injected disk-full error")),
        )
        save()

    def failed():
        return (
            "Saving the results failed" in app.status_label["text"] and str(app.save_results_btn["state"]) == "normal"
        )

    def check_previous_intact():
        assert "injected disk-full error" in app.status_label["text"]
        assert {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()} == before  # manifest included
        assert mask_digests(load_saved_result(dest).result) == mask_digests(previous.result)
        monkeypatch.setattr(
            saved_results,
            "_write_npz",
            saved_results._write_npz.__wrapped__
            if hasattr(saved_results._write_npz, "__wrapped__")
            else original_write_npz,
        )

    original_write_npz = saved_results._write_npz
    (
        _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
        .step("failed save", failing_save, failed, 30)
        .step("previous run intact", check_previous_intact, None, 5)
        .step("replacing save", save, lambda: "Results saved" in app.status_label["text"], 30)
        .run()
    )
    assert not (dest / "run_manifest.json").exists()
    assert mask_digests(load_saved_result(dest).result) == mask_digests(app.video_segments)
    assert (dest / "raw_cyst_data.csv").read_bytes() == before["raw_cyst_data.csv"]  # exports left alone


def test_loading_another_video_invalidates_the_results(app, monkeypatch, small_disc, small_video, tmp_path):
    """Review finding: after loading another video through the same model, the old masks could be saved under the
    new video's identity (and exported over its frames). The results now belong to the video they were tracked on."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.saved_results import load_saved_result

    other_disc = DiscVideo(n_frames=N, size=128, radius=10, start=40, step=8)
    other_video = other_disc.write(tmp_path / "other.mp4")
    box = tuple(int(v) for v in small_disc.box(N - 1))
    cx, cy = small_disc.centers[N - 1]
    dest = tmp_path / "stale"
    state = {}

    def load_other_video():
        state["first_run_id"] = app.tracking_run_id
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(other_video))
        app.load_video()

    def other_loaded():
        return app.current_video_path == str(other_video) and str(app.track_btn["state"]) == "normal"

    def stale_requests_refused():
        assert app.video_segments is None and app.result_video is None
        for button in (app.generate_btn, app.analysis_btn, app.save_results_btn):
            assert str(button["state"]) == "disabled", button
        assert "previous tracking results were discarded" in log_of(app)
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(dest))
        app.save_results()
        app.generate_videos()
        app.generate_analysis_report()
        assert "No tracking results available" in app.status_label["text"]
        assert not dest.exists()

    def track_other():
        app.on_canvas_click(5, 5)
        app.on_canvas_bbox(*tuple(int(v) for v in other_disc.box(N - 1)))
        app.start_tracking()

    def tracked_other():
        return not app.tracking_in_progress and app.video_segments is not None and app.result_video is not None

    def save_other():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(dest))
        app.save_results()

    (
        _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
        .step("other video loaded", load_other_video, other_loaded, 30)
        .step("stale requests refused", stale_requests_refused, None, 5)
        .step("other video tracked", track_other, tracked_other, 60)
        .step("saved", save_other, lambda: "Results saved" in app.status_label["text"], 30)
        .run()
    )
    saved = load_saved_result(dest)
    assert saved.video.sha256 == hashlib.sha256(other_video.read_bytes()).hexdigest()
    assert mask_digests(saved.result) == mask_digests(app.video_segments)  # the fake's masks do not depend on the video
    assert saved.run_id == app.tracking_run_id and saved.run_id != state["first_run_id"]
    assert len(saved.session.annotations.organoids) == 1


def test_a_run_opened_inside_a_chooser_is_refused_and_the_next_request_uses_its_inputs(
    app, monkeypatch, small_disc, small_video, tmp_path
):
    """Review finding: run B installed while the report chooser was open was analyzed with run A's calibration and
    duration. A request whose results changed inside the dialog is refused; the next one uses B's inputs."""
    from dataclasses import replace

    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.session import Calibration, Timing
    from organoidtracker.services.video_frames import load_video_frames

    box = tuple(int(v) for v in small_disc.box(N - 1))
    session_a = equivalent_session(
        small_video, [{"organoid_id": 1, "point": [1, 1], "cysts": [{"cyst_id": 1, "bbox": list(box)}]}]
    )
    run_a = run_session(
        session_a,
        tmp_path / "a",
        videos=False,
        tracking_service_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse)
        ),
    )
    saved_a = run_a.saved_result
    saved_b = replace(
        saved_a,
        run_id="run-b",
        session=replace(saved_a.session, calibration=Calibration(2.0), timing=Timing(time_lapse_days=14.0)),
    )
    frames = load_video_frames(small_video, saved_a.video)
    out = tmp_path / "report"
    out.mkdir()

    def chooser_during_which_run_b_opens(*args, **kwargs):
        app.on_results_opened(saved_b, frames, None, 0.0)  # what the main loop does while the real chooser is open
        return str(out)

    def report_with_swap():
        assert app.conversion_factor_var.get() == 1.0 and app.time_lapse_var.get() == 7.0
        monkeypatch.setattr(main_window.filedialog, "askdirectory", chooser_during_which_run_b_opens)
        app.generate_analysis_report()
        assert "results changed while the dialog was open" in app.status_label["text"]
        assert not list(out.iterdir())
        assert app.conversion_factor_var.get() == 2.0 and app.time_lapse_var.get() == 14.0  # B is shown now

    def videos_with_swap():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", chooser_during_which_run_b_opens)
        app.generate_videos()
        assert "results changed while the dialog was open" in app.status_label["text"]
        assert not list(out.iterdir())

    def plain_report():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(out))
        app.generate_analysis_report()

    def report_done():
        return str(app.analysis_btn["state"]) == "normal" and "analysis report generated successfully" in log_of(app)

    (
        TkDriver(app.root)
        .step("run A opened", lambda: app.on_results_opened(saved_a, frames, None, 0.0), None, 5)
        .step("report refused after the swap", report_with_swap, None, 5)
        .step("videos refused after the swap", videos_with_swap, None, 5)
        .step("report of B", plain_report, report_done, 120)
        .run()
    )
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["experiment_info"]["conversion_factor_um_per_pixel"] == 2.0
    assert summary["experiment_info"]["time_lapse_days"] == 14.0
