"""Qt coordinate and lifecycle checks use synthetic videos and an isolated settings file."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pytestqt", reason="install the qt extra and qt-test dependency group")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QSettings, Qt  # noqa: E402

from helpers import FakeTracker  # noqa: E402
from organoidtracker.gui_qt.canvas import AnnotationCanvas  # noqa: E402
from organoidtracker.gui_qt.window import MainWindow  # noqa: E402
from organoidtracker.gui_qt.worker import preview_session  # noqa: E402
from organoidtracker.services.annotations import AnnotationSet  # noqa: E402
from organoidtracker.services.export_service import ExportError, ExportService  # noqa: E402
from organoidtracker.services.run_manifest import mask_digests  # noqa: E402
from organoidtracker.services.saved_results import load_saved_result  # noqa: E402
from organoidtracker.services.session import load_session  # noqa: E402
from organoidtracker.services.tracking_service import TrackingService  # noqa: E402
from test_services import make_session, run  # noqa: E402
from test_services import small_disc as small_disc  # noqa: E402
from test_services import small_video as small_video  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def window(qtbot, tmp_path, small_disc):
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    window = MainWindow(
        settings=settings,
        tracking_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse, frame_delay_s=0.03)
        ),
    )
    window.run_root = tmp_path / "实验 runs"
    window.videos.setChecked(False)
    qtbot.addWidget(window)
    window.show()
    yield window
    if window.busy:
        window.cancel_tracking()
        qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    window.close()


def opened(window, small_disc, small_video):
    session = make_session(
        small_disc, small_video, extra_empty=True, timing={"frame_times_days": [1, 1.2, 1.7, 2.8, 3, 4.4, 5.9, 8]}
    )
    window.use_opened(preview_session(session))
    return session


@pytest.mark.parametrize("zoom", [0.25, 1, 3])
def test_canvas_source_coordinates_and_mouse_gestures(qtbot, zoom):
    canvas = AnnotationCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(700, 700)
    canvas.show()
    canvas.set_frame(np.zeros((256, 256, 3), dtype=np.uint8), fit=True)
    canvas.resetTransform()
    canvas.zoom(zoom)
    canvas.editable = True
    point = QPointF(128, 128)
    assert (canvas.source_point(canvas.mapFromScene(point)) - point).manhattanLength() <= 2 / zoom
    with qtbot.waitSignal(canvas.organoid_clicked) as signal:
        qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=canvas.mapFromScene(point))
    assert signal.args == pytest.approx([128, 128], abs=1 / zoom)
    start, end = canvas.mapFromScene(QPointF(80, 80)), canvas.mapFromScene(QPointF(180, 180))
    with qtbot.waitSignal(canvas.cyst_drawn) as signal:
        qtbot.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
        qtbot.mouseMove(canvas.viewport(), end)
        qtbot.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert signal.args == pytest.approx([80, 80, 180, 180], abs=1 / zoom)


def test_annotations_undo_and_shorter_reopen(window, small_disc, small_video, qtbot):
    session = opened(window, small_disc, small_video)
    window.use_opened(replace(window.opened, session=replace(session, annotations=AnnotationSet(()))))
    window.add_cyst(10, 10, 20, 20)
    assert not window.annotations.cysts
    window.add_organoid(20, 20)
    window.add_cyst(10, 10, 30, 30)
    window.add_organoid(50, 50)  # empty organoids stay in the denominator
    assert len(window.annotations.organoids) == 2 and len(window.annotations.cysts) == 1
    window.undo_stack.undo()
    assert len(window.annotations.organoids) == 1
    window.undo_stack.undo()
    assert not window.annotations.cysts
    window.undo_stack.redo()
    assert len(window.annotations.cysts) == 1
    one = replace(
        window.opened,
        source=replace(window.opened.source, frame_map=(0,), annotation_frame=0),
        frames=window.opened.frames[:1],
    )
    window.use_opened(one)
    assert window.slider.maximum() == 0 and window.slider.value() == 0


def test_qt_run_equals_cli_reopens_and_exports_new_analysis(
    window, small_disc, small_video, qtbot, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QFileDialog

    session = opened(window, small_disc, small_video)
    reference = run(session, tmp_path / "reference", small_disc)
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    saved = window.opened.saved
    assert saved is not None and saved.result.status == "completed"
    assert mask_digests(saved.result) == mask_digests(reference.result)
    assert (saved.directory / "raw_cyst_data.csv").read_bytes() == (
        reference.output_dir / "raw_cyst_data.csv"
    ).read_bytes()
    original = saved.path.read_bytes()
    window._begin("open", path=saved.path)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window.opened.saved.run_id == saved.run_id and not window.canvas.editable
    window.calibration.setText("2.000123456789123")
    window.explicit_times.setText("1, 2, 3, 5, 7, 9, 10, 12")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path / "copies"))
    window.export_copy()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    copied = load_saved_result(window.opened.saved.path)
    assert copied.path != saved.path and saved.path.read_bytes() == original
    assert copied.session.calibration.um_per_pixel == 2.000123456789123
    assert copied.session.timing.frame_times_days == (1, 2, 3, 5, 7, 9, 10, 12)
    assert load_session(copied.directory / "session.json").calibration == copied.session.calibration
    assert load_session(copied.directory / "prompts.json").timing == copied.session.timing
    assert mask_digests(copied.result) == mask_digests(saved.result)
    # Reopening and exporting the copy again must retain the changed measurements.
    raw = (copied.directory / "raw_cyst_data.csv").read_bytes()
    window._begin("open", path=copied.path)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    window.export_copy()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    assert (window.opened.saved.directory / "raw_cyst_data.csv").read_bytes() == raw


def test_cancel_close_and_reopen_preserve_partial_status(window, small_disc, small_video, qtbot):
    opened(window, small_disc, small_video)
    window.tracking_factory = lambda spec: TrackingService(FakeTracker(small_disc, frame_delay_s=0.1))
    window.start_tracking()
    qtbot.waitUntil(lambda: window.progress.maximum() == 8 and window.progress.value() >= 1, timeout=15000)
    window.close()  # waits cooperatively instead of destroying a live QThread
    assert window.busy and window._closing
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    saved = load_saved_result(window.opened.saved.path)
    assert saved.result.status == "cancelled" and 0 < saved.result.frames_done < 8
    assert not window.isVisible()


def test_export_failure_recovers_masks_and_failed_model_keeps_previous_state(
    window, small_disc, small_video, qtbot, monkeypatch
):
    opened(window, small_disc, small_video)

    def fail(*args, **kwargs):
        raise ExportError("injected report failure")

    monkeypatch.setattr(ExportService, "write_report", fail)
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    saved = window.opened.saved
    assert saved is not None and saved.path.is_file()
    assert "injected report failure" in window.banner.text() and "Masks were saved" in window.banner.text()
    window.edit_saved_prompts()
    window.tracking_factory = lambda spec: TrackingService(FakeTracker(small_disc, fail_load=True))
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window.opened.saved is None and window.annotations.cysts
    assert load_saved_result(saved.path).result.status == "completed"


def test_stale_progress_and_invalid_analysis_cannot_start_work(window, small_disc, small_video):
    opened(window, small_disc, small_video)
    text = window.banner.text()
    window._progress("old-token", "tracking", 7, 8, "stale")
    assert window.banner.text() == text
    window.explicit_times.setText("1, 2")
    with pytest.raises(ValueError, match="selected frames"):
        window.start_tracking()
    assert not window.busy and not window.run_root.exists()


def test_missing_video_relocation_checks_content(window, small_disc, small_video, qtbot, tmp_path):
    import shutil

    session = opened(window, small_disc, small_video)
    source = tmp_path / "原始.mp4"
    shutil.copyfile(small_video, source)
    saved = run(session.with_video(source), tmp_path / "run", small_disc).saved_result
    moved = tmp_path / "新位置.mp4"
    source.rename(moved)
    window._begin("open", path=saved.path)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert not window.opened.frames and not window.videos.isChecked()
    assert "Source video unavailable" in window.banner.text()
    wrong = tmp_path / "wrong.mp4"
    wrong.write_bytes(b"wrong content")
    window._begin("relocate", saved=saved, path=wrong)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert not window.opened.frames and "does not match" in window.banner.text()
    window._begin("relocate", saved=saved, path=moved)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert len(window.opened.frames) == 8 and window.current_session().video.path == moved


def test_cancel_before_first_mask_can_retry(window, small_disc, small_video, qtbot):
    opened(window, small_disc, small_video)
    window.tracking_factory = lambda spec: TrackingService(FakeTracker(small_disc, start_delay_s=0.15))
    window.start_tracking()
    window.cancel_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window.opened.saved is None
    assert "Cancelled before any masks" in window.banner.text()
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    assert window.opened.saved.result.status == "completed"


def test_qt_video_preset_and_viewer_keep_chronological_frames(window, small_disc, small_video, qtbot):
    from helpers import read_video

    opened(window, small_disc, small_video)
    window.videos.setChecked(True)
    window.quality.setCurrentText("mid")
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    saved = window.opened.saved
    assert saved is not None
    manifest = json.loads((saved.directory / "run_manifest.json").read_text())
    assert manifest["video_export"]["quality_scale"] == 0.5
    for name, width in [("overlay", 64), ("mask", 64), ("side_by_side", 128)]:
        frames = read_video(saved.directory / f"videos/multi_object_{name}.mp4")
        assert len(frames) == 8 and frames[0].shape[:2] == (64, width)
    window.view_mode.setCurrentText("Mask")
    first_x, first_y = small_disc.centers[0]
    last_x, last_y = small_disc.centers[7]
    window.slider.setValue(0)
    image = window.canvas.image_item.pixmap().toImage()
    assert image.pixelColor(first_x, first_y).red() > 0 and image.pixelColor(last_x, last_y).red() == 0
    window.slider.setValue(7)
    image = window.canvas.image_item.pixmap().toImage()
    assert image.pixelColor(last_x, last_y).red() > 0 and image.pixelColor(first_x, first_y).red() == 0


def test_stop_bridge_saves_cancelled_run_before_closing(window, small_disc, small_video, qtbot):
    from organoidtracker.gui_qt.signals import StopBridge

    opened(window, small_disc, small_video)
    bridge = StopBridge(window)
    bridge.timer.start()
    window.start_tracking()
    qtbot.waitUntil(lambda: window.progress.value() > 0 and "tracking" in window.banner.text(), timeout=15000)
    bridge.request()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=60000)
    assert not window.busy
    saved = load_saved_result(window.opened.saved.path)
    assert saved.result.status == "cancelled"
    assert 0 < saved.result.frames_done < saved.result.frames_total
    assert (saved.directory / "run_manifest.json").is_file()


def test_persistent_output_root_rejects_ephemeral_and_escaping_paths(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    root = tmp_path / "persistent"
    root.mkdir()
    window.output_root = root.resolve()
    outside = tmp_path / "ephemeral"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    for path in (outside, root / "escape" / "run"):
        with pytest.raises(ValueError, match="persistent results folder"):
            window.output_path(path)
    assert window.output_path(root / "new") == root / "new"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(outside))
    previous = window.run_root
    with pytest.raises(ValueError, match="persistent results folder"):
        window.choose_run_root()
    assert window.run_root == previous


def test_import_missing_source_session_relocates_with_hash_check(
    window, small_disc, small_video, qtbot, tmp_path, monkeypatch
):
    import hashlib

    from PySide6.QtWidgets import QFileDialog

    session = opened(window, small_disc, small_video)
    document = session.to_document()
    document["video"] = {
        "path": str(tmp_path / "old-computer" / "video.mp4"),
        "sha256": hashlib.sha256(small_video.read_bytes()).hexdigest(),
    }
    path = tmp_path / "imported.json"
    path.write_text(json.dumps(document))
    wrong = tmp_path / "wrong.mp4"
    wrong.write_bytes(b"different content")
    for replacement, accepted in ((wrong, False), (small_video, True)):
        choices = iter((str(path), str(replacement)))
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, choices=choices: (next(choices), ""))
        previous = window.opened
        window.choose_session()
        qtbot.waitUntil(lambda: not window.busy, timeout=15000)
        if accepted:
            assert window.opened.session.video.path == small_video
            assert window.opened.session.video.sha256 == document["video"]["sha256"]
            assert window.annotations == session.annotations
        else:
            assert window.opened is previous
            assert "match" in window.banner.text().lower()


def test_selected_organoid_survives_two_boxes_and_undo(window, small_disc, small_video):
    opened(window, small_disc, small_video)
    window.set_annotations(AnnotationSet(()))
    window.add_organoid(20, 20)
    window.add_organoid(60, 60)
    assert window.active_organoid == 2
    window.tree.setCurrentItem(window.tree.topLevelItem(0))
    window.add_cyst(10, 10, 25, 25)
    window.add_cyst(30, 30, 45, 45)
    assert [len(o.cysts) for o in window.current_session().annotations.organoids] == [2, 0]
    assert window.active_organoid == 1
    window.undo_stack.undo()
    window.undo_stack.redo()
    assert window.active_organoid == 1
    window.tree.setCurrentItem(window.tree.topLevelItem(0))
    window.remove_selected()
    assert window.active_organoid == 2
    window.undo_stack.undo()
    assert window.active_organoid == 1
    window.undo_stack.redo()
    assert window.active_organoid == 2


def test_frame_confirmation_keeps_quiet_observations_and_locks_prompt_coordinates(window, small_disc, small_video):
    from organoidtracker.services.session import Timing, VideoReference

    session = make_session(small_disc, small_video)
    session = replace(
        session, video=VideoReference(small_video), annotations=AnnotationSet(()), timing=Timing(time_lapse_days=6)
    )
    window.use_opened(preview_session(session))
    assert not window.canvas.editable and not window.track_button.isEnabled()
    assert window.slider.isEnabled()
    window.source_indices.setText("1, 3, 5, 8")
    window.explicit_times.setText("1, 2, 4, 7")
    window.confirm_frames()
    assert window.current_session().video.frame_indices == (0, 2, 4, 7)
    assert window.current_session().timing.frame_times_days == (1, 2, 4, 7)
    window.slider.setValue(0)
    assert not window.canvas.editable
    window.slider.setValue(3)
    window.add_organoid(20, 20)
    window.add_cyst(10, 10, 30, 30)
    # With annotations the source frames are locked; the days of the same frames stay correctable.
    assert not window.source_indices.isEnabled() and not window.all_frames_button.isEnabled()
    before = window.current_session().video
    window.source_indices.setText("1, 3, 5, 7")
    with pytest.raises(ValueError, match="Clear annotations"):
        window.confirm_frames()
    window.source_indices.setText("1, 3, 5, 8")
    assert window.current_session().video == before
    window.clear_annotations()
    window.select_all_frames()
    assert not window.canvas.editable
    with pytest.raises(ValueError, match="Confirm the changed"):
        window.current_session()


def test_daily_export_preset_requires_confirmation(window, small_disc, tmp_path):
    from organoidtracker.services.session import Timing, VideoReference

    pattern = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6]
    video = small_disc.write(tmp_path / "daily.mp4", repeat_pattern=pattern)
    session = replace(
        make_session(small_disc, video),
        video=VideoReference(video),
        annotations=AnnotationSet(()),
        timing=Timing(time_lapse_days=6),
    )
    window.use_opened(preview_session(session))
    window.select_daily_preset()
    assert not window.track_button.isEnabled() and window.opened.source.n_frames == 11
    window.confirm_frames()
    assert window.current_session().video.frame_indices == (0, 2, 3, 5, 6, 8, 9)
    assert window.current_session().timing.frame_times_days == (1, 2, 3, 4, 5, 6, 7)
    assert window.opened.source.annotation_frame == 6


def confirmed_empty(window, small_disc, small_video, days="1, 2, 4, 7"):
    """A new run on four selected frames, confirmed, without annotations (reverse: annotate on frame 4)."""
    from organoidtracker.services.session import Timing, VideoReference

    session = replace(
        make_session(small_disc, small_video),
        video=VideoReference(small_video),
        annotations=AnnotationSet(()),
        timing=Timing(time_lapse_days=6),
    )
    window.use_opened(preview_session(session))
    window.source_indices.setText("1, 3, 5, 8")
    window.explicit_times.setText(days)
    window.confirm_frames()
    assert window.canvas.editable


def test_direction_cannot_change_under_existing_annotations(window, small_disc, small_video):
    confirmed_empty(window, small_disc, small_video)
    # The reason is stated before and after: the note never appears or disappears (no layout change).
    assert window.direction.isEnabled() and window.direction_note.isVisibleTo(window)
    heights = (window.model_group.height(), window.checkpoint_status.height())
    window.add_organoid(20, 20)
    window.add_cyst(10, 10, 30, 30)
    assert not window.direction.isEnabled() and window.direction_note.isVisibleTo(window)
    assert (window.model_group.height(), window.checkpoint_status.height()) == heights
    # Even a change that bypasses the disabled control cannot move the boxes to the first frame.
    window.direction.setCurrentIndex(window.direction.findData("forward"))
    with pytest.raises(ValueError, match="drawn on the last time point"):
        window.current_session()
    with pytest.raises(ValueError, match="drawn on the last time point"):
        window.start_tracking()
    assert not window.busy and not window.run_root.exists()
    window.direction.setCurrentIndex(window.direction.findData("reverse"))
    assert window.current_session().tracking.direction == "reverse"
    # Clearing unlocks the direction; undo cannot then bring the last-frame boxes back onto frame 1.
    window.clear_annotations()
    assert window.direction.isEnabled() and window.undo_stack.canUndo()
    window.direction.setCurrentIndex(window.direction.findData("forward"))
    assert not window.undo_stack.canUndo() and window.slider.value() == 0
    window.add_organoid(20, 20)
    window.add_cyst(10, 10, 30, 30)
    assert window.current_session().tracking.direction == "forward"


def test_direction_is_not_remembered_by_the_next_window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    settings.setValue("direction", "First → last")  # written by an earlier version
    first = MainWindow(settings=settings)
    qtbot.addWidget(first)
    assert first.direction.currentData() == "reverse"
    first.direction.setCurrentIndex(first.direction.findData("forward"))
    first.model_size.setCurrentIndex(first.model_size.findData("sam2_hiera_t"))
    first.close()
    second = MainWindow(settings=settings)
    qtbot.addWidget(second)
    assert second.direction.currentData() == "reverse" and second.model_size.currentData() == "sam2_hiera_t"
    second.close()


def test_new_video_has_no_default_days_and_cannot_be_confirmed_without_them(
    window, small_disc, small_video, qtbot, monkeypatch
):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(small_video), ""))
    window.choose_video()
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window.opened.source.n_frames == 8 and window.opened.session.video.frame_indices is None
    assert window.duration.text() == "" and window.explicit_times.text() == ""
    assert [window.mapping.item(1, column).text() for column in range(8)] == ["?"] * 8
    with pytest.raises(ValueError, match="span in days, or one day per selected frame"):
        window.confirm_frames()
    assert window.opened.session.video.frame_indices is None and not window.canvas.editable
    window.source_indices.setText("1; 2")
    with pytest.raises(ValueError, match="frame numbers separated by commas"):
        window.confirm_frames()
    window.source_indices.setText("1, 9")
    with pytest.raises(ValueError, match="between 1 and 8"):
        window.confirm_frames()
    window.source_indices.setText("1, 2, 4, 5, 8")
    window.duration.setText("six")
    with pytest.raises(ValueError, match="enter a number of days"):
        window.confirm_frames()
    window.duration.clear()
    window.explicit_times.setText("1, 2, 4, 5, 8")
    # The table shows what will be recorded before the confirmation is given.
    assert [window.mapping.item(0, c).text() for c in range(5)] == ["1", "2", "4", "5", "8"]
    assert [window.mapping.item(1, c).text() for c in range(5)] == ["1", "2", "4", "5", "8"]
    window.confirm_frames()
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window.current_session().video.frame_indices == (0, 1, 3, 4, 7)
    assert window.current_session().timing.frame_times_days == (1, 2, 4, 5, 8)


def test_days_changed_after_confirmation_need_a_new_confirmation(window, small_disc, small_video):
    confirmed_empty(window, small_disc, small_video)
    window.add_organoid(20, 20)
    window.add_cyst(10, 10, 30, 30)
    assert window.track_button.isEnabled() and "day 7" in window.frame_label.text()
    window.explicit_times.setText("1, 2, 4, 30")
    # Not confirmed: nothing can be tracked or saved with days that are not the recorded ones.
    assert not window.track_button.isEnabled() and not window.canvas.editable
    assert "day 7" in window.frame_label.text() and "Not confirmed" in window.selection_status.text()
    with pytest.raises(ValueError, match="Confirm the changed acquisition days"):
        window.current_session()
    with pytest.raises(ValueError, match="Confirm the changed acquisition days"):
        window.start_tracking()
    assert not window.run_root.exists()
    window.explicit_times.setText("1, 2")
    with pytest.raises(ValueError, match="selected frames"):
        window.confirm_frames()
    # Confirming corrected days for the same frames keeps the annotations and their undo history.
    window.explicit_times.setText("1, 2, 4, 30")
    window.confirm_frames()
    assert len(window.annotations.cysts) == 1 and window.undo_stack.canUndo()
    assert window.track_button.isEnabled() and "day 30" in window.frame_label.text()
    assert window.current_session().timing.frame_times_days == (1, 2, 4, 30)
    assert window.current_session().video.frame_indices == (0, 2, 4, 7)
    # Returning the inputs to the confirmed values needs no further confirmation.
    window.explicit_times.setText("1, 2, 4, 8")
    assert not window.track_button.isEnabled()
    window.explicit_times.setText("1, 2, 4, 30")
    assert window.track_button.isEnabled()


def test_a_click_that_cannot_annotate_says_why(window, small_disc, small_video, qtbot):
    from organoidtracker.gui_qt.window import START_TEXT
    from organoidtracker.services.session import Timing, VideoReference

    assert "Confirm" in START_TEXT and window.banner.text() == START_TEXT
    session = replace(
        make_session(small_disc, small_video),
        video=VideoReference(small_video),
        annotations=AnnotationSet(()),
        timing=Timing(time_lapse_days=6),
    )
    window.use_opened(preview_session(session))
    assert "Confirm frames and days" in window.banner.text()

    def click():
        position = window.canvas.mapFromScene(QPointF(40, 40))
        qtbot.mouseClick(window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=position)

    click()
    assert not window.annotations.organoids
    assert "locked until the acquisition time points are confirmed" in window.banner.text()
    assert window.confirm_frames_button.styleSheet() != ""
    window.explicit_times.setText("1, 2, 3, 4, 5, 6, 7, 8")
    window.confirm_frames()
    assert window.confirm_frames_button.styleSheet() == ""
    window.slider.setValue(2)
    click()
    assert not window.annotations.organoids and "Move the frame slider to frame 8" in window.banner.text()
    window.slider.setValue(7)
    click()
    assert len(window.annotations.organoids) == 1


def test_result_view_shows_which_mask_is_which_cyst(window, small_disc, small_video, qtbot):
    from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsSimpleTextItem

    from organoidtracker.core.cyst_identity import cyst_color

    session = make_session(small_disc, small_video, two_objects=True)
    window.use_opened(preview_session(session))
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    assert window.opened.saved is not None

    def texts():
        return sorted({i.text() for i in window.canvas.mask_items if isinstance(i, QGraphicsSimpleTextItem)})

    window.slider.setValue(7)
    assert texts() == ["C1", "C2"]
    window.view_mode.setCurrentText("Original")
    assert texts() == []
    window.view_mode.setCurrentText("Mask")
    x, y = small_disc.centers[7]
    pixel = window.canvas.image_item.pixmap().toImage().pixelColor(x - 20, y)  # inside cyst 1 only
    assert (pixel.red(), pixel.green(), pixel.blue()) == cyst_color(1) == (255, 0, 0)
    # The annotation list carries the same colours and selecting a cyst outlines its mask.
    organoid = window.tree.topLevelItem(0)
    assert not organoid.child(1).icon(0).isNull()
    swatch = organoid.child(1).icon(0).pixmap(12, 12).toImage().pixelColor(6, 6)
    assert (swatch.red(), swatch.green(), swatch.blue()) == cyst_color(2)
    assert not any(isinstance(i, QGraphicsRectItem) for i in window.canvas.mask_items)
    window.tree.setCurrentItem(organoid.child(1))
    boxes = [i for i in window.canvas.mask_items if isinstance(i, QGraphicsRectItem)]
    assert len(boxes) == 1 and boxes[0].pen().color().getRgb()[:3] == cyst_color(2)
    window.view_mode.setCurrentText("Side by side")
    assert all(i.pos().x() >= 128 for i in window.canvas.mask_items if isinstance(i, QGraphicsSimpleTextItem))


def test_wrapped_status_lines_keep_the_height_their_text_needs(window, qtbot):
    """On the container desktop a relayout clipped these lines; each now reserves what its text needs."""
    labels = (window.checkpoint_status, window.selection_status, window.direction_note)

    def fitted():
        return all(label.minimumHeight() == label.heightForWidth(label.width()) <= label.height() for label in labels)

    qtbot.waitUntil(fitted, timeout=5000)
    status = window.checkpoint_status
    one_line = status.fontMetrics().lineSpacing()
    window.checkpoint.setText("/w/" + "a-very-long-directory-name/" * 12 + "sam2.1_hiera_base_plus.pt")
    qtbot.waitUntil(lambda: fitted() and status.height() >= 3 * one_line, timeout=5000)
    window.checkpoint.setText("/w/x.pt")  # shorter text gives the space back
    qtbot.waitUntil(lambda: fitted() and status.minimumHeight() < 3 * one_line, timeout=5000)
    for data in ("sam2_hiera_t", "sam2_hiera_l", "sam2_hiera_b"):
        window.model_size.setCurrentIndex(window.model_size.findData(data))
        qtbot.waitUntil(fitted, timeout=5000)
    before = status.minimumHeight()
    window.update_checkpoint_status()  # the container refreshes this every two seconds
    assert status.minimumHeight() == before


def test_confirming_days_keeps_the_calibration_entered_before_it(window, small_disc, small_video, qtbot):
    """Confirmation wrote the previous calibration back into the field, and the run measured with it."""
    import csv

    opened(window, small_disc, small_video)  # annotated and confirmed at 1 µm/pixel
    window.calibration.setText("2")
    window.explicit_times.setText("1, 1.2, 1.7, 2.8, 3, 4.4, 5.9, 9")
    window.confirm_frames()  # the same frames with corrected days: annotations stay
    assert float(window.calibration.text()) == 2 and len(window.annotations.cysts) == 1
    assert window.current_session().calibration.um_per_pixel == 2
    window.calibration.setText("0")
    with pytest.raises(ValueError, match="um_per_pixel|calibration"):
        window.confirm_frames()
    assert window.calibration.text() == "0" and window.opened.session.calibration.um_per_pixel == 2
    window.calibration.setText("2")
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    saved = load_saved_result(window.opened.saved.path)
    assert saved.session.calibration.um_per_pixel == 2 and saved.session.timing.frame_times_days[-1] == 9
    with (saved.directory / "raw_cyst_data.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:  # 2 µm/pixel: every written area is four times its mask's pixel count
        pixels = int(saved.result[int(row["Frame"])][int(row["Cyst_ID"])].numpy().sum())
        assert pixels > 0 and float(row["Area_um2"]) == 4 * pixels


def test_first_confirmation_keeps_the_calibration_entered_before_it(window, small_disc, small_video):
    from organoidtracker.services.session import Timing, VideoReference

    session = replace(
        make_session(small_disc, small_video),
        video=VideoReference(small_video),
        annotations=AnnotationSet(()),
        timing=Timing(time_lapse_days=6),
    )
    window.use_opened(preview_session(session))
    window.source_indices.setText("1, 3, 5, 8")
    window.explicit_times.setText("1, 2, 4, 7")
    window.calibration.setText("1.6934")
    window.confirm_frames()  # other frames than the preview's: the window reopens the selection
    assert window.opened.source.frame_map == (0, 2, 4, 7) and window.calibration.text() == "1.6934"
    window.add_organoid(20, 20)
    window.add_cyst(10, 10, 30, 30)
    assert window.current_session().calibration.um_per_pixel == 1.6934


def test_reports_export_from_saved_masks_after_the_source_video_moved(
    window, small_disc, small_video, qtbot, tmp_path, monkeypatch
):
    """The window named the recorded video path as a relocation, which made a report-only export read it."""
    import shutil

    from PySide6.QtWidgets import QFileDialog

    session = opened(window, small_disc, small_video)
    source = tmp_path / "source.mp4"
    shutil.copyfile(small_video, source)
    saved = run(session.with_video(source), tmp_path / "run", small_disc).saved_result
    moved = tmp_path / "moved.mp4"
    source.rename(moved)
    window._begin("open", path=saved.path)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert not window.opened.frames and not window.videos.isChecked()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path / "copies"))
    window.export_copy()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    assert window.banner.text().startswith("Completed"), window.banner.text()
    first = load_saved_result(window.opened.saved.path)
    assert first.directory.parent == tmp_path / "copies" and first.session.video.path == source  # not relocated
    assert (first.directory / "organoid_analysis_report.pdf").stat().st_size > 1000
    assert (first.directory / "raw_cyst_data.csv").read_bytes() == (saved.directory / "raw_cyst_data.csv").read_bytes()
    assert mask_digests(first.result) == mask_digests(saved.result)
    # Videos still need the video: asking for them without it fails and writes no second copy.
    window.videos.setChecked(True)
    window.export_copy()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    assert "video not found" in window.banner.text()
    assert len(list((tmp_path / "copies").rglob("organoid_analysis_report.pdf"))) == 1
    # A re-analysis typed before locating the video survives it; the located video is checked and recorded.
    window.calibration.setText("2")
    window._begin("relocate", saved=window.opened.saved, path=moved)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert len(window.opened.frames) == 8 and window.calibration.text() == "2"
    window.videos.setChecked(True)
    window.export_copy()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    second = load_saved_result(window.opened.saved.path)
    assert second.session.video.path == moved and second.session.calibration.um_per_pixel == 2
    assert (second.directory / "videos" / "multi_object_overlay.mp4").is_file()


def test_report_export_survives_a_located_video_going_offline(
    window, small_disc, small_video, qtbot, tmp_path, monkeypatch
):
    """Once a moved video had been located, its path was required even for a report without videos."""
    import shutil

    from PySide6.QtWidgets import QFileDialog

    session = opened(window, small_disc, small_video)
    source = tmp_path / "source.mp4"
    shutil.copyfile(small_video, source)
    saved = run(session.with_video(source), tmp_path / "run", small_disc).saved_result
    external = tmp_path / "external.mp4"
    source.rename(external)
    window._begin("open", path=saved.path)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    window._begin("relocate", saved=saved, path=external)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert len(window.opened.frames) == 8 and window.current_session().video.path == external
    offline = tmp_path / "unplugged.mp4"
    external.rename(offline)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path / "copies"))
    window.videos.setChecked(True)  # videos cannot be written without the video: refused, nothing exported
    window.export_copy()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    assert "video not found" in window.banner.text()
    assert not list((tmp_path / "copies").rglob("organoid_analysis_report.pdf"))
    window.videos.setChecked(False)
    window.export_copy()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    assert window.banner.text().startswith("Completed"), window.banner.text()
    copy = load_saved_result(window.opened.saved.path)
    assert copy.session.video.path == source  # the unread location is not recorded as the run's video
    assert (copy.directory / "organoid_analysis_report.pdf").stat().st_size > 1000
    assert (copy.directory / "raw_cyst_data.csv").read_bytes() == (saved.directory / "raw_cyst_data.csv").read_bytes()
    assert mask_digests(copy.result) == mask_digests(saved.result)


def test_an_older_prompt_record_opens_unconfirmed_in_the_window(window, small_disc, small_video, tmp_path):
    """Its frames were selected by the older software; the window must ask before a new run uses them."""
    from organoidtracker.services.session import Timing

    session = replace(
        make_session(small_disc, small_video),
        video=replace(make_session(small_disc, small_video).video, frame_indices=(0, 4, 7)),
        timing=Timing(time_lapse_days=7),
    )
    record = json.loads((run(session, tmp_path / "run", small_disc).output_dir / "prompts.json").read_text())
    current, legacy = tmp_path / "current.json", tmp_path / "legacy.json"
    current.write_text(json.dumps(record))
    legacy.write_text(json.dumps({**record, "results_version": 4}))
    window.use_opened(preview_session(load_session(current)))
    assert window.track_button.isEnabled() and window.current_session().video.frame_indices == (0, 4, 7)
    window.use_opened(preview_session(load_session(legacy)))
    assert not window.track_button.isEnabled() and not window.canvas.editable
    assert window.opened.source.frame_map == tuple(range(8)) and len(window.annotations.cysts) == 1
    with pytest.raises(ValueError, match="Confirm source frames"):
        window.current_session()
    with pytest.raises(ValueError, match="Confirm source frames"):
        window.start_tracking()
    assert not window.run_root.exists()
