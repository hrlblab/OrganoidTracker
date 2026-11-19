"""Qt workflow over the headless services; one operation owns inference at a time."""

import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QObject, QSaveFile, QSettings, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QPixmap, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.cyst_identity import cyst_color, cyst_label, mask_box
from ..services.annotations import AnnotationSet, CystAnnotation, OrganoidAnnotation
from ..services.export_service import DEFAULT_RUN_ROOT, new_run_directory
from ..services.session import (
    Calibration,
    Session,
    SessionError,
    Timing,
    TrackingSpec,
    VideoReference,
    load_session,
)
from .canvas import AnnotationCanvas
from .worker import Opened, Operation

START_TEXT = (
    "1. Open a video.   2. Confirm its acquisition time points (left panel).   "
    "3. On the annotation frame, click an organoid and drag boxes around its cysts.   4. Track and save run."
)
BANNER_STYLES = {
    "info": "",
    "attention": "background: #fff3cd; color: #5c4400; font-weight: bold; padding: 4px;",
    "error": "background: #f8d7da; color: #7a1c24; font-weight: bold; padding: 4px;",
}
ACTION_NEEDED_STYLE = "font-weight: bold; border: 2px solid #c98a00; padding: 3px;"


class _WrappedLabel(QLabel):
    """A word-wrapped label that always keeps the height its text needs at its current width.

    A form layout sizes a wrapped label from a hint computed at the label's preferred width. When the
    text needs one more line at the real width, the next relayout clips it (seen on the container
    desktop, where "Weights found: …" for the default model sits exactly at that threshold).
    """

    def __init__(self, text=""):
        super().__init__(text)
        self.setWordWrap(True)

    def _fit(self):
        if self.width() > 0:
            self.setMinimumHeight(0)  # the current minimum must not count as needed height
            self.setMinimumHeight(self.heightForWidth(self.width()))

    def setText(self, text):
        if text != self.text():  # a periodic refresh with the same text must not disturb the layout
            super().setText(text)
            self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class _AnnotationEdit(QUndoCommand):
    def __init__(self, window, before, after, text, selected=None):
        super().__init__(text)
        self.window, self.before, self.after = window, before, after
        self.before_selection = window.active_organoid
        ids = [o.organoid_id for o in after.organoids]
        desired = selected if selected is not None else self.before_selection
        self.after_selection = desired if desired in ids else (ids[-1] if ids else None)

    def undo(self):
        self.window.set_annotations(self.before, self.before_selection)

    def redo(self):
        self.window.set_annotations(self.after, self.after_selection)


class _LogBridge(QObject):
    message = Signal(str)


class _LogHandler(logging.Handler):
    def __init__(self, bridge):
        super().__init__(logging.INFO)
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    def emit(self, record):
        self.bridge.message.emit(self.format(record))


class MainWindow(QMainWindow):
    def __init__(self, *, settings=None, tracking_factory=None, output_root=None):
        super().__init__()
        self.setWindowTitle("Organoid Tracker")
        self.resize(1320, 900)
        self.preferences = settings if settings is not None else QSettings("OrganoidTracker", "Desktop")
        self.tracking_factory = tracking_factory
        self.output_root = Path(output_root).resolve() if output_root is not None else None
        self.opened: Opened | None = None
        self.annotations = AnnotationSet(())
        self.active_organoid = None
        # The direction under which the current annotations were drawn; boxes belong to that frame.
        self.annotation_direction: str | None = None
        self.selected_cyst: int | None = None
        self._selection_hint = ""
        self._confirmed = False
        self._mapping_key: object = None
        self._box_cache: dict[int, dict[int, tuple[int, int, int, int]]] = {}
        self.operation: Operation | None = None
        self._closing = False
        self.undo_stack = QUndoStack(self)
        self.run_root = Path(str(self.preferences.value("run_root", str(DEFAULT_RUN_ROOT)))).expanduser().absolute()
        self._build_ui()
        self.undo_stack.indexChanged.connect(self._sync)
        # The tracking direction is not a preference: every start uses the paper's reverse workflow.
        for name in ("backend", "family", "model_size", "device"):
            combo = getattr(self, name)
            value = self.preferences.value(name)
            if value is not None:
                index = combo.findText(str(value))
                if index >= 0:
                    combo.setCurrentIndex(index)
        self.checkpoint.setText(str(self.preferences.value("checkpoint", "")))
        geometry = self.preferences.value("geometry")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        self.bridge = _LogBridge(self)
        self.bridge.message.connect(self.log.appendPlainText)
        self.log_handler = _LogHandler(self.bridge)
        logging.getLogger().addHandler(self.log_handler)
        self._sync()

    @property
    def busy(self):
        return self.operation is not None

    def _button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(lambda: self._guard(callback))
        return button

    def _guard(self, callback):
        try:
            callback()
        except (ValueError, OSError, RuntimeError) as error:
            self.show_error(str(error))

    def _build_ui(self):
        toolbar = QToolBar("Files", self)
        self.addToolBar(toolbar)
        self.file_actions = []
        for text, callback in [
            ("Open video", self.choose_video),
            ("Open session", self.choose_session),
            ("Open results", self.choose_results),
            ("Locate source video", self.locate_video),
            ("Save session", self.save_session),
            ("Run folder", self.choose_run_root),
        ]:
            action = QAction(text, self)
            action.triggered.connect(lambda checked=False, callback=callback: self._guard(callback))
            toolbar.addAction(action)
            self.file_actions.append(action)
        self.undo_action = self.undo_stack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut("Ctrl+Z")
        toolbar.addAction(self.undo_action)
        self.redo_action = self.undo_stack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        toolbar.addAction(self.redo_action)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.banner = QLabel(START_TEXT)
        self.banner.setWordWrap(True)
        layout.addWidget(self.banner)
        split = QSplitter()
        layout.addWidget(split, 1)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        self.model_group = QGroupBox("Tracking model")
        form = QFormLayout(self.model_group)
        self.backend = QComboBox()
        self.backend.addItem("SAM 2 (reference)", "sam2")
        self.backend.addItem("SAM 2 (Transformers, experimental)", "hf-sam2")
        self.family = QComboBox()
        self.family.addItems(["2.1", "2"])
        self.model_size = QComboBox()
        for label, value in [
            ("Tiny", "sam2_hiera_t"),
            ("Small", "sam2_hiera_s"),
            ("Base-plus", "sam2_hiera_b"),
            ("Large", "sam2_hiera_l"),
        ]:
            self.model_size.addItem(label, value)
        self.model_size.setCurrentIndex(2)
        self.device = QComboBox()
        self.device.addItems(["cuda", "cpu"])
        self.device.setEditable(True)
        self.direction = QComboBox()
        self.direction.addItem("Last → first", "reverse")
        self.direction.addItem("First → last", "forward")
        self.checkpoint = QLineEdit()
        self.checkpoint.setPlaceholderText("Default checkpoint location")
        self.checkpoint_button = self._button("Browse weights", self.choose_checkpoint)
        self.checkpoint_status = _WrappedLabel()
        for label, widget in [
            ("Backend", self.backend),
            ("Family", self.family),
            ("Size", self.model_size),
            ("Device", self.device),
            ("Direction", self.direction),
            ("Weights", self.checkpoint),
        ]:
            form.addRow(label, widget)
        # Always shown, so that annotating does not change the panel's layout.
        self.direction_note = _WrappedLabel(
            "The direction can be changed only while there are no annotations: boxes belong to the "
            "frame they were drawn on."
        )
        form.addRow(self.direction_note)
        form.addRow(self.checkpoint_button)
        form.addRow(self.checkpoint_status)
        side_layout.addWidget(self.model_group)
        self.frame_group = QGroupBox("Acquisition time points")
        selection_form = QFormLayout(self.frame_group)
        self.source_indices = QLineEdit()
        self.source_indices.setToolTip(
            "Chronological source frame numbers, starting at 1. Inspect with the slider before confirming."
        )
        selection_form.addRow("Source frames", self.source_indices)
        self.all_frames_button = self._button("Use every source frame", self.select_all_frames)
        self.daily_preset_button = self._button("7 daily points (11-frame export)", self.select_daily_preset)
        for button in (self.all_frames_button, self.daily_preset_button):
            selection_form.addRow(button)
        # No default: a span nobody entered is not acquisition metadata.
        self.duration = QLineEdit()
        self.duration.setPlaceholderText("e.g. 6")
        self.duration.setToolTip(
            "Days from the first to the last selected frame, spread evenly from day 1 "
            "(6 for daily frames on days 1 to 7). Leave empty when you enter one day per frame."
        )
        self.explicit_times = QLineEdit()
        self.explicit_times.setPlaceholderText("e.g. 1, 2.5, 4")
        self.explicit_times.setToolTip("One chronological day value per selected frame. Replaces the span.")
        selection_form.addRow("First-to-last span (days)", self.duration)
        selection_form.addRow("Days, one per frame", self.explicit_times)
        self.mapping = QTableWidget(2, 0)
        self.mapping.setVerticalHeaderLabels(["Source frame", "Day"])
        self.mapping.horizontalHeader().setVisible(False)
        self.mapping.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.mapping.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.mapping.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mapping.verticalHeader().setDefaultSectionSize(22)
        self.mapping.setFixedHeight(66)
        self.mapping.setToolTip("What will be recorded: each selected source frame and its acquisition day.")
        selection_form.addRow(self.mapping)
        self.confirm_frames_button = self._button("Confirm frames and days", self.confirm_frames)
        selection_form.addRow(self.confirm_frames_button)
        self.selection_status = _WrappedLabel("Open a video to confirm its acquisition time points.")
        selection_form.addRow(self.selection_status)
        side_layout.addWidget(self.frame_group)
        self.analysis_group = QGroupBox("Analysis and export")
        form = QFormLayout(self.analysis_group)
        self.calibration = QLineEdit("1.6934")
        self.quality = QComboBox()
        self.quality.addItems(["original", "mid", "low"])
        self.quality.setCurrentText(str(self.preferences.value("video_quality", "original")))
        self.videos = QCheckBox("Include overlay, mask and side-by-side videos")
        self.videos.setChecked(True)
        self.counts = QLabel("0 organoids · 0 cysts")
        form.addRow("µm / source pixel", self.calibration)
        form.addRow("Video quality", self.quality)
        form.addRow(self.videos)
        form.addRow(self.counts)
        side_layout.addWidget(self.analysis_group)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Annotations"])
        self.tree.itemSelectionChanged.connect(self.select_organoid)
        side_layout.addWidget(self.tree, 1)
        controls = QHBoxLayout()
        self.remove_button = self._button("Remove selected", self.remove_selected)
        self.clear_button = self._button("Clear annotations", self.clear_annotations)
        controls.addWidget(self.remove_button)
        controls.addWidget(self.clear_button)
        side_layout.addLayout(controls)
        self.track_button = self._button("Track and save run", self.start_tracking)
        self.edit_button = self._button("Edit prompts for a new run", self.edit_saved_prompts)
        self.export_button = self._button("Export reports to a new copy", self.export_copy)
        self.report_button = self._button("Open report folder", self.open_report_folder)
        for button in (self.track_button, self.edit_button, self.export_button, self.report_button):
            side_layout.addWidget(button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(350)
        self.side_scroll = scroll
        split.addWidget(scroll)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs, 1)
        viewer = QWidget()
        view_layout = QVBoxLayout(viewer)
        self.canvas = AnnotationCanvas()
        view_layout.addWidget(self.canvas, 1)
        self.canvas.organoid_clicked.connect(self.add_organoid)
        self.canvas.cyst_drawn.connect(self.add_cyst)
        self.canvas.reports_blocked_clicks = True
        self.canvas.blocked_click.connect(self.explain_blocked_click)
        view_controls = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Original", "Overlay", "Mask", "Side by side"])
        self.view_mode.setCurrentText("Overlay")
        self.view_mode.currentTextChanged.connect(self.render_frame)
        view_controls.addWidget(self.view_mode)
        for text, callback in [
            ("Fit", self.canvas.fit_image),
            ("+", lambda: self.canvas.zoom(1.2)),
            ("−", lambda: self.canvas.zoom(1 / 1.2)),
        ]:
            view_controls.addWidget(self._button(text, callback))
        self.frame_label = QLabel()
        view_controls.addWidget(self.frame_label, 1)
        view_layout.addLayout(view_controls)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.render_frame)
        view_layout.addWidget(self.slider)
        self.tabs.addTab(viewer, "Video and masks")
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.tabs.addTab(self.summary, "Run details")
        self.figure = AnnotationCanvas()
        self.figure.editable = False
        plots = QWidget()
        plots_layout = QVBoxLayout(plots)
        self.plot_choices = QComboBox()
        self.plot_choices.currentIndexChanged.connect(self.show_plot)
        plots_layout.addWidget(self.plot_choices)
        plots_layout.addWidget(self.figure, 1)
        self.tabs.addTab(plots, "Figures")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.tabs.addTab(self.log, "Log")
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 940])
        row = QHBoxLayout()
        row.addWidget(self.track_button)
        row.addWidget(self.export_button)
        self.progress = QProgressBar()
        row.addWidget(self.progress, 1)
        self.cancel_button = self._button("Cancel", self.cancel_tracking)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        for combo in (self.backend, self.family, self.model_size):
            combo.currentIndexChanged.connect(self.update_checkpoint_status)
        self.checkpoint.textChanged.connect(self.update_checkpoint_status)
        self.direction.currentIndexChanged.connect(self._direction_changed)
        self.update_checkpoint_status()
        for field in (self.source_indices, self.duration, self.explicit_times):
            field.textChanged.connect(self._sync)

    def _set_banner(self, text, level="info"):
        self.banner.setStyleSheet(BANNER_STYLES[level])
        self.banner.setText(text)
        if level == "info":
            self.statusBar().clearMessage()

    @staticmethod
    def _day_values(resolved, count):
        return tuple(
            resolved.frame_timestamps or [1 + i * resolved.time_lapse_days / max(count - 1, 1) for i in range(count)]
        )

    def _selected_indices(self):
        """Zero-based source indices from the one-based input; a readable error for anything else."""
        if not self.opened:
            raise ValueError("Open a video first.")
        total = self.opened.source.decoded_frames
        example = f"for example 1, 3, 4; this video has frames 1 to {total}"
        try:
            numbers = [int(part.strip()) for part in self.source_indices.text().split(",")]
        except ValueError:
            raise ValueError(f"Source frames: enter frame numbers separated by commas ({example}).") from None
        if any(number < 1 or number > total for number in numbers):
            raise ValueError(f"Source frames: every number must be between 1 and {total}.")
        if any(b <= a for a, b in zip(numbers, numbers[1:], strict=False)):
            raise ValueError("Source frames: list them in increasing order, each frame once.")
        return tuple(number - 1 for number in numbers)

    def _frames_match(self):
        return bool(
            self.opened
            and self.opened.session.video.frame_indices is not None
            and self.source_indices.text().replace(" ", "")
            == ",".join(str(i + 1) for i in self.opened.source.frame_map)
        )

    def _pending_days(self, count=None):
        """The days the inputs describe for ``count`` frames; ``None`` while they are missing or invalid."""
        if not self.opened:
            return None
        count = self.opened.source.n_frames if count is None else count
        try:
            return self._day_values(self.selected_timing().resolve(count), count)
        except ValueError:
            return None

    def _confirmed_days(self):
        if not self.opened or (self.opened.session.video.frame_indices is None and not self.opened.saved):
            return None
        count = self.opened.source.n_frames
        try:
            return self._day_values(self.opened.session.timing.resolve(count), count)
        except ValueError:
            return None  # the recorded timing does not describe these frames

    def _update_mapping(self, editable):
        """Show what would be recorded (source frame and day per time point) and whether it is confirmed."""
        saved = self.opened.saved if self.opened else None
        try:
            indices = self._selected_indices() if self.opened else ()
            problem = ""
        except ValueError as error:
            indices, problem = (), str(error)
        days = self._pending_days(len(indices)) if indices else None
        key = (id(self.opened), indices, days, self._confirmed, bool(self.annotations.organoids), problem, editable)
        if key == self._mapping_key:
            return
        self._mapping_key = key
        self.mapping.setColumnCount(len(indices))
        metrics = self.mapping.fontMetrics()
        for column, index in enumerate(indices):
            texts = (str(index + 1), f"{days[column]:g}" if days else "?")
            for row, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.mapping.setItem(row, column, item)
            # Seven daily points fit without scrolling.
            self.mapping.setColumnWidth(column, max(28, max(metrics.horizontalAdvance(text) for text in texts) + 12))
        if not self.opened:
            status = "Open a video to confirm its acquisition time points."
        elif saved:
            status = (
                "Recorded with the saved run. Changed days or calibration apply to an exported copy; "
                "the saved run is not altered."
            )
        elif problem:
            status = problem
        elif days is None:
            try:
                self.selected_timing().resolve(len(indices))
                status = ""
            except ValueError as error:
                status = str(error)
        elif self._confirmed:
            status = "Confirmed: these frames and days will be recorded."
            if self.annotations.organoids:
                status += " Clear annotations to change the source frames; the days can still be corrected."
        else:
            status = "Not confirmed yet. Check each source frame and its day in the table, then confirm."
            if self._selection_hint:
                status = self._selection_hint + " " + status
        self.selection_status.setText(status)

    def explain_blocked_click(self):
        """Say why a click on the image did nothing, and point at the control that unlocks it."""
        if self.busy:
            return
        if not self.opened:
            message = "Open a video first (toolbar, top left)."
        elif self.opened.saved:
            message = "Saved results are read-only. Choose 'Edit prompts for a new run' to annotate again."
        elif not self._confirmed:
            message = (
                "Annotation is locked until the acquisition time points are confirmed. Check the source frames "
                "and days on the left, then click 'Confirm frames and days'."
            )
            self.side_scroll.ensureWidgetVisible(self.confirm_frames_button)
        else:
            reverse = self.direction.currentData() == "reverse"
            frame = self.opened.source.n_frames if reverse else 1
            message = (
                f"Annotations are drawn on the {'last' if reverse else 'first'} time point. "
                f"Move the frame slider to frame {frame}."
            )
        self._set_banner(message, "attention")

    def _sync(self):
        saved = self.opened.saved if self.opened else None
        editable = bool(self.opened and not saved and not self.busy)
        # Confirmed means: the frames and the days on screen are exactly the recorded ones.
        pending = self._pending_days()
        confirmed = self._frames_match() and (
            bool(saved) or (pending is not None and pending == self._confirmed_days())
        )
        self._confirmed = confirmed
        selection_editable = editable and not self.annotations.organoids
        self.source_indices.setEnabled(selection_editable)
        self.all_frames_button.setEnabled(selection_editable)
        self.daily_preset_button.setEnabled(
            bool(selection_editable and self.opened and self.opened.source.decoded_frames == 11)
        )
        days_enabled = bool(self.opened) and not self.busy
        self.explicit_times.setEnabled(days_enabled)
        self.duration.setEnabled(days_enabled and not self.explicit_times.text().strip())
        self.confirm_frames_button.setEnabled(editable)
        self.confirm_frames_button.setStyleSheet(ACTION_NEEDED_STYLE if editable and not confirmed else "")
        self._update_mapping(editable)
        for action in self.file_actions:
            action.setEnabled(not self.busy)
        self.model_group.setEnabled(not self.busy and not saved)
        # Boxes belong to the frame they were drawn on, so the direction cannot change under them.
        self.direction.setEnabled(not self.annotations.organoids)
        self.analysis_group.setEnabled(not self.busy)
        annotation_frame = (
            (self.opened.source.n_frames - 1 if self.direction.currentData() == "reverse" else 0) if self.opened else 0
        )
        self.canvas.editable = editable and confirmed and self.slider.value() == annotation_frame
        self.tree.setEnabled(not self.busy)
        self.remove_button.setEnabled(editable)
        self.clear_button.setEnabled(editable)
        self.track_button.setEnabled(editable and confirmed and bool(self.annotations.cysts))
        self.edit_button.setEnabled(bool(saved) and not self.busy)
        self.export_button.setEnabled(bool(saved) and not self.busy)
        self.report_button.setEnabled(bool(saved) and not self.busy)
        self.undo_action.setEnabled(editable and self.undo_stack.canUndo())
        self.redo_action.setEnabled(editable and self.undo_stack.canRedo())
        self.cancel_button.setEnabled(
            bool(self.operation is not None and self.operation.action == "track" and not self.operation.cancel.is_set())
        )
        self.slider.setEnabled(bool(self.opened) and not self.busy)
        self.view_mode.setEnabled(bool(saved))

    def _begin(self, action, **inputs):
        if self.busy:
            raise RuntimeError("Wait for the current operation to finish.")
        self.operation = Operation(uuid.uuid4().hex, action, inputs, self, self.tracking_factory)
        self.operation.progress.connect(self._progress)
        self.operation.finished.connect(self._finished)
        self.progress.setRange(0, 0)
        self._set_banner(
            {
                "track": "Preparing model and video…",
                "export": "Exporting a complete copy…",
                "open": "Opening saved results…",
                "preview": "Reading video…",
                "relocate": "Checking source video content…",
            }[action]
        )
        self._sync()
        self.operation.start()

    @Slot(str, str, int, int, str)
    def _progress(self, token, phase, current, total, message):
        if not self.operation or token != self.operation.token:
            return
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(current)
        prefix = "Cancellation requested; saving retained frames. " if self.operation.cancel.is_set() else ""
        self._set_banner(f"{prefix}{phase}: {message}")

    @Slot()
    def _finished(self):
        sender = self.sender()
        if not isinstance(sender, Operation) or sender is not self.operation:
            return
        operation = sender
        self.operation = None
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        if operation.cancelled_without_masks:
            self._set_banner("Cancelled before any masks were kept. Prompts remain ready for a new run.")
        elif operation.error:
            if operation.recovery is not None:
                self._use_saved(operation.recovery)
            self.show_error(
                operation.error
                + (
                    f"\nMasks were saved in {operation.recovery.directory}; reopen or export again."
                    if operation.recovery
                    else ""
                )
            )
        elif isinstance(operation.value, Opened):
            self.use_opened(operation.value)
        else:
            self._use_saved(operation.value.saved_result)
            self._set_banner(
                f"{operation.value.status.capitalize()}: {operation.value.result.frames_done}/{operation.value.result.frames_total} frames. Saved in {operation.value.output_dir}"
            )
        operation.deleteLater()
        self._sync()
        if self._closing:
            self.close()

    def show_error(self, message):
        self._set_banner(message, "error")
        self.log.appendPlainText("ERROR " + message)
        self.statusBar().showMessage(message)

    def _use_saved(self, saved):
        frames = (
            self.opened.frames
            if self.opened
            and self.opened.source.sha256 == saved.video.sha256
            and self.opened.source.frame_map == saved.video.frame_map
            else []
        )
        self.use_opened(Opened(saved.session, saved.video, frames, saved))

    def use_opened(self, opened):
        self.opened = opened
        self.undo_stack.clear()
        self.slider.blockSignals(True)
        self.slider.setRange(0, opened.source.n_frames - 1)
        self.slider.blockSignals(False)
        self.apply_session_fields(opened.session)
        unconfirmed = not opened.saved and opened.session.video.frame_indices is None
        if unconfirmed and not opened.timing_given:
            # A video opened without a session has no acquisition days yet; the user supplies them.
            self.duration.clear()
            self.explicit_times.clear()
        self.source_indices.setText(", ".join(str(i + 1) for i in opened.source.frame_map))
        self._selection_hint = (
            "Imported prompts must be cleared before selecting other frames."
            if unconfirmed and opened.session.annotations.organoids
            else ""
        )
        self._mapping_key = None
        self._box_cache = {}
        self.annotation_direction = None  # taken from the session's direction, which the fields now show
        self.set_annotations(opened.session.annotations)
        self.slider.setValue(opened.source.annotation_frame)
        self.render_frame(fit=True)
        self._set_banner(
            opened.warning
            or f"{opened.source.decoded_frames} decoded frames → {opened.source.n_frames} displayed frames. "
            + (
                "Saved report inputs are preserved; changes apply to an exported copy."
                if opened.saved
                else (
                    "Click an organoid, then drag its cyst boxes. Click again for the next organoid."
                    if opened.session.video.frame_indices is not None
                    else "Next: check the source frames and their days on the left, then click "
                    "'Confirm frames and days'. Annotation unlocks after that."
                )
            ),
            "info" if opened.saved or opened.session.video.frame_indices is not None else "attention",
        )
        self.plot_choices.clear()
        self.figure.set_frame(np.zeros((1, 1, 3), dtype=np.uint8), fit=True)
        if opened.saved:
            if not opened.frames:
                self.videos.setChecked(False)
            self.summary.setPlainText(
                json.dumps(
                    {
                        "status": opened.saved.result.status,
                        "frames_done": opened.saved.result.frames_done,
                        "frames_total": opened.saved.result.frames_total,
                        "directory": str(opened.saved.directory),
                        "video": opened.source.to_document(),
                        "provenance": opened.saved.provenance,
                    },
                    indent=2,
                    default=str,
                )
            )
            for path in sorted(opened.saved.directory.rglob("*.png")):
                self.plot_choices.addItem(path.stem.replace("_", " "), str(path))
        else:
            self.summary.clear()
        self._sync()

    def apply_session_fields(self, session):
        for combo, value in [
            (self.backend, session.tracking.backend),
            (self.model_size, session.tracking.model_config),
            (self.direction, session.tracking.direction),
        ]:
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(value))
            combo.blockSignals(False)
        self.family.setCurrentText(session.tracking.checkpoint_family or "2.1")
        self.device.setCurrentText(session.tracking.device)
        self.checkpoint.setText(str(session.tracking.checkpoint_path or ""))
        self.calibration.setText(str(session.calibration.um_per_pixel))
        times = session.timing.frame_times_days
        self.duration.setText(str(session.timing.time_lapse_days) if times is None else f"{times[-1] - times[0]:g}")
        self.explicit_times.setText(
            ", ".join(str(v) for v in session.timing.frame_times_days) if session.timing.frame_times_days else ""
        )
        self.update_checkpoint_status()

    def selected_calibration(self):
        try:
            value = float(self.calibration.text().strip())
        except ValueError:
            raise ValueError("µm / source pixel: enter a number (for example 1.6934).") from None
        return Calibration(value)

    def selected_timing(self):
        explicit = self.explicit_times.text().strip()
        if explicit:
            try:
                values = tuple(float(v.strip()) for v in explicit.split(","))
            except ValueError:
                raise ValueError(
                    "Days: enter one number per selected frame, separated by commas (for example 1, 2, 4)."
                ) from None
            return Timing(frame_times_days=values)
        span = self.duration.text().strip()
        if not span:
            raise ValueError("Enter the first-to-last span in days, or one day per selected frame.")
        try:
            return Timing(time_lapse_days=float(span))
        except ValueError as error:
            if isinstance(error, SessionError):
                raise
            raise ValueError("First-to-last span: enter a number of days (for example 6).") from None

    def _check_annotation_direction(self):
        if self.annotations.organoids and self.annotation_direction not in (None, self.direction.currentData()):
            where = "last" if self.annotation_direction == "reverse" else "first"
            raise ValueError(
                f"The annotations were drawn on the {where} time point. Clear them before changing the "
                "tracking direction: boxes cannot move to another frame."
            )

    def current_session(self):
        if not self.opened:
            raise ValueError("Open a video first.")
        if self.opened.saved:
            # Re-analysis inputs for an exported copy; the saved run keeps its own.
            timing = self.selected_timing()
            timing.resolve(self.opened.source.n_frames)
        else:
            if self.opened.session.video.frame_indices is None:
                raise ValueError("Confirm source frames and acquisition days before saving or tracking.")
            if not self._frames_match():
                raise ValueError("Confirm the changed source frame selection before saving or tracking.")
            pending = self._pending_days()
            if pending is None or pending != self._confirmed_days():
                self.selected_timing().resolve(self.opened.source.n_frames)  # a readable reason, when there is one
                raise ValueError("Confirm the changed acquisition days before saving or tracking.")
            self._check_annotation_direction()
            timing = self.opened.session.timing  # exactly what was confirmed and is shown per frame
        return Session(
            VideoReference(
                self.opened.source.path,
                self.opened.source.sha256,
                self.opened.source.frame_map,
                self.opened.source.decoded_frames,
            ),
            self.tracking_spec(),
            self.selected_calibration(),
            timing,
            self.annotations,
        )

    def select_all_frames(self):
        if self.opened:
            self.source_indices.setText(", ".join(str(i + 1) for i in range(self.opened.source.decoded_frames)))
            self._selection_hint = "Every source frame selected: enter its days."
            self._mapping_key = None
            self._sync()

    def select_daily_preset(self):
        if self.opened and self.opened.source.decoded_frames == 11:
            self.source_indices.setText("1, 3, 4, 6, 7, 9, 10")
            self.explicit_times.setText("1, 2, 3, 4, 5, 6, 7")
            self._selection_hint = "Preset for the confirmed 11-frame daily export only."
            self._mapping_key = None
            self._sync()

    def confirm_frames(self):
        if not self.opened or self.opened.saved or self.busy:
            raise ValueError("Open a video for a new run before confirming its acquisition time points.")
        source = self.opened.source
        indices = self._selected_indices()
        same_frames = self.opened.session.video.frame_indices is not None and indices == tuple(source.frame_map)
        if self.annotations.organoids and not same_frames:
            raise ValueError(
                "Clear annotations before changing the source frames: boxes cannot move to another observation."
            )
        video = VideoReference(source.path, source.sha256, indices, source.decoded_frames)
        timing = self.selected_timing()
        resolved = timing.resolve(len(indices))
        # Persist the actual day assigned to every selected source frame, even for uniform spacing.
        days = resolved.frame_timestamps or [
            1 + i * resolved.time_lapse_days / max(len(indices) - 1, 1) for i in range(len(indices))
        ]
        session = replace(
            self.opened.session,
            video=video,
            timing=Timing(frame_times_days=tuple(days)),
            tracking=self.tracking_spec(),
            annotations=self.annotations,
        )
        self._selection_hint = ""
        if same_frames:
            # The same observations with corrected days: annotations, undo history and view stay.
            self.opened = replace(self.opened, session=session, timing_given=True)
            self.apply_session_fields(session)
            self._mapping_key = None
            self.render_frame()
            self._set_banner("Acquisition days confirmed for the selected frames.")
            return
        if self.opened.source_frames is None:
            self._begin("preview", session=session)
        else:
            source = replace(
                source,
                frame_map=indices,
                direction=session.tracking.direction,
                annotation_frame=len(indices) - 1 if session.tracking.reverse else 0,
            )
            self.use_opened(
                Opened(
                    session,
                    source,
                    [self.opened.source_frames[i] for i in indices],
                    source_frames=self.opened.source_frames,
                )
            )

    def clear_annotations(self):
        if self.opened and not self.opened.saved and not self.busy:
            self.undo_stack.push(_AnnotationEdit(self, self.annotations, AnnotationSet(()), "Clear annotations"))

    def tracking_spec(self):
        checkpoint = self.checkpoint.text().strip()
        return TrackingSpec(
            backend=self.backend.currentData(),
            direction=self.direction.currentData(),
            model_config=self.model_size.currentData(),
            checkpoint_family=self.family.currentText(),
            device=self.device.currentText(),
            checkpoint_path=Path(checkpoint).expanduser().absolute() if checkpoint else None,
        )

    def set_annotations(self, annotations, selected=None):
        self.annotations = annotations
        if not annotations.organoids:
            self.annotation_direction = None
        elif self.annotation_direction is None:
            self.annotation_direction = self.direction.currentData()
        if self.selected_cyst not in annotations.cyst_ids():
            self.selected_cyst = None
        ids = [o.organoid_id for o in annotations.organoids]
        desired = selected if selected is not None else self.active_organoid
        self.active_organoid = desired if desired in ids else (ids[-1] if ids else None)
        self.tree.blockSignals(True)
        self.tree.clear()
        for organoid in annotations.organoids:
            parent = QTreeWidgetItem([f"Organoid {organoid.organoid_id}"])
            parent.setData(0, Qt.ItemDataRole.UserRole, ("organoid", organoid.organoid_id))
            self.tree.addTopLevelItem(parent)
            for cyst in organoid.cysts:
                child = QTreeWidgetItem([f"Cyst {cyst.cyst_id}"])
                child.setData(0, Qt.ItemDataRole.UserRole, ("cyst", cyst.cyst_id))
                swatch = QPixmap(12, 12)
                swatch.fill(QColor(*cyst_color(cyst.cyst_id)))
                child.setIcon(0, QIcon(swatch))  # the colour of this cyst's mask in results and videos
                parent.addChild(child)
            parent.setExpanded(True)
            if organoid.organoid_id == self.active_organoid:
                self.tree.setCurrentItem(parent)
        self.tree.blockSignals(False)
        self.counts.setText(f"{len(annotations.organoids)} organoids · {len(annotations.cysts)} cysts")
        self.render_frame()
        self._sync()

    def edit_annotations(self, annotations, text, selected=None):
        if not self.canvas.editable:
            return
        self.undo_stack.push(_AnnotationEdit(self, self.annotations, annotations, text, selected))
        self._sync()

    @Slot(float, float)
    def add_organoid(self, x, y):
        identifier = max((o.organoid_id for o in self.annotations.organoids), default=0) + 1
        self.edit_annotations(
            AnnotationSet((*self.annotations.organoids, OrganoidAnnotation(identifier, (x, y)))),
            "Add organoid",
            identifier,
        )

    @Slot(float, float, float, float)
    def add_cyst(self, x1, y1, x2, y2):
        if self.active_organoid is None:
            self.statusBar().showMessage("Click an organoid before drawing its cyst boxes.")
            return
        cyst = CystAnnotation(max(self.annotations.cyst_ids(), default=0) + 1, (x1, y1, x2, y2))
        organs = tuple(
            replace(o, cysts=(*o.cysts, cyst)) if o.organoid_id == self.active_organoid else o
            for o in self.annotations.organoids
        )
        self.edit_annotations(AnnotationSet(organs), "Add cyst")

    def select_organoid(self):
        item = self.tree.currentItem()
        selected_cyst = None
        if item:
            parent = item.parent()
            if parent is not None:
                selected_cyst = item.data(0, Qt.ItemDataRole.UserRole)[1]
                item = parent
            self.active_organoid = item.data(0, Qt.ItemDataRole.UserRole)[1]
        if selected_cyst != self.selected_cyst:
            self.selected_cyst = selected_cyst
            if self.opened and self.opened.saved:
                self.render_frame()  # highlight the selected cyst's mask

    def remove_selected(self):
        item = self.tree.currentItem()
        if not item:
            return
        kind, identifier = item.data(0, Qt.ItemDataRole.UserRole)
        organs = tuple(
            replace(o, cysts=tuple(c for c in o.cysts if kind != "cyst" or c.cyst_id != identifier))
            for o in self.annotations.organoids
            if kind != "organoid" or o.organoid_id != identifier
        )
        self.edit_annotations(AnnotationSet(organs), "Remove annotation")

    def _direction_changed(self):
        # An undo must not bring back boxes that were drawn on the other end of the video.
        self.undo_stack.clear()
        if self.opened and not self.opened.saved:
            self.slider.setValue(self.opened.source.n_frames - 1 if self.direction.currentData() == "reverse" else 0)
            self.render_frame()

    def render_frame(self, *args, fit=False):
        if not self.opened:
            return
        source = self.opened.source
        index = self.slider.value()
        rgb = (
            self.opened.frames[index].copy()
            if self.opened.frames
            else np.zeros((source.height, source.width, 3), dtype=np.uint8)
        )
        saved = self.opened.saved
        labels = []
        if saved and self.view_mode.currentText() != "Original":
            colors = np.zeros_like(rgb)
            foreground = np.zeros(rgb.shape[:2], dtype=bool)
            boxes = self._box_cache.get(index)
            measured: dict[int, tuple[int, int, int, int]] = {}
            for identifier, mask in saved.result.get(index, {}).items():
                pixels = mask.numpy().astype(bool)
                foreground |= pixels
                colors[pixels] = cyst_color(identifier)  # the same colour as in the exported videos
                if boxes is None:
                    box = mask_box(pixels)
                    if box is not None:
                        measured[identifier] = box
            if boxes is None:
                boxes = self._box_cache[index] = measured
            offset = rgb.shape[1] if self.view_mode.currentText() == "Side by side" else 0
            labels = [
                (
                    x + offset,
                    y,
                    width,
                    height,
                    cyst_label(identifier),
                    cyst_color(identifier),
                    identifier == self.selected_cyst,
                )
                for identifier, (x, y, width, height) in sorted(boxes.items())
            ]
            overlay = rgb.copy()
            overlay[foreground] = (rgb[foreground] * 0.6 + colors[foreground] * 0.4).astype(np.uint8)
            rgb = (
                colors
                if self.view_mode.currentText() == "Mask"
                else np.concatenate((rgb, overlay), axis=1)
                if self.view_mode.currentText() == "Side by side"
                else overlay
            )
        self.canvas.set_frame(rgb, fit=fit)
        self.canvas.set_mask_labels(labels)
        annotation_frame = source.n_frames - 1 if self.direction.currentData() == "reverse" else 0
        self.canvas.set_annotations(self.annotations, visible=not saved and index == annotation_frame)
        times = self.opened.session.timing.frame_times_days
        confirmed = saved or self.opened.session.video.frame_indices is not None
        day = (
            times[index]
            if times and confirmed
            else 1 + index * (self.opened.session.timing.time_lapse_days or 0) / max(source.n_frames - 1, 1)
        )
        observed = "" if not saved else (" · tracked" if index in saved.result.tracked_frames else " · not tracked")
        time_label = f"day {day:g}" if confirmed else "acquisition day unconfirmed"
        self.frame_label.setText(
            f"Frame {index + 1}/{source.n_frames} · source {source.frame_map[index] + 1} · {time_label}{observed}"
        )
        self._sync()

    def show_plot(self):
        path = self.plot_choices.currentData()
        if path:
            from PIL import Image

            try:
                with Image.open(path) as image:
                    self.figure.set_frame(np.asarray(image.convert("RGB")), fit=True)
            except (OSError, ValueError) as error:
                self.show_error(f"Cannot display figure: {error}")

    def update_checkpoint_status(self):
        from ..paths import checkpoints_dir

        names = {"sam2_hiera_t": "tiny", "sam2_hiera_s": "small", "sam2_hiera_b": "base_plus", "sam2_hiera_l": "large"}
        size = names[self.model_size.currentData()]
        hf = self.backend.currentData() == "hf-sam2"
        default = (
            checkpoints_dir()
            / ("hf" if hf else "")
            / (
                f"sam{self.family.currentText()}-hiera-{size.replace('_', '-')}"
                if hf
                else f"sam{self.family.currentText()}_hiera_{size}.pt"
            )
        )
        path = Path(self.checkpoint.text().strip()).expanduser() if self.checkpoint.text().strip() else default
        found = (path / "model.safetensors").is_file() if hf else path.is_file()
        self.checkpoint_status.setText(("Weights found: " if found else "Weights missing: ") + str(path))

    def choose_checkpoint(self):
        if self.backend.currentData() == "hf-sam2":
            path = QFileDialog.getExistingDirectory(self, "Local Hugging Face snapshot")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "SAM 2 checkpoint", "", "Checkpoint (*.pt)")
        if path and not self.busy:
            self.checkpoint.setText(path)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", str(self.preferences.value("last_video_dir", "")), "Video (*.mp4 *.avi *.mov)"
        )
        if path and not self.busy:
            self.preferences.setValue("last_video_dir", str(Path(path).parent))
            session = Session(
                VideoReference(Path(path)),
                self.tracking_spec(),
                self.selected_calibration(),
                Timing(time_lapse_days=1.0),  # placeholder, never shown or used: days are entered and confirmed
                AnnotationSet(()),
            )
            self._begin("preview", session=session, timing_given=False)

    def choose_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open session or prompts", "", "JSON (*.json)")
        if path and not self.busy:
            session = load_session(path)
            if not session.video.path.is_file():
                replacement, _ = QFileDialog.getOpenFileName(
                    self,
                    "Locate the original source video for this session",
                    str(self.preferences.value("last_video_dir", "")),
                    "Video (*.mp4 *.avi *.mov)",
                )
                if not replacement or self.busy:
                    return
                # with_video retains the recorded digest; preview verifies the
                # selected content before publishing the imported annotations.
                session = session.with_video(Path(replacement))
            self._begin("preview", session=session)

    def choose_results(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open saved results", str(self.run_root), "Saved result (results.json)"
        )
        if path and not self.busy:
            self._begin("open", path=Path(path))

    def locate_video(self):
        if not self.opened or not self.opened.saved:
            raise ValueError("Open saved results before locating their source video.")
        path, _ = QFileDialog.getOpenFileName(self, "Locate the same source video", "", "Video (*.mp4 *.avi *.mov)")
        if path and not self.busy:
            self._begin("relocate", saved=self.opened.saved, path=Path(path))

    def output_path(self, path):
        path = Path(path).resolve()
        if self.output_root is not None and not path.is_relative_to(self.output_root):
            raise ValueError(f"Save outputs inside {self.output_root}, the persistent results folder.")
        return path

    def choose_run_root(self):
        path = QFileDialog.getExistingDirectory(self, "Parent folder for future runs", str(self.run_root))
        if path and not self.busy:
            self.run_root = self.output_path(path)
            self.preferences.setValue("run_root", str(self.run_root))

    def save_session(self):
        session = self.current_session()
        path, _ = QFileDialog.getSaveFileName(self, "Save session", str(self.output_root or ""), "JSON (*.json)")
        if not path or self.busy:
            return
        stream = QSaveFile(str(self.output_path(path)))
        if not stream.open(QSaveFile.OpenModeFlag.WriteOnly):
            raise OSError(stream.errorString())
        if stream.write(json.dumps(session.to_document(), indent=2).encode("utf-8")) == -1 or not stream.commit():
            raise OSError(stream.errorString())
        self.statusBar().showMessage(f"Session saved: {path}")

    def start_tracking(self):
        session = self.current_session()
        if not session.annotations.cysts:
            raise ValueError("Draw at least one cyst box.")
        output = new_run_directory(session.video.path, uuid.uuid4().hex[:12], self.output_path(self.run_root))
        output.mkdir(parents=True, exist_ok=False)
        self._begin(
            "track",
            session=session,
            output_dir=output,
            videos=self.videos.isChecked(),
            video_quality=self.quality.currentText(),
        )

    def cancel_tracking(self):
        if self.operation and self.operation.action == "track":
            self.operation.cancel.set()
            self._set_banner("Cancelling after the current frame; retained masks will be saved.")
            self._sync()

    def edit_saved_prompts(self):
        if self.opened and self.opened.saved:
            if not self.opened.frames:
                raise ValueError("Reopen the session with its source video before editing prompts.")
            self.use_opened(replace(self.opened, session=self.current_session(), saved=None))

    def export_copy(self):
        if not self.opened or not self.opened.saved:
            return
        session = self.current_session()
        path = QFileDialog.getExistingDirectory(self, "Parent folder for a new export copy", str(self.run_root))
        if not path or self.busy:
            return
        output = new_run_directory(session.video.path, uuid.uuid4().hex[:12], self.output_path(path))
        output.mkdir(parents=True, exist_ok=False)
        self._begin(
            "export",
            saved=self.opened.saved,
            output_dir=output,
            videos=self.videos.isChecked(),
            video_quality=self.quality.currentText(),
            calibration=session.calibration,
            timing=session.timing,
            video_path=session.video.path,
        )

    def open_report_folder(self):
        if self.opened and self.opened.saved:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.opened.saved.directory)))

    def closeEvent(self, event):
        if self.busy:
            self._closing = True
            self.cancel_tracking()
            self._set_banner("Closing after the operation finishes and retained results are saved…")
            event.ignore()
            return
        self.preferences.setValue("geometry", self.saveGeometry())
        self.preferences.setValue("video_quality", self.quality.currentText())
        for name in ("backend", "family", "model_size", "device"):
            self.preferences.setValue(name, getattr(self, name).currentText())
        self.preferences.remove("direction")
        self.preferences.setValue("checkpoint", self.checkpoint.text())
        logging.getLogger().removeHandler(self.log_handler)
        self.log_handler.close()
        event.accept()
