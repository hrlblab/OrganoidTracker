#!/usr/bin/env python3
"""
Main GUI Window for Multi-Model Video Object Tracking
Built with Tkinter for beginner-friendly GUI development
"""

import logging
import queue
import sys
import threading
import time
import tkinter as tk
import uuid
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from ..config import (
    AUTO_OPEN_OUTPUT_DIRECTORY,
    DEFAULT_CONVERSION_FACTOR,
    DEFAULT_TIME_LAPSE_DAYS,
    GUI_LOG_LEVEL,
    SAM2_CHECKPOINT_FAMILY,
)
from ..core.model_registry import get_model_registry
from ..core.sam2_tracker import checkpoint_filename
from ..services.analysis_service import AnalysisService
from ..services.annotations import AnnotationError, AnnotationSet, CystAnnotation
from ..services.export_service import ExportError, ExportService
from ..services.prompt_record import build_prompt_record
from ..services.saved_results import SavedResult, load_saved_result
from ..services.session import Calibration, SessionError, Timing, TrackingSpec, session_from_document
from ..services.tracking_service import TrackingError, TrackingService
from ..services.video_frames import load_video_frames
from .progress_dialog import ProgressDialog
from .video_canvas import VideoCanvas

logger = logging.getLogger(__name__)


@dataclass
class OpenedResult:
    """A saved run reopened without a model: its result and, when its video was found, the frames."""

    saved: SavedResult
    frames: list | None


class LogPanelHandler(logging.Handler):
    """Shows application log records in the GUI log panel; safe to call from worker threads."""

    def __init__(self, app: "VideoTrackerApp", level: str = GUI_LOG_LEVEL):
        super().__init__(level=logging.getLevelNamesMapping().get(str(level).upper(), logging.WARNING))
        self.app = app
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.app.post(self.app.log_event, self.format(record))
        except Exception:  # the Tk main loop has gone away
            pass


class VideoTrackerApp:
    """
    Main GUI window for video object tracking

    Features:
    - Model selection (SAM2 families, future models)
    - Video loading and preview
    - Click-based object prompting
    - Progress tracking
    - Results viewing
    """

    def __init__(self):
        """Initialize the main window"""
        self.root = tk.Tk()
        self.root.title("Multi-Model Video Object Tracker")

        # Set proper minimum size for 720p compatibility
        self.root.minsize(1280, 720)
        self.root.geometry("1400x900")

        # Make window resizable and properly scalable
        self.root.state("zoomed") if sys.platform == "win32" else None

        # Enable window resizing
        self.root.resizable(True, True)

        # Application state: one tracking service owns the backend (model, video, prompts, run)
        self.tracking: TrackingService | None = None
        self.current_video_path: str | None = None
        self.video_segments: dict | None = None
        self.tracking_in_progress = False
        self.opened: OpenedResult | None = None  # a saved run reopened without a model (Open Results...)
        self.tracking_run_id: str | None = None  # identifies the results shown when they are saved
        self.result_video = None  # the VideoSource the live results belong to (the backend's at tracking time)
        self.result_provenance: dict | None = None  # the backend's provenance at tracking time
        self._results_serial = 0  # bumped whenever the shown results change (ids of objects can be reused)
        self._cancel_request: threading.Event | None = None  # the running tracking's cancel request

        # Progress dialog references
        self.tracking_dialog = None
        self.generation_dialog = None

        # Analysis parameter variables (simplified - no manual organoid input needed)
        self.time_lapse_var = tk.DoubleVar(value=7.0)  # Default: 7 days
        self.conversion_factor_var = tk.DoubleVar(value=1.0)  # Default: 1.0 μm/pixel

        # New organoid-cyst tracking state
        self.current_organoid_id = None  # Currently selected organoid
        self.organoid_mode = True  # True = waiting for organoid click, False = adding cysts
        self.organoid_data = {}  # organoid_id -> {'point': (x,y), 'cysts': [(x1,y1,x2,y2), ...]}
        self.next_organoid_id = 1
        self.next_cyst_id = 1

        # Configuration options (config.py, with user_config.py overrides)
        self.auto_open_directory = AUTO_OPEN_OUTPUT_DIRECTORY
        self.time_lapse_var.set(DEFAULT_TIME_LAPSE_DAYS)
        self.conversion_factor_var.set(DEFAULT_CONVERSION_FACTOR)

        # Worker threads never touch Tk: they post callbacks here and the main loop drains the queue.
        self._ui_queue: queue.Queue = queue.Queue()
        self._ui_after_id = self.root.after(self.UI_POLL_MS, self._drain_ui_queue)
        self.root.bind("<Destroy>", self._on_destroy, add="+")

        # GUI setup
        self.setup_styles()
        self.create_widgets()
        self.setup_layout()
        self.setup_bindings()

        # Initialize model list and set initial checkpoint display
        self.update_model_list()
        self.on_model_size_changed()  # Set initial checkpoint display

        # Application log records (warnings and errors by default) also go to the log panel
        self.log_handler = LogPanelHandler(self)
        logging.getLogger("organoidtracker").addHandler(self.log_handler)

        # Status
        self.set_status("Select a model and load it to begin.")

    UI_POLL_MS = 50

    def post(self, callback, *args) -> None:
        """Run ``callback(*args)`` on the GUI thread; safe to call from any thread."""
        self._ui_queue.put((callback, args))

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                callback, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args)
            except Exception as error:  # a failing callback must not stop the loop
                logger.error(f"GUI callback {getattr(callback, '__name__', callback)} failed: {error}")
        self._ui_after_id = self.root.after(self.UI_POLL_MS, self._drain_ui_queue)

    def _on_destroy(self, event) -> None:
        """Stop the queue poll when the window goes away (a pending ``after`` on a destroyed root errors)."""
        if event.widget is self.root and self._ui_after_id is not None:
            try:
                self.root.after_cancel(self._ui_after_id)
            except tk.TclError:
                pass
            self._ui_after_id = None

    @property
    def current_model(self):
        """The loaded backend, or None (read-only view; the window acts through ``self.tracking``)."""
        return self.tracking.tracker if self.tracking is not None else None

    def setup_styles(self):
        """Setup custom styles for the GUI"""
        self.style = ttk.Style()

        # Configure styles for better appearance
        self.style.configure("Title.TLabel", font=("Arial", 14, "bold"))
        self.style.configure("Header.TLabel", font=("Arial", 10, "bold"))
        self.style.configure("Status.TLabel", font=("Arial", 9))
        self.style.configure("VideoLoad.TButton", font=("Arial", 10, "bold"))
        self.style.configure("Action.TButton", font=("Arial", 10))

        # Error style for analysis input validation (MobaXterm-friendly)
        try:
            self.style.configure("Error.TEntry", fieldbackground="#ffe6e6", bordercolor="red")
        except Exception:
            # Fallback if style configuration fails in MobaXterm
            pass

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main title
        self.title_label = ttk.Label(self.root, text="Multi-Model Video Object Tracker", style="Title.TLabel")

        # Step 1: Model selection frame
        self.model_frame = ttk.LabelFrame(self.root, text="Step 1: Model Configuration", padding=15)

        self.model_label = ttk.Label(self.model_frame, text="Model:", style="Header.TLabel")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self.model_frame, textvariable=self.model_var, state="readonly", width=25)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)

        self.load_model_btn = ttk.Button(self.model_frame, text="Load Model", command=self.load_selected_model)

        # Model configuration options
        self.config_frame = ttk.LabelFrame(self.model_frame, text="Model Settings", padding=10)

        # Device selection
        self.device_label = ttk.Label(self.config_frame, text="Device:")
        self.device_var = tk.StringVar(value="cuda")
        self.device_combo = ttk.Combobox(
            self.config_frame, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly", width=15
        )

        # Model config selection
        self.model_config_label = ttk.Label(self.config_frame, text="Model Size:")
        self.model_config_var = tk.StringVar(value="sam2_hiera_b")
        self.model_config_combo = ttk.Combobox(
            self.config_frame,
            textvariable=self.model_config_var,
            values=["sam2_hiera_s", "sam2_hiera_b", "sam2_hiera_l"],
            state="readonly",
            width=15,
        )
        self.model_config_combo.bind("<<ComboboxSelected>>", self.on_model_size_changed)

        # Debug mode toggle
        self.debug_label = ttk.Label(self.config_frame, text="Debug Mode:")
        self.debug_var = tk.BooleanVar(value=False)
        self.debug_check = ttk.Checkbutton(self.config_frame, text="Enable Debug Output", variable=self.debug_var)

        # Reverse tracking toggle
        self.reverse_label = ttk.Label(self.config_frame, text="Temporal Direction:")
        self.reverse_var = tk.BooleanVar(value=True)  # Default to reverse (biological use case)
        self.reverse_check = ttk.Checkbutton(
            self.config_frame, text="Enable Reverse Tracking (last→first frame)", variable=self.reverse_var
        )

        # Video quality selection
        self.quality_label = ttk.Label(self.config_frame, text="Output Quality:")
        self.quality_var = tk.StringVar(value="original")
        self.quality_combo = ttk.Combobox(
            self.config_frame,
            textvariable=self.quality_var,
            values=["original", "mid", "low"],
            state="readonly",
            width=15,
        )

        # Checkpoint info (read-only display)
        self.checkpoint_label = ttk.Label(self.config_frame, text="Checkpoint:")
        self.checkpoint_info = ttk.Label(self.config_frame, text="sam2.1_hiera_small.pt", style="Status.TLabel")

        # Step 2: Video loading frame
        self.video_frame = ttk.LabelFrame(self.root, text="Step 2: Video Loading", padding=15)

        self.load_video_btn = ttk.Button(
            self.video_frame,
            text="📹 Load Video",
            command=self.load_video,
            state="disabled",
            width=18,
            style="VideoLoad.TButton",
        )

        # A saved run (results.json and its mask file) opens without a model, for viewing and exporting
        self.open_results_btn = ttk.Button(
            self.video_frame, text="📂 Open Results...", command=self.open_results, width=18
        )

        self.video_info_label = ttk.Label(
            self.video_frame, text="No video loaded", style="Status.TLabel", wraplength=250
        )

        # Step 3: Video display and interaction frame
        self.display_frame = ttk.LabelFrame(self.root, text="Step 3: Multi-Object Video Tracking", padding=10)

        self.video_canvas = VideoCanvas(self.display_frame, width=800, height=600)
        self.video_canvas.set_bbox_callback(self.on_canvas_bbox)
        self.video_canvas.set_click_callback(self.on_canvas_click)

        # Object selection frame
        self.object_frame = ttk.Frame(self.display_frame)

        # Object management controls
        self.object_mgmt_frame = ttk.Frame(self.object_frame)
        self.object_label = ttk.Label(self.object_mgmt_frame, text="Objects:", style="Header.TLabel")

        # Object management settings
        self.max_objects = 20  # Maximum number of objects
        self.active_object_ids = set()  # No background by default - separate feature
        self.action_history = []  # Stack for multi-level undo (like Ctrl+Z)
        self.next_object_id = 1  # Track next ID to assign

        # Object colors - expanded palette for 20+ objects
        self.object_colors = {
            0: "#808080",  # Gray - Background
            1: "#FF0000",  # Red
            2: "#00FF00",  # Green
            3: "#0000FF",  # Blue
            4: "#FFFF00",  # Yellow
            5: "#FF00FF",  # Magenta
            6: "#00FFFF",  # Cyan
            7: "#FFA500",  # Orange
            8: "#800080",  # Purple
            9: "#FFC0CB",  # Pink
            10: "#A52A2A",  # Brown
            11: "#90EE90",  # Light Green
            12: "#87CEEB",  # Sky Blue
            13: "#DDA0DD",  # Plum
            14: "#F0E68C",  # Khaki
            15: "#FF6347",  # Tomato
            16: "#40E0D0",  # Turquoise
            17: "#EE82EE",  # Violet
            18: "#FFB6C1",  # Light Pink
            19: "#98FB98",  # Pale Green
            20: "#F5DEB3",  # Wheat
        }

        # Object management - simplified (add/remove through canvas clicks and revert button)

        # Object display (no dropdown needed - automatic sequential numbering)

        # New organoid-cyst workflow section
        self.workflow_frame = ttk.LabelFrame(self.object_frame, text="Organoid-Cyst Workflow", padding=5)

        # Workflow status display
        self.workflow_status_label = ttk.Label(
            self.workflow_frame, text="Click on an organoid location", font=("Arial", 10, "bold")
        )

        # Simplified workflow - no need for next/finish buttons
        # Users can simply click on the next organoid when ready

        # Active objects display (scrollable for many objects)
        self.active_objects_frame = ttk.Frame(self.object_frame)
        self.active_objects_label = ttk.Label(
            self.active_objects_frame, text="Active Objects: None", style="Status.TLabel"
        )

        # Object list (scrollable text widget for many objects)
        self.object_list_text = tk.Text(
            self.active_objects_frame, height=3, width=50, state="disabled", wrap="word", font=("Arial", 9)
        )

        # Initialize object list
        self.update_active_objects_display()

        # Controls frame
        self.controls_frame = ttk.LabelFrame(self.display_frame, text="Tracking Controls", padding=10)

        self.clear_prompts_btn = ttk.Button(
            self.controls_frame, text="Clear All Objects", command=self.clear_prompts, state="disabled"
        )

        self.revert_btn = ttk.Button(
            self.controls_frame, text="/Revert Last", command=self.revert_last_action, state="disabled"
        )

        self.track_btn = ttk.Button(
            self.controls_frame,
            text="🚀 Start Multi-Object Tracking",
            command=self.start_tracking,
            state="disabled",
            width=25,
        )

        # Prompt info
        self.prompt_info_label = ttk.Label(
            self.controls_frame,
            text="Left click anywhere on the video to add a new object\nNumbers will appear to identify each object",
            style="Status.TLabel",
        )

        # Step 4: Output frame (video generation)
        self.output_frame = ttk.LabelFrame(self.root, text="Step 4: Video Output", padding=10)

        # Video generation button
        self.generate_btn = ttk.Button(
            self.output_frame, text="🎬 Generate Videos", command=self.generate_videos, state="disabled"
        )
        self.save_results_btn = ttk.Button(
            self.output_frame, text="💾 Save Results...", command=self.save_results, state="disabled"
        )

        # Step 5: Analysis frame (separate section)
        self.analysis_frame = ttk.LabelFrame(self.root, text="Step 5: Organoid Cyst Analysis", padding=10)

        # Analysis parameters sub-frame
        self.analysis_params_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Parameters", padding=5)

        # Analysis parameter inputs (simplified - organoid count detected automatically)
        self.time_lapse_label = ttk.Label(self.analysis_params_frame, text="Time Lapse (days, first to last frame):")
        self.time_lapse_entry = ttk.Entry(
            self.analysis_params_frame, width=10, textvariable=self.time_lapse_var, font=("Arial", 10)
        )

        self.conversion_factor_label = ttk.Label(self.analysis_params_frame, text="Conversion Factor (μm/pixel):")
        self.conversion_factor_entry = ttk.Entry(
            self.analysis_params_frame, width=15, textvariable=self.conversion_factor_var, font=("Arial", 10)
        )

        # Organoid count display (auto-detected)
        self.organoid_count_label = ttk.Label(self.analysis_params_frame, text="Detected Organoids:")
        self.organoid_count_display = ttk.Label(self.analysis_params_frame, text="0", font=("Arial", 10, "bold"))

        # Add validation for better user experience (especially with MobaXterm)
        self.time_lapse_entry.bind("<KeyRelease>", self._validate_analysis_inputs)
        self.conversion_factor_entry.bind("<KeyRelease>", self._validate_analysis_inputs)

        # Add focus events for visual feedback
        self.time_lapse_entry.bind("<FocusIn>", lambda e: self._on_analysis_entry_focus(e, "time"))
        self.conversion_factor_entry.bind("<FocusIn>", lambda e: self._on_analysis_entry_focus(e, "conversion"))

        # Analysis report button
        self.analysis_btn = ttk.Button(
            self.analysis_frame,
            text="📊 Generate Analysis Report",
            command=self.generate_analysis_report,
            state="disabled",
        )

        # Output and results information text area (shared between output and analysis)
        self.results_frame = ttk.LabelFrame(self.root, text="Results & Log", padding=10)
        self.output_info_text = scrolledtext.ScrolledText(self.results_frame, height=8, width=50, state="disabled")

        # Status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_label = ttk.Label(self.status_frame, text="Ready", style="Status.TLabel", relief="sunken")

        # Progress bar (initially hidden)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_frame, variable=self.progress_var, mode="determinate")

    def setup_layout(self):
        """Setup the scalable layout of all widgets with proper step organization"""
        # Main title
        self.title_label.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")

        # Left column - Steps 1 & 2
        self.model_frame.grid(row=1, column=0, padx=5, pady=5, sticky="new")
        self.video_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # Middle column - Step 3 (main tracking area)
        self.display_frame.grid(row=1, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")

        # Right column - Steps 4 & 5
        self.output_frame.grid(row=1, column=2, padx=5, pady=5, sticky="new")
        self.analysis_frame.grid(row=2, column=2, padx=5, pady=5, sticky="new")

        # Bottom section - Results and status (spans all columns)
        self.results_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.status_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=2)

        # Model frame layout
        self.model_label.grid(row=0, column=0, sticky="w", pady=2)
        self.model_combo.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.load_model_btn.grid(row=0, column=2, padx=5, pady=2)
        self.config_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky="ew")

        # Configuration frame layout
        self.device_label.grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.device_combo.grid(row=0, column=1, padx=5, pady=2)

        self.model_config_label.grid(row=0, column=2, sticky="w", padx=10, pady=2)
        self.model_config_combo.grid(row=0, column=3, padx=5, pady=2)

        # Row 1: Debug and Quality controls
        self.debug_label.grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.debug_check.grid(row=1, column=1, padx=5, pady=2, sticky="w")

        self.quality_label.grid(row=1, column=2, sticky="w", padx=10, pady=2)
        self.quality_combo.grid(row=1, column=3, padx=5, pady=2)

        # Row 3: Reverse tracking controls
        self.reverse_label.grid(row=3, column=0, sticky="w", padx=2, pady=2)
        self.reverse_check.grid(row=3, column=1, columnspan=2, padx=5, pady=2, sticky="w")

        # Row 4: Checkpoint info
        self.checkpoint_label.grid(row=4, column=0, sticky="w", padx=2, pady=2)
        self.checkpoint_info.grid(row=4, column=1, columnspan=3, padx=5, pady=2, sticky="ew")

        self.model_frame.columnconfigure(1, weight=1)
        self.config_frame.columnconfigure(3, weight=1)

        # Video frame layout
        self.load_video_btn.grid(row=0, column=0, pady=5, sticky="ew")
        self.open_results_btn.grid(row=1, column=0, pady=5, sticky="ew")
        self.video_info_label.grid(row=2, column=0, pady=10, sticky="new")

        # Make video frame expand properly
        self.video_frame.columnconfigure(0, weight=1)

        # Display frame layout
        self.video_canvas.grid(row=0, column=0, columnspan=2, pady=5)
        self.object_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        self.controls_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10, padx=5)

        # Object management layout - simplified
        self.object_mgmt_frame.grid(row=0, column=0, sticky="w", pady=2)
        self.object_label.grid(row=0, column=0, sticky="w", padx=5)

        # Workflow layout - simplified
        self.workflow_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=5)
        self.workflow_status_label.grid(row=0, column=0, padx=5, pady=2)

        self.active_objects_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        self.active_objects_label.grid(row=0, column=0, sticky="w")
        self.object_list_text.grid(row=1, column=0, sticky="ew", pady=2)

        # Configure frame weights for proper expansion
        self.object_frame.columnconfigure(0, weight=1)
        self.active_objects_frame.columnconfigure(0, weight=1)

        # CRITICAL FIX: Configure display_frame rows to ensure controls are visible
        self.display_frame.rowconfigure(0, weight=2)  # Video canvas - main content
        self.display_frame.rowconfigure(1, weight=1)  # Object management
        self.display_frame.rowconfigure(2, weight=0)  # Controls - fixed height
        self.display_frame.columnconfigure(0, weight=1)

        # Controls layout
        self.clear_prompts_btn.grid(row=0, column=0, padx=8, pady=5)
        self.revert_btn.grid(row=0, column=1, padx=8, pady=5)
        self.track_btn.grid(row=0, column=2, padx=8, pady=5)
        self.prompt_info_label.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        self.controls_frame.columnconfigure(0, weight=1)
        self.controls_frame.columnconfigure(1, weight=1)
        self.controls_frame.columnconfigure(2, weight=1)

        # Step 4: Output frame layout (simple video generation)
        self.generate_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.save_results_btn.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.output_frame.columnconfigure(0, weight=1)

        # Step 5: Analysis frame layout
        self.analysis_params_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.analysis_btn.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Analysis parameters layout (simplified)
        self.organoid_count_label.grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.organoid_count_display.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        self.time_lapse_label.grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.time_lapse_entry.grid(row=1, column=1, padx=5, pady=2)

        self.conversion_factor_label.grid(row=2, column=0, sticky="w", padx=2, pady=2)
        self.conversion_factor_entry.grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # Configure analysis frame weights
        self.analysis_frame.columnconfigure(0, weight=1)
        self.analysis_params_frame.columnconfigure(1, weight=1)

        # Results frame layout
        self.output_info_text.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)

        # Status frame layout
        self.status_label.grid(row=0, column=0, sticky="ew", padx=2)
        self.progress_bar.grid(row=0, column=1, sticky="ew", padx=2)

        self.status_frame.columnconfigure(0, weight=1)

        # Configure grid weights for proper 3-column scalable layout
        self.root.columnconfigure(0, weight=1)  # Left column (Steps 1-2)
        self.root.columnconfigure(1, weight=4)  # Middle column (Step 3 - main tracking area)
        self.root.columnconfigure(2, weight=1)  # Right column (Steps 4-5)

        self.root.rowconfigure(0, weight=0)  # Title - fixed height
        self.root.rowconfigure(1, weight=1)  # Main content row 1
        self.root.rowconfigure(2, weight=1)  # Main content row 2
        self.root.rowconfigure(3, weight=0)  # Results frame - expandable but controlled
        self.root.rowconfigure(4, weight=0)  # Status frame - fixed height

        # Bind window resize for auto-zoom canvas
        self.root.bind("<Configure>", self.on_window_resize)

    def setup_bindings(self):
        """Setup event bindings"""
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)

        # Ensure window focuses on video loading button when model is loaded
        self.root.bind("<Button-1>", self.ensure_focus)

        # Debug shortcut - Press Ctrl+D to check button states
        self.root.bind("<Control-d>", lambda e: self.debug_button_states())

    def on_window_resize(self, event=None):
        """Handle window resize events to maintain widget visibility"""
        if event and event.widget == self.root:
            # Ensure minimum height for all widgets to be visible
            current_height = self.root.winfo_height()
            if current_height < 700:
                self.root.geometry(f"{self.root.winfo_width()}x700")

    def ensure_focus(self, event=None):
        """Ensure the main window has focus for proper event handling"""
        self.root.focus_set()

    def update_model_list(self):
        """Update the model selection dropdown"""
        try:
            registry = get_model_registry()
            available_models = registry.get_available_models()

            model_options = []
            for model in available_models:
                model_options.append(f"{model.display_name} ({model.name})")

            self.model_combo["values"] = model_options

            if model_options:
                self.model_combo.current(0)  # Select first model
                self.on_model_selected()
            else:
                self.set_status("No models available! Check installation.")

        except Exception as e:
            self.set_status(f"Error loading models: {str(e)}")

    def on_model_size_changed(self, event=None):
        """Update checkpoint display when model size changes"""
        model_config = self.model_config_var.get()
        try:
            checkpoint_file = checkpoint_filename(model_config, SAM2_CHECKPOINT_FAMILY)
        except Exception:
            checkpoint_file = "sam2.1_hiera_small.pt"
        self.checkpoint_info.config(text=checkpoint_file)

    def on_model_selected(self, event=None):
        """Handle model selection - simplified without description"""
        selected = self.model_var.get()
        if selected:
            # Extract model name for status update
            model_name = selected.split("(")[0].strip()
            self.set_status(f"Selected {model_name}. Configure settings and click 'Load Model'.")

    def load_selected_model(self):
        """Load the selected model with user configuration"""
        selected = self.model_var.get()
        if not selected:
            self.set_status("Please select a model first")
            self.log_event("❌ No model selected")
            return

        # Get user configuration
        device = self.device_var.get()
        model_config = self.model_config_var.get()

        # Let the model's auto-detection handle checkpoint selection
        # Each model knows its own correct checkpoint paths and filenames
        checkpoint_path = None  # This will trigger auto-detection in the model

        try:
            import time

            start_time = time.time()

            self.set_status("Loading model... Please wait.")
            self.load_model_btn.config(state="disabled")
            self.log_event(f"🔄 Loading {selected} with {device.upper()} device...")
            enable_reverse = self.reverse_var.get()  # read Tk variables on the GUI thread

            # Use threading to prevent GUI freeze
            def load_model_thread():
                try:
                    spec = TrackingSpec(
                        direction="reverse" if enable_reverse else "forward",
                        model_config=model_config,
                        checkpoint_path=checkpoint_path,
                        device=device,
                    )
                    service = TrackingService.create(spec)
                    service.load_model()
                    load_time = time.time() - start_time
                    self.post(self.on_model_loaded_success, service, load_time)

                except (TrackingError, SessionError) as e:
                    load_time = time.time() - start_time
                    self.post(self.on_model_loaded_error, str(e), load_time)
                except Exception as e:
                    load_time = time.time() - start_time
                    self.post(self.on_model_loaded_error, str(e), load_time)

            threading.Thread(target=load_model_thread, daemon=True).start()

        except Exception as e:
            load_time = time.time() - start_time
            self.on_model_loaded_error(str(e), load_time)

    def on_model_loaded_success(self, service, load_time):
        """Install the loaded backend (GUI thread) and discard whatever the previous one produced."""
        previous = self.tracking
        self.tracking = service
        if previous is not None:
            self._discard_downstream_state("a new model replaced the previous one")
        self.load_model_btn.config(state="normal")
        self.load_video_btn.config(state="normal")

        # Check tracking settings and show feedback
        direction_status = "reverse" if self.reverse_var.get() else "forward"
        self.set_status(f"Model loaded successfully with {direction_status} tracking! Ready to load video.")

        # Get model name for more specific logging
        selected = self.model_var.get()
        registry = get_model_registry()
        metadata = registry.get_model_metadata(selected)
        model_name = metadata.display_name if metadata else selected

        direction_icon = "⏪" if self.reverse_var.get() else "⏩"
        self.log_event(
            f"✅ {model_name} loaded successfully in {load_time:.2f}s ({direction_icon} {direction_status} tracking)"
        )

    def _discard_downstream_state(self, reason: str) -> None:
        """Forget the video, annotations and results of the previous backend; disable what depends on them.

        A new backend holds no video, so results and prompts of the old one must not look usable.
        """
        had_results = self.video_segments is not None or self.current_video_path is not None or bool(self.organoid_data)
        self.current_video_path = None
        self.video_segments = None
        self.opened = None
        self.tracking_run_id = None
        self.result_video = None
        self.result_provenance = None
        if self.tracking_in_progress and self.tracking_dialog is not None:
            self.tracking_dialog.close()  # the run of the replaced backend is over for the window; its callback is stale
            self.tracking_dialog = None
        self.tracking_in_progress = False
        self._results_changed()
        self.video_canvas.clear_markers()
        self.video_canvas.show_placeholder()
        self.organoid_data.clear()
        self.active_object_ids.clear()
        self.action_history.clear()
        self.next_organoid_id = 1
        self.next_cyst_id = 1
        self.current_organoid_id = None
        self.organoid_mode = True
        for button in (
            self.track_btn,
            self.clear_prompts_btn,
            self.revert_btn,
            self.generate_btn,
            self.analysis_btn,
            self.save_results_btn,
        ):
            button.config(state="disabled")
        self.video_info_label.config(text="No video loaded")
        self.update_active_objects_display()
        self.update_organoid_count_display()
        self.update_workflow_status()
        if had_results:
            self.log_event(
                f"🔁 {reason}: the previous video, annotations and results were discarded; load a video to continue"
            )

    def _ready_for_export(self, what: str) -> bool:
        """Results and the video they belong to are needed: a reopened run's, or this backend's.

        Live results are tied to the video the backend held when they were produced; once another
        video is loaded through the backend (even the same file again) they are no longer exportable.
        """
        if self.tracking_in_progress:
            self.set_status("Tracking is running; wait for it to finish")
            self.log_event(f"❌ Cannot start {what}: tracking is in progress")
            return False
        if self.video_segments is None or not self.video_segments:
            self.set_status("No tracking results available. Please run tracking first")
            self.log_event(f"❌ No tracking results available for {what}")
            return False
        if self.opened is not None:
            return True
        if (
            self.tracking is None
            or self.tracking.video is None
            or self.result_video is None
            or self.tracking.video is not self.result_video
        ):
            self.set_status("Load a video and run tracking first")
            self.log_event(f"❌ Cannot start {what}: the loaded video is not the one the results were tracked on")
            return False
        return True

    def on_model_loaded_error(self, error_msg, load_time):
        """Handle model loading error"""
        self.load_model_btn.config(state="normal")
        self.set_status(f"Error loading model: {error_msg}")
        self.log_event(f"❌ Model loading failed after {load_time:.2f}s: {error_msg}")
        # Remove popup - just use status and log (already handled above)

    def log_event(self, message):
        """Log an event with timestamp to the results area"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        log_message = f"[{timestamp}] {message}\n"

        self.output_info_text.config(state="normal")
        self.output_info_text.insert(tk.END, log_message)
        self.output_info_text.see(tk.END)  # Auto-scroll to bottom
        self.output_info_text.config(state="disabled")

    def load_video(self):
        """Load a video file"""
        if not self.tracking:
            self.set_status("Please load a model first")
            self.log_event("❌ No model loaded")
            return

        # File dialog for video selection
        file_types = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("MP4 files", "*.mp4"), ("All files", "*.*")]

        file_path = filedialog.askopenfilename(
            title="Select a video file", filetypes=file_types, initialdir="./data/input_videos"
        )

        if not file_path:
            return
        if self.tracking_in_progress:  # the chooser is modal: tracking may have started meanwhile
            self.set_status("Tracking is running; cancel it or wait for it to finish before loading a video")
            return

        # The backend is about to hold another video: results tracked on the previous one (or reopened
        # ones) must not be exported, saved or viewed under the new video's identity
        self._close_opened_result()
        self._invalidate_results("a new video is being loaded")

        try:
            import time

            start_time = time.time()

            self.set_status("Loading video... Please wait.")
            self.load_video_btn.config(state="disabled")
            self.log_event(f"🎬 Loading video: {Path(file_path).name}")

            # Load video in thread to prevent GUI freeze
            def load_video_thread():
                try:
                    source = self.tracking.open_video(file_path)
                    video_info = {
                        "num_frames": source.n_frames,
                        "decoded_frames": source.decoded_frames,
                        "duplicate_frames_removed": source.duplicate_frames_removed,
                        "fps": source.nominal_fps,
                        "dimensions": (source.height, source.width),
                        "direction": source.direction,
                    }
                    load_time = time.time() - start_time
                    self.post(self.on_video_loaded_success, file_path, video_info, load_time)
                except Exception as e:
                    load_time = time.time() - start_time
                    self.post(self.on_video_loaded_error, str(e), load_time)

            threading.Thread(target=load_video_thread, daemon=True).start()

        except Exception as e:
            load_time = time.time() - start_time
            self.on_video_loaded_error(str(e), load_time)

    def on_video_loaded_success(self, file_path, video_info, load_time):
        """Handle successful video loading"""
        self.load_video_btn.config(state="normal")
        self._close_opened_result()  # a video loaded through the backend replaces a reopened run
        self.current_video_path = file_path

        # Clear all objects from previous video (if model is loaded)
        if self.tracking:
            self.clear_prompts()
            self.log_event("🧹 Cleared all objects from previous video")
        else:
            self.log_event("📹 Video loaded - no previous objects to clear")

        # Update video info display
        info_text = f"✅ Video loaded: {Path(file_path).name}\n"
        num_frames = video_info.get("num_frames", "Unknown")
        decoded = video_info.get("decoded_frames", num_frames)
        removed = video_info.get("duplicate_frames_removed", 0)
        if removed:
            info_text += f"Frames: {num_frames} unique time points ({decoded} decoded, {removed} duplicates removed)\n"
        else:
            info_text += f"Frames: {num_frames}\n"
        info_text += f"FPS: {video_info.get('fps', 'Unknown'):.1f}\n"

        # Extract dimensions correctly (dimensions is a tuple: height, width)
        dimensions = video_info.get("dimensions", (0, 0))
        if isinstance(dimensions, tuple) and len(dimensions) >= 2:
            height, width = dimensions[:2]
            info_text += f"Size: {width}x{height}"
        else:
            # Fallback for individual width/height keys
            width = video_info.get("width", "?")
            height = video_info.get("height", "?")
            info_text += f"Size: {width}x{height}"

            self.video_info_label.config(text=info_text)

        # Display the annotation frame (the last chronological frame in reverse mode)
        if self.tracking and self.tracking.video is not None:
            self.video_canvas.display_frame(self.tracking.annotation_frame())
            if video_info.get("direction") == "reverse":
                self.log_event("🖼️ Showing the last frame of the video for annotation (reverse tracking)")

            # Enable object controls (simplified UI)
            self.clear_prompts_btn.config(state="normal")
            self.track_btn.config(state="normal")

        # Initialize workflow status
        self.update_workflow_status()

        self.set_status("Video loaded! Drag a bounding box around an organoid to start.")
        self.log_event(f"✅ Video loaded in {load_time:.2f}s ({video_info.get('num_frames', '?')} frames)")

    def on_video_loaded_error(self, error_msg, load_time):
        """Handle video loading error"""
        self.load_video_btn.config(state="normal")
        self.set_status(f"Error loading video: {error_msg}")
        self.log_event(f"❌ Video loading failed after {load_time:.2f}s: {error_msg}")
        # Remove popup - already handled with status and log above

    def _validate_analysis_inputs(self, event=None):
        """Validate analysis input fields in real-time (simplified for new workflow)"""
        try:
            # Validate time lapse
            try:
                time_val = self.time_lapse_var.get()
                if time_val <= 0:
                    self.time_lapse_entry.config(style="Error.TEntry")
                else:
                    self.time_lapse_entry.config(style="TEntry")
            except (tk.TclError, ValueError):
                self.time_lapse_entry.config(style="Error.TEntry")

            # Validate conversion factor
            try:
                conv_val = self.conversion_factor_var.get()
                if conv_val <= 0:
                    self.conversion_factor_entry.config(style="Error.TEntry")
                else:
                    self.conversion_factor_entry.config(style="TEntry")
            except (tk.TclError, ValueError):
                self.conversion_factor_entry.config(style="Error.TEntry")

        except Exception:
            # Silently handle validation errors
            pass

    def _on_analysis_entry_focus(self, event, field_type):
        """Handle focus events for analysis entry fields with MobaXterm compatibility"""
        try:
            widget = event.widget
            widget.select_range(0, "end")  # Select all text for easy editing

            # Provide helpful status messages
            if field_type == "time":
                self.set_status("Enter time lapse period in days")
            elif field_type == "conversion":
                self.set_status("Enter conversion factor (micrometers per pixel)")

        except Exception:
            # Silently handle focus errors (common with MobaXterm)
            pass

    def on_canvas_click(self, x, y):
        """Handle click events for organoid placement"""
        if self.opened is not None:
            self.set_status("Reopened results are not editable: load the video through a model to annotate again")
            return
        if not self.tracking or not self.current_video_path:
            return

        self.tracking.set_debug(self.debug_var.get())

        try:
            # Create new organoid entry at click location
            organoid_id = self.next_organoid_id
            self.organoid_data[organoid_id] = {"point": (x, y), "cysts": []}
            self.current_organoid_id = organoid_id
            self.next_organoid_id += 1

            # Visual feedback - add organoid marker
            self.video_canvas.add_organoid_marker(x, y, organoid_id)

            # Switch to cyst addition mode for this organoid
            self.organoid_mode = False
            self.update_workflow_status()
            # Update organoid count display in analysis section
            self.update_organoid_count_display()

            # Store action for revert functionality
            action = {"type": "add_organoid", "organoid_id": organoid_id, "point": (x, y)}
            self.action_history.append(action)

            # Enable controls
            self.revert_btn.config(state="normal")

            self.set_status(f"Organoid {organoid_id} placed. Now drag bounding boxes around its cysts.")
            self.log_event(f"🔴 Added organoid {organoid_id} at ({x},{y})")

        except Exception as e:
            error_msg = f"Error placing organoid: {str(e)}"
            self.set_status(error_msg)
            self.log_event(f"❌ {error_msg}")
            logger.error(f"DEBUG: Error in on_canvas_click: {e}")

    def on_canvas_bbox(self, x1, y1, x2, y2):
        """Handle bounding box creation for cyst addition"""
        if self.opened is not None:
            self.set_status("Reopened results are not editable: load the video through a model to annotate again")
            return
        if not self.tracking or not self.current_video_path:
            return

        self.tracking.set_debug(self.debug_var.get())

        try:
            # Cyst addition mode - send bounding box to SAM2 for tracking
            if self.current_organoid_id is None:
                self.set_status("Please click on an organoid location first")
                return

            cyst_id = self.next_cyst_id

            # Add the cyst box as a prompt on the annotation frame (object id = cyst id)
            try:
                self.tracking.add_cyst(CystAnnotation(cyst_id=cyst_id, bbox=(x1, y1, x2, y2)))
                success = True
            except (AnnotationError, TrackingError) as error:
                success = False
                self.log_event(f"❌ Cyst box rejected: {error}")

            if success:
                # Store cyst information
                self.organoid_data[self.current_organoid_id]["cysts"].append(
                    {"cyst_id": cyst_id, "bbox": (x1, y1, x2, y2)}
                )
                self.next_cyst_id += 1

                # Add to active objects for tracking
                self.active_object_ids.add(cyst_id)

                # Store action in history for revert functionality
                action = {
                    "type": "add_cyst",
                    "organoid_id": self.current_organoid_id,
                    "cyst_id": cyst_id,
                    "bbox": (x1, y1, x2, y2),
                }
                self.action_history.append(action)

                # Visual feedback - add cyst bounding box
                self.video_canvas.add_bbox_marker(x1, y1, x2, y2, obj_id=cyst_id)

                # Update displays
                self.update_active_objects_display()
                self.update_organoid_count_display()

                # Enable controls
                self.clear_prompts_btn.config(state="normal")
                self.revert_btn.config(state="normal")
                self.track_btn.config(state="normal")

                cyst_count = len(self.organoid_data[self.current_organoid_id]["cysts"])
                self.set_status(f"Added cyst {cyst_count} to organoid {self.current_organoid_id}")
                self.log_event(
                    f"🔵 Added cyst {cyst_id} to organoid {self.current_organoid_id}: ({x1},{y1})-({x2},{y2})"
                )

                # Update workflow status to show current cyst count
                self.update_workflow_status()
                # Update organoid count display in analysis section
                self.update_organoid_count_display()

            else:
                self.set_status(f"Failed to add cyst to organoid {self.current_organoid_id}")
                self.log_event(f"❌ Failed to add cyst to organoid {self.current_organoid_id}")

        except Exception as e:
            error_msg = f"Error adding cyst: {str(e)}"
            self.set_status(error_msg)
            self.log_event(f"❌ {error_msg}")
            logger.error(f"DEBUG: Error in on_canvas_bbox: {e}")
            import traceback

            traceback.print_exc()

    def _get_next_object_id(self):
        """Get the next available object ID (legacy method)"""
        return self.next_object_id

    def update_workflow_status(self):
        """Update the workflow status display"""
        if self.current_organoid_id is None:
            self.workflow_status_label.config(text="Click to place an organoid")
        else:
            cyst_count = len(self.organoid_data[self.current_organoid_id]["cysts"])
            total_organoids = len(self.organoid_data)
            self.workflow_status_label.config(
                text=f"Organoid {self.current_organoid_id} ({cyst_count} cysts) | Total: {total_organoids} organoids | Click for new organoid, drag for cysts"
            )

    def update_organoid_count_display(self):
        """Update the organoid count display"""
        total_organoids = len(self.organoid_data)
        self.organoid_count_display.config(text=str(total_organoids))

    def revert_last_action(self):
        """Revert the last action in the organoid-cyst workflow"""
        if not self.action_history:
            self.set_status("No action to revert")
            self.log_event("⚠️ No action to revert")
            return

        # Pop the last action from history
        last_action = self.action_history.pop()

        try:
            if last_action["type"] == "add_cyst":
                # Reverting cyst addition
                organoid_id = last_action["organoid_id"]
                cyst_id = last_action["cyst_id"]

                # Drop the cyst's prompts from the backend
                if self.tracking:
                    try:
                        self.tracking.remove_cyst(cyst_id)
                    except TrackingError as error:
                        logger.warning(f"Failed to clear cyst {cyst_id}: {error}")

                # Remove from active objects
                self.active_object_ids.discard(cyst_id)

                # Remove from organoid data
                if organoid_id in self.organoid_data:
                    self.organoid_data[organoid_id]["cysts"] = [
                        c for c in self.organoid_data[organoid_id]["cysts"] if c["cyst_id"] != cyst_id
                    ]

                # Clear visual markers
                self.video_canvas.clear_markers(cyst_id)

                # Fix ID continuity: if this was the highest cyst ID, adjust next_cyst_id
                if cyst_id == self.next_cyst_id - 1:
                    # Find the actual highest cyst ID still in use
                    max_cyst_id = 0
                    for org_data in self.organoid_data.values():
                        for cyst in org_data["cysts"]:
                            max_cyst_id = max(max_cyst_id, cyst["cyst_id"])
                    self.next_cyst_id = max_cyst_id + 1

                # Check if organoid has no cysts left - auto remove organoid
                if organoid_id in self.organoid_data and len(self.organoid_data[organoid_id]["cysts"]) == 0:
                    # Remove empty organoid
                    self.video_canvas.clear_markers(f"organoid_{organoid_id}")
                    del self.organoid_data[organoid_id]

                    # Reset current organoid if this was it
                    if self.current_organoid_id == organoid_id:
                        self.current_organoid_id = None

                    # IMPORTANT: Remove the corresponding add_organoid action from history to prevent redundant removal
                    self.action_history = [
                        action
                        for action in self.action_history
                        if not (action["type"] == "add_organoid" and action["organoid_id"] == organoid_id)
                    ]

                    # Fix ID continuity for auto-removed organoid: if this was the highest organoid ID, adjust next_organoid_id
                    if organoid_id == self.next_organoid_id - 1:
                        # Find the actual highest organoid ID still in use
                        max_organoid_id = 0
                        for org_id in self.organoid_data.keys():
                            max_organoid_id = max(max_organoid_id, org_id)
                        self.next_organoid_id = max_organoid_id + 1

                    self.set_status(f"Reverted cyst {cyst_id}. Auto-removed empty organoid {organoid_id}")
                    self.log_event(f"↩️ Reverted cyst {cyst_id} and auto-removed empty organoid {organoid_id}")
                else:
                    self.set_status(f"Reverted cyst {cyst_id} from organoid {organoid_id}")
                    self.log_event(f"↩️ Reverted cyst {cyst_id} from organoid {organoid_id}")

            elif last_action["type"] == "add_organoid":
                # Reverting organoid addition (would also remove all its cysts)
                organoid_id = last_action["organoid_id"]

                # Remove all cysts for this organoid
                if organoid_id in self.organoid_data:
                    for cyst in self.organoid_data[organoid_id]["cysts"]:
                        cyst_id = cyst["cyst_id"]
                        if self.tracking:
                            try:
                                self.tracking.remove_cyst(cyst_id)
                            except TrackingError as error:
                                logger.warning(f"Failed to clear cyst {cyst_id}: {error}")
                        self.active_object_ids.discard(cyst_id)
                        self.video_canvas.clear_markers(cyst_id)

                    # Remove organoid data
                    del self.organoid_data[organoid_id]

                # Clear organoid marker
                self.video_canvas.clear_markers(f"organoid_{organoid_id}")

                # Reset workflow state if this was current organoid
                if self.current_organoid_id == organoid_id:
                    self.current_organoid_id = None

                # Fix ID continuity: if this was the highest organoid ID, adjust next_organoid_id
                if organoid_id == self.next_organoid_id - 1:
                    # Find the actual highest organoid ID still in use
                    max_organoid_id = 0
                    for org_id in self.organoid_data.keys():
                        max_organoid_id = max(max_organoid_id, org_id)
                    self.next_organoid_id = max_organoid_id + 1

                self.set_status(f"Reverted organoid {organoid_id} and all its cysts")
                self.log_event(f"↩️ Reverted organoid {organoid_id}")

            # Update displays
            self.update_active_objects_display()
            self.update_organoid_count_display()
            self.update_workflow_status()

            # Disable revert button if no more actions
            if not self.action_history:
                self.revert_btn.config(state="disabled")

            # Update button states
            if not self.active_object_ids:
                self.track_btn.config(state="disabled")
                self.clear_prompts_btn.config(state="disabled")

        except Exception as e:
            error_msg = f"Failed to revert: {str(e)}"
            self.set_status(error_msg)
            self.log_event(f"❌ {error_msg}")

    def debug_button_states(self):
        """Debug method to check button states - simplified for new interface"""
        logger.debug("DEBUG: Button States Check:")
        logger.debug(f"- Clear All Objects: {self.clear_prompts_btn['state']}")
        logger.debug(f"- Revert Last: {self.revert_btn['state']}")
        logger.debug(f"- Track: {self.track_btn['state']}")
        logger.debug(f"- Active Objects: {self.active_object_ids}")
        logger.debug(f"- Action History: {len(self.action_history)} actions")
        logger.debug(f"- Background Mode: {self.background_mode}")
        logger.debug(f"- Next Object ID: {self.next_object_id}")

    def add_new_object(self):
        """Legacy method - objects now added by clicking on canvas"""
        self.set_status("Left click on the video to add objects automatically")
        self.log_event("💡 Hint: Left click on video to add objects")

    # Removed _get_current_object_id - no object selection needed with sequential numbering

    def remove_current_object(self):
        """Legacy method - use revert button to undo last action"""
        self.set_status("Use the '⮪ Revert Last' button to undo the last object addition")
        self.log_event("💡 Hint: Use Revert Last button to remove objects")

    # Removed update_object_combo and related methods - no dropdown needed with sequential numbering

    def update_active_objects_display(self):
        """Update the display showing active objects - simplified for sequential numbering"""
        if not self.tracking:
            self.active_objects_label.config(text="Active Objects: None")
            self.object_list_text.config(state="normal")
            self.object_list_text.delete(1.0, tk.END)
            self.object_list_text.config(state="disabled")
            return

        try:
            object_info = []
            total_organoids = len(self.organoid_data)
            total_cysts = len(self.active_object_ids)

            # Show organoid and cyst information
            for organoid_id, organoid in self.organoid_data.items():
                cyst_count = len(organoid["cysts"])
                object_info.append(f"🔴 Organoid {organoid_id}: {cyst_count} cysts")

                # Show individual cysts for current organoid
                if organoid_id == self.current_organoid_id and organoid["cysts"]:
                    for cyst in organoid["cysts"]:
                        cyst_id = cyst["cyst_id"]
                        object_info.append(f"  🔵 Cyst {cyst_id}")

            # Update display
            if self.organoid_data:
                self.active_objects_label.config(text=f"Organoids: {total_organoids}, Cysts: {total_cysts}")
            else:
                self.active_objects_label.config(text="Organoids: 0, Cysts: 0")

            self.object_list_text.config(state="normal")
            self.object_list_text.delete(1.0, tk.END)
            self.object_list_text.insert(1.0, "\n".join(object_info) if object_info else "Ready to start workflow...")
            self.object_list_text.config(state="disabled")

        except Exception as e:
            logger.error(f"DEBUG: Error in update_active_objects_display: {e}")
            self.active_objects_label.config(text="Active Objects: None")
            self.object_list_text.config(state="normal")
            self.object_list_text.delete(1.0, tk.END)
            self.object_list_text.config(state="disabled")

    def clear_prompts(self):
        """Clear all organoids and cysts - reset to initial state"""
        if not self.tracking:
            return

        try:
            # Clear all prompts from the backend
            self.tracking.clear_prompts()

            # Clear all visual markers (including organoid markers)
            self.video_canvas.clear_markers()

            # Reset organoid-cyst workflow state
            self.organoid_data.clear()
            self.active_object_ids.clear()
            self.action_history.clear()
            self.next_organoid_id = 1
            self.next_cyst_id = 1
            self.current_organoid_id = None
            self.organoid_mode = True

            # Update displays
            self.update_active_objects_display()
            self.update_organoid_count_display()
            self.update_workflow_status()

            # Disable workflow buttons
            self.track_btn.config(state="disabled")
            self.revert_btn.config(state="disabled")
            self.clear_prompts_btn.config(state="disabled")

            self.set_status("All organoids and cysts cleared - ready to start fresh")
            self.log_event("🧹 All organoids and cysts cleared, workflow reset")

        except Exception as e:
            self.set_status(f"Error clearing: {str(e)}")

    # Removed clear_current_object method - replaced by revert button and clear all functionality

    def start_tracking(self):
        """Start multi-object tracking with timing"""
        if not self.tracking or self.tracking.video is None:
            self.set_status("Error: Please load a video first")
            self.log_event("❌ Cannot start tracking - no video loaded")
            return

        # Check if there are any active cyst objects to track
        if not self.active_object_ids:
            self.set_status("Error: Please add at least one cyst by dragging bounding boxes")
            self.log_event("❌ Cannot start tracking - no cysts added")
            return

        if self.tracking.prompt_count() == 0:
            self.set_status("Error: No tracking prompts available")
            self.log_event("❌ Cannot start tracking - no prompts in model")
            return

        if self.tracking_in_progress:
            self.set_status("Tracking is already running")
            return

        start_time = time.time()

        self.tracking_in_progress = True
        self.tracking_run_id = uuid.uuid4().hex[:12]  # identifies these results when they are saved
        run_id = self.tracking_run_id
        service = self.tracking
        self._cancel_request = threading.Event()  # this run's cancel request; honoured even before the backend starts
        cancel_request = self._cancel_request
        self.track_btn.config(state="disabled")
        self.save_results_btn.config(state="disabled")
        self.set_status("Running tracking... Please wait.")

        # Count total prompts
        total_prompts = self.tracking.prompt_count()
        active_objects = self.tracking.active_object_ids()
        self.log_event(f"🎯 Starting tracking for {len(active_objects)} objects ({total_prompts} prompts)")
        self._write_prompt_record()

        # Create progress dialog; its Cancel button asks the service to stop after the frame in progress
        self.tracking_dialog = ProgressDialog(self.root, "Running Object Tracking", on_cancel=self.cancel_tracking)
        dialog = self.tracking_dialog

        def tracking_thread():
            # Everything posted back names the run, the service and the dialog it belongs to, so that the
            # window can ignore callbacks of a run that is no longer the current one
            try:

                def progress_callback(current, total, message):
                    progress = (current / total) * 100 if total > 0 else 0
                    self.post(self._on_tracking_progress, run_id, dialog, progress, message)

                result = service.run(progress_callback, should_stop=cancel_request.is_set)
                final = "Tracking completed!" if result.is_complete else f"Tracking {result.status}"
                self.post(self._on_tracking_progress, run_id, dialog, 100, final)
                self.post(self.on_tracking_complete_success, run_id, service, dialog, result, time.time() - start_time)

            except Exception as e:
                self.post(self.on_tracking_complete_error, run_id, service, dialog, str(e), time.time() - start_time)

        # Start tracking in background thread
        threading.Thread(target=tracking_thread, daemon=True).start()
        self.tracking_dialog.show()

    def cancel_tracking(self):
        """Ask the running tracking to stop after the frame in progress (the progress dialog's Cancel button)."""
        if not self.tracking_in_progress or self.tracking is None or self._cancel_request is None:
            self.set_status("No tracking is running")
            return
        self._cancel_request.set()  # polled by the backend between frames, from before the first one
        self.tracking.cancel()  # the service's own state, when its run has already begun
        self.set_status("Cancelling... finishing the frame in progress")
        self.log_event("⏹ Cancel requested: tracking stops after the frame in progress")

    def _on_tracking_progress(self, run_id, dialog, progress, message):
        if run_id != self.tracking_run_id:
            return  # a stale run's progress
        dialog.update_progress(progress, message)

    def _stale_tracking_callback(self, run_id, service, dialog, kind: str) -> bool:
        """True (and the callback's own dialog closed) when the callback belongs to a run that is no longer current."""
        if run_id == self.tracking_run_id and service is self.tracking:
            return False
        if dialog is not None:
            dialog.close()  # the superseded run's dialog, whether or not it is still the window's current reference
            if self.tracking_dialog is dialog:
                self.tracking_dialog = None
        self.log_event(
            f"🔁 Ignored a stale tracking {kind} of run {run_id}: the window moved on to another run or model"
        )
        return True

    def _write_prompt_record(self):
        """Save prompts, organoid associations and provenance so a run can be reproduced."""
        try:
            from ..services.prompt_record import build_prompt_record, default_prompt_record_path, write_prompt_record

            def tk_value(var):
                try:
                    return var.get()
                except Exception:
                    return None

            record = build_prompt_record(
                self.tracking.tracker,
                str(self.current_video_path),
                self.organoid_data,
                tk_value(self.time_lapse_var),
                tk_value(self.conversion_factor_var),
            )
            path = write_prompt_record(record, default_prompt_record_path(str(self.current_video_path)))
            self.log_event(f"💾 Prompt record saved: {path}")
        except Exception as e:
            self.log_event(f"⚠️ Could not save prompt record: {e}")

    def on_tracking_complete_success(self, run_id, service, dialog, result, tracking_time):
        """Install the results of the current run (GUI thread); a stale run's completion is ignored."""
        if self._stale_tracking_callback(run_id, service, dialog, "completion"):
            return
        # Close progress dialog
        if dialog is not None:
            dialog.close()
        if self.tracking_dialog is dialog:
            self.tracking_dialog = None

        self.tracking_in_progress = False
        self.track_btn.config(state="normal")
        self.video_segments = result if result else None
        status = getattr(result, "status", "completed")
        frames_total = getattr(result, "frames_total", "?")
        frames_done = getattr(result, "frames_done", "?")

        if self.video_segments:
            # The results belong to the video and backend state of this moment
            self.result_video = service.video
            self.result_provenance = service.provenance()
            self._results_changed()

            # Enable results buttons
            self.generate_btn.config(state="normal")
            self.analysis_btn.config(state="normal")
            self.save_results_btn.config(state="normal")

            num_frames = len(self.video_segments)
            active_objects = service.active_object_ids()
            if status == "partial":
                error = getattr(result, "error", "unknown error")
                self.log_event(f"⚠️ Tracking stopped early after {tracking_time:.2f}s: {error}")
                self.log_event(f"⚠️ Results cover {num_frames} of {frames_total} frames; treat exports as partial")
                self.set_status("Tracking stopped early; results are partial. Check the log.")
            elif status == "cancelled":
                self.log_event(
                    f"⏹ Tracking cancelled after {tracking_time:.2f}s: {frames_done} of {frames_total} frames tracked"
                )
                self.log_event(f"⚠️ Results cover {num_frames} of {frames_total} frames; treat exports as partial")
                self.set_status("Tracking cancelled: the tracked frames can be exported or saved, or track again.")
            else:
                self.log_event(f"✅ Tracking completed in {tracking_time:.2f}s")
                self.set_status("Tracking completed! Ready to generate videos.")
            self.log_event(f"📊 Processed {num_frames} frames for {len(active_objects)} objects")
        else:
            self.result_video = None
            self.result_provenance = None
            self._results_changed()
            for button in (self.generate_btn, self.analysis_btn, self.save_results_btn):
                button.config(state="disabled")
            if status == "cancelled":
                self.log_event(
                    f"⏹ Tracking cancelled after {tracking_time:.2f}s before any mask was kept "
                    f"({frames_done} of {frames_total} frames); track again when ready"
                )
                self.set_status("Tracking cancelled before any mask was kept. Track again when ready.")
            else:
                self.log_event("⚠️ Tracking completed but no results generated")
                self.set_status("Tracking produced no results")

    def on_tracking_complete_error(self, run_id, service, dialog, error_msg, tracking_time):
        """Handle tracking completion error"""
        if self._stale_tracking_callback(run_id, service, dialog, "error"):
            return
        # Close progress dialog
        if dialog is not None:
            dialog.close()
        if self.tracking_dialog is dialog:
            self.tracking_dialog = None

        self.tracking_in_progress = False
        self.track_btn.config(state="normal")

        self.set_status(f"Error during tracking: {error_msg}")
        self.log_event(f"❌ Tracking failed after {tracking_time:.2f}s: {error_msg}")

    def generate_videos(self):
        """Generate output videos with timing"""
        if not self._ready_for_export("video generation"):
            return

        # Ask for output directory
        token = self._results_token()
        output_dir = filedialog.askdirectory(title="Select Output Directory", initialdir="./data/output_videos")

        if not output_dir:
            return

        # The chooser is modal and runs the event loop: a model reload may have completed meanwhile
        # and discarded the results, or another run may have been opened, so the readiness and the
        # identity of the results are checked again before any control changes.
        if not self._ready_for_export("video generation") or not self._same_results(token, "video generation"):
            return
        frames = self._export_frames()
        if frames is None:
            self.set_status("The video of the reopened run was not found; videos cannot be generated")
            self.log_event("❌ Video generation needs the reopened run's video file (not found at its recorded path)")
            return

        start_time = time.time()

        self.generate_btn.config(state="disabled")
        self.set_status("Generating videos... This may take a while.")

        # The worker exports this snapshot; a later reload cannot pull the results from under it
        result = self.video_segments

        # Count objects for logging
        active_objects = self._result_object_ids(result)
        num_frames = len(result)
        video_types = ["overlay", "mask", "side_by_side"]

        # DETAILED DEBUG OUTPUT
        self.log_event("🎬" + "=" * 60)
        self.log_event("🎬 DETAILED VIDEO GENERATION DEBUG START")
        self.log_event("🎬" + "=" * 60)
        self.log_event("📊 Video Generation Configuration:")
        self.log_event(f"   • Active Objects: {active_objects} (count: {len(active_objects)})")
        self.log_event(f"   • Video Segments: {num_frames} frames")
        self.log_event(f"   • Video Types: {video_types}")
        self.log_event(f"   • Quality: {self.quality_var.get()}")
        self.log_event(f"   • Debug Mode: {self.debug_var.get()}")
        self.log_event(f"   • Output Directory: {output_dir}")

        # Video frames analysis
        self.log_event(f"   • Source Frames: {len(frames)} frames")
        self.log_event(f"   • Frame Dimensions: {frames[0].shape if frames else 'N/A'}")

        # Tracking data analysis
        frame_indices = list(result.keys())
        self.log_event(
            f"   • Tracking Frame Indices: {sorted(frame_indices)[:5]}{'...' if len(frame_indices) > 5 else ''}"
        )
        sample_frame = frame_indices[0] if frame_indices else None
        if sample_frame is not None:
            self.log_event(f"   • Sample Frame Objects: {list(result[sample_frame].keys())}")

        # Organoid-cyst mapping
        total_organoids = len(self.organoid_data)
        total_cysts = sum(len(org["cysts"]) for org in self.organoid_data.values())
        self.log_event(f"   • Organoid-Cyst Mapping: {total_organoids} organoids, {total_cysts} cysts")

        for org_id, org_data in list(self.organoid_data.items())[:3]:  # Show first 3
            cyst_ids = [c["cyst_id"] for c in org_data["cysts"]]
            self.log_event(f"     ◦ Organoid {org_id}: cysts {cyst_ids}")
        if len(self.organoid_data) > 3:
            self.log_event(f"     ◦ ... and {len(self.organoid_data) - 3} more organoids")

        self.log_event(f"🎬 Starting video generation for {len(active_objects)} objects")
        self.log_event(f"📹 Creating {len(video_types)} video types ({num_frames} frames each)")

        # Read the Tk variables on the GUI thread before starting the worker
        quality = self.quality_var.get()
        debug = self.debug_var.get()
        objects_text = f"Objects: {', '.join(map(str, sorted(active_objects)))}"
        output_dir_path = Path(output_dir)

        def generation_thread():
            try:

                def progress_callback(current, total, message):
                    progress = (current / total) * 100 if total > 0 else 0
                    self.post(self.generation_dialog.update_progress, progress, f"{message} ({objects_text})")

                # The export service writes the three videos straight into the chosen directory
                created_videos = ExportService(output_dir_path).write_videos(
                    frames,
                    result,
                    quality=quality,
                    progress=progress_callback,
                    debug=debug,
                    directory=output_dir_path,
                )
                total_time = time.time() - start_time
                self.post(self.generation_dialog.update_progress, 100, "Video generation completed!")
                self.post(self.on_generation_complete_success, created_videos, output_dir, total_time)

            except Exception as e:
                total_time = time.time() - start_time
                self.post(self.on_generation_complete_error, str(e), total_time)

        # Show progress dialog
        self.generation_dialog = ProgressDialog(self.root, "Generating Multi-Object Videos")

        # Start generation in background
        threading.Thread(target=generation_thread, daemon=True).start()
        self.generation_dialog.show()

    def on_generation_complete_success(self, created_videos, output_dir, total_time):
        """Handle successful video generation"""
        # ✅ CRITICAL FIX: Always ensure button is enabled, regardless of any exceptions
        self.generate_btn.config(state="normal")

        try:
            # Force garbage collection before GUI updates to prevent memory pressure
            import gc

            gc.collect()

            # Close progress dialog
            if self.generation_dialog:
                self.generation_dialog.close()
                self.generation_dialog = None

            self.generate_btn.config(state="normal")

            # Count successful videos
            successful_videos = [v for v in created_videos.values() if v is not None]
            failed_videos = [k for k, v in created_videos.items() if v is None]

            # DETAILED DEBUG OUTPUT - COMPLETION
            self.log_event("🎬" + "=" * 60)
            self.log_event("🎬 DETAILED VIDEO GENERATION DEBUG COMPLETE")
            self.log_event("🎬" + "=" * 60)
            self.log_event(f"🎉 Video generation completed in {total_time:.2f}s")
            self.log_event(f"📊 Generated {len(successful_videos)}/{len(created_videos)} videos successfully")

            # Detailed results breakdown
            self.log_event("📋 Detailed Results:")
            for video_type, path in created_videos.items():
                if path:
                    file_size = Path(path).stat().st_size / (1024 * 1024) if Path(path).exists() else 0
                    self.log_event(f"   ✅ {video_type}: {Path(path).name} ({file_size:.1f}MB)")
                else:
                    self.log_event(f"   ❌ {video_type}: FAILED")

            if failed_videos:
                self.log_event(f"⚠️ Failed videos: {', '.join(failed_videos)}")

            # Performance metrics
            num_frames = len(self.video_segments) if self.video_segments else 0
            if num_frames > 0 and total_time > 0:
                frames_per_second = num_frames / total_time
                self.log_event(
                    f"⚡ Performance: {frames_per_second:.1f} frames/sec across {len(successful_videos)} video types"
                )

            self.log_event("🎬" + "=" * 60)

            self.set_status(f"Video generation completed! {len(successful_videos)} videos created.")

            # Show result message
            if successful_videos:
                result_msg = f"Video generation completed in {total_time:.2f}s!\n\n"
                result_msg += f"Successfully generated {len(successful_videos)} videos:\n"
                for video_type, path in created_videos.items():
                    if path:
                        result_msg += f"✅ {video_type}: {Path(path).name}\n"
                    else:
                        result_msg += f"❌ {video_type}: Failed\n"
                result_msg += f"\nOutput directory: {output_dir}"

                # Remove popup - use log instead
                self.log_event("✅ Video generation completed")
            else:
                # Remove popup - use log instead
                self.log_event("❌ All video generation failed")

        except Exception as e:
            # Handle any errors in completion callback
            # ✅ CRITICAL FIX: Always ensure button is enabled, even on exceptions
            self.generate_btn.config(state="normal")
            self.log_event(f"❌ Error in video generation completion: {str(e)}")
            if self.generation_dialog:
                self.generation_dialog.close()
                self.generation_dialog = None

    def on_generation_complete_error(self, error_msg, total_time):
        """Handle video generation error"""
        # ✅ CRITICAL FIX: Always ensure button is enabled first
        self.generate_btn.config(state="normal")

        # Close progress dialog
        if self.generation_dialog:
            self.generation_dialog.close()
            self.generation_dialog = None
        self.set_status(f"Error generating videos: {error_msg}")
        self.log_event(f"❌ Video generation failed after {total_time:.2f}s: {error_msg}")
        # Remove popup - already handled with status and log

    def view_results(self):
        """View tracking results"""
        frames = self._export_frames()
        if not self.video_segments or frames is None:
            return

        # Show a simple results viewer
        from .results_viewer import ResultsViewer

        try:
            viewer = ResultsViewer(self.root, frames, self.video_segments, obj_id=1)
            viewer.show()

        except Exception as e:
            error_msg = f"Failed to open results viewer: {str(e)}"
            self.set_status(error_msg)
            self.log_event(f"❌ {error_msg}")

    # ------------------------------------------------------------------ saved results: open and save
    def _export_frames(self) -> list | None:
        """The frames the shown results belong to: the reopened run's (None if its video was not found) or the backend's."""
        if self.opened is not None:
            return self.opened.frames
        return self.tracking.frames if self.tracking is not None else None

    @staticmethod
    def _result_object_ids(result) -> list[int]:
        if hasattr(result, "object_ids"):
            return list(result.object_ids())
        return sorted({int(obj) for frame_masks in result.values() for obj in frame_masks})

    def open_results(self):
        """Reopen a saved run (results.json and its mask file) without a model, for viewing and exporting."""
        directory = filedialog.askdirectory(
            title="Select a run directory holding results.json", initialdir="./data/output_videos"
        )
        if not directory:
            return
        if self.tracking_in_progress:  # the chooser is modal: tracking may have started meanwhile
            self.set_status("Tracking is running; open saved results when it has finished")
            return
        run_dir = Path(directory)
        start_time = time.time()
        self.open_results_btn.config(state="disabled")
        self.set_status("Opening the saved results... Please wait.")
        self.log_event(f"📂 Opening results from {run_dir}")

        def open_thread():
            try:
                saved = load_saved_result(run_dir)
                frames = None
                video_error = None
                try:
                    frames = load_video_frames(saved.session.video.path, saved.video)
                except SessionError as error:
                    video_error = str(error)
                self.post(self.on_results_opened, saved, frames, video_error, time.time() - start_time)
            except Exception as error:
                self.post(self.on_results_open_error, str(error), time.time() - start_time)

        threading.Thread(target=open_thread, daemon=True).start()

    def on_results_opened(self, saved, frames, video_error, elapsed):
        """Install a reopened run (GUI thread): it replaces whatever the window showed."""
        self.open_results_btn.config(state="normal")
        if self.tracking_in_progress:
            self.set_status("Tracking is running; open the saved results again when it has finished")
            self.log_event("❌ Opening results refused: tracking is in progress")
            return
        self._discard_downstream_state("a saved run was opened")
        self.opened = OpenedResult(saved, frames)
        self.video_segments = saved.result
        self.tracking_run_id = saved.run_id
        self._results_changed()
        self.current_video_path = str(saved.session.video.path)
        self.organoid_data = saved.session.annotations.organoid_data()
        self.next_organoid_id = max(self.organoid_data, default=0) + 1
        self.next_cyst_id = max(saved.session.annotations.cyst_ids(), default=0) + 1
        self.active_object_ids = set(saved.result.object_ids())
        timing = saved.session.timing.resolve(saved.video.n_frames)
        self.time_lapse_var.set(timing.time_lapse_days)
        self.conversion_factor_var.set(saved.session.calibration.um_per_pixel)

        video = saved.video
        info = f"📂 Reopened run {saved.run_id} ({saved.created})\n"
        info += f"Video: {Path(video.path).name}\nFrames: {video.n_frames} unique"
        if video.duplicate_frames_removed:
            info += f" ({video.decoded_frames} decoded, {video.duplicate_frames_removed} duplicates removed)"
        info += f"\nSize: {video.width}x{video.height}\nTracking: {saved.result.summary()}"
        self.video_info_label.config(text=info)
        if frames is not None:
            self.video_canvas.display_frame(frames[video.annotation_frame])
            for organoid_id, organoid in self.organoid_data.items():
                self.video_canvas.add_organoid_marker(*organoid["point"], organoid_id)
                for cyst in organoid["cysts"]:
                    self.video_canvas.add_bbox_marker(*cyst["bbox"], obj_id=cyst["cyst_id"])
        else:
            self.video_canvas.show_placeholder()
            self.log_event(f"⚠️ The run's video was not found, so videos cannot be generated: {video_error}")
        self.generate_btn.config(state="normal" if frames is not None else "disabled")
        self.analysis_btn.config(state="normal")
        self.save_results_btn.config(state="disabled")  # already saved where it was opened from
        self.update_active_objects_display()
        self.update_organoid_count_display()
        self.workflow_status_label.config(text="Reopened results: the annotations are shown, not editable")
        if saved.result.is_partial:
            self.log_event(f"⚠️ The reopened run is partial: {saved.result.summary()}")
        if timing.frame_timestamps is not None:
            self.log_event("🕒 The run used explicit frame times; the analysis report will use them")
        self.log_event(f"✅ Opened results of run {saved.run_id} in {elapsed:.2f}s: {saved.result.summary()}")
        self.set_status("Saved results opened. Generate videos or the analysis report from them.")

    def on_results_open_error(self, error_msg, elapsed):
        self.open_results_btn.config(state="normal")
        self.set_status(f"Cannot open the saved results: {error_msg}")
        self.log_event(f"❌ Opening results failed after {elapsed:.2f}s: {error_msg}")

    def _close_opened_result(self) -> None:
        """Set a reopened run aside (a video loaded through the backend replaces it)."""
        if self.opened is None:
            return
        self.opened = None
        self.video_segments = None
        self.tracking_run_id = None
        self.result_video = None
        self.result_provenance = None
        self._results_changed()
        self.organoid_data.clear()
        self.active_object_ids.clear()
        self.action_history.clear()
        self.next_organoid_id = 1
        self.next_cyst_id = 1
        self.current_organoid_id = None
        self.organoid_mode = True
        self.video_canvas.clear_markers()
        for button in (self.generate_btn, self.analysis_btn, self.save_results_btn):
            button.config(state="disabled")
        self.log_event("🔁 The reopened results were set aside")

    def _invalidate_results(self, reason: str) -> None:
        """Forget the live results and disable what depends on them (the video they belong to is going away)."""
        had_results = self.video_segments is not None
        self.video_segments = None
        self.tracking_run_id = None
        self.result_video = None
        self.result_provenance = None
        self._results_changed()
        for button in (self.generate_btn, self.analysis_btn, self.save_results_btn):
            button.config(state="disabled")
        if had_results:
            self.log_event(f"🔁 {reason}: the previous tracking results were discarded; track again to export or save")

    def _results_changed(self) -> None:
        """Note that the shown results were replaced or discarded (a token taken before no longer matches)."""
        self._results_serial += 1

    def _results_token(self) -> int:
        """Identity of the results shown; compared after a modal dialog, which runs the event loop."""
        return self._results_serial

    def _same_results(self, token: int, what: str) -> bool:
        if token == self._results_token():
            return True
        self.set_status(f"The results changed while the dialog was open; request {what} again")
        self.log_event(f"❌ {what} refused: the results changed while the dialog was open")
        return False

    def save_results(self):
        """Save the shown results with their session and prompt record into a directory of the user's choice."""
        if not self._ready_for_export("saving the results"):
            return
        if self.opened is not None:
            self.set_status("These results are already saved (they were opened from a run directory)")
            return
        token = self._results_token()
        directory = filedialog.askdirectory(
            title="Select a directory for the saved results", initialdir="./data/output_videos"
        )
        if not directory:
            return
        # The chooser is modal and runs the event loop: re-check after it returns (see generate_videos)
        if (
            self.opened is not None
            or not self._ready_for_export("saving the results")
            or not self._same_results(token, "saving the results")
        ):
            return
        target = Path(directory)
        exporter = ExportService(target)
        existing = exporter.previous_saved_run()
        replace_existing = False
        if existing:
            names = ", ".join(path.name for path in existing)
            if not messagebox.askyesno(
                "Replace the saved run?",
                f"{target} already holds a saved run ({names}).\n\nReplace those files with the current results? "
                "Exported videos, tables and figures in the directory are left alone.",
            ):
                self.set_status("Saving cancelled: the directory already holds a saved run")
                return
            replace_existing = True
            if (  # the question box is modal too
                self.opened is not None
                or not self._ready_for_export("saving the results")
                or not self._same_results(token, "saving the results")
            ):
                return

        # Read the Tk variables and the annotations on the GUI thread before starting the worker
        try:
            time_lapse_days = float(self.time_lapse_var.get())
            conversion_factor = float(self.conversion_factor_var.get())
        except (tk.TclError, ValueError) as error:
            self.set_status(f"Cannot save: check the time lapse and the conversion factor ({error})")
            return
        organoid_data = {
            oid: {"point": info["point"], "cysts": list(info["cysts"])} for oid, info in self.organoid_data.items()
        }
        result = self.video_segments
        service = self.tracking
        video = self.result_video  # the video the results were tracked on, checked by _ready_for_export
        provenance = self.result_provenance or service.provenance()
        run_id = self.tracking_run_id or service.run_id
        video_path = self.current_video_path
        start_time = time.time()
        self.save_results_btn.config(state="disabled")
        self.set_status("Saving the results... Please wait.")

        def save_thread():
            try:
                record = build_prompt_record(
                    service.tracker, video_path, organoid_data, time_lapse_days, conversion_factor
                )
                session = session_from_document(record)
                saved = exporter.save_run(
                    run_id=run_id,
                    session=session,
                    video=video,
                    provenance=provenance,
                    result=result,
                    prompt_record=record,
                    replace_existing=replace_existing,
                )
                self.post(self.on_results_saved, saved, time.time() - start_time)
            except Exception as error:
                self.post(self.on_results_save_error, str(error), time.time() - start_time)

        threading.Thread(target=save_thread, daemon=True).start()

    def on_results_saved(self, saved, elapsed):
        self.save_results_btn.config(state="normal")
        self.log_event(
            f"💾 Results saved in {elapsed:.2f}s to {saved.directory} "
            f"({saved.path.name}, {saved.masks_path.name}, session.json, prompts.json)"
        )
        self.log_event(
            f'   Export them again without a model: organoidtracker export --run "{saved.directory}" --out DIR'
        )
        self.set_status(f"Results saved to {saved.directory}")

    def on_results_save_error(self, error_msg, elapsed):
        self.save_results_btn.config(state="normal")
        self.set_status(f"Saving the results failed: {error_msg}")
        self.log_event(f"❌ Saving the results failed after {elapsed:.2f}s: {error_msg}")

    def set_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.root.quit()
        finally:
            logging.getLogger("organoidtracker").removeHandler(self.log_handler)

    def generate_analysis_report(self):
        """Generate comprehensive organoid-cyst analysis report using new workflow"""
        if not self._ready_for_export("the analysis report"):
            return

        if not self.organoid_data:
            self.set_status("No organoid data available. Please add organoids and cysts first")
            self.log_event("❌ No organoid-cyst data available for analysis")
            return

        # Get analysis parameters
        try:
            try:
                time_lapse_days = self.time_lapse_var.get()
            except (tk.TclError, ValueError):
                time_lapse_days = 7.0  # Fallback to default
                self.log_event("⚠️ Using default value for time lapse: 7.0 days")

            try:
                conversion_factor = self.conversion_factor_var.get()
            except (tk.TclError, ValueError):
                conversion_factor = 1.0  # Fallback to default
                self.log_event("⚠️ Using default value for conversion factor: 1.0")

            # Validate ranges
            if time_lapse_days <= 0:
                time_lapse_days = 7.0
                self.time_lapse_var.set(7.0)
                self.log_event("⚠️ Invalid time lapse value, reset to default: 7.0 days")

            if conversion_factor <= 0:
                conversion_factor = 1.0
                self.conversion_factor_var.set(1.0)
                self.log_event("⚠️ Invalid conversion factor value, reset to default: 1.0 μm/px")

            # Count detected organoids and cysts
            total_organoids = len(self.organoid_data)
            total_cysts = sum(len(org["cysts"]) for org in self.organoid_data.values())

            # DETAILED DEBUG OUTPUT - ANALYSIS START
            self.log_event("🧬" + "=" * 60)
            self.log_event("🧬 DETAILED ORGANOID ANALYSIS DEBUG START")
            self.log_event("🧬" + "=" * 60)
            self.log_event("📊 Analysis Configuration:")
            self.log_event(f"   • Total Organoids: {total_organoids}")
            self.log_event(f"   • Total Cysts: {total_cysts}")
            self.log_event(f"   • Time Lapse: {time_lapse_days} days")
            self.log_event(f"   • Conversion Factor: {conversion_factor} μm/pixel")

            # Tracking data verification
            tracking_frame_count = len(self.video_segments) if self.video_segments else 0
            self.log_event(f"   • Tracking Frames: {tracking_frame_count}")

            if self.video_segments:
                sample_frame_idx = list(self.video_segments.keys())[0]
                sample_objects = list(self.video_segments[sample_frame_idx].keys())
                self.log_event(f"   • Sample Tracked Objects: {sample_objects}")

            # Model verification
            if self.opened is not None:
                self.log_event(f"   • Results: reopened from {self.opened.saved.directory}")
            elif self.tracking:
                self.log_event(f"   • Current Model: {type(self.tracking.tracker).__name__}")
                self.log_event(f"   • Original Frames Available: {len(self.tracking.frames)}")
            else:
                self.log_event("   • Current Model: MISSING")

            self.log_event("🧬" + "=" * 60)

        except Exception as e:
            self.set_status(f"Error reading analysis parameters: {str(e)}")
            self.log_event(f"❌ Parameter validation error: {e}")
            return

        # Ask for output directory
        token = self._results_token()
        output_dir = filedialog.askdirectory(
            title="Select Output Directory for Organoid Analysis Report", initialdir="./data/output_videos"
        )

        if not output_dir:
            return

        # The chooser is modal and runs the event loop: re-check after it returns (see generate_videos)
        if not self._ready_for_export("the analysis report") or not self._same_results(token, "the analysis report"):
            return

        # The parameters are read again now: they belong to the results shown after the chooser
        try:
            time_lapse_days = float(self.time_lapse_var.get())
            conversion_factor = float(self.conversion_factor_var.get())
        except (tk.TclError, ValueError) as error:
            self.set_status(f"Error reading analysis parameters: {error}")
            self.log_event(f"❌ Parameter validation error: {error}")
            return
        if time_lapse_days <= 0 or conversion_factor <= 0:
            self.set_status("The time lapse and the conversion factor must be positive")
            self.log_event("❌ Invalid analysis parameters")
            return

        start_time = time.time()

        self.analysis_btn.config(state="disabled")
        self.set_status("Generating organoid-cyst analysis report... This may take a moment.")

        # Log analysis start
        self.log_event(f"🧬 Starting organoid-cyst analysis for {total_organoids} organoids with {total_cysts} cysts")

        # Read the Tk variables and the annotations on the GUI thread before starting the worker
        debug = self.debug_var.get()
        organoid_data = {
            oid: {"point": info["point"], "cysts": list(info["cysts"])} for oid, info in self.organoid_data.items()
        }
        frames = self._export_frames()
        result = self.video_segments  # the worker measures this snapshot
        timing = Timing(time_lapse_days=time_lapse_days)
        if self.opened is not None and self.opened.saved.session.timing.frame_times_days is not None:
            timing = self.opened.saved.session.timing  # the run's explicit frame times, not a uniform span
            self.log_event("🕒 Using the explicit frame times recorded with the reopened run")

        def analysis_thread():
            try:
                # The same measurements and files as a headless run of this session
                annotations = AnnotationSet.from_organoid_data(organoid_data)
                analysis = AnalysisService().analyze(
                    result,
                    annotations,
                    Calibration(conversion_factor),
                    timing,
                    debug_mode=debug,
                )
                analysis_summary = ExportService(output_dir).write_report(
                    analysis, debug_mode=debug, original_frames=frames
                )

                analysis_time = time.time() - start_time

                # Update GUI in main thread with results for display
                self.post(self.on_organoid_analysis_complete_success, analysis_summary, analysis_time)

            except (AnnotationError, SessionError, ExportError) as e:
                analysis_time = time.time() - start_time
                self.post(self.on_organoid_analysis_complete_error, str(e), analysis_time)
            except Exception as e:
                analysis_time = time.time() - start_time
                self.post(self.on_organoid_analysis_complete_error, str(e), analysis_time)

        # Run analysis in background thread
        import threading

        analysis_thread = threading.Thread(target=analysis_thread, daemon=True)
        analysis_thread.start()

    def on_organoid_analysis_complete_success(self, analysis_summary, analysis_time):
        """Handle successful organoid analysis completion"""
        self.analysis_btn.config(state="normal")

        if not analysis_summary.get("success", False):
            self.on_organoid_analysis_complete_error(analysis_summary.get("error", "Unknown error"), analysis_time)
            return

        # DETAILED DEBUG OUTPUT - ANALYSIS COMPLETION
        self.log_event("🧬" + "=" * 60)
        self.log_event("🧬 DETAILED ORGANOID ANALYSIS DEBUG COMPLETE")
        self.log_event("🧬" + "=" * 60)
        self.log_event(f"🎉 Analysis completed in {analysis_time:.2f}s")
        tracking = analysis_summary.get("tracking") or {}
        if tracking and tracking.get("status") != "completed":
            self.log_event(
                f"⚠️ {str(tracking.get('status', 'partial')).upper()} TRACKING RUN: "
                f"{tracking.get('frames_done')} of {tracking.get('frames_total')} frames "
                f"were tracked ({tracking.get('error') or 'no error recorded'}); the report covers only those frames"
            )

        # Display analysis results summary in the results area
        self.log_event("🧬 ORGANOID-CYST ANALYSIS RESULTS")
        self.log_event("🧬" + "=" * 60)

        # Display experiment information
        exp_info = analysis_summary.get("experiment_info", {})
        self.log_event("🔬 Experiment Summary:")
        self.log_event(f"   • Total Organoids: {exp_info.get('total_organoids', 0)}")
        self.log_event(f"   • Total Cysts: {exp_info.get('total_cysts', 0)}")
        self.log_event(f"   • Frames Analyzed: {exp_info.get('total_frames', 0)}")
        self.log_event(f"   • Time Period: {exp_info.get('time_lapse_days', 0)} days")
        self.log_event(f"   • Conversion Factor: {exp_info.get('conversion_factor_um_per_pixel', 0)} μm/pixel")

        # Display quality metrics
        quality = analysis_summary.get("quality_metrics", {})
        self.log_event("📊 Data Quality:")
        self.log_event(f"   • Tracking Coverage: {quality.get('tracking_coverage_percent', 0):.1f}%")
        self.log_event(f"   • Mean Trajectory Length: {quality.get('mean_trajectory_length_frames', 0):.1f} frames")
        self.log_event(f"   • Organoids with Cysts: {quality.get('organoids_with_cysts', 0)}")

        # Display growth statistics
        growth = analysis_summary.get("growth_statistics", {})
        if growth.get("mean_growth_rate_um2_per_day", 0) > 0:
            self.log_event("📈 Growth Statistics:")
            self.log_event(f"   • Mean Growth Rate: {growth.get('mean_growth_rate_um2_per_day', 0):.4f} μm²/day")
            self.log_event(f"   • Max Growth Rate: {growth.get('max_growth_rate_um2_per_day', 0):.4f} μm²/day")
            self.log_event(f"   • Min Growth Rate: {growth.get('min_growth_rate_um2_per_day', 0):.4f} μm²/day")

        # Display output files
        output_files = analysis_summary.get("output_files", {})
        self.log_event("📄 Generated Files:")

        # CSV files
        csv_files = output_files.get("csv_files", {})
        if csv_files:
            self.log_event("   📊 CSV Data Files:")
            if csv_files.get("raw_data"):
                self.log_event(f"      • Raw Data: {Path(csv_files['raw_data']).name}")
            if csv_files.get("cyst_summary"):
                self.log_event(f"      • Cyst Summary: {Path(csv_files['cyst_summary']).name}")
            if csv_files.get("organoid_summary"):
                self.log_event(f"      • Organoid Summary: {Path(csv_files['organoid_summary']).name}")

        # Visualizations
        visualizations = output_files.get("visualizations", {})
        if visualizations:
            viz_count = len([v for v in visualizations.values() if v])
            self.log_event(f"   🎨 Visualizations: {viz_count} advanced plots generated")
            viz_names = {
                "organoids_with_cysts": "Organoids with Cysts vs Time",
                "cyst_organoid_ratio": "Cyst/Organoid Ratio vs Time",
                "cyst_areas_multiline": "Individual Cyst Area Trajectories",
                "cyst_circularity_multiline": "Individual Cyst Circularity Trajectories",
                "circularity_scatter": "Circularity Scatter (sized by area)",
                "lasagna_plot": "Organoid Growth Heatmap (Lasagna Plot)",
            }
            for viz_key, viz_path in visualizations.items():
                if viz_path and Path(viz_path).exists():
                    viz_name = viz_names.get(viz_key, viz_key.replace("_", " ").title())
                    self.log_event(f"      • {viz_name}")

        # PDF report
        pdf_path = output_files.get("pdf_report")
        if pdf_path and Path(pdf_path).exists():
            self.log_event(f"   📋 Enhanced PDF Report: {Path(pdf_path).name}")

        # Analysis timing
        self.log_event(f"⏱️ Analysis completed in {analysis_time:.2f} seconds")

        # Validation warnings
        validation = analysis_summary.get("validation_results", {})
        warnings = validation.get("warnings", [])
        if warnings:
            self.log_event("⚠️ Quality Warnings:")
            for warning in warnings:
                self.log_event(f"   • {warning}")

        self.log_event("=" * 60)
        self.log_event("✅ Comprehensive organoid analysis report generated successfully!")

        # Show completion status
        output_dir = Path(csv_files.get("raw_data", "")).parent if csv_files.get("raw_data") else None
        if output_dir:
            self.log_event(f"📁 All files saved to: {output_dir}")
            self.set_status(f"Analysis complete! Files saved to: {output_dir}")

            # Optional: Open directory automatically (with GTK-safe method)
            if self.auto_open_directory:
                try:
                    import os
                    import platform
                    import subprocess

                    if platform.system() == "Windows":
                        subprocess.run(["explorer", str(output_dir)], check=False)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", str(output_dir)], check=False)
                    else:  # Linux - suppress GTK warnings and run in background
                        with open(os.devnull, "w") as devnull:
                            subprocess.Popen(["xdg-open", str(output_dir)], stderr=devnull, stdout=devnull)

                    self.log_event("📂 Output directory opened automatically")
                except Exception:
                    self.log_event(f"ℹ️ Directory: {output_dir}")
        else:
            self.set_status("Analysis complete!")

    def on_organoid_analysis_complete_error(self, error_msg, analysis_time):
        """Handle organoid analysis error"""
        self.analysis_btn.config(state="normal")
        self.set_status(f"Analysis failed: {error_msg}")

        self.log_event("=" * 60)
        self.log_event("❌ ORGANOID ANALYSIS FAILED")
        self.log_event("=" * 60)
        self.log_event(f"Error: {error_msg}")
        self.log_event(f"Duration: {analysis_time:.2f} seconds")
        self.log_event("=" * 60)

        # Provide helpful suggestions
        self.log_event("💡 Troubleshooting suggestions:")
        self.log_event("   • Ensure tracking was completed successfully")
        self.log_event("   • Check that organoids and cysts were properly defined")
        self.log_event("   • Verify analysis parameters are valid")
        self.log_event("   • Try running with debug mode enabled")

    def on_analysis_complete_success(self, report_paths, analysis_time, results):
        """Handle successful analysis completion and display metrics in results area"""
        self.analysis_btn.config(state="normal")

        # Display analysis results summary in the results area
        self.log_event("=" * 50)
        self.log_event("📊 ORGANOID CYST ANALYSIS RESULTS")
        self.log_event("=" * 50)

        # Display parameters used
        params = results["parameters"]
        self.log_event("📋 Analysis Parameters:")
        self.log_event(f"   • Total Organoids: {params['total_organoids']}")
        self.log_event(f"   • Time Lapse: {params['time_lapse_days']} days")
        self.log_event(f"   • Conversion Factor: {params['conversion_factor_um_per_pixel']} μm/pixel")

        # Display cyst tracking summary
        summary = results["cyst_data_summary"]
        self.log_event("🎯 Tracking Summary:")
        self.log_event(f"   • Cysts Tracked: {summary['num_cysts_tracked']}")
        self.log_event(f"   • Object IDs: {summary['cyst_ids']}")

        # Display key metrics
        self.log_event("📈 Key Metrics:")
        for metric_name, metric_data in results["metrics"].items():
            if "error" in metric_data:
                self.log_event(f"   ❌ {metric_name}: Error - {metric_data['error']}")
                continue

            metric_data["info"]
            metric_results = metric_data["results"]

            if metric_name == "Cyst Formation Efficiency":
                value = metric_results.get("value", 0)
                organoids_with_cysts = metric_results.get("organoids_with_cysts", 0)
                total_organoids = metric_results.get("total_organoids", 0)
                self.log_event(f"   • {metric_name}: {value:.1f}% ({organoids_with_cysts}/{total_organoids} organoids)")

            elif metric_name == "De Novo Cyst Formation Rate":
                value = metric_results.get("value", 0)
                self.log_event(f"   • {metric_name}: {value:.2f} cysts/day")

            elif metric_name == "Radial Expansion Velocity":
                mean_val = metric_results.get("mean_value", 0)
                std_val = metric_results.get("std_value", 0)
                max_val = metric_results.get("max_value", 0)
                min_val = metric_results.get("min_value", 0)
                num_cysts = metric_results.get("num_cysts", 0)
                self.log_event(f"   • {metric_name}:")
                self.log_event(f"     - Mean: {mean_val:.2f} ± {std_val:.2f} μm/day")
                self.log_event(f"     - Range: {min_val:.2f} to {max_val:.2f} μm/day")
                self.log_event(f"     - Analyzed Cysts: {num_cysts}")

        self.log_event("=" * 50)

        # Log file generation results
        self.log_event(f"✅ Analysis completed in {analysis_time:.2f}s")

        if "csv" in report_paths:
            self.log_event(f"📄 CSV report: {Path(report_paths['csv']).name}")

        if "pdf" in report_paths:
            self.log_event(f"📋 PDF report: {Path(report_paths['pdf']).name}")

        if "json" in report_paths:
            self.log_event(f"💾 Analysis data: {Path(report_paths['json']).name}")

        # Log advanced visualizations
        viz_count = 0
        for key in report_paths:
            if key.startswith("viz_"):
                viz_count += 1

        if viz_count > 0:
            self.log_event(f"🎨 Advanced visualizations generated: {viz_count} plots")
            self.log_event("   📊 Collective outcome plots (bar charts, time series, box plots)")
            self.log_event("   🌱 De novo formation dynamics (cumulative counts, dual-axis)")
            self.log_event("   📏 Radial expansion heterogeneity (lasagna plots, velocity analysis)")
            self.log_event("   🔬 Morphological & spatial analysis (morphospace, density maps)")

        if "enhanced_pdf" in report_paths:
            self.log_event(f"📋 Enhanced PDF (with visualizations): {Path(report_paths['enhanced_pdf']).name}")

        if "visualization_summary" in report_paths:
            self.log_event(f"📝 Visualization summary: {Path(report_paths['visualization_summary']).name}")

        # Log any errors
        if "csv_error" in report_paths:
            self.log_event(f"❌ CSV generation failed: {report_paths['csv_error']}")

        if "pdf_error" in report_paths:
            self.log_event(f"❌ PDF generation failed: {report_paths['pdf_error']}")

        if "visualization_error" in report_paths:
            self.log_event(f"⚠️ Visualization warning: {report_paths['visualization_error']}")

        if "enhanced_pdf_error" in report_paths:
            self.log_event(f"⚠️ Enhanced PDF warning: {report_paths['enhanced_pdf_error']}")

        self.set_status("Comprehensive analysis report generated successfully!")

        # Add completion info
        if "csv" in report_paths:
            self.log_event(f"📊 Files saved to: {Path(report_paths['csv']).parent}")
        self.log_event("💡 Tip: Check the output directory for comprehensive analysis reports and visualizations")

        # Optional: Open directory automatically (configurable and GTK-safe)
        if self.auto_open_directory:
            try:
                if "csv" in report_paths:
                    import os
                    import platform
                    import subprocess

                    output_dir = Path(report_paths["csv"]).parent

                    if platform.system() == "Windows":
                        subprocess.run(["explorer", str(output_dir)], check=False)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", str(output_dir)], check=False)
                    else:  # Linux - suppress GTK warnings and run in background
                        # Suppress GTK warnings by redirecting stderr and run detached
                        with open(os.devnull, "w") as devnull:
                            subprocess.Popen(["xdg-open", str(output_dir)], stderr=devnull, stdout=devnull)

                    self.log_event("📂 Output directory opened automatically")
            except Exception:
                # Silently handle directory opening failures - just show path
                if "csv" in report_paths:
                    self.log_event(f"ℹ️ Directory: {Path(report_paths['csv']).parent}")
        else:
            # Just show the directory path when auto-open is disabled
            if "csv" in report_paths:
                self.log_event(f"ℹ️ Directory: {Path(report_paths['csv']).parent}")

    def on_analysis_complete_error(self, error_msg, analysis_time):
        """Handle analysis error"""
        self.analysis_btn.config(state="normal")
        self.set_status(f"Error during analysis: {error_msg}")
        self.log_event(f"❌ Analysis failed after {analysis_time:.2f}s: {error_msg}")


if __name__ == "__main__":
    app = VideoTrackerApp()
    app.run()
