#!/usr/bin/env python3
"""
Main GUI Window for Multi-Model Video Object Tracking
Built with Tkinter for beginner-friendly GUI development
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import sys
from typing import Optional, Dict, Any
import queue
import time

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import modules using absolute imports after path setup
try:
    from src.core.model_registry import get_model_registry, ModelFactory
    from src.core.base_model import BaseVideoTracker
    from src.utils.video_output import VideoOutputGenerator
    from src.analysis import MetricsCalculator, ReportGenerator, AnalysisParameters
    from src.gui.video_canvas import VideoCanvas
    from src.gui.progress_dialog import ProgressDialog
except ImportError:
    # Fallback to relative imports if absolute imports fail
    from ..core.model_registry import get_model_registry, ModelFactory
    from ..core.base_model import BaseVideoTracker
    from ..utils.video_output import VideoOutputGenerator
    from ..analysis import MetricsCalculator, ReportGenerator, AnalysisParameters
    from .video_canvas import VideoCanvas
    from .progress_dialog import ProgressDialog


class VideoTrackerApp:
    """
    Main GUI window for video object tracking

    Features:
    - Model selection (Medical-SAM2, SAM2, future models)
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
        self.root.state('zoomed') if sys.platform == 'win32' else None

        # Enable window resizing
        self.root.resizable(True, True)

        # Application state
        self.current_model: Optional[BaseVideoTracker] = None
        self.current_video_path: Optional[str] = None
        self.video_segments: Optional[Dict] = None
        self.tracking_in_progress = False

        # Progress dialog references
        self.tracking_dialog = None
        self.generation_dialog = None

        # Analysis parameter variables (simplified - no manual organoid input needed)
        self.time_lapse_var = tk.DoubleVar(value=7.0)   # Default: 7 days
        self.conversion_factor_var = tk.DoubleVar(value=1.0)  # Default: 1.0 μm/pixel

        # New organoid-cyst tracking state
        self.current_organoid_id = None  # Currently selected organoid
        self.organoid_mode = True  # True = waiting for organoid click, False = adding cysts
        self.organoid_data = {}  # organoid_id -> {'point': (x,y), 'cysts': [(x1,y1,x2,y2), ...]}
        self.next_organoid_id = 1
        self.next_cyst_id = 1

        # Configuration options (load from config file)
        try:
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from config import AUTO_OPEN_OUTPUT_DIRECTORY, DEFAULT_TIME_LAPSE_DAYS, DEFAULT_CONVERSION_FACTOR
            self.auto_open_directory = AUTO_OPEN_OUTPUT_DIRECTORY
            # Update defaults from config (no more manual organoid count needed)
            self.time_lapse_var.set(DEFAULT_TIME_LAPSE_DAYS)
            self.conversion_factor_var.set(DEFAULT_CONVERSION_FACTOR)
        except ImportError:
            # Fallback to hardcoded defaults if config not available
            self.auto_open_directory = True

        # GUI setup
        self.setup_styles()
        self.create_widgets()
        self.setup_layout()
        self.setup_bindings()

        # Initialize model list and set initial checkpoint display
        self.update_model_list()
        self.on_model_size_changed()  # Set initial checkpoint display

        # Status
        self.set_status("Select a model and load it to begin.")

    def setup_styles(self):
        """Setup custom styles for the GUI"""
        self.style = ttk.Style()

        # Configure styles for better appearance
        self.style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        self.style.configure('Header.TLabel', font=('Arial', 10, 'bold'))
        self.style.configure('Status.TLabel', font=('Arial', 9))
        self.style.configure('VideoLoad.TButton', font=('Arial', 10, 'bold'))
        self.style.configure('Action.TButton', font=('Arial', 10))

        # Error style for analysis input validation (MobaXterm-friendly)
        try:
            self.style.configure('Error.TEntry', fieldbackground='#ffe6e6', bordercolor='red')
        except:
            # Fallback if style configuration fails in MobaXterm
            pass

    def create_widgets(self):
        """Create all GUI widgets"""
        # Main title
        self.title_label = ttk.Label(
            self.root,
            text="Multi-Model Video Object Tracker",
            style='Title.TLabel'
        )

        # Step 1: Model selection frame
        self.model_frame = ttk.LabelFrame(self.root, text="Step 1: Model Configuration", padding=15)

        self.model_label = ttk.Label(self.model_frame, text="Model:", style='Header.TLabel')
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self.model_frame,
            textvariable=self.model_var,
            state='readonly',
            width=25
        )
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)

        self.load_model_btn = ttk.Button(
            self.model_frame,
            text="Load Model",
            command=self.load_selected_model
        )

        # Model configuration options
        self.config_frame = ttk.LabelFrame(self.model_frame, text="Model Settings", padding=10)

        # Device selection
        self.device_label = ttk.Label(self.config_frame, text="Device:")
        self.device_var = tk.StringVar(value="cuda")
        self.device_combo = ttk.Combobox(
            self.config_frame,
            textvariable=self.device_var,
            values=["cuda", "cpu"],
            state='readonly',
            width=15
        )

        # Model config selection
        self.model_config_label = ttk.Label(self.config_frame, text="Model Size:")
        self.model_config_var = tk.StringVar(value="sam2_hiera_b")
        self.model_config_combo = ttk.Combobox(
            self.config_frame,
            textvariable=self.model_config_var,
            values=["sam2_hiera_s", "sam2_hiera_b", "sam2_hiera_l"],
            state='readonly',
            width=15
        )
        self.model_config_combo.bind('<<ComboboxSelected>>', self.on_model_size_changed)

        # Debug mode toggle
        self.debug_label = ttk.Label(self.config_frame, text="Debug Mode:")
        self.debug_var = tk.BooleanVar(value=False)
        self.debug_check = ttk.Checkbutton(
            self.config_frame,
            text="Enable Debug Output",
            variable=self.debug_var
        )

        # Adaptive tracking toggle
        self.adaptive_label = ttk.Label(self.config_frame, text="Tracking Mode:")
        self.adaptive_var = tk.BooleanVar(value=True)  # Default to enabled
        self.adaptive_check = ttk.Checkbutton(
            self.config_frame,
            text="Enable Adaptive Tracking (reduces drift)",
            variable=self.adaptive_var
        )
        
        # Reverse tracking toggle
        self.reverse_label = ttk.Label(self.config_frame, text="Temporal Direction:")
        self.reverse_var = tk.BooleanVar(value=True)  # Default to reverse (biological use case)
        self.reverse_check = ttk.Checkbutton(
            self.config_frame,
            text="Enable Reverse Tracking (last→first frame)",
            variable=self.reverse_var
        )

        # Video quality selection
        self.quality_label = ttk.Label(self.config_frame, text="Output Quality:")
        self.quality_var = tk.StringVar(value="original")
        self.quality_combo = ttk.Combobox(
            self.config_frame,
            textvariable=self.quality_var,
            values=["original", "mid", "low"],
            state='readonly',
            width=15
        )

        # Checkpoint info (read-only display)
        self.checkpoint_label = ttk.Label(self.config_frame, text="Checkpoint:")
        self.checkpoint_info = ttk.Label(
            self.config_frame,
            text="sam2.1_hiera_small.pt",
            style='Status.TLabel'
        )

        # Step 2: Video loading frame
        self.video_frame = ttk.LabelFrame(self.root, text="Step 2: Video Loading", padding=15)

        self.load_video_btn = ttk.Button(
            self.video_frame,
            text="📹 Load Video",
            command=self.load_video,
            state='disabled',
            width=18,
            style='VideoLoad.TButton'
        )

        self.video_info_label = ttk.Label(
            self.video_frame,
            text="No video loaded",
            style='Status.TLabel',
            wraplength=250
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
        self.object_label = ttk.Label(self.object_mgmt_frame, text="Objects:", style='Header.TLabel')

        # Object management settings
        self.max_objects = 20  # Maximum number of objects
        self.active_object_ids = set()  # No background by default - separate feature
        self.action_history = []  # Stack for multi-level undo (like Ctrl+Z)
        self.next_object_id = 1  # Track next ID to assign

        # Object colors - expanded palette for 20+ objects
        self.object_colors = {
            0: '#808080',   # Gray - Background
            1: '#FF0000',   # Red
            2: '#00FF00',   # Green
            3: '#0000FF',   # Blue
            4: '#FFFF00',   # Yellow
            5: '#FF00FF',   # Magenta
            6: '#00FFFF',   # Cyan
            7: '#FFA500',   # Orange
            8: '#800080',   # Purple
            9: '#FFC0CB',   # Pink
            10: '#A52A2A',  # Brown
            11: '#90EE90',  # Light Green
            12: '#87CEEB',  # Sky Blue
            13: '#DDA0DD',  # Plum
            14: '#F0E68C',  # Khaki
            15: '#FF6347',  # Tomato
            16: '#40E0D0',  # Turquoise
            17: '#EE82EE',  # Violet
            18: '#FFB6C1',  # Light Pink
            19: '#98FB98',  # Pale Green
            20: '#F5DEB3',  # Wheat
        }

        # Object management - simplified (add/remove through canvas clicks and revert button)

        # Object display (no dropdown needed - automatic sequential numbering)

        # New organoid-cyst workflow section
        self.workflow_frame = ttk.LabelFrame(self.object_frame, text="Organoid-Cyst Workflow", padding=5)

        # Workflow status display
        self.workflow_status_label = ttk.Label(
            self.workflow_frame,
            text="Click on an organoid location",
            font=('Arial', 10, 'bold')
        )

        # Simplified workflow - no need for next/finish buttons
        # Users can simply click on the next organoid when ready

        # Active objects display (scrollable for many objects)
        self.active_objects_frame = ttk.Frame(self.object_frame)
        self.active_objects_label = ttk.Label(
            self.active_objects_frame,
            text="Active Objects: None",
            style='Status.TLabel'
        )

        # Object list (scrollable text widget for many objects)
        self.object_list_text = tk.Text(
            self.active_objects_frame,
            height=3,
            width=50,
            state='disabled',
            wrap='word',
            font=('Arial', 9)
        )

        # Initialize object list
        self.update_active_objects_display()

        # Controls frame
        self.controls_frame = ttk.LabelFrame(self.display_frame, text="Tracking Controls", padding=10)

        self.clear_prompts_btn = ttk.Button(
            self.controls_frame,
            text="Clear All Objects",
            command=self.clear_prompts,
            state='disabled'
        )

        self.revert_btn = ttk.Button(
            self.controls_frame,
            text="/Revert Last",
            command=self.revert_last_action,
            state='disabled'
        )

        self.track_btn = ttk.Button(
            self.controls_frame,
            text="🚀 Start Multi-Object Tracking",
            command=self.start_tracking,
            state='disabled',
            width=25
        )

        # Prompt info
        self.prompt_info_label = ttk.Label(
            self.controls_frame,
            text="Left click anywhere on the video to add a new object\nNumbers will appear to identify each object",
            style='Status.TLabel'
        )

        # Step 4: Output frame (video generation)
        self.output_frame = ttk.LabelFrame(self.root, text="Step 4: Video Output", padding=10)

        # Video generation button
        self.generate_btn = ttk.Button(
            self.output_frame,
            text="🎬 Generate Videos",
            command=self.generate_videos,
            state='disabled'
        )

        # Step 5: Analysis frame (separate section)
        self.analysis_frame = ttk.LabelFrame(self.root, text="Step 5: Organoid Cyst Analysis", padding=10)

        # Analysis parameters sub-frame
        self.analysis_params_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Parameters", padding=5)

        # Analysis parameter inputs (simplified - organoid count detected automatically)
        self.time_lapse_label = ttk.Label(self.analysis_params_frame, text="Time Lapse (days):")
        self.time_lapse_entry = ttk.Entry(
            self.analysis_params_frame,
            width=10,
            textvariable=self.time_lapse_var,
            font=('Arial', 10)
        )

        self.conversion_factor_label = ttk.Label(self.analysis_params_frame, text="Conversion Factor (μm/pixel):")
        self.conversion_factor_entry = ttk.Entry(
            self.analysis_params_frame,
            width=15,
            textvariable=self.conversion_factor_var,
            font=('Arial', 10)
        )

        # Organoid count display (auto-detected)
        self.organoid_count_label = ttk.Label(self.analysis_params_frame, text="Detected Organoids:")
        self.organoid_count_display = ttk.Label(self.analysis_params_frame, text="0", font=('Arial', 10, 'bold'))

        # Add validation for better user experience (especially with MobaXterm)
        self.time_lapse_entry.bind('<KeyRelease>', self._validate_analysis_inputs)
        self.conversion_factor_entry.bind('<KeyRelease>', self._validate_analysis_inputs)

        # Add focus events for visual feedback
        self.time_lapse_entry.bind('<FocusIn>', lambda e: self._on_analysis_entry_focus(e, 'time'))
        self.conversion_factor_entry.bind('<FocusIn>', lambda e: self._on_analysis_entry_focus(e, 'conversion'))

        # Analysis report button
        self.analysis_btn = ttk.Button(
            self.analysis_frame,
            text="📊 Generate Analysis Report",
            command=self.generate_analysis_report,
            state='disabled'
        )

        # Output and results information text area (shared between output and analysis)
        self.results_frame = ttk.LabelFrame(self.root, text="Results & Log", padding=10)
        self.output_info_text = scrolledtext.ScrolledText(
            self.results_frame,
            height=8,
            width=50,
            state='disabled'
        )

        # Status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready",
            style='Status.TLabel',
            relief='sunken'
        )

        # Progress bar (initially hidden)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.status_frame,
            variable=self.progress_var,
            mode='determinate'
        )

    def setup_layout(self):
        """Setup the scalable layout of all widgets with proper step organization"""
        # Main title
        self.title_label.grid(row=0, column=0, columnspan=3, pady=10, sticky='ew')

        # Left column - Steps 1 & 2
        self.model_frame.grid(row=1, column=0, padx=5, pady=5, sticky='new')
        self.video_frame.grid(row=2, column=0, padx=5, pady=5, sticky='ew')

        # Middle column - Step 3 (main tracking area)
        self.display_frame.grid(row=1, column=1, rowspan=2, padx=5, pady=5, sticky='nsew')

        # Right column - Steps 4 & 5
        self.output_frame.grid(row=1, column=2, padx=5, pady=5, sticky='new')
        self.analysis_frame.grid(row=2, column=2, padx=5, pady=5, sticky='new')

        # Bottom section - Results and status (spans all columns)
        self.results_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        self.status_frame.grid(row=4, column=0, columnspan=3, sticky='ew', padx=5, pady=2)

        # Model frame layout
        self.model_label.grid(row=0, column=0, sticky='w', pady=2)
        self.model_combo.grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        self.load_model_btn.grid(row=0, column=2, padx=5, pady=2)
        self.config_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky='ew')

        # Configuration frame layout
        self.device_label.grid(row=0, column=0, sticky='w', padx=2, pady=2)
        self.device_combo.grid(row=0, column=1, padx=5, pady=2)

        self.model_config_label.grid(row=0, column=2, sticky='w', padx=10, pady=2)
        self.model_config_combo.grid(row=0, column=3, padx=5, pady=2)

        # Row 1: Debug and Quality controls
        self.debug_label.grid(row=1, column=0, sticky='w', padx=2, pady=2)
        self.debug_check.grid(row=1, column=1, padx=5, pady=2, sticky='w')

        self.quality_label.grid(row=1, column=2, sticky='w', padx=10, pady=2)
        self.quality_combo.grid(row=1, column=3, padx=5, pady=2)

        # Row 2: Adaptive tracking controls
        self.adaptive_label.grid(row=2, column=0, sticky='w', padx=2, pady=2)
        self.adaptive_check.grid(row=2, column=1, columnspan=2, padx=5, pady=2, sticky='w')

        # Row 3: Reverse tracking controls
        self.reverse_label.grid(row=3, column=0, sticky='w', padx=2, pady=2)
        self.reverse_check.grid(row=3, column=1, columnspan=2, padx=5, pady=2, sticky='w')

        # Row 4: Checkpoint info
        self.checkpoint_label.grid(row=4, column=0, sticky='w', padx=2, pady=2)
        self.checkpoint_info.grid(row=4, column=1, columnspan=3, padx=5, pady=2, sticky='ew')

        self.model_frame.columnconfigure(1, weight=1)
        self.config_frame.columnconfigure(3, weight=1)

        # Video frame layout
        self.load_video_btn.grid(row=0, column=0, pady=5, sticky='ew')
        self.video_info_label.grid(row=1, column=0, pady=10, sticky='new')

        # Make video frame expand properly
        self.video_frame.columnconfigure(0, weight=1)

        # Display frame layout
        self.video_canvas.grid(row=0, column=0, columnspan=2, pady=5)
        self.object_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        self.controls_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10, padx=5)

        # Object management layout - simplified
        self.object_mgmt_frame.grid(row=0, column=0, sticky='w', pady=2)
        self.object_label.grid(row=0, column=0, sticky='w', padx=5)

        # Workflow layout - simplified
        self.workflow_frame.grid(row=1, column=0, sticky='ew', pady=5, padx=5)
        self.workflow_status_label.grid(row=0, column=0, padx=5, pady=2)

        self.active_objects_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        self.active_objects_label.grid(row=0, column=0, sticky='w')
        self.object_list_text.grid(row=1, column=0, sticky='ew', pady=2)

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
        self.prompt_info_label.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky='w')

        self.controls_frame.columnconfigure(0, weight=1)
        self.controls_frame.columnconfigure(1, weight=1)
        self.controls_frame.columnconfigure(2, weight=1)

        # Step 4: Output frame layout (simple video generation)
        self.generate_btn.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        self.output_frame.columnconfigure(0, weight=1)

        # Step 5: Analysis frame layout
        self.analysis_params_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        self.analysis_btn.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

        # Analysis parameters layout (simplified)
        self.organoid_count_label.grid(row=0, column=0, sticky='w', padx=2, pady=2)
        self.organoid_count_display.grid(row=0, column=1, padx=5, pady=2, sticky='w')

        self.time_lapse_label.grid(row=1, column=0, sticky='w', padx=2, pady=2)
        self.time_lapse_entry.grid(row=1, column=1, padx=5, pady=2)

        self.conversion_factor_label.grid(row=2, column=0, sticky='w', padx=2, pady=2)
        self.conversion_factor_entry.grid(row=2, column=1, padx=5, pady=2, sticky='ew')

        # Configure analysis frame weights
        self.analysis_frame.columnconfigure(0, weight=1)
        self.analysis_params_frame.columnconfigure(1, weight=1)

        # Results frame layout
        self.output_info_text.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)

        # Status frame layout
        self.status_label.grid(row=0, column=0, sticky='ew', padx=2)
        self.progress_bar.grid(row=0, column=1, sticky='ew', padx=2)

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
        self.root.bind('<Configure>', self.on_window_resize)

    def on_window_resize(self, event):
        """Handle window resize to auto-zoom canvas"""
        # Only resize for the root window, not child widgets
        if event.widget == self.root:
            # Get available space for display frame
            self.root.update_idletasks()  # Ensure geometry is updated

            # Calculate new canvas size based on available space
            display_frame_width = self.display_frame.winfo_width()
            display_frame_height = self.display_frame.winfo_height()

            if display_frame_width > 100 and display_frame_height > 100:  # Valid size
                # Reserve space for controls and padding
                canvas_width = max(400, display_frame_width - 40)  # Min 400px
                canvas_height = max(300, display_frame_height - 200)  # Min 300px, reserve 200px for controls

                # Update canvas size
                self.video_canvas.config(width=canvas_width, height=canvas_height)

    def setup_bindings(self):
        """Setup event bindings"""
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)

        # Ensure window focuses on video loading button when model is loaded
        self.root.bind('<Button-1>', self.ensure_focus)

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

            self.model_combo['values'] = model_options

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

        # Map model config to checkpoint filename
        checkpoint_map = {
            'sam2_hiera_s': 'sam2.1_hiera_small.pt',
            'sam2_hiera_b': 'sam2.1_hiera_base_plus.pt',
            'sam2_hiera_l': 'sam2.1_hiera_large.pt',
            'sam2_hiera_t': 'sam2.1_hiera_tiny.pt'
        }

        checkpoint_file = checkpoint_map.get(model_config, 'sam2.1_hiera_small.pt')
        self.checkpoint_info.config(text=checkpoint_file)

    def on_model_selected(self, event=None):
        """Handle model selection - simplified without description"""
        selected = self.model_var.get()
        if selected:
            # Extract model name for status update
            model_name = selected.split('(')[0].strip()
            self.set_status(f"Selected {model_name}. Configure settings and click 'Load Model'.")

    def load_selected_model(self):
        """Load the selected model with user configuration"""
        selected = self.model_var.get()
        if not selected:
            self.set_status("Please select a model first")
            self.log_event("❌ No model selected")
            return

        model_name = selected.split('(')[-1].rstrip(')')

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
            self.load_model_btn.config(state='disabled')
            self.log_event(f"🔄 Loading {selected} with {device.upper()} device...")

            # Use threading to prevent GUI freeze
            def load_model_thread():
                try:
                    registry = get_model_registry()
                    
                    # Get tracking settings from GUI
                    enable_adaptive = self.adaptive_var.get()
                    enable_reverse = self.reverse_var.get()
                    
                    model = registry.create_model_instance(
                        model_name,
                        device=device,
                        model_config=model_config,
                        checkpoint_path=checkpoint_path,
                        enable_adaptive_tracking=enable_adaptive,
                        enable_reverse_tracking=enable_reverse
                    )

                    if model and model.load_model():
                        load_time = time.time() - start_time
                        self.current_model = model
                        self.root.after(0, self.on_model_loaded_success, load_time)
                    else:
                        load_time = time.time() - start_time
                        self.root.after(0, self.on_model_loaded_error, "Failed to load model", load_time)

                except Exception as e:
                    load_time = time.time() - start_time
                    self.root.after(0, self.on_model_loaded_error, str(e), load_time)

            threading.Thread(target=load_model_thread, daemon=True).start()

        except Exception as e:
            load_time = time.time() - start_time
            self.on_model_loaded_error(str(e), load_time)

    def on_model_loaded_success(self, load_time):
        """Handle successful model loading"""
        self.load_model_btn.config(state='normal')
        self.load_video_btn.config(state='normal')
        
        # Check tracking settings and show feedback
        adaptive_status = "adaptive" if self.adaptive_var.get() else "static"
        direction_status = "reverse" if self.reverse_var.get() else "forward"
        tracking_status = f"with {adaptive_status} {direction_status} tracking"
        self.set_status(f"Model loaded successfully {tracking_status}! Ready to load video.")

        # Get model name for more specific logging
        selected = self.model_var.get()
        registry = get_model_registry()
        metadata = registry.get_model_metadata(selected)
        model_name = metadata.display_name if metadata else selected
        
        # Enhanced logging with complete tracking status
        adaptive_icon = "🎯" if self.adaptive_var.get() else "📦"
        direction_icon = "⏪" if self.reverse_var.get() else "⏩"
        tracking_mode = f"{adaptive_icon} {adaptive_status.title()} {direction_icon} {direction_status.title()}"
        self.log_event(f"✅ {model_name} loaded successfully in {load_time:.2f}s ({tracking_mode} tracking)")

    def on_model_loaded_error(self, error_msg, load_time):
        """Handle model loading error"""
        self.load_model_btn.config(state='normal')
        self.set_status(f"Error loading model: {error_msg}")
        self.log_event(f"❌ Model loading failed after {load_time:.2f}s: {error_msg}")
        # Remove popup - just use status and log (already handled above)

    def log_event(self, message):
        """Log an event with timestamp to the results area"""
        import time
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        log_message = f"[{timestamp}] {message}\n"

        self.output_info_text.config(state='normal')
        self.output_info_text.insert(tk.END, log_message)
        self.output_info_text.see(tk.END)  # Auto-scroll to bottom
        self.output_info_text.config(state='disabled')

    def load_video(self):
        """Load a video file"""
        if not self.current_model:
            self.set_status("Please load a model first")
            self.log_event("❌ No model loaded")
            return

        # File dialog for video selection
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=file_types,
            initialdir="./data/input_videos"
        )

        if not file_path:
            return

        try:
            import time
            start_time = time.time()

            self.set_status("Loading video... Please wait.")
            self.load_video_btn.config(state='disabled')
            self.log_event(f"🎬 Loading video: {Path(file_path).name}")

            # Load video in thread to prevent GUI freeze
            def load_video_thread():
                try:
                    video_info = self.current_model.load_video(file_path)
                    load_time = time.time() - start_time
                    self.root.after(0, self.on_video_loaded_success, file_path, video_info, load_time)
                except Exception as e:
                    load_time = time.time() - start_time
                    self.root.after(0, self.on_video_loaded_error, str(e), load_time)

            threading.Thread(target=load_video_thread, daemon=True).start()

        except Exception as e:
            load_time = time.time() - start_time
            self.on_video_loaded_error(str(e), load_time)

    def on_video_loaded_success(self, file_path, video_info, load_time):
        """Handle successful video loading"""
        self.load_video_btn.config(state='normal')
        self.current_video_path = file_path

        # Clear all objects from previous video (if model is loaded)
        if hasattr(self, 'current_model') and self.current_model:
            self.clear_prompts()
            self.log_event("🧹 Cleared all objects from previous video")
        else:
            self.log_event("📹 Video loaded - no previous objects to clear")

        # Update video info display
        info_text = f"✅ Video loaded: {Path(file_path).name}\n"
        info_text += f"Frames: {video_info.get('num_frames', 'Unknown')}\n"
        info_text += f"FPS: {video_info.get('fps', 'Unknown'):.1f}\n"

        # Extract dimensions correctly (dimensions is a tuple: height, width)
        dimensions = video_info.get('dimensions', (0, 0))
        if isinstance(dimensions, tuple) and len(dimensions) >= 2:
            height, width = dimensions[:2]
            info_text += f"Size: {width}x{height}"
        else:
            # Fallback for individual width/height keys
            width = video_info.get('width', '?')
            height = video_info.get('height', '?')
            info_text += f"Size: {width}x{height}"

            self.video_info_label.config(text=info_text)

        # Display first frame
        if hasattr(self.current_model, 'video_frames') and self.current_model.video_frames:
            self.video_canvas.display_frame(self.current_model.video_frames[0])

        # Enable object controls (simplified UI)
            self.clear_prompts_btn.config(state='normal')
            self.track_btn.config(state='normal')

        # Initialize workflow status
        self.update_workflow_status()

        self.set_status("Video loaded! Drag a bounding box around an organoid to start.")
        self.log_event(f"✅ Video loaded in {load_time:.2f}s ({video_info.get('num_frames', '?')} frames)")

    def on_video_loaded_error(self, error_msg, load_time):
        """Handle video loading error"""
        self.load_video_btn.config(state='normal')
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
                    self.time_lapse_entry.config(style='Error.TEntry')
                else:
                    self.time_lapse_entry.config(style='TEntry')
            except (tk.TclError, ValueError):
                self.time_lapse_entry.config(style='Error.TEntry')

            # Validate conversion factor
            try:
                conv_val = self.conversion_factor_var.get()
                if conv_val <= 0:
                    self.conversion_factor_entry.config(style='Error.TEntry')
                else:
                    self.conversion_factor_entry.config(style='TEntry')
            except (tk.TclError, ValueError):
                self.conversion_factor_entry.config(style='Error.TEntry')

        except Exception as e:
            # Silently handle validation errors
            pass

    def _on_analysis_entry_focus(self, event, field_type):
        """Handle focus events for analysis entry fields with MobaXterm compatibility"""
        try:
            widget = event.widget
            widget.select_range(0, 'end')  # Select all text for easy editing

            # Provide helpful status messages
            if field_type == 'time':
                self.set_status("Enter time lapse period in days")
            elif field_type == 'conversion':
                self.set_status("Enter conversion factor (micrometers per pixel)")

        except Exception as e:
            # Silently handle focus errors (common with MobaXterm)
            pass

    def on_canvas_click(self, x, y):
        """Handle click events for organoid placement"""
        if not self.current_model or not self.current_video_path:
            return

        # Set debug mode on current model
        self.current_model.debug_mode = self.debug_var.get()

        try:
            # Create new organoid entry at click location
            organoid_id = self.next_organoid_id
            self.organoid_data[organoid_id] = {
                'point': (x, y),
                'cysts': []
            }
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
            action = {
                'type': 'add_organoid',
                'organoid_id': organoid_id,
                'point': (x, y)
            }
            self.action_history.append(action)

            # Enable controls
            self.revert_btn.config(state='normal')

            self.set_status(f"Organoid {organoid_id} placed. Now drag bounding boxes around its cysts.")
            self.log_event(f"🔴 Added organoid {organoid_id} at ({x},{y})")

        except Exception as e:
            error_msg = f"Error placing organoid: {str(e)}"
            self.set_status(error_msg)
            self.log_event(f"❌ {error_msg}")
            print(f"❌ DEBUG: Error in on_canvas_click: {e}")

    def on_canvas_bbox(self, x1, y1, x2, y2):
        """Handle bounding box creation for cyst addition"""
        if not self.current_model or not self.current_video_path:
            return

        # Set debug mode on current model
        self.current_model.debug_mode = self.debug_var.get()

        try:
            # Cyst addition mode - send bounding box to SAM2 for tracking
            if self.current_organoid_id is None:
                self.set_status("Please click on an organoid location first")
                return

            cyst_id = self.next_cyst_id

            # Add cyst bounding box to SAM2 for tracking
            success = self.current_model.add_bbox_prompt(x1, y1, x2, y2, obj_id=cyst_id)

            if success:
                # Store cyst information
                self.organoid_data[self.current_organoid_id]['cysts'].append({
                    'cyst_id': cyst_id,
                    'bbox': (x1, y1, x2, y2)
                })
                self.next_cyst_id += 1

                # Add to active objects for tracking
                self.active_object_ids.add(cyst_id)

                # Store action in history for revert functionality
                action = {
                    'type': 'add_cyst',
                    'organoid_id': self.current_organoid_id,
                    'cyst_id': cyst_id,
                    'bbox': (x1, y1, x2, y2)
                }
                self.action_history.append(action)

                # Visual feedback - add cyst bounding box
                self.video_canvas.add_bbox_marker(x1, y1, x2, y2, obj_id=cyst_id)

                # Update displays
                self.update_active_objects_display()
                self.update_organoid_count_display()

                # Enable controls
                self.clear_prompts_btn.config(state='normal')
                self.revert_btn.config(state='normal')
                self.track_btn.config(state='normal')

                cyst_count = len(self.organoid_data[self.current_organoid_id]['cysts'])
                self.set_status(f"Added cyst {cyst_count} to organoid {self.current_organoid_id}")
                self.log_event(f"🔵 Added cyst {cyst_id} to organoid {self.current_organoid_id}: ({x1},{y1})-({x2},{y2})")

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
            print(f"❌ DEBUG: Error in on_canvas_bbox: {e}")
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
            cyst_count = len(self.organoid_data[self.current_organoid_id]['cysts'])
            total_organoids = len(self.organoid_data)
            self.workflow_status_label.config(text=f"Organoid {self.current_organoid_id} ({cyst_count} cysts) | Total: {total_organoids} organoids | Click for new organoid, drag for cysts")

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
            if last_action['type'] == 'add_cyst':
                # Reverting cyst addition
                organoid_id = last_action['organoid_id']
                cyst_id = last_action['cyst_id']

                # Clear cyst prompts from the model
                if self.current_model:
                    success = self.current_model.clear_prompts(cyst_id)
                    if not success:
                        print(f"⚠️ Warning: Failed to clear cyst {cyst_id}")

                # Remove from active objects
                self.active_object_ids.discard(cyst_id)

                # Remove from organoid data
                if organoid_id in self.organoid_data:
                    self.organoid_data[organoid_id]['cysts'] = [
                        c for c in self.organoid_data[organoid_id]['cysts']
                        if c['cyst_id'] != cyst_id
                    ]

                # Clear visual markers
                self.video_canvas.clear_markers(cyst_id)

                # Fix ID continuity: if this was the highest cyst ID, adjust next_cyst_id
                if cyst_id == self.next_cyst_id - 1:
                    # Find the actual highest cyst ID still in use
                    max_cyst_id = 0
                    for org_data in self.organoid_data.values():
                        for cyst in org_data['cysts']:
                            max_cyst_id = max(max_cyst_id, cyst['cyst_id'])
                    self.next_cyst_id = max_cyst_id + 1

                # Check if organoid has no cysts left - auto remove organoid
                if organoid_id in self.organoid_data and len(self.organoid_data[organoid_id]['cysts']) == 0:
                    # Remove empty organoid
                    self.video_canvas.clear_markers(f"organoid_{organoid_id}")
                    del self.organoid_data[organoid_id]

                    # Reset current organoid if this was it
                    if self.current_organoid_id == organoid_id:
                        self.current_organoid_id = None

                    # IMPORTANT: Remove the corresponding add_organoid action from history to prevent redundant removal
                    self.action_history = [
                        action for action in self.action_history
                        if not (action['type'] == 'add_organoid' and action['organoid_id'] == organoid_id)
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

            elif last_action['type'] == 'add_organoid':
                # Reverting organoid addition (would also remove all its cysts)
                organoid_id = last_action['organoid_id']

                # Remove all cysts for this organoid
                if organoid_id in self.organoid_data:
                    for cyst in self.organoid_data[organoid_id]['cysts']:
                        cyst_id = cyst['cyst_id']
                        if self.current_model:
                            self.current_model.clear_prompts(cyst_id)
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
                self.revert_btn.config(state='disabled')

            # Update button states
            if not self.active_object_ids:
                self.track_btn.config(state='disabled')
                self.clear_prompts_btn.config(state='disabled')

        except Exception as e:
            error_msg = f"Failed to revert: {str(e)}"
            self.set_status(error_msg)
            self.log_event(f"❌ {error_msg}")





    def debug_button_states(self):
        """Debug method to check button states - simplified for new interface"""
        print(f"🔍 DEBUG: Button States Check:")
        print(f"  - Clear All Objects: {self.clear_prompts_btn['state']}")
        print(f"  - Revert Last: {self.revert_btn['state']}")
        print(f"  - Track: {self.track_btn['state']}")
        print(f"  - Active Objects: {self.active_object_ids}")
        print(f"  - Action History: {len(self.action_history)} actions")
        print(f"  - Background Mode: {self.background_mode}")
        print(f"  - Next Object ID: {self.next_object_id}")

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
        if not self.current_model:
            self.active_objects_label.config(text="Active Objects: None")
            self.object_list_text.config(state='normal')
            self.object_list_text.delete(1.0, tk.END)
            self.object_list_text.config(state='disabled')
            return

        try:
            object_info = []
            total_organoids = len(self.organoid_data)
            total_cysts = len(self.active_object_ids)

            # Show organoid and cyst information
            for organoid_id, organoid in self.organoid_data.items():
                cyst_count = len(organoid['cysts'])
                object_info.append(f"🔴 Organoid {organoid_id}: {cyst_count} cysts")

                # Show individual cysts for current organoid
                if organoid_id == self.current_organoid_id and organoid['cysts']:
                    for cyst in organoid['cysts']:
                        cyst_id = cyst['cyst_id']
                        object_info.append(f"  🔵 Cyst {cyst_id}")

            # Update display
            if self.organoid_data:
                self.active_objects_label.config(text=f"Organoids: {total_organoids}, Cysts: {total_cysts}")
            else:
                self.active_objects_label.config(text="Organoids: 0, Cysts: 0")

            self.object_list_text.config(state='normal')
            self.object_list_text.delete(1.0, tk.END)
            self.object_list_text.insert(1.0, "\n".join(object_info) if object_info else "Ready to start workflow...")
            self.object_list_text.config(state='disabled')

        except Exception as e:
            print(f"❌ DEBUG: Error in update_active_objects_display: {e}")
            self.active_objects_label.config(text="Active Objects: None")
            self.object_list_text.config(state='normal')
            self.object_list_text.delete(1.0, tk.END)
            self.object_list_text.config(state='disabled')

    def clear_prompts(self):
        """Clear all organoids and cysts - reset to initial state"""
        if not self.current_model:
            return

        try:
            # Clear all prompts from model
            self.current_model.clear_prompts()

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
            self.track_btn.config(state='disabled')
            self.revert_btn.config(state='disabled')
            self.clear_prompts_btn.config(state='disabled')

            self.set_status("All organoids and cysts cleared - ready to start fresh")
            self.log_event("🧹 All organoids and cysts cleared, workflow reset")

        except Exception as e:
            self.set_status(f"Error clearing: {str(e)}")

    # Removed clear_current_object method - replaced by revert button and clear all functionality

    def start_tracking(self):
        """Start multi-object tracking with timing"""
        if not self.current_model or not self.current_model.video_frames:
            self.set_status("Error: Please load a video first")
            self.log_event("❌ Cannot start tracking - no video loaded")
            return

        # Check if there are any active cyst objects to track
        if not self.active_object_ids:
            self.set_status("Error: Please add at least one cyst by dragging bounding boxes")
            self.log_event("❌ Cannot start tracking - no cysts added")
            return

        if not self.current_model.prompts:
            self.set_status("Error: No tracking prompts available")
            self.log_event("❌ Cannot start tracking - no prompts in model")
            return

        import time
        start_time = time.time()

        self.tracking_in_progress = True
        self.track_btn.config(state='disabled')
        self.set_status("Running tracking... Please wait.")

        # Count total prompts
        total_prompts = sum(len(prompts) for prompts in self.current_model.prompts.values())
        active_objects = list(self.current_model.get_active_objects())
        self.log_event(f"🎯 Starting tracking for {len(active_objects)} objects ({total_prompts} prompts)")

        # Create progress dialog
        self.tracking_dialog = ProgressDialog(self.root, "Running Object Tracking")

        def tracking_thread():
            try:
                # Progress callback
                def progress_callback(current, total, message):
                    progress = (current / total) * 100 if total > 0 else 0
                    self.root.after(0, self.tracking_dialog.update_progress, progress, message)

                # Run tracking
                self.video_segments = self.current_model.run_tracking(progress_callback)

                # Ensure completion progress is shown
                self.root.after(0, self.tracking_dialog.update_progress, 100, "Tracking completed!")

                tracking_time = time.time() - start_time
                self.root.after(0, self.on_tracking_complete_success, tracking_time)

            except Exception as e:
                tracking_time = time.time() - start_time
                self.root.after(0, self.on_tracking_complete_error, str(e), tracking_time)

        # Start tracking in background thread
        threading.Thread(target=tracking_thread, daemon=True).start()
        self.tracking_dialog.show()

    def on_tracking_complete_success(self, tracking_time):
        """Handle successful tracking completion"""
        # Close progress dialog
        if self.tracking_dialog:
            self.tracking_dialog.close()
            self.tracking_dialog = None

        self.tracking_in_progress = False
        self.track_btn.config(state='normal')

        # Enable results buttons
        self.generate_btn.config(state='normal')
        self.analysis_btn.config(state='normal')
        # self.view_results_btn.config(state='normal') # This line is removed

        # Update results info
        if self.video_segments:
            num_frames = len(self.video_segments)
            active_objects = list(self.current_model.get_active_objects())

            self.log_event(f"✅ Tracking completed in {tracking_time:.2f}s")
            self.log_event(f"📊 Processed {num_frames} frames for {len(active_objects)} objects")
            
            # Log adaptive tracking statistics if available
            if hasattr(self.current_model, 'get_adaptive_tracking_stats'):
                try:
                    adaptive_stats = self.current_model.get_adaptive_tracking_stats()
                    if adaptive_stats:
                        total_updates = sum(stats.get('total_updates', 0) for stats in adaptive_stats.values())
                        if total_updates > 0:
                            self.log_event(f"🎯 Adaptive tracking: {total_updates} bbox updates across {len(adaptive_stats)} objects")
                        else:
                            self.log_event(f"📦 Static tracking: No adaptive updates needed")
                except Exception as e:
                    pass  # Don't fail tracking on stats error

            self.set_status("Tracking completed! Ready to generate videos.")
        else:
            self.log_event(f"⚠️ Tracking completed but no results generated")

    def on_tracking_complete_error(self, error_msg, tracking_time):
        """Handle tracking completion error"""
        # Close progress dialog
        if self.tracking_dialog:
            self.tracking_dialog.close()
            self.tracking_dialog = None

        self.tracking_in_progress = False
        self.track_btn.config(state='normal')

        self.set_status(f"Error during tracking: {error_msg}")
        self.log_event(f"❌ Tracking failed after {tracking_time:.2f}s: {error_msg}")
        # Remove popup - already handled with status and log

    def generate_videos(self):
        """Generate output videos with timing"""
        if not self.video_segments:
            self.set_status("No tracking results available. Please run tracking first")
            self.log_event("❌ No tracking results available")
            return

        # Ask for output directory
        output_dir = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir="./data/output_videos"
        )

        if not output_dir:
            return

        import time
        start_time = time.time()

        self.generate_btn.config(state='disabled')
        self.set_status("Generating videos... This may take a while.")

        # Count objects for logging
        active_objects = self.current_model.get_active_objects()
        num_frames = len(self.video_segments)
        video_types = ['overlay', 'mask', 'side_by_side']

        # DETAILED DEBUG OUTPUT
        self.log_event("🎬" + "=" * 60)
        self.log_event("🎬 DETAILED VIDEO GENERATION DEBUG START")
        self.log_event("🎬" + "=" * 60)
        self.log_event(f"📊 Video Generation Configuration:")
        self.log_event(f"   • Active Objects: {active_objects} (count: {len(active_objects)})")
        self.log_event(f"   • Video Segments: {num_frames} frames")
        self.log_event(f"   • Video Types: {video_types}")
        self.log_event(f"   • Quality: {self.quality_var.get()}")
        self.log_event(f"   • Debug Mode: {self.debug_var.get()}")
        self.log_event(f"   • Output Directory: {output_dir}")

        # Video frames analysis
        if hasattr(self.current_model, 'video_frames') and self.current_model.video_frames:
            frame_count = len(self.current_model.video_frames)
            first_frame_shape = self.current_model.video_frames[0].shape if frame_count > 0 else "N/A"
            self.log_event(f"   • Source Frames: {frame_count} frames")
            self.log_event(f"   • Frame Dimensions: {first_frame_shape}")
        else:
            self.log_event(f"   • Source Frames: MISSING - check video loading")

        # Tracking data analysis
        if self.video_segments:
            frame_indices = list(self.video_segments.keys())
            self.log_event(f"   • Tracking Frame Indices: {sorted(frame_indices)[:5]}{'...' if len(frame_indices) > 5 else ''}")

            sample_frame = frame_indices[0] if frame_indices else None
            if sample_frame is not None and sample_frame in self.video_segments:
                sample_objects = list(self.video_segments[sample_frame].keys())
                self.log_event(f"   • Sample Frame Objects: {sample_objects}")
        else:
            self.log_event(f"   • Tracking Data: MISSING")

        # Organoid-cyst mapping
        total_organoids = len(self.organoid_data)
        total_cysts = sum(len(org['cysts']) for org in self.organoid_data.values())
        self.log_event(f"   • Organoid-Cyst Mapping: {total_organoids} organoids, {total_cysts} cysts")

        for org_id, org_data in list(self.organoid_data.items())[:3]:  # Show first 3
            cyst_ids = [c['cyst_id'] for c in org_data['cysts']]
            self.log_event(f"     ◦ Organoid {org_id}: cysts {cyst_ids}")
        if len(self.organoid_data) > 3:
            self.log_event(f"     ◦ ... and {len(self.organoid_data) - 3} more organoids")

        self.log_event(f"🎬 Starting video generation for {len(active_objects)} objects")
        self.log_event(f"📹 Creating {len(video_types)} video types ({num_frames} frames each)")

        def generation_thread():
            try:
                generator = VideoOutputGenerator()
                output_dir_path = Path(output_dir)

                # Get active objects for display
                objects_text = f"Objects: {', '.join(map(str, sorted(active_objects)))}"

                # Set debug mode on generator
                generator.debug_mode = self.debug_var.get()

                # Determine quality scale factor
                quality = self.quality_var.get()
                quality_scale = {"original": 1.0, "mid": 0.5, "low": 0.25}[quality]

                # 🚀 USE OPTIMIZED APPROACH - single mask processing pass!
                self.log_event(f"🚀 Using optimized video generation (2-6x faster)")

                def optimized_progress_callback(current, total, message):
                    # Calculate overall progress
                    progress = (current / total) * 100 if total > 0 else 0
                    self.root.after(0, self.generation_dialog.update_progress, progress, f"{message} ({objects_text})")

                created_videos = generator.create_optimized_multi_object_videos(
                    frames=self.current_model.video_frames,
                    video_segments=self.video_segments,
                    output_dir=str(output_dir_path),
                    fps=5.0,  # Slower for easier viewing
                    alpha=0.4,  # Good visibility
                    progress_callback=optimized_progress_callback,
                    quality_scale=quality_scale,
                    tracker=self.current_model  # Pass tracker for reverse state
                )

                # ✅ FIX: Explicitly set progress to 100% when optimization completes
                self.root.after(0, self.generation_dialog.update_progress, 100, "Video generation completed!")

                # ✅ CRITICAL FIX: Always re-enable button, even if completion callback fails
                self.root.after(0, lambda: self.generate_btn.config(state='normal'))

                # Log results
                successful_videos = [v for v in created_videos.values() if v is not None]
                for video_type, path in created_videos.items():
                    if path:
                        self.root.after(0, lambda vt=video_type: self.log_event(f"✅ {vt} video created (optimized)"))
                    else:
                        self.root.after(0, lambda vt=video_type: self.log_event(f"❌ {vt} video failed"))

                # FALLBACK TO ORIGINAL METHOD if optimization fails
                if not successful_videos:
                    self.root.after(0, lambda: self.log_event(f"⚠️ Optimization failed, falling back to original method"))

                    created_videos = {}
                    total_videos = len(video_types)

                    for i, video_type in enumerate(video_types):
                        video_start_time = time.time()

                        # Base progress for this video type
                        base_progress = (i / total_videos) * 100
                        video_progress_range = 100 / total_videos  # e.g., 33.33% per video

                        # Create a closure that captures the current values
                        def make_progress_callback(base_prog, prog_range, vid_type):
                            def video_progress_callback(current_frame, total_frames, frame_message):
                                # Calculate progress within this video (0-33.33% for first video, etc.)
                                if total_frames > 0:
                                    video_completion = (current_frame / total_frames) * prog_range
                                    overall_progress = base_prog + video_completion
                                else:
                                    overall_progress = base_prog

                                message = f"Creating {vid_type} video: {frame_message} ({objects_text})"
                                self.root.after(0, self.generation_dialog.update_progress, overall_progress, message)
                            return video_progress_callback

                        video_progress_callback = make_progress_callback(base_progress, video_progress_range, video_type)

                        output_path = output_dir_path / f"multi_object_{video_type}.mp4"
                        try:
                            # Set debug mode on generator
                            generator.debug_mode = self.debug_var.get()

                            # Determine quality scale factor
                            quality = self.quality_var.get()
                            quality_scale = {"original": 1.0, "mid": 0.5, "low": 0.25}[quality]

                            result_path = generator.create_multi_object_video(
                                frames=self.current_model.video_frames,
                                video_segments=self.video_segments,
                                output_path=str(output_path),
                                fps=5.0,  # Slower for easier viewing
                                video_type=video_type,
                                alpha=0.4,  # Good visibility
                                progress_callback=video_progress_callback,
                                quality_scale=quality_scale,
                                tracker=self.current_model  # Pass tracker for reverse state
                            )
                            created_videos[video_type] = result_path

                            video_time = time.time() - video_start_time
                            # Log individual video completion in main thread
                            self.root.after(0, lambda vt=video_type, t=video_time:
                                           self.log_event(f"✅ {vt} video created in {t:.2f}s"))

                        except Exception as e:
                            print(f"❌ Error creating {video_type} video: {str(e)}")
                            created_videos[video_type] = None

                            video_time = time.time() - video_start_time
                            self.root.after(0, lambda vt=video_type, t=video_time, err=str(e):
                                           self.log_event(f"❌ {vt} video failed after {t:.2f}s: {err}"))

                    total_time = time.time() - start_time

                    # Ensure completion progress is shown
                    self.root.after(0, self.generation_dialog.update_progress, 100, "Video generation completed!")

                    # ✅ CRITICAL FIX: Always re-enable button, even if completion callback fails
                    self.root.after(0, lambda: self.generate_btn.config(state='normal'))

                    # Add delay before callback to allow cleanup and reduce memory pressure
                    self.root.after(100, self.on_generation_complete_success, created_videos, output_dir, total_time)

            except Exception as e:
                total_time = time.time() - start_time
                # ✅ CRITICAL FIX: Always re-enable button, even on exceptions
                self.root.after(0, lambda: self.generate_btn.config(state='normal'))
                self.root.after(0, self.on_generation_complete_error, str(e), total_time)

        # ✅ FIX: Ensure button is disabled during generation and dialog can be restarted
        self.generate_btn.config(state='disabled')

        # Show progress dialog
        from .progress_dialog import ProgressDialog
        self.generation_dialog = ProgressDialog(self.root, "Generating Multi-Object Videos")

        # Start generation in background
        threading.Thread(target=generation_thread, daemon=True).start()
        self.generation_dialog.show()

    def on_generation_complete_success(self, created_videos, output_dir, total_time):
        """Handle successful video generation"""
        # ✅ CRITICAL FIX: Always ensure button is enabled, regardless of any exceptions
        self.generate_btn.config(state='normal')

        try:
            # Force garbage collection before GUI updates to prevent memory pressure
            import gc
            gc.collect()

            # Close progress dialog
            if self.generation_dialog:
                self.generation_dialog.close()
                self.generation_dialog = None

            self.generate_btn.config(state='normal')

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
            self.log_event(f"📋 Detailed Results:")
            for video_type, path in created_videos.items():
                if path:
                    file_size = Path(path).stat().st_size / (1024*1024) if Path(path).exists() else 0
                    self.log_event(f"   ✅ {video_type}: {Path(path).name} ({file_size:.1f}MB)")
                else:
                    self.log_event(f"   ❌ {video_type}: FAILED")

            if failed_videos:
                self.log_event(f"⚠️ Failed videos: {', '.join(failed_videos)}")

            # Performance metrics
            active_objects = self.current_model.get_active_objects() if self.current_model else []
            num_frames = len(self.video_segments) if self.video_segments else 0
            if num_frames > 0 and total_time > 0:
                frames_per_second = num_frames / total_time
                self.log_event(f"⚡ Performance: {frames_per_second:.1f} frames/sec across {len(successful_videos)} video types")

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
                self.log_event(f"✅ Video generation completed")
            else:
                # Remove popup - use log instead
                self.log_event(f"❌ All video generation failed")

        except Exception as e:
            # Handle any errors in completion callback
            # ✅ CRITICAL FIX: Always ensure button is enabled, even on exceptions
            self.generate_btn.config(state='normal')
            self.log_event(f"❌ Error in video generation completion: {str(e)}")
            if self.generation_dialog:
                self.generation_dialog.close()
                self.generation_dialog = None

    def on_generation_complete_error(self, error_msg, total_time):
        """Handle video generation error"""
        # ✅ CRITICAL FIX: Always ensure button is enabled first
        self.generate_btn.config(state='normal')

        # Close progress dialog
        if self.generation_dialog:
            self.generation_dialog.close()
            self.generation_dialog = None
        self.set_status(f"Error generating videos: {error_msg}")
        self.log_event(f"❌ Video generation failed after {total_time:.2f}s: {error_msg}")
        # Remove popup - already handled with status and log

    def view_results(self):
        """View tracking results"""
        if not self.video_segments or not self.current_model:
            return

        # Show a simple results viewer
        from .results_viewer import ResultsViewer

        try:
            viewer = ResultsViewer(
                self.root,
                self.current_model.video_frames,
                self.video_segments,
                obj_id=1
            )
            viewer.show()

        except Exception as e:
            error_msg = f"Failed to open results viewer: {str(e)}"
            self.set_status(error_msg)
            self.log_event(f"❌ {error_msg}")

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

    def generate_analysis_report(self):
        """Generate comprehensive organoid-cyst analysis report using new workflow"""
        if not self.video_segments:
            self.set_status("No tracking results available. Please run tracking first")
            self.log_event("❌ No tracking results available for analysis")
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
            total_cysts = sum(len(org['cysts']) for org in self.organoid_data.values())

            # DETAILED DEBUG OUTPUT - ANALYSIS START
            self.log_event("🧬" + "=" * 60)
            self.log_event("🧬 DETAILED ORGANOID ANALYSIS DEBUG START")
            self.log_event("🧬" + "=" * 60)
            self.log_event(f"📊 Analysis Configuration:")
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
            if self.current_model:
                self.log_event(f"   • Current Model: {type(self.current_model).__name__}")
                if hasattr(self.current_model, 'original_frames'):
                    self.log_event(f"   • Original Frames Available: {len(self.current_model.original_frames) if self.current_model.original_frames else 0}")
                else:
                    self.log_event(f"   • Original Frames Available: No")
            else:
                self.log_event(f"   • Current Model: MISSING")

            self.log_event("🧬" + "=" * 60)

        except Exception as e:
            self.set_status(f"Error reading analysis parameters: {str(e)}")
            self.log_event(f"❌ Parameter validation error: {e}")
            return

        # Ask for output directory
        output_dir = filedialog.askdirectory(
            title="Select Output Directory for Organoid Analysis Report",
            initialdir="./data/output_videos"
        )

        if not output_dir:
            return

        import time
        start_time = time.time()

        self.analysis_btn.config(state='disabled')
        self.set_status("Generating organoid-cyst analysis report... This may take a moment.")

        # Log analysis start
        self.log_event(f"🧬 Starting organoid-cyst analysis for {total_organoids} organoids with {total_cysts} cysts")

        def analysis_thread():
            try:
                # Import new analysis system
                try:
                    from src.analysis import OrganoidAnalysisReportGenerator
                except ImportError:
                    from analysis import OrganoidAnalysisReportGenerator

                # Create new analysis report generator
                report_generator = OrganoidAnalysisReportGenerator()

                # Get original frames from current model if available
                original_frames = None
                if hasattr(self.current_model, 'original_frames') and self.current_model.original_frames:
                    original_frames = self.current_model.original_frames
                    self.log_event(f"   📸 Using {len(original_frames)} original frames for PDF comparison")
                else:
                    self.log_event("   ⚠️ Original frames not available from current model")

                # Generate complete analysis report
                analysis_summary = report_generator.generate_complete_analysis_report(
                    tracking_results=self.video_segments,
                    organoid_data=self.organoid_data,
                    time_lapse_days=time_lapse_days,
                    conversion_factor=conversion_factor,
                    output_dir=output_dir,
                    debug_mode=self.debug_var.get(),
                    original_frames=original_frames
                )

                # Export data for standalone visualization (after successful analysis)
                try:
                    from src.utils.data_export import TrackingDataExporter
                    
                    exporter = TrackingDataExporter()
                    export_dir = output_dir_path / "exported_data"
                    
                    exported_files = exporter.export_tracking_data(
                        tracking_results=self.video_segments,
                        organoid_data=self.organoid_data,
                        time_lapse_days=time_lapse_days,
                        conversion_factor=conversion_factor,
                        original_frames=original_frames,
                        debug_mode=self.debug_var.get(),
                        video_path=self.current_video_path,
                        output_dir=str(export_dir)
                    )
                    
                    if exported_files:
                        self.log_event(f"💾 Data exported for standalone visualization")
                        self.log_event(f"   📁 Export directory: {export_dir}")
                        if 'combined_data' in exported_files:
                            combined_file = Path(exported_files['combined_data']).name
                            self.log_event(f"   🚀 Run: python scripts/standalone_visualizer.py {combined_file}")
                    
                except Exception as e:
                    self.log_event(f"⚠️ Data export failed: {e}")
                    # Don't fail the analysis if export fails

                analysis_time = time.time() - start_time

                # Update GUI in main thread with results for display
                self.root.after(0, self.on_organoid_analysis_complete_success, analysis_summary, analysis_time)

            except Exception as e:
                analysis_time = time.time() - start_time
                self.root.after(0, self.on_organoid_analysis_complete_error, str(e), analysis_time)

        # Run analysis in background thread
        import threading
        analysis_thread = threading.Thread(target=analysis_thread, daemon=True)
        analysis_thread.start()

    def on_organoid_analysis_complete_success(self, analysis_summary, analysis_time):
        """Handle successful organoid analysis completion"""
        self.analysis_btn.config(state='normal')

        if not analysis_summary.get('success', False):
            self.on_organoid_analysis_complete_error(
                analysis_summary.get('error', 'Unknown error'), analysis_time
            )
            return

        # DETAILED DEBUG OUTPUT - ANALYSIS COMPLETION
        self.log_event("🧬" + "=" * 60)
        self.log_event("🧬 DETAILED ORGANOID ANALYSIS DEBUG COMPLETE")
        self.log_event("🧬" + "=" * 60)
        self.log_event(f"🎉 Analysis completed in {analysis_time:.2f}s")

        # Display analysis results summary in the results area
        self.log_event("🧬 ORGANOID-CYST ANALYSIS RESULTS")
        self.log_event("🧬" + "=" * 60)

        # Display experiment information
        exp_info = analysis_summary.get('experiment_info', {})
        self.log_event(f"🔬 Experiment Summary:")
        self.log_event(f"   • Total Organoids: {exp_info.get('total_organoids', 0)}")
        self.log_event(f"   • Total Cysts: {exp_info.get('total_cysts', 0)}")
        self.log_event(f"   • Frames Analyzed: {exp_info.get('total_frames', 0)}")
        self.log_event(f"   • Time Period: {exp_info.get('time_lapse_days', 0)} days")
        self.log_event(f"   • Conversion Factor: {exp_info.get('conversion_factor_um_per_pixel', 0)} μm/pixel")

        # Display quality metrics
        quality = analysis_summary.get('quality_metrics', {})
        self.log_event(f"📊 Data Quality:")
        self.log_event(f"   • Tracking Coverage: {quality.get('tracking_coverage_percent', 0):.1f}%")
        self.log_event(f"   • Mean Trajectory Length: {quality.get('mean_trajectory_length_frames', 0):.1f} frames")
        self.log_event(f"   • Organoids with Cysts: {quality.get('organoids_with_cysts', 0)}")

        # Display growth statistics
        growth = analysis_summary.get('growth_statistics', {})
        if growth.get('mean_growth_rate_um2_per_day', 0) > 0:
            self.log_event(f"📈 Growth Statistics:")
            self.log_event(f"   • Mean Growth Rate: {growth.get('mean_growth_rate_um2_per_day', 0):.4f} μm²/day")
            self.log_event(f"   • Max Growth Rate: {growth.get('max_growth_rate_um2_per_day', 0):.4f} μm²/day")
            self.log_event(f"   • Min Growth Rate: {growth.get('min_growth_rate_um2_per_day', 0):.4f} μm²/day")

        # Display output files
        output_files = analysis_summary.get('output_files', {})
        self.log_event(f"📄 Generated Files:")

        # CSV files
        csv_files = output_files.get('csv_files', {})
        if csv_files:
            self.log_event(f"   📊 CSV Data Files:")
            if csv_files.get('raw_data'):
                self.log_event(f"      • Raw Data: {Path(csv_files['raw_data']).name}")
            if csv_files.get('cyst_summary'):
                self.log_event(f"      • Cyst Summary: {Path(csv_files['cyst_summary']).name}")
            if csv_files.get('organoid_summary'):
                self.log_event(f"      • Organoid Summary: {Path(csv_files['organoid_summary']).name}")

        # Visualizations
        visualizations = output_files.get('visualizations', {})
        if visualizations:
            viz_count = len([v for v in visualizations.values() if v])
            self.log_event(f"   🎨 Visualizations: {viz_count} advanced plots generated")
            viz_names = {
                'organoids_with_cysts': 'Organoids with Cysts vs Time',
                'cyst_organoid_ratio': 'Cyst/Organoid Ratio vs Time',
                'cyst_areas_multiline': 'Individual Cyst Area Trajectories',
                'cyst_circularity_multiline': 'Individual Cyst Circularity Trajectories',
                'circularity_scatter': 'Circularity Scatter (sized by area)',
                'lasagna_plot': 'Organoid Growth Heatmap (Lasagna Plot)'
            }
            for viz_key, viz_path in visualizations.items():
                if viz_path and Path(viz_path).exists():
                    viz_name = viz_names.get(viz_key, viz_key.replace('_', ' ').title())
                    self.log_event(f"      • {viz_name}")

        # PDF report
        pdf_path = output_files.get('pdf_report')
        if pdf_path and Path(pdf_path).exists():
            self.log_event(f"   📋 Enhanced PDF Report: {Path(pdf_path).name}")

        # Analysis timing
        self.log_event(f"⏱️ Analysis completed in {analysis_time:.2f} seconds")

        # Validation warnings
        validation = analysis_summary.get('validation_results', {})
        warnings = validation.get('warnings', [])
        if warnings:
            self.log_event(f"⚠️ Quality Warnings:")
            for warning in warnings:
                self.log_event(f"   • {warning}")

        self.log_event("=" * 60)
        self.log_event("✅ Comprehensive organoid analysis report generated successfully!")

        # Show completion status
        output_dir = Path(csv_files.get('raw_data', '')).parent if csv_files.get('raw_data') else None
        if output_dir:
            self.log_event(f"📁 All files saved to: {output_dir}")
            self.set_status(f"Analysis complete! Files saved to: {output_dir}")

            # Optional: Open directory automatically (with GTK-safe method)
            if self.auto_open_directory:
                try:
                    import subprocess
                    import platform
                    import os

                    if platform.system() == "Windows":
                        subprocess.run(["explorer", str(output_dir)], check=False)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", str(output_dir)], check=False)
                    else:  # Linux - suppress GTK warnings and run in background
                        with open(os.devnull, 'w') as devnull:
                            subprocess.Popen(["xdg-open", str(output_dir)],
                                           stderr=devnull, stdout=devnull)

                    self.log_event("📂 Output directory opened automatically")
                except Exception:
                    self.log_event(f"ℹ️ Directory: {output_dir}")
        else:
            self.set_status("Analysis complete!")

    def on_organoid_analysis_complete_error(self, error_msg, analysis_time):
        """Handle organoid analysis error"""
        self.analysis_btn.config(state='normal')
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
        self.analysis_btn.config(state='normal')

        # Display analysis results summary in the results area
        self.log_event("=" * 50)
        self.log_event("📊 ORGANOID CYST ANALYSIS RESULTS")
        self.log_event("=" * 50)

        # Display parameters used
        params = results['parameters']
        self.log_event(f"📋 Analysis Parameters:")
        self.log_event(f"   • Total Organoids: {params['total_organoids']}")
        self.log_event(f"   • Time Lapse: {params['time_lapse_days']} days")
        self.log_event(f"   • Conversion Factor: {params['conversion_factor_um_per_pixel']} μm/pixel")

        # Display cyst tracking summary
        summary = results['cyst_data_summary']
        self.log_event(f"🎯 Tracking Summary:")
        self.log_event(f"   • Cysts Tracked: {summary['num_cysts_tracked']}")
        self.log_event(f"   • Object IDs: {summary['cyst_ids']}")

        # Display key metrics
        self.log_event(f"📈 Key Metrics:")
        for metric_name, metric_data in results['metrics'].items():
            if 'error' in metric_data:
                self.log_event(f"   ❌ {metric_name}: Error - {metric_data['error']}")
                continue

            info = metric_data['info']
            metric_results = metric_data['results']

            if metric_name == "Cyst Formation Efficiency":
                value = metric_results.get('value', 0)
                organoids_with_cysts = metric_results.get('organoids_with_cysts', 0)
                total_organoids = metric_results.get('total_organoids', 0)
                self.log_event(f"   • {metric_name}: {value:.1f}% ({organoids_with_cysts}/{total_organoids} organoids)")

            elif metric_name == "De Novo Cyst Formation Rate":
                value = metric_results.get('value', 0)
                self.log_event(f"   • {metric_name}: {value:.2f} cysts/day")

            elif metric_name == "Radial Expansion Velocity":
                mean_val = metric_results.get('mean_value', 0)
                std_val = metric_results.get('std_value', 0)
                max_val = metric_results.get('max_value', 0)
                min_val = metric_results.get('min_value', 0)
                num_cysts = metric_results.get('num_cysts', 0)
                self.log_event(f"   • {metric_name}:")
                self.log_event(f"     - Mean: {mean_val:.2f} ± {std_val:.2f} μm/day")
                self.log_event(f"     - Range: {min_val:.2f} to {max_val:.2f} μm/day")
                self.log_event(f"     - Analyzed Cysts: {num_cysts}")

        self.log_event("=" * 50)

        # Log file generation results
        self.log_event(f"✅ Analysis completed in {analysis_time:.2f}s")

        if 'csv' in report_paths:
            self.log_event(f"📄 CSV report: {Path(report_paths['csv']).name}")

        if 'pdf' in report_paths:
            self.log_event(f"📋 PDF report: {Path(report_paths['pdf']).name}")

        if 'json' in report_paths:
            self.log_event(f"💾 Analysis data: {Path(report_paths['json']).name}")

        # Log advanced visualizations
        viz_count = 0
        for key in report_paths:
            if key.startswith('viz_'):
                viz_count += 1

        if viz_count > 0:
            self.log_event(f"🎨 Advanced visualizations generated: {viz_count} plots")
            self.log_event("   📊 Collective outcome plots (bar charts, time series, box plots)")
            self.log_event("   🌱 De novo formation dynamics (cumulative counts, dual-axis)")
            self.log_event("   📏 Radial expansion heterogeneity (lasagna plots, velocity analysis)")
            self.log_event("   🔬 Morphological & spatial analysis (morphospace, density maps)")

        if 'enhanced_pdf' in report_paths:
            self.log_event(f"📋 Enhanced PDF (with visualizations): {Path(report_paths['enhanced_pdf']).name}")

        if 'visualization_summary' in report_paths:
            self.log_event(f"📝 Visualization summary: {Path(report_paths['visualization_summary']).name}")

        # Log any errors
        if 'csv_error' in report_paths:
            self.log_event(f"❌ CSV generation failed: {report_paths['csv_error']}")

        if 'pdf_error' in report_paths:
            self.log_event(f"❌ PDF generation failed: {report_paths['pdf_error']}")

        if 'visualization_error' in report_paths:
            self.log_event(f"⚠️ Visualization warning: {report_paths['visualization_error']}")

        if 'enhanced_pdf_error' in report_paths:
            self.log_event(f"⚠️ Enhanced PDF warning: {report_paths['enhanced_pdf_error']}")

        self.set_status("Comprehensive analysis report generated successfully!")

        # Add completion info
        if 'csv' in report_paths:
            self.log_event(f"📊 Files saved to: {Path(report_paths['csv']).parent}")
        self.log_event("💡 Tip: Check the output directory for comprehensive analysis reports and visualizations")

        # Optional: Open directory automatically (configurable and GTK-safe)
        if self.auto_open_directory:
            try:
                if 'csv' in report_paths:
                    import subprocess
                    import platform
                    import os

                    output_dir = Path(report_paths['csv']).parent

                    if platform.system() == "Windows":
                        subprocess.run(["explorer", str(output_dir)], check=False)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", str(output_dir)], check=False)
                    else:  # Linux - suppress GTK warnings and run in background
                        # Suppress GTK warnings by redirecting stderr and run detached
                        with open(os.devnull, 'w') as devnull:
                            subprocess.Popen(["xdg-open", str(output_dir)],
                                           stderr=devnull, stdout=devnull)

                    self.log_event("📂 Output directory opened automatically")
            except Exception:
                # Silently handle directory opening failures - just show path
                if 'csv' in report_paths:
                    self.log_event(f"ℹ️ Directory: {Path(report_paths['csv']).parent}")
        else:
            # Just show the directory path when auto-open is disabled
            if 'csv' in report_paths:
                self.log_event(f"ℹ️ Directory: {Path(report_paths['csv']).parent}")

    def on_analysis_complete_error(self, error_msg, analysis_time):
        """Handle analysis error"""
        self.analysis_btn.config(state='normal')
        self.set_status(f"Error during analysis: {error_msg}")
        self.log_event(f"❌ Analysis failed after {analysis_time:.2f}s: {error_msg}")


if __name__ == "__main__":
    app = VideoTrackerApp()
    app.run()