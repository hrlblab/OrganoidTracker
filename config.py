#!/usr/bin/env python3
"""
Configuration settings for the Kidney Organoid Video Tracker

Modify these settings to customize the application behavior.
"""

# GUI Configuration
AUTO_OPEN_OUTPUT_DIRECTORY = False  # Set to False to disable automatic file manager opening
                                    # This prevents GTK warnings on some Linux systems

# Analysis Configuration
DEFAULT_TOTAL_ORGANOIDS = 13      # Default number of organoids
DEFAULT_TIME_LAPSE_DAYS = 6.0     # Default time lapse duration in days
DEFAULT_CONVERSION_FACTOR = 1.6934   # Default μm per pixel conversion factor

# Video Processing Configuration
DEFAULT_VIDEO_QUALITY = "mid"     # "original", "mid", or "low"
DEBUG_MODE_ENABLED = False        # Enable debug frame generation by default

# Advanced Analysis Configuration
ENABLE_COMPREHENSIVE_REPORTS = True    # Generate enhanced reports with visualizations
MATPLOTLIB_DPI = 150                   # DPI for plot output (150=normal, 300=publication)
VISUALIZATION_FORMAT = 'png'           # Output format for visualizations ('png' or 'svg')
FIGURE_SIZE = (10, 6)                  # Default figure size in inches

# System Configuration
SUPPRESS_MATPLOTLIB_WARNINGS = True   # Suppress matplotlib 3D projection warnings
FALLBACK_FOR_OPTIONAL_DEPS = True     # Use fallbacks when seaborn/scipy unavailable

# Report Configuration
GENERATE_DEBUG_FRAMES = False         # Generate debug frames for research validation
AUTO_SAVE_ANALYSIS_DATA = True        # Automatically save JSON analysis data
INCLUDE_SESSION_METADATA = True       # Include session info in reports

# Color Scheme for Visualizations (RGB tuples)
PLOT_COLORS = {
    'primary': (70, 130, 180),        # Steel blue
    'secondary': (220, 20, 60),       # Crimson
    'accent': (255, 165, 0),          # Orange
    'success': (34, 139, 34),         # Forest green
    'warning': (255, 140, 0),         # Dark orange
    'error': (178, 34, 34)            # Fire brick
}

# File Output Configuration
CSV_ENCODING = 'utf-8'               # Character encoding for CSV files
JSON_ENSURE_ASCII = False            # Allow Unicode characters in JSON
PDF_PAGE_SIZE = 'A4'                 # PDF page size: 'A4', 'letter', etc.

# Visualization Font Scaling
FONT_SCALE_FACTOR = 1.0               # Scale factor for all visualization fonts
DISABLE_VISUALIZATION_TEXT = False    # Enable axis labels and legends
DISABLE_VISUALIZATION_TITLES = False  # Enable titles for normal use (True=publication mode)

# Performance Configuration
MAX_VIDEO_FRAMES = None               # Maximum frames to load (None = no limit)
MEMORY_OPTIMIZATION = True           # Enable memory optimization for large videos
GARBAGE_COLLECTION_FREQUENCY = 10    # Frames between garbage collection calls

# Expert Configuration (Advanced Users Only)
TORCH_DEVICE_OVERRIDE = None         # Override device selection: 'cpu', 'cuda', None=auto
SAM2_MODEL_CACHE = True              # Cache loaded SAM2 models
ENABLE_MODEL_VALIDATION = True       # Validate model compatibility

# SAM2 Tracking Quality Configuration - STRICT SETTINGS FOR NOISY BACKGROUNDS
SAM2_IMPROVED_TRACKING = True        # Enable improved tracking for disappearing objects
SAM2_USE_IMPROVED_CONFIG = True      # Use improved SAM2 configuration file
SAM2_MIN_MASK_AREA = 50              # Minimum mask area (pixels) - Allow small cyst detection
SAM2_MIN_CONFIDENCE = 0.5            # Minimum confidence score (0-1) - FURTHER INCREASED for even stricter filtering
SAM2_MEMORY_FRAMES = 1000             # Number of previous frames to depend on for memory (unlimited - depends on all previous frames)

# Adaptive Tracking Configuration
ENABLE_ADAPTIVE_TRACKING = True           # Enable adaptive bounding box tracking
ADAPTIVE_BBOX_UPDATE_THRESHOLD = 0.6      # Overlap ratio threshold for bbox updates (0-1)
ADAPTIVE_EXPANSION_FACTOR = 1.3           # Factor for expanding search region when object lost
ADAPTIVE_MIN_MASK_AREA = 50               # Minimum mask area for adaptive decisions (pixels)
ADAPTIVE_MAX_EXPANSION_FACTOR = 2.0       # Maximum expansion allowed
ADAPTIVE_CONFIDENCE_THRESHOLD = 0.7       # High confidence threshold for adaptive decisions
ADAPTIVE_LOW_CONFIDENCE_THRESHOLD = 0.4   # Low confidence threshold triggering expansion
ADAPTIVE_PADDING_FACTOR = 0.2             # Padding around fitted bbox (ratio of bbox size)
ADAPTIVE_VELOCITY_SMOOTHING = 0.7         # Velocity smoothing factor for predictive updates
ADAPTIVE_ENABLE_LOGGING = True            # Enable detailed adaptive tracking logs
ADAPTIVE_ENABLE_VELOCITY_PREDICTION = True # Enable velocity-based bbox prediction

# Logging Configuration
LOG_LEVEL = "INFO"                   # "DEBUG", "INFO", "WARNING", "ERROR"
ENABLE_PERFORMANCE_LOGGING = False   # Log performance metrics
LOG_ANALYSIS_DETAILS = True          # Log detailed analysis information

# Load user-specific configuration overrides
try:
    import importlib.util
    import os

    user_config_path = os.path.join(os.path.dirname(__file__), 'user_config.py')
    if os.path.exists(user_config_path):
        spec = importlib.util.spec_from_file_location("user_config", user_config_path)
        user_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_config)

        # Override settings from user config
        for attr in dir(user_config):
            if not attr.startswith('_') and attr.isupper():
                globals()[attr] = getattr(user_config, attr)

        print("✅ User configuration loaded successfully")
except ImportError:
    # No user config file found, use defaults
    pass
except Exception as e:
    print(f"⚠️ Warning: Error loading user config: {e}")
    print("   Using default configuration")