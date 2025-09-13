# Example User Configuration
# Copy this file to 'user_config.py' and modify as needed

# Disable automatic directory opening to avoid GTK warnings
AUTO_OPEN_OUTPUT_DIRECTORY = False

# Customize default analysis parameters
DEFAULT_TOTAL_ORGANOIDS = 20     # Your typical organoid count
DEFAULT_TIME_LAPSE_DAYS = 10.0   # Your typical experiment duration
DEFAULT_CONVERSION_FACTOR = 0.65 # Your microscope's μm/pixel ratio

# Customize visualization settings
MATPLOTLIB_DPI = 600             # Higher DPI for publication-quality figures
FIGURE_SIZE = (14, 10)           # Larger figures

# Enable debug features
DEBUG_MODE_ENABLED = True        # Generate debug frames by default
ENABLE_PERFORMANCE_LOGGING = True # Log performance metrics

# Disable warnings for cleaner output
SUPPRESS_MATPLOTLIB_WARNINGS = True