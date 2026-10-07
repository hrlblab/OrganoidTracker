# Utility Scripts

This directory contains a utility script for advanced visualization workflows.

## csv_visualizer.py

A standalone script that generates publication-quality visualizations directly from CSV data files produced by the main application. This ensures exact data fidelity and provides researchers with reproducible visualization capabilities.

### Usage

The script automatically looks for CSV data in the `data/output_videos/` directory and generates corresponding visualizations.

```bash
uv run python scripts/csv_visualizer.py   # from the repository root (it reads data/output_videos/)
```

### Note

Most users will not need to use this script directly, as the main application provides all necessary functionality through the GUI interface. This script is provided for advanced analysis workflows and custom visualization requirements.

For standard usage, please use the main application: `uv run organoidtracker-tk`
