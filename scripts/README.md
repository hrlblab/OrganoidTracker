# Utility Scripts

## csv_visualizer.py

Re-creates the analysis figures from the CSV data a report exported, so plots can be regenerated
(for example with other visualization settings in `organoidtracker.toml`) without re-running the
tracking.

### Usage

```bash
uv run python scripts/csv_visualizer.py                      # reads data/output_videos/, writes csv_output/visualizations/
uv run python scripts/csv_visualizer.py --data-dir data/output_videos/my_run --output-dir replot
```

Keep `raw_cyst_data.csv`, `organoid_summary.csv` and `analysis_summary.json` together in the data
directory: the raw table holds the measurements and their timestamps, the organoid summary the whole
organoid population (including organoids without cysts), and the analysis summary the experiment
parameters and the frames the tracker visited. With all three the figures match the report; a missing
file is reported and limits what can be rebuilt (for example, organoids without cysts cannot be
recovered without the organoid summary).

### Note

Most users will not need this script; the application produces the same figures as part of the
analysis report. For standard usage run the application: `uv run organoidtracker-tk`.
