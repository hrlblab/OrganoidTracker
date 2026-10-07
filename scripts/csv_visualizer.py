#!/usr/bin/env python3
"""
Re-create the visualizations from exported CSV data.

Reads ``raw_cyst_data.csv`` and ``analysis_summary.json`` from the data directory (default
``data/output_videos`` under the current directory) and writes the plots to the output directory
(default ``csv_output/visualizations``). The exported time axis and the tracked frames are
preserved, so growth rates and gaps match the original report.

    uv run python scripts/csv_visualizer.py [--data-dir DIR] [--output-dir DIR]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from organoidtracker.analysis.csv_import import experiment_from_csv
from organoidtracker.analysis.organoid_visualizations import OrganoidVisualizationSuite
from organoidtracker.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/output_videos", help="directory with the exported CSV and summary")
    parser.add_argument("--output-dir", default="csv_output", help="directory for the re-created visualizations")
    args = parser.parse_args(argv)
    configure_logging(log_file=None)

    data_dir = Path(args.data_dir)
    raw_csv = data_dir / "raw_cyst_data.csv"
    summary = data_dir / "analysis_summary.json"
    if not raw_csv.is_file():
        logger.error(f"Raw data CSV not found: {raw_csv}")
        return 1
    if not summary.is_file():
        logger.warning(f"{summary} not found; frame count and timing are inferred from the CSV")
        summary = None

    experiment = experiment_from_csv(raw_csv, summary)
    logger.info(
        f"Loaded {experiment.get_total_organoid_count()} organoids, {len(experiment.get_all_cysts())} cysts, "
        f"{experiment.total_frames} frames ({len(experiment.frames_observed())} tracked) from {raw_csv}"
    )

    viz_dir = Path(args.output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = OrganoidVisualizationSuite().create_all_visualizations(experiment, str(viz_dir))
    created = {name: path for name, path in results.items() if path}
    if not created:
        logger.error("Visualization generation failed")
        return 1
    logger.info(f"Generated {len(created)} visualizations in {time.time() - started:.1f}s under {viz_dir}")
    for name, path in created.items():
        logger.debug(f"• {name}: {Path(path).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
