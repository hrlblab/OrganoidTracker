#!/usr/bin/env python3
"""
CSV-Based Visualization Script

This script generates visualizations directly from the CSV data files
produced by the GUI analysis, ensuring exact data fidelity.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from organoidtracker.analysis.organoid_cyst_data import CystFrameData, CystTrajectory, ExperimentData, OrganoidData
from organoidtracker.analysis.organoid_visualizations import OrganoidVisualizationSuite

# Outputs are read from and written to data/ under the current working directory, as the
# application does; run this script from the repository root (or the directory holding data/).
logger = logging.getLogger(__name__)

project_root = Path.cwd()


def load_csv_data():
    """Load real data from CSV files"""

    data_dir = project_root / "data" / "output_videos"
    raw_data_file = data_dir / "raw_cyst_data.csv"
    summary_file = data_dir / "analysis_summary.json"

    if not raw_data_file.exists():
        logger.error(f"Raw data CSV not found: {raw_data_file}")
        return None, None

    logger.info(f"Loading real CSV data from: {raw_data_file}")

    # Load CSV data
    df = pd.read_csv(raw_data_file)
    logger.debug(f"Loaded {len(df)} data points")

    # Load metadata
    import json

    with open(summary_file) as f:
        metadata = json.load(f)

    experiment_info = metadata["experiment_info"]

    logger.debug(f"Experiment: {experiment_info['total_organoids']} organoids, {experiment_info['total_cysts']} cysts")
    logger.debug(f"Duration: {experiment_info['time_lapse_days']} days")
    logger.debug(f"Conversion: {experiment_info['conversion_factor_um_per_pixel']} μm/pixel")

    return df, experiment_info


def create_experiment_from_csv(df, experiment_info):
    """Create ExperimentData object from CSV data"""

    # Create experiment
    experiment = ExperimentData(
        total_frames=experiment_info["total_frames"],
        time_lapse_days=experiment_info["time_lapse_days"],
        conversion_factor_um_per_pixel=experiment_info["conversion_factor_um_per_pixel"],
        frame_timestamps=list(range(experiment_info["total_frames"])),  # 0, 1, 2, ..., frames-1
    )

    # Group data by organoid and cyst
    organoid_groups = df.groupby("Organoid_ID")

    for org_id, org_data in organoid_groups:
        # Create organoid (use first cyst's centroid as marker point)
        first_row = org_data.iloc[0]
        marker_point = (first_row["Centroid_X"], first_row["Centroid_Y"])

        organoid = OrganoidData(organoid_id=org_id, marker_point=marker_point)

        # Process cysts for this organoid
        cyst_groups = org_data.groupby("Cyst_ID")

        for cyst_id, cyst_data in cyst_groups:
            # Create cyst trajectory
            cyst_trajectory = CystTrajectory(cyst_id=cyst_id, organoid_id=org_id)

            # Add frame data from CSV
            for _, row in cyst_data.iterrows():
                frame_data = CystFrameData(
                    frame_index=int(row["Frame"]),
                    area_pixels=row["Area_um2"] / (experiment_info["conversion_factor_um_per_pixel"] ** 2),
                    circularity=row["Circularity"],
                    centroid=(row["Centroid_X"], row["Centroid_Y"]),
                )
                cyst_trajectory.add_frame_data(frame_data)

            organoid.add_cyst(cyst_trajectory)

        experiment.add_organoid(organoid)

    return experiment


def generate_visualizations_from_csv(output_dir="csv_output"):
    """Generate visualizations from CSV data"""

    logger.info("CSV-Based Visualization Generator")

    # Load CSV data
    df, experiment_info = load_csv_data()
    if df is None:
        return 1

    # Create experiment structure
    logger.info("Creating experiment structure from CSV...")
    experiment = create_experiment_from_csv(df, experiment_info)
    logger.debug(f"Created experiment with {experiment.get_total_organoid_count()} organoids")

    # Create visualizer
    visualizer = OrganoidVisualizationSuite()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    viz_dir = output_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    logger.info(f"Output directory: {viz_dir}")

    # Generate visualizations
    logger.info("Generating visualizations...")
    start_time = time.time()

    results = visualizer.create_all_visualizations(experiment, str(viz_dir))

    elapsed_time = time.time() - start_time

    if results and isinstance(results, dict):
        logger.info(f"Generated {len(results)} visualizations in {elapsed_time:.1f}s:")
        for viz_name, file_path in results.items():
            if file_path:
                logger.debug(f"• {viz_name}: {Path(file_path).name}")

        logger.info(f"Your cyst heatmap: {viz_dir}/f_lasagna_plot.svg")
        logger.info(f"All results in: {viz_dir}")

        return 0
    else:
        logger.error("Visualization generation failed")
        return 1


def main():
    return generate_visualizations_from_csv()


if __name__ == "__main__":
    sys.exit(main())
