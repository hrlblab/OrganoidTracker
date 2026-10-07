"""
Data Export/Import Module for Standalone Visualization

This module handles saving and loading all data needed for standalone visualization
generation without re-running SAM2 tracking.
"""

import gzip
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)


class TrackingDataExporter:
    """
    Handles export and import of tracking data for standalone analysis
    """

    def __init__(self):
        self.version = "1.0"

    def export_tracking_data(
        self,
        tracking_results: dict[str, Any],
        organoid_data: dict[int, dict],
        time_lapse_days: float,
        conversion_factor: float,
        original_frames: list | None = None,
        debug_mode: bool = False,
        video_path: str | None = None,
        output_dir: str = "data/exported_tracking_data",
    ) -> dict[str, str]:
        """
        Export all data needed for standalone visualization generation

        Args:
            tracking_results: SAM2 tracking results (video_segments)
            organoid_data: GUI workflow organoid-cyst relationships
            time_lapse_days: Experiment duration in days
            conversion_factor: μm per pixel conversion factor
            original_frames: Original video frames (optional)
            debug_mode: Debug mode setting
            video_path: Original video file path
            output_dir: Directory to save exported data

        Returns:
            Dictionary with paths to exported files
        """
        logger.info("Exporting tracking data for standalone visualization...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare metadata
        metadata = {
            "version": self.version,
            "timestamp": timestamp,
            "video_path": video_path,
            "time_lapse_days": time_lapse_days,
            "conversion_factor_um_per_pixel": conversion_factor,
            "debug_mode": debug_mode,
            "total_frames": self._get_total_frames(tracking_results),
            "has_original_frames": original_frames is not None,
            "export_info": {
                "total_organoids": len(organoid_data),
                "total_cysts": sum(len(org["cysts"]) for org in organoid_data.values()),
                "active_object_ids": self._get_tracked_object_ids(tracking_results),
            },
        }

        exported_files = {}

        try:
            # 1. Save metadata (JSON - human readable)
            metadata_path = output_path / f"experiment_metadata_{timestamp}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            exported_files["metadata"] = str(metadata_path)
            logger.debug(f"Metadata saved: {metadata_path.name}")

            # 2. Save organoid data (JSON - human readable)
            organoid_path = output_path / f"organoid_data_{timestamp}.json"
            with open(organoid_path, "w") as f:
                json.dump(organoid_data, f, indent=2)
            exported_files["organoid_data"] = str(organoid_path)
            logger.debug(f"Organoid data saved: {organoid_path.name}")

            # 3. Save tracking results (compressed pickle - handles tensors/arrays)
            tracking_path = output_path / f"tracking_results_{timestamp}.pkl.gz"

            # Convert PyTorch tensors to numpy arrays for better serialization
            serializable_tracking = self._convert_tensors_to_numpy(tracking_results)

            with gzip.open(tracking_path, "wb") as f:
                pickle.dump(serializable_tracking, f, protocol=pickle.HIGHEST_PROTOCOL)
            exported_files["tracking_results"] = str(tracking_path)
            logger.debug(f"Tracking results saved: {tracking_path.name}")

            # 4. Save original frames (optional - as compressed video to save space)
            if original_frames is not None:
                frames_path = output_path / f"original_frames_{timestamp}.mp4"
                self._save_frames_as_video(original_frames, str(frames_path))
                exported_files["original_frames"] = str(frames_path)
                logger.debug(f"Original frames saved: {frames_path.name}")
            else:
                logger.warning("No original frames to save")

            # 5. Create a combined data file for easy loading
            combined_path = output_path / f"combined_data_{timestamp}.pkl.gz"
            combined_data = {
                "metadata": metadata,
                "organoid_data": organoid_data,
                "tracking_results": serializable_tracking,
                "original_frames_available": original_frames is not None,
            }

            with gzip.open(combined_path, "wb") as f:
                pickle.dump(combined_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            exported_files["combined_data"] = str(combined_path)
            logger.debug(f"Combined data saved: {combined_path.name}")

            # 6. Create usage instructions
            instructions_path = output_path / f"README_{timestamp}.txt"
            instructions = f"""
Standalone Visualization Data Export
===================================

Generated: {timestamp}
Video: {video_path or "Unknown"}

Usage:
    python scripts/standalone_visualizer.py {combined_path.name}

Files exported:
    • {metadata_path.name} - Experiment parameters (JSON)
    • {organoid_path.name} - Organoid-cyst relationships (JSON)
    • {tracking_path.name} - SAM2 tracking results (compressed pickle)
    • {frames_path.name if original_frames else "N/A"} - Original video frames (MP4)
    • {combined_path.name} - Combined data file (recommended)

Data Summary:
    • Total organoids: {metadata["export_info"]["total_organoids"]}
    • Total cysts: {metadata["export_info"]["total_cysts"]}
    • Total frames: {metadata["total_frames"]}
    • Time period: {time_lapse_days} days
    • Conversion: {conversion_factor} μm/pixel
"""
            with open(instructions_path, "w") as f:
                f.write(instructions)
            exported_files["instructions"] = str(instructions_path)

            logger.info(f"All data exported to: {output_dir}")
            logger.info(f"Use: python scripts/standalone_visualizer.py {combined_path.name}")

            return exported_files

        except Exception as e:
            logger.error(f"Export failed: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def load_tracking_data(self, data_file: str) -> dict[str, Any]:
        """
        Load exported tracking data for standalone analysis

        Args:
            data_file: Path to combined data file or metadata file

        Returns:
            Dictionary with all loaded data
        """
        data_path = Path(data_file)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        logger.info(f"Loading tracking data from: {data_path.name}")

        try:
            # Try loading as combined data file first
            if data_path.suffix == ".gz" or "combined_data" in data_path.name:
                with gzip.open(data_path, "rb") as f:
                    combined_data = pickle.load(f)

                logger.debug(
                    f"Loaded combined data (version: {combined_data.get('metadata', {}).get('version', 'unknown')})"
                )
                return combined_data

            # Fallback: load individual files based on metadata
            elif data_path.suffix == ".json":
                # Load metadata to find other files
                with open(data_path) as f:
                    metadata = json.load(f)

                timestamp = metadata.get("timestamp", "")
                data_dir = data_path.parent

                # Load organoid data
                organoid_file = data_dir / f"organoid_data_{timestamp}.json"
                with open(organoid_file) as f:
                    organoid_data = json.load(f)

                # Load tracking results
                tracking_file = data_dir / f"tracking_results_{timestamp}.pkl.gz"
                with gzip.open(tracking_file, "rb") as f:
                    tracking_results = pickle.load(f)

                return {
                    "metadata": metadata,
                    "organoid_data": organoid_data,
                    "tracking_results": tracking_results,
                    "original_frames_available": False,  # Would need separate loading
                }

            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _convert_tensors_to_numpy(self, data: Any) -> Any:
        """
        Recursively convert PyTorch tensors to numpy arrays for serialization
        """
        if hasattr(data, "cpu") and hasattr(data, "numpy"):
            # PyTorch tensor
            return data.cpu().numpy()
        elif isinstance(data, dict):
            return {k: self._convert_tensors_to_numpy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_tensors_to_numpy(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._convert_tensors_to_numpy(item) for item in data)
        else:
            return data

    def _get_total_frames(self, tracking_results: dict[str, Any]) -> int:
        """Extract total frame count from tracking results"""
        if isinstance(tracking_results, dict):
            if "video_segments" in tracking_results:
                return len(tracking_results["video_segments"])
            elif all(isinstance(k, int) for k in tracking_results.keys()):
                return len(tracking_results)
        return 0

    def _get_tracked_object_ids(self, tracking_results: dict[str, Any]) -> list[int]:
        """Extract tracked object IDs from tracking results"""
        object_ids = set()

        if isinstance(tracking_results, dict):
            # Handle direct SAM2 format: {frame_idx: {obj_id: mask}}
            for frame_data in tracking_results.values():
                if isinstance(frame_data, dict):
                    object_ids.update(frame_data.keys())

        return sorted(list(object_ids))

    def _save_frames_as_video(self, frames: list, output_path: str, fps: float = 5.0):
        """Save frames as compressed MP4 video"""
        if not frames:
            return

        try:
            # Get frame dimensions
            height, width = frames[0].shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Write frames
            for frame in frames:
                if len(frame.shape) == 3:
                    # RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)

            out.release()
            logger.debug(f"Saved {len(frames)} frames as video: {Path(output_path).name}")

        except Exception as e:
            logger.error(f"Failed to save frames as video: {e}")


class StandaloneVisualizationRunner:
    """
    Runs visualization generation using exported tracking data
    """

    def __init__(self):
        self.exporter = TrackingDataExporter()

    def generate_visualizations_from_file(
        self, data_file: str, output_dir: str = "standalone_visualizations", debug_mode: bool = False
    ) -> dict[str, Any]:
        """
        Generate visualizations from exported tracking data

        Args:
            data_file: Path to exported data file
            output_dir: Output directory for visualizations
            debug_mode: Enable debug output

        Returns:
            Analysis summary with visualization paths
        """
        logger.info("Starting standalone visualization generation...")
        logger.debug(f"Data file: {data_file}")
        logger.debug(f"Output directory: {output_dir}")

        try:
            # Load data
            data = self.exporter.load_tracking_data(data_file)

            # Extract components
            metadata = data["metadata"]
            organoid_data = data["organoid_data"]
            tracking_results = data["tracking_results"]

            logger.debug(
                f"Loaded: {metadata['export_info']['total_organoids']} organoids, {metadata['export_info']['total_cysts']} cysts"
            )

            # Import analysis system
            from ..analysis import OrganoidAnalysisReportGenerator

            # Create report generator
            report_generator = OrganoidAnalysisReportGenerator()

            # Generate complete analysis (same as GUI workflow)
            analysis_summary = report_generator.generate_complete_analysis_report(
                tracking_results=tracking_results,
                organoid_data=organoid_data,
                time_lapse_days=metadata["time_lapse_days"],
                conversion_factor=metadata["conversion_factor_um_per_pixel"],
                output_dir=output_dir,
                debug_mode=debug_mode or metadata.get("debug_mode", False),
                original_frames=None,  # Skip frame comparison for faster iteration
            )

            logger.info("Standalone visualization generation completed!")
            logger.info(f"Results saved to: {output_dir}")

            return analysis_summary

        except Exception as e:
            logger.error(f"Standalone visualization failed: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}
