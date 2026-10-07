"""
Organoid Analysis Report Generator

Integrates the new organoid-cyst analysis system with comprehensive reporting.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import RESULTS_VERSION, __version__
from ..paths import source_revision
from .organoid_analysis_engine import OrganoidAnalysisEngine, OrganoidAnalysisValidator
from .organoid_csv_exporter import OrganoidCSVExporter
from .organoid_cyst_data import ExperimentData
from .organoid_visualizations import OrganoidVisualizationSuite

# Optional imports for enhanced reporting
logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.warning("ReportLab not available. PDF reports will be basic.")


@dataclass
class Analysis:
    """The measured experiment and the facts about the tracking run it came from (no files yet)."""

    experiment: ExperimentData
    run: dict[str, Any]  # tracking facts (status, frame counts, frame map, tracked frames, object ids)
    validation: dict[str, Any]
    untracked_cysts: list[int] = field(default_factory=list)  # annotated cysts without any tracked mask
    unannotated_objects: list[int] = field(default_factory=list)  # tracked objects that are not annotated cysts

    @property
    def complete(self) -> bool:
        """False for a partial tracking run; its exports cover only the tracked frames."""
        return self.run.get("status", "completed") == "completed"


class OrganoidAnalysisReportGenerator:
    """
    Comprehensive report generator for organoid-cyst analysis
    """

    def __init__(self):
        self.analysis_engine = OrganoidAnalysisEngine()
        self.csv_exporter = OrganoidCSVExporter()
        self.visualizer = OrganoidVisualizationSuite()
        self.validator = OrganoidAnalysisValidator()

    def generate_complete_analysis_report(
        self,
        tracking_results: dict[str, Any],
        organoid_data: dict[int, dict],  # From GUI workflow
        time_lapse_days: float,
        conversion_factor: float,
        output_dir: str,
        debug_mode: bool = False,
        original_frames: list | None = None,
        frame_timestamps: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive analysis report with all components

        :meth:`analyze` followed by :meth:`write_report`. A failure in either step is returned
        as an error summary (``success: False``) instead of raised, as the GUI expects.

        Returns:
            Dictionary with paths to all generated files and analysis summary
        """
        logger.info("Starting comprehensive organoid analysis...")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Time lapse: {time_lapse_days} days")
        logger.debug(f"Conversion factor: {conversion_factor} μm/pixel")
        logger.debug(f"Debug mode: {debug_mode}")

        # Store data for frame comparison section
        self._original_frames = original_frames
        self._tracking_results = tracking_results
        if original_frames:
            logger.debug(f"Frame comparison: {len(original_frames)} original frames available")
        else:
            logger.debug("Frame comparison: Original frames not available")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            analysis = self.analyze(
                tracking_results,
                organoid_data,
                time_lapse_days,
                conversion_factor,
                frame_timestamps=frame_timestamps,
                debug_mode=debug_mode,
            )
            summary = self.write_report(analysis, output_dir, debug_mode=debug_mode)

            logger.info("Complete analysis finished successfully!")
            logger.info(f"All files saved to: {output_dir}")

            return summary

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback

            traceback.print_exc()

            # Return error summary
            return {
                "success": False,
                "error": str(e),
                "output_directory": str(output_dir),
                "timestamp": datetime.now().isoformat(),
            }

    def analyze(
        self,
        tracking_results: dict[str, Any],
        organoid_data: dict[int, dict],
        time_lapse_days: float,
        conversion_factor: float,
        frame_timestamps: Sequence[float] | None = None,
        debug_mode: bool = False,
    ) -> Analysis:
        """Measure the tracked cysts: the analysis half of the report, writing no file.

        Describes the tracking run and compares the annotations with the tracked objects,
        extracts the experiment data and validates it. ``frame_timestamps`` (days per
        chronological frame) replaces the uniform time axis derived from ``time_lapse_days``.
        """
        # Set up analysis engine
        self.analysis_engine.conversion_factor = conversion_factor
        self.analysis_engine.debug_mode = debug_mode

        # Step 1: Describe the run and compare the annotations with the tracked objects.
        # Nothing is invented: a cyst without masks has no trajectory, a tracked object that
        # is not an annotated cyst is ignored, and both cases are reported.
        logger.info("Step 1: Checking the tracking results...")
        run = self._describe_tracking(tracking_results)
        total_frames = run["frames_total"]
        logger.debug(f"Frames: {total_frames} tracked, {run['frames_with_masks']} with masks; tracking {run['status']}")
        annotated_ids = sorted(
            int(cyst["cyst_id"]) for info in organoid_data.values() for cyst in info.get("cysts", [])
        )
        untracked = sorted(set(annotated_ids) - set(run["object_ids"]))
        unannotated = sorted(set(run["object_ids"]) - set(annotated_ids))
        if untracked:
            logger.warning(f"Annotated cysts without any tracked mask (no trajectory): {untracked}")
        if unannotated:
            logger.warning(f"Tracked objects that are not annotated cysts (ignored): {unannotated}")

        # Step 2: Extract experiment data from tracking results
        logger.info("Step 2: Extracting experiment data...")

        experiment = self.analysis_engine.extract_experiment_data_from_tracking(
            tracking_results=tracking_results,
            organoid_data=organoid_data,
            time_lapse_days=time_lapse_days,
            total_frames=total_frames,
            observed_frames=run["tracked_frames"],
            frame_timestamps=frame_timestamps,
        )

        # Step 3: Validate data quality
        logger.info("Step 3: Validating data quality...")
        validation_results = self.validator.validate_experiment_data(experiment)
        if run["status"] != "completed":
            validation_results["warnings"].append(
                f"Tracking {run['status']}: {run['frames_done']} of {run['frames_total']} frames were tracked"
                + (f" ({run['error']})" if run["error"] else "")
            )
        if untracked:
            validation_results["warnings"].append(f"Annotated cysts without tracked masks: {untracked}")

        logger.debug("Validation summary:")
        logger.debug(f"• Total organoids: {validation_results['total_organoids']}")
        logger.debug(f"• Total cysts: {validation_results['total_cysts']}")
        logger.debug(f"• Frames analyzed: {validation_results['frames_analyzed']}")
        for warning in validation_results.get("warnings", []):
            logger.warning(f"{warning}")

        return Analysis(
            experiment=experiment,
            run=run,
            validation=validation_results,
            untracked_cysts=untracked,
            unannotated_objects=unannotated,
        )

    def write_report(self, analysis: Analysis, output_dir: str, debug_mode: bool = False) -> dict[str, Any]:
        """Write the CSV tables, figures, PDF and ``analysis_summary.json`` of an analysis; returns the summary."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        experiment, validation_results, run = analysis.experiment, analysis.validation, analysis.run

        # Save experiment data for debugging
        if debug_mode:
            experiment_json_path = output_path / "experiment_data_debug.json"
            self.analysis_engine.save_experiment_data(experiment, str(experiment_json_path))

        # Step 4: Export CSV data
        logger.info("Step 4: Exporting CSV data...")
        csv_paths = self._export_csv_data(experiment, output_path)

        # Step 5: Generate visualizations
        logger.info("Step 5: Creating visualizations...")
        viz_paths = self.visualizer.create_all_visualizations(experiment, str(output_path / "visualizations"))

        # Step 5.1: Generate frame comparison visualization (TEMPORARILY DISABLED)
        logger.info("Step 5.1: Frame comparison visualization temporarily disabled")
        logger.debug("Frame comparison generation has been temporarily disabled per user request")

        # Step 6: Generate enhanced PDF report
        logger.info("Step 6: Generating PDF report...")
        pdf_path = self._generate_enhanced_pdf_report(
            experiment, validation_results, csv_paths, viz_paths, output_path, run
        )

        # Step 7: Create analysis summary
        logger.info("Step 7: Creating analysis summary...")
        summary = self._create_analysis_summary(experiment, validation_results, csv_paths, viz_paths, pdf_path, run)

        # Save summary as JSON
        summary_json_path = output_path / "analysis_summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def _describe_tracking(self, tracking_results: Any) -> dict[str, Any]:
        """Facts about the tracking run behind the results.

        A ``TrackingResult`` carries them (status, frames_total, frames_done, error, direction,
        frame_map). A plain ``{frame_index: {object_id: mask}}`` dict can only say which frames
        have masks; its frame count is the highest frame index plus one, because a frame whose
        masks were all rejected is still a frame of the video.
        """
        results = tracking_results
        if isinstance(results, dict) and "video_segments" in results:
            results = results["video_segments"]
        elif isinstance(results, dict) and "masks" in results:
            results = results["masks"]
        if not isinstance(results, dict):
            raise ValueError("tracking results must be a mapping {frame_index: {object_id: mask}}")
        frame_keys = [key for key in results if isinstance(key, int)]
        frame_map = list(getattr(tracking_results, "frame_map", None) or [])
        frames_total = int(getattr(tracking_results, "frames_total", 0) or 0)
        if frames_total <= 0 and frame_map:
            frames_total = len(frame_map)
        if frames_total <= 0:
            if not frame_keys:
                raise ValueError("cannot determine the number of frames: the tracking results are empty")
            frames_total = max(frame_keys) + 1
        frames_done = int(getattr(tracking_results, "frames_done", 0) or 0) or len(frame_keys)
        tracked_frames = sorted(getattr(tracking_results, "tracked_frames", None) or frame_keys)
        object_ids = sorted({int(obj) for key in frame_keys for obj in results[key]})
        return {
            "status": getattr(tracking_results, "status", "completed"),
            "frames_total": frames_total,
            "frames_done": frames_done,
            "frames_with_masks": len(frame_keys),
            "error": getattr(tracking_results, "error", None),
            "direction": getattr(tracking_results, "direction", None),
            "annotation_frame": getattr(tracking_results, "annotation_frame", None),
            "frame_map": frame_map,
            "tracked_frames": tracked_frames,
            "object_ids": object_ids,
        }

    def _determine_total_frames(self, tracking_results: dict[str, Any]) -> int:
        """Number of frames of the tracked video (see ``_describe_tracking``)."""
        return self._describe_tracking(tracking_results)["frames_total"]

    @staticmethod
    def _tracking_label(run: dict[str, Any] | None) -> str:
        if run is None:
            return "not recorded"
        label = f"{run['status']}: {run['frames_done']} of {run['frames_total']} frames"
        if run.get("direction"):
            label += f", {run['direction']}"
        return label

    def _export_csv_data(self, experiment: ExperimentData, output_path: Path) -> dict[str, str]:
        """
        Export all CSV data formats
        """
        csv_paths = {}

        try:
            # Raw data table
            raw_csv_path = output_path / "raw_cyst_data.csv"
            csv_paths["raw_data"] = self.csv_exporter.export_raw_data_table(experiment, str(raw_csv_path))

            # Cyst summary
            summary_csv_path = output_path / "cyst_summary.csv"
            csv_paths["cyst_summary"] = self.csv_exporter.export_summary_table(experiment, str(summary_csv_path))

            # Organoid summary
            organoid_csv_path = output_path / "organoid_summary.csv"
            csv_paths["organoid_summary"] = self.csv_exporter.export_organoid_summary(
                experiment, str(organoid_csv_path)
            )

            logger.debug(f"CSV files exported: {len(csv_paths)}")

        except Exception as e:
            logger.error(f"CSV export error: {e}")

        return csv_paths

    def _generate_enhanced_pdf_report(
        self,
        experiment: ExperimentData,
        validation_results: dict[str, Any],
        csv_paths: dict[str, str],
        viz_paths: dict[str, str],
        output_path: Path,
        run: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Generate enhanced PDF report with visualizations
        """
        if not HAS_REPORTLAB:
            logger.warning("ReportLab not available, skipping PDF generation")
            return None

        try:
            pdf_path = output_path / "organoid_analysis_report.pdf"

            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path), pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18
            )

            # Build PDF content
            story = []
            styles = getSampleStyleSheet()

            # Title
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # Center
            )

            story.append(Paragraph("Organoid Cyst Analysis Report", title_style))
            story.append(Spacer(1, 20))

            if run is not None and run["status"] != "completed":
                notice_style = ParagraphStyle("Notice", parent=styles["Normal"], textColor=colors.red, fontSize=11)
                error_text = f" ({run['error']})" if run.get("error") else ""
                story.append(
                    Paragraph(
                        f"<b>Partial tracking run:</b> {run['frames_done']} of {run['frames_total']} frames were "
                        f"tracked{error_text}. Measurements cover only the tracked frames.",
                        notice_style,
                    )
                )
                story.append(Spacer(1, 12))

            # Analysis summary
            story.append(Paragraph("Analysis Summary", styles["Heading2"]))

            summary_data = [
                ["Metric", "Value"],
                ["Total Organoids", str(validation_results["total_organoids"])],
                ["Total Cysts", str(validation_results["total_cysts"])],
                ["Frames Analyzed", str(validation_results["frames_analyzed"])],
                ["Tracking", self._tracking_label(run)],
                ["Time Period", f"{experiment.time_lapse_days} days"],
                ["Conversion Factor", f"{experiment.conversion_factor_um_per_pixel} μm/pixel"],
                ["Analysis Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ]

            summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
            summary_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(summary_table)
            story.append(Spacer(1, 20))

            # Note: Frame comparison is now handled as a standard visualization (g_frame_comparison.png)

            # Add visualizations
            story.append(Paragraph("Visualizations", styles["Heading2"]))

            for viz_name, viz_path in viz_paths.items():
                if viz_path and Path(viz_path).exists():
                    try:
                        # Add visualization title
                        viz_titles = {
                            "organoids_with_cysts": "Percentage of Organoids with Cysts Over Time",
                            "cyst_organoid_ratio": "Cyst to Organoid Ratio Over Time",
                            "cyst_areas_multiline": "Individual Cyst Area Trajectories",
                            "cyst_circularity_multiline": "Individual Cyst Circularity Trajectories",
                            "circularity_scatter": "Circularity vs Time (Sized by Area)",
                            "lasagna_plot": "Organoid Growth Heatmap (Lasagna Plot)",
                            "frame_comparison": "Frame-by-Frame Comparison: Original vs Tracked Cysts",
                        }

                        title = viz_titles.get(viz_name, viz_name.replace("_", " ").title())
                        story.append(Paragraph(title, styles["Heading3"]))

                        # Add image
                        img = Image(viz_path, width=6 * inch, height=4 * inch)
                        story.append(img)
                        story.append(Spacer(1, 12))

                    except Exception as e:
                        logger.warning(f"Could not add visualization {viz_name}: {e}")

            # Add data files information
            story.append(Paragraph("Generated Data Files", styles["Heading2"]))

            file_info = []
            file_info.append(["File Type", "Description", "Filename"])

            if csv_paths.get("raw_data"):
                file_info.append(["Raw Data CSV", "Frame-by-frame cyst measurements", Path(csv_paths["raw_data"]).name])
            if csv_paths.get("cyst_summary"):
                file_info.append(
                    ["Cyst Summary CSV", "Aggregate metrics per cyst", Path(csv_paths["cyst_summary"]).name]
                )
            if csv_paths.get("organoid_summary"):
                file_info.append(
                    ["Organoid Summary CSV", "Aggregate metrics per organoid", Path(csv_paths["organoid_summary"]).name]
                )

            if len(file_info) > 1:
                files_table = Table(file_info, colWidths=[1.5 * inch, 3 * inch, 1.5 * inch])
                files_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("FONTSIZE", (0, 1), (-1, -1), 9),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )

                story.append(files_table)

            # Build PDF
            doc.build(story)

            logger.debug(f"Enhanced PDF report generated: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            return None

    def _create_analysis_summary(
        self,
        experiment: ExperimentData,
        validation_results: dict[str, Any],
        csv_paths: dict[str, str],
        viz_paths: dict[str, str],
        pdf_path: str | None,
        run: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create comprehensive analysis summary
        """
        # Calculate key metrics
        all_cysts = experiment.get_all_cysts()

        # Growth rate statistics (μm² per day on the experiment's time axis)
        growth_rate_values = [rate for _, rate in experiment.sort_organoids_by_growth_rate()]

        # Time coverage statistics
        if all_cysts:
            trajectory_lengths = [len(cyst.frame_data) for cyst in all_cysts]
            mean_trajectory_length = sum(trajectory_lengths) / len(trajectory_lengths)
            coverage_percent = mean_trajectory_length / experiment.total_frames * 100
        else:
            mean_trajectory_length = 0
            coverage_percent = 0

        experiment_info: dict[str, Any] = {
            "total_organoids": len(experiment.organoids),
            "total_cysts": len(all_cysts),
            "total_frames": experiment.total_frames,
            "time_lapse_days": experiment.time_lapse_days,
            "conversion_factor_um_per_pixel": experiment.conversion_factor_um_per_pixel,
        }
        if experiment.has_explicit_time_axis:
            # The CSV carries times only where a cyst was measured; a reload needs the whole axis.
            # The uniform axis is reconstructed from the span, so it is not repeated here.
            experiment_info["frame_timestamps"] = list(experiment.frame_timestamps)

        summary = {
            "success": True,
            "complete": run is None or run["status"] == "completed",
            "tracking": run,
            "timestamp": datetime.now().isoformat(),
            "results_version": RESULTS_VERSION,
            "software": {"organoidtracker": __version__, "source_revision": source_revision()},
            "experiment_info": experiment_info,
            "quality_metrics": {
                "mean_trajectory_length_frames": round(mean_trajectory_length, 1),
                "tracking_coverage_percent": round(coverage_percent, 1),
                "organoids_with_cysts": sum(1 for org in experiment.organoids.values() if len(org.cysts) > 0),
            },
            "growth_statistics": {
                "mean_growth_rate_um2_per_day": round(sum(growth_rate_values) / len(growth_rate_values), 4)
                if growth_rate_values
                else 0,
                "max_growth_rate_um2_per_day": round(max(growth_rate_values), 4) if growth_rate_values else 0,
                "min_growth_rate_um2_per_day": round(min(growth_rate_values), 4) if growth_rate_values else 0,
            },
            "output_files": {"csv_files": csv_paths, "visualizations": viz_paths, "pdf_report": pdf_path},
            "validation_results": validation_results,
        }

        return summary

        # Note: Frame comparison is now handled as a Stage 2 visualization (g_frame_comparison.png)

    # The old _create_frame_comparison_section method has been removed and replaced with
    # create_frame_comparison_visualization in OrganoidVisualizationSuite
