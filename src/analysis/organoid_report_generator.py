"""
Organoid Analysis Report Generator

Integrates the new organoid-cyst analysis system with comprehensive reporting.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .organoid_analysis_engine import OrganoidAnalysisEngine, OrganoidAnalysisValidator
from .organoid_csv_exporter import OrganoidCSVExporter
from .organoid_visualizations import OrganoidVisualizationSuite
from .organoid_cyst_data import ExperimentData
from .data_reconstruction import DataReconstructionEngine

# Optional imports for enhanced reporting
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("⚠️ ReportLab not available. PDF reports will be basic.")


class OrganoidAnalysisReportGenerator:
    """
    Comprehensive report generator for organoid-cyst analysis
    """

    def __init__(self):
        self.analysis_engine = OrganoidAnalysisEngine()
        self.csv_exporter = OrganoidCSVExporter()
        self.visualizer = OrganoidVisualizationSuite()
        self.validator = OrganoidAnalysisValidator()
        self.reconstruction_engine = DataReconstructionEngine()

    def generate_complete_analysis_report(
        self,
        tracking_results: Dict[str, Any],
        organoid_data: Dict[int, Dict],  # From GUI workflow
        time_lapse_days: float,
        conversion_factor: float,
        output_dir: str,
        debug_mode: bool = False,
        original_frames: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with all components

        Returns:
            Dictionary with paths to all generated files and analysis summary
        """
        print(f"🧬 Starting comprehensive organoid analysis...")
        print(f"   📁 Output directory: {output_dir}")
        print(f"   🕐 Time lapse: {time_lapse_days} days")
        print(f"   📏 Conversion factor: {conversion_factor} μm/pixel")
        print(f"   🔍 Debug mode: {debug_mode}")

        # Store data for frame comparison section
        self._original_frames = original_frames
        self._tracking_results = tracking_results
        if original_frames:
            print(f"   📸 Frame comparison: {len(original_frames)} original frames available")
        else:
            print(f"   📸 Frame comparison: Original frames not available")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set up analysis engine
        self.analysis_engine.conversion_factor = conversion_factor
        self.analysis_engine.debug_mode = debug_mode

        try:
            # Step 1: Check for data reconstruction needs
            print("\n🔍 Step 1: Checking data integrity...")

            # Determine total frames from tracking results
            total_frames = self._determine_total_frames(tracking_results)
            print(f"   📊 Total frames detected: {total_frames}")

            # Set debug mode for reconstruction engine
            self.reconstruction_engine.debug_mode = debug_mode

            # Detect potential data mismatch
            mismatch_info = self.reconstruction_engine.detect_data_mismatch(
                tracking_results, organoid_data, total_frames
            )

            # Reconstruct data if needed
            final_organoid_data = organoid_data
            reconstruction_performed = False

            if mismatch_info['has_mismatch'] and mismatch_info['reconstruction_possible']:
                print(f"\n🛠️ Step 1b: Reconstructing missing cyst data...")
                print(f"   ⚠️ Detected {mismatch_info['missing_cysts']} missing cysts from {mismatch_info['tracked_objects']} tracked objects")

                final_organoid_data = self.reconstruction_engine.reconstruct_organoid_data(
                    tracking_results, organoid_data, mismatch_info
                )
                reconstruction_performed = True

                # Save reconstruction report
                if debug_mode:
                    reconstruction_report_path = output_path / "data_reconstruction_report.json"
                    self.reconstruction_engine.save_reconstruction_report(
                        mismatch_info, final_organoid_data, str(reconstruction_report_path)
                    )
                    print(f"   📄 Reconstruction report saved: {reconstruction_report_path}")

            elif mismatch_info['has_mismatch']:
                print(f"\n⚠️ Warning: Data mismatch detected but reconstruction not possible")
                print(f"   • Tracked objects: {mismatch_info['tracked_objects']}")
                print(f"   • Expected objects: {mismatch_info['expected_total']}")
                print(f"   • Proceeding with available data...")

            # Step 2: Extract experiment data from tracking results
            print("\n🔬 Step 2: Extracting experiment data...")

            experiment = self.analysis_engine.extract_experiment_data_from_tracking(
                tracking_results=tracking_results,
                organoid_data=final_organoid_data,  # Use reconstructed data
                time_lapse_days=time_lapse_days,
                total_frames=total_frames
            )

            # Save experiment data for debugging
            if debug_mode:
                experiment_json_path = output_path / "experiment_data_debug.json"
                self.analysis_engine.save_experiment_data(experiment, str(experiment_json_path))

            # Step 3: Validate data quality
            print("\n✅ Step 3: Validating data quality...")
            validation_results = self.validator.validate_experiment_data(experiment)

            print(f"   📊 Validation summary:")
            print(f"      • Total organoids: {validation_results['total_organoids']}")
            print(f"      • Total cysts: {validation_results['total_cysts']}")
            print(f"      • Frames analyzed: {validation_results['frames_analyzed']}")

            if validation_results.get('warnings'):
                for warning in validation_results['warnings']:
                    print(f"   ⚠️ {warning}")

            # Add reconstruction info to validation results
            if reconstruction_performed:
                if 'warnings' not in validation_results:
                    validation_results['warnings'] = []
                validation_results['warnings'].append(f"Data reconstruction performed: {mismatch_info['missing_cysts']} cysts recovered")

            # Step 4: Export CSV data
            print("\n📊 Step 4: Exporting CSV data...")
            csv_paths = self._export_csv_data(experiment, output_path)

            # Step 5: Generate visualizations
            print("\n🎨 Step 5: Creating visualizations...")
            viz_paths = self.visualizer.create_all_visualizations(
                experiment, str(output_path / "visualizations")
            )

            # Step 5.1: Generate frame comparison visualization (TEMPORARILY DISABLED)
            print("\n📸 Step 5.1: Frame comparison visualization temporarily disabled")
            print(f"   ⏸️ Frame comparison generation has been temporarily disabled per user request")
            # if self._original_frames and self._tracking_results:
            #     frame_comparison_path = self.visualizer.create_frame_comparison_visualization(
            #         self._original_frames,
            #         self._tracking_results,
            #         str(output_path / "visualizations" / "g_frame_comparison.png")
            #     )
            #     if frame_comparison_path:
            #         viz_paths['frame_comparison'] = frame_comparison_path
            #         print(f"   ✅ Frame comparison visualization: {Path(frame_comparison_path).name}")
            #     else:
            #         print(f"   ❌ Frame comparison visualization failed")
            # else:
            #     missing = []
            #     if not self._original_frames:
            #         missing.append("original frames")
            #     if not self._tracking_results:
            #         missing.append("tracking results")
            #     print(f"   ⚠️ Skipping frame comparison - missing: {', '.join(missing)}")

            # Step 6: Generate enhanced PDF report
            print("\n📄 Step 6: Generating PDF report...")
            pdf_path = self._generate_enhanced_pdf_report(
                experiment, validation_results, csv_paths, viz_paths, output_path
            )

            # Step 7: Create analysis summary
            print("\n📋 Step 7: Creating analysis summary...")
            summary = self._create_analysis_summary(
                experiment, validation_results, csv_paths, viz_paths, pdf_path
            )

            # Save summary as JSON
            summary_json_path = output_path / "analysis_summary.json"
            with open(summary_json_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"\n✅ Complete analysis finished successfully!")
            print(f"📁 All files saved to: {output_dir}")

            return summary

        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()

            # Return error summary
            return {
                'success': False,
                'error': str(e),
                'output_directory': str(output_dir),
                'timestamp': datetime.now().isoformat()
            }

    def _determine_total_frames(self, tracking_results: Dict[str, Any]) -> int:
        """
        Determine total number of frames from tracking results
        """
        try:
            # Try different possible keys for frame count
            if 'total_frames' in tracking_results:
                return tracking_results['total_frames']
            elif 'num_frames' in tracking_results:
                return tracking_results['num_frames']
            elif 'video_segments' in tracking_results:
                # Count frames in video_segments
                segments = tracking_results['video_segments']
                if isinstance(segments, dict):
                    return len(segments)
                elif isinstance(segments, list):
                    return len(segments)
            elif 'masks' in tracking_results:
                # Count frames in masks
                masks = tracking_results['masks']
                if isinstance(masks, dict):
                    return len(masks)
                elif isinstance(masks, list):
                    return len(masks)
            elif isinstance(tracking_results, dict) and all(isinstance(k, int) for k in tracking_results.keys()):
                # Direct SAM2 format: {frame_idx: {obj_id: mask}}
                frame_count = len(tracking_results)
                print(f"✅ Detected frame count from direct SAM2 format: {frame_count}")
                return frame_count

            # Default fallback
            print("⚠️ Could not determine frame count from tracking results, using default: 100")
            return 100

        except Exception as e:
            print(f"⚠️ Error determining frame count: {e}, using default: 100")
            return 100

    def _export_csv_data(self, experiment: ExperimentData, output_path: Path) -> Dict[str, str]:
        """
        Export all CSV data formats
        """
        csv_paths = {}

        try:
            # Raw data table
            raw_csv_path = output_path / "raw_cyst_data.csv"
            csv_paths['raw_data'] = self.csv_exporter.export_raw_data_table(
                experiment, str(raw_csv_path)
            )

            # Cyst summary
            summary_csv_path = output_path / "cyst_summary.csv"
            csv_paths['cyst_summary'] = self.csv_exporter.export_summary_table(
                experiment, str(summary_csv_path)
            )

            # Organoid summary
            organoid_csv_path = output_path / "organoid_summary.csv"
            csv_paths['organoid_summary'] = self.csv_exporter.export_organoid_summary(
                experiment, str(organoid_csv_path)
            )

            print(f"   ✅ CSV files exported: {len(csv_paths)}")

        except Exception as e:
            print(f"   ❌ CSV export error: {e}")

        return csv_paths

    def _generate_enhanced_pdf_report(
        self,
        experiment: ExperimentData,
        validation_results: Dict[str, Any],
        csv_paths: Dict[str, str],
        viz_paths: Dict[str, str],
        output_path: Path
    ) -> Optional[str]:
        """
        Generate enhanced PDF report with visualizations
        """
        if not HAS_REPORTLAB:
            print("   ⚠️ ReportLab not available, skipping PDF generation")
            return None

        try:
            pdf_path = output_path / "organoid_analysis_report.pdf"

            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )

            # Build PDF content
            story = []
            styles = getSampleStyleSheet()

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center
            )

            story.append(Paragraph("Organoid Cyst Analysis Report", title_style))
            story.append(Spacer(1, 20))

            # Analysis summary
            story.append(Paragraph("Analysis Summary", styles['Heading2']))

            summary_data = [
                ['Metric', 'Value'],
                ['Total Organoids', str(validation_results['total_organoids'])],
                ['Total Cysts', str(validation_results['total_cysts'])],
                ['Frames Analyzed', str(validation_results['frames_analyzed'])],
                ['Time Period', f"{experiment.time_lapse_days} days"],
                ['Conversion Factor', f"{experiment.conversion_factor_um_per_pixel} μm/pixel"],
                ['Analysis Date', datetime.now().strftime("%Y-%m-%d %H:%M")]
            ]

            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(summary_table)
            story.append(Spacer(1, 20))

            # Note: Frame comparison is now handled as a standard visualization (g_frame_comparison.png)

            # Add visualizations
            story.append(Paragraph("Visualizations", styles['Heading2']))

            for viz_name, viz_path in viz_paths.items():
                if viz_path and Path(viz_path).exists():
                    try:
                        # Add visualization title
                        viz_titles = {
                            'organoids_with_cysts': 'Percentage of Organoids with Cysts Over Time',
                            'cyst_organoid_ratio': 'Cyst to Organoid Ratio Over Time',
                            'cyst_areas_multiline': 'Individual Cyst Area Trajectories',
                            'cyst_circularity_multiline': 'Individual Cyst Circularity Trajectories',
                            'circularity_scatter': 'Circularity vs Time (Sized by Area)',
                            'lasagna_plot': 'Organoid Growth Heatmap (Lasagna Plot)',
                            'frame_comparison': 'Frame-by-Frame Comparison: Original vs Tracked Cysts'
                        }

                        title = viz_titles.get(viz_name, viz_name.replace('_', ' ').title())
                        story.append(Paragraph(title, styles['Heading3']))

                        # Add image
                        img = Image(viz_path, width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 12))

                    except Exception as e:
                        print(f"   ⚠️ Could not add visualization {viz_name}: {e}")

            # Add data files information
            story.append(Paragraph("Generated Data Files", styles['Heading2']))

            file_info = []
            file_info.append(['File Type', 'Description', 'Filename'])

            if csv_paths.get('raw_data'):
                file_info.append(['Raw Data CSV', 'Frame-by-frame cyst measurements', Path(csv_paths['raw_data']).name])
            if csv_paths.get('cyst_summary'):
                file_info.append(['Cyst Summary CSV', 'Aggregate metrics per cyst', Path(csv_paths['cyst_summary']).name])
            if csv_paths.get('organoid_summary'):
                file_info.append(['Organoid Summary CSV', 'Aggregate metrics per organoid', Path(csv_paths['organoid_summary']).name])

            if len(file_info) > 1:
                files_table = Table(file_info, colWidths=[1.5*inch, 3*inch, 1.5*inch])
                files_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                story.append(files_table)

            # Build PDF
            doc.build(story)

            print(f"   ✅ Enhanced PDF report generated: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            print(f"   ❌ PDF generation error: {e}")
            return None

    def _create_analysis_summary(
        self,
        experiment: ExperimentData,
        validation_results: Dict[str, Any],
        csv_paths: Dict[str, str],
        viz_paths: Dict[str, str],
        pdf_path: Optional[str]
    ) -> Dict[str, Any]:
        """
        Create comprehensive analysis summary
        """
        # Calculate key metrics
        all_cysts = experiment.get_all_cysts()

        # Growth rate statistics
        growth_rates = experiment.sort_organoids_by_growth_rate()
        # Convert from μm²/frame to μm²/day using actual time lapse (accounting for Day 0)
        growth_rate_values = [rate * experiment.time_lapse_days / max(1, experiment.total_frames - 1) 
                             for _, rate in growth_rates] if growth_rates else []

        # Time coverage statistics
        if all_cysts:
            trajectory_lengths = [len(cyst.frame_data) for cyst in all_cysts]
            mean_trajectory_length = sum(trajectory_lengths) / len(trajectory_lengths)
            coverage_percent = mean_trajectory_length / experiment.total_frames * 100
        else:
            mean_trajectory_length = 0
            coverage_percent = 0

        summary = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'experiment_info': {
                'total_organoids': len(experiment.organoids),
                'total_cysts': len(all_cysts),
                'total_frames': experiment.total_frames,
                'time_lapse_days': experiment.time_lapse_days,
                'conversion_factor_um_per_pixel': experiment.conversion_factor_um_per_pixel
            },
            'quality_metrics': {
                'mean_trajectory_length_frames': round(mean_trajectory_length, 1),
                'tracking_coverage_percent': round(coverage_percent, 1),
                'organoids_with_cysts': sum(1 for org in experiment.organoids.values() if len(org.cysts) > 0)
            },
            'growth_statistics': {
                'mean_growth_rate_um2_per_day': round(sum(growth_rate_values) / len(growth_rate_values), 4) if growth_rate_values else 0,
                'max_growth_rate_um2_per_day': round(max(growth_rate_values), 4) if growth_rate_values else 0,
                'min_growth_rate_um2_per_day': round(min(growth_rate_values), 4) if growth_rate_values else 0
            },
            'output_files': {
                'csv_files': csv_paths,
                'visualizations': viz_paths,
                'pdf_report': pdf_path
            },
            'validation_results': validation_results
        }

        return summary

        # Note: Frame comparison is now handled as a Stage 2 visualization (g_frame_comparison.png)
    # The old _create_frame_comparison_section method has been removed and replaced with
    # create_frame_comparison_visualization in OrganoidVisualizationSuite