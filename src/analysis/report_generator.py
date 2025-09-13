"""
Report generation for kidney organoid cyst analysis

Generates CSV and PDF reports from metrics calculations
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from datetime import datetime
import json

# Optional imports for enhanced functionality
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ Pandas not available. CSV functionality may be limited.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib not available. Plot generation disabled.")

# Import configuration settings
try:
    from ...config import MATPLOTLIB_DPI, VISUALIZATION_FORMAT, FONT_SCALE_FACTOR, DISABLE_VISUALIZATION_TEXT
except ImportError:
    # Fallback values if config import fails
    MATPLOTLIB_DPI = 300
    VISUALIZATION_FORMAT = 'svg'
    FONT_SCALE_FACTOR = 5.0
    DISABLE_VISUALIZATION_TEXT = True

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ ReportLab not available. Install with: pip install reportlab")


class ReportGenerator:
    """Generate comprehensive reports from metrics analysis"""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _sanitize_for_csv(self, text):
        """Sanitize text for CSV output by replacing special characters"""
        if not isinstance(text, str):
            return text

        # Replace special characters with ASCII equivalents
        replacements = {
            'μ': 'u',  # micro symbol
            '±': '+/-',  # plus-minus
            '°': 'deg',  # degree symbol
            '–': '-',   # en dash
            '—': '-',   # em dash
            ''': "'",   # left single quote
            ''': "'",   # right single quote
            '"': '"',   # left double quote
            '"': '"',   # right double quote
            '…': '...',  # ellipsis
        }

        for special, replacement in replacements.items():
            text = text.replace(special, replacement)

        return text

    def generate_csv_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """
        Generate CSV report with all metrics data

        Args:
            results: Results from MetricsCalculator
            output_dir: Output directory path

        Returns:
            Path to generated CSV file
        """
        output_path = Path(output_dir) / f"organoid_analysis_{self.timestamp}.csv"

        # Prepare data for CSV
        csv_data = []

        # Add summary information
        params = results['parameters']
        summary = results['cyst_data_summary']

        csv_data.append(['ANALYSIS SUMMARY', '', '', ''])
        csv_data.append(['Timestamp', self.timestamp, '', ''])
        csv_data.append(['Total Organoids', params['total_organoids'], '', ''])
        csv_data.append(['Time Lapse (days)', params['time_lapse_days'], '', ''])
        csv_data.append(['Conversion Factor (um/pixel)', params['conversion_factor_um_per_pixel'], '', ''])
        csv_data.append(['Cysts Tracked', summary['num_cysts_tracked'], '', ''])
        csv_data.append(['', '', '', ''])

        # Add metrics results
        csv_data.append(['METRICS RESULTS', '', '', ''])
        csv_data.append(['Metric', 'Value', 'Unit', 'Details'])

        for metric_name, metric_data in results['metrics'].items():
            if 'error' in metric_data:
                csv_data.append([metric_name, 'ERROR', '', metric_data['error']])
                continue

            info = metric_data['info']
            metric_results = metric_data['results']

            if metric_name == "Radial Expansion Velocity":
                # Special handling for velocity metric
                csv_data.append([
                    metric_name + ' (Mean)',
                    f"{metric_results['mean_value']:.3f}",
                    info['unit'],
                    f"±{metric_results['std_value']:.3f}"
                ])
                csv_data.append([
                    metric_name + ' (Max)',
                    f"{metric_results['max_value']:.3f}",
                    info['unit'],
                    ''
                ])
                csv_data.append([
                    metric_name + ' (Min)',
                    f"{metric_results['min_value']:.3f}",
                    info['unit'],
                    ''
                ])
            else:
                # Standard metrics
                value = metric_results.get('value', 'N/A')
                if isinstance(value, float):
                    value = f"{value:.3f}"
                csv_data.append([metric_name, value, info['unit'], ''])

        csv_data.append(['', '', '', ''])

        # Add individual cyst data for radial expansion
        if 'Radial Expansion Velocity' in results['metrics']:
            velocity_data = results['metrics']['Radial Expansion Velocity']['results']
            if 'cyst_details' in velocity_data:
                csv_data.append(['INDIVIDUAL CYST DATA', '', '', ''])
                csv_data.append([
                    'Cyst ID',
                    'Velocity (μm/day)',
                    'Initial Radius (μm)',
                    'Final Radius (μm)'
                ])

                for cyst_detail in velocity_data['cyst_details']:
                    csv_data.append([
                        cyst_detail['object_id'],
                        f"{cyst_detail['velocity_um_per_day']:.3f}",
                        f"{cyst_detail['initial_radius_um']:.2f}",
                        f"{cyst_detail['final_radius_um']:.2f}"
                    ])

        # Sanitize CSV data to remove problematic characters
        sanitized_csv_data = []
        for row in csv_data:
            sanitized_row = [self._sanitize_for_csv(cell) for cell in row]
            sanitized_csv_data.append(sanitized_row)

        # Write to CSV with proper encoding
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(sanitized_csv_data)
            df.to_csv(output_path, index=False, header=False, encoding='utf-8')
        else:
            # Fallback CSV writing without pandas
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(sanitized_csv_data)

        return str(output_path)

    def generate_pdf_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """
        Generate PDF report with metrics and visualizations

        Args:
            results: Results from MetricsCalculator
            output_dir: Output directory path

        Returns:
            Path to generated PDF file
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")

        output_path = Path(output_dir) / f"organoid_analysis_report_{self.timestamp}.pdf"

        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Kidney Organoid Cyst Analysis Report", title_style))
        story.append(Spacer(1, 12))

        # Analysis parameters
        story.append(Paragraph("Analysis Parameters", styles['Heading2']))
        params = results['parameters']
        param_data = [
            ['Parameter', 'Value'],
            ['Analysis Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Total Organoids', str(params['total_organoids'])],
            ['Time Lapse Period', f"{params['time_lapse_days']} days"],
            ['Conversion Factor', f"{params['conversion_factor_um_per_pixel']:.4f} μm/pixel"],
            ['Cysts Tracked', str(results['cyst_data_summary']['num_cysts_tracked'])]
        ]

        param_table = Table(param_data, colWidths=[3*inch, 2*inch])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(param_table)
        story.append(Spacer(1, 20))

        # Metrics results
        story.append(Paragraph("Metrics Results", styles['Heading2']))

        metrics_data = [['Metric', 'Value', 'Unit', 'Description']]

        for metric_name, metric_data in results['metrics'].items():
            if 'error' in metric_data:
                metrics_data.append([metric_name, 'ERROR', '', metric_data['error']])
                continue

            info = metric_data['info']
            metric_results = metric_data['results']

            if metric_name == "Radial Expansion Velocity":
                metrics_data.append([
                    metric_name + ' (Mean)',
                    f"{metric_results['mean_value']:.3f} ± {metric_results['std_value']:.3f}",
                    info['unit'],
                    info['description']
                ])
            else:
                value = metric_results.get('value', 'N/A')
                if isinstance(value, float):
                    value = f"{value:.3f}"
                metrics_data.append([metric_name, str(value), info['unit'], info['description']])

        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 1*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(metrics_table)

        # Generate and include plots
        plot_path = self._generate_plots(results, output_dir)
        if plot_path:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Data Visualization", styles['Heading2']))
            story.append(Spacer(1, 12))
            story.append(Image(plot_path, width=6*inch, height=4*inch))

        # Build PDF
        doc.build(story)

        return str(output_path)

    def _generate_plots(self, results: Dict[str, Any], output_dir: str) -> str:
        """Generate visualization plots for the report"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Matplotlib not available. Skipping plot generation.")
            return None

        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            if not DISABLE_VISUALIZATION_TEXT:
                fig.suptitle('Kidney Organoid Cyst Analysis', fontsize=int(16 * FONT_SCALE_FACTOR), fontweight='bold')

            # Plot 1: Cyst Formation Efficiency
            if 'Cyst Formation Efficiency' in results['metrics']:
                efficiency_data = results['metrics']['Cyst Formation Efficiency']['results']

                ax1 = axes[0, 0]
                organoids_with_cysts = efficiency_data['organoids_with_cysts']
                total_organoids = efficiency_data['total_organoids']
                organoids_without_cysts = total_organoids - organoids_with_cysts

                labels = ['With Cysts', 'Without Cysts']
                sizes = [organoids_with_cysts, organoids_without_cysts]
                colors = ['#66b3ff', '#ff9999']

                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Cyst Formation Efficiency')

            # Plot 2: De Novo Formation Rate
            if 'De Novo Cyst Formation Rate' in results['metrics']:
                rate_data = results['metrics']['De Novo Cyst Formation Rate']['results']

                ax2 = axes[0, 1]
                ax2.bar(['Formation Rate'], [rate_data['value']], color='#99ff99')
                ax2.set_ylabel('Cysts/day')
                ax2.set_title('De Novo Cyst Formation Rate')
                ax2.tick_params(axis='x', rotation=45)

            # Plot 3: Radial Expansion Velocity Distribution
            if 'Radial Expansion Velocity' in results['metrics']:
                velocity_data = results['metrics']['Radial Expansion Velocity']['results']

                if 'individual_velocities' in velocity_data and velocity_data['individual_velocities']:
                    ax3 = axes[1, 0]
                    velocities = velocity_data['individual_velocities']
                    ax3.hist(velocities, bins=min(10, len(velocities)), color='#ffcc99', alpha=0.7, edgecolor='black')
                    ax3.set_xlabel('Velocity (μm/day)')
                    ax3.set_ylabel('Frequency')
                    ax3.set_title('Radial Expansion Velocity Distribution')
                    ax3.axvline(velocity_data['mean_value'], color='red', linestyle='--',
                              label=f'Mean: {velocity_data["mean_value"]:.2f}')
                    ax3.legend()

            # Plot 4: Individual Cyst Velocities
            if 'Radial Expansion Velocity' in results['metrics']:
                velocity_data = results['metrics']['Radial Expansion Velocity']['results']

                if 'cyst_details' in velocity_data and velocity_data['cyst_details']:
                    ax4 = axes[1, 1]
                    cyst_ids = [detail['object_id'] for detail in velocity_data['cyst_details']]
                    velocities = [detail['velocity_um_per_day'] for detail in velocity_data['cyst_details']]

                    ax4.bar(range(len(cyst_ids)), velocities, color='#ff99cc')
                    ax4.set_xlabel('Cyst ID')
                    ax4.set_ylabel('Velocity (μm/day)')
                    ax4.set_title('Individual Cyst Velocities')
                    ax4.set_xticks(range(len(cyst_ids)))
                    ax4.set_xticklabels(cyst_ids)

            plt.tight_layout()

            # Save plot
            plot_path = Path(output_dir) / f"analysis_plots_{self.timestamp}.{VISUALIZATION_FORMAT}"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(plot_path)

        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
            return None

    def generate_reports(self, results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """
        Generate both CSV and PDF reports

        Args:
            results: Results from MetricsCalculator
            output_dir: Output directory path

        Returns:
            Dictionary with paths to generated reports
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        report_paths = {}

        try:
            # Generate CSV report
            csv_path = self.generate_csv_report(results, output_dir)
            report_paths['csv'] = csv_path
            print(f"✅ CSV report generated: {csv_path}")
        except Exception as e:
            print(f"❌ Error generating CSV report: {e}")
            report_paths['csv_error'] = str(e)

        try:
            # Generate PDF report
            pdf_path = self.generate_pdf_report(results, output_dir)
            report_paths['pdf'] = pdf_path
            print(f"✅ PDF report generated: {pdf_path}")
        except Exception as e:
            print(f"❌ Error generating PDF report: {e}")
            report_paths['pdf_error'] = str(e)

        # Save analysis data as JSON for future reference
        try:
            json_path = output_dir_path / f"analysis_data_{self.timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            report_paths['json'] = str(json_path)
            print(f"✅ Analysis data saved: {json_path}")
        except Exception as e:
            print(f"❌ Error saving analysis data: {e}")

        return report_paths

    def generate_comprehensive_reports(self, results: Dict[str, Any], output_dir: str,
                                     time_points: List[float] = None) -> Dict[str, str]:
        """
        Generate comprehensive reports with advanced visualizations

        Args:
            results: Results from MetricsCalculator
            output_dir: Output directory path
            time_points: Time points for temporal analyses

        Returns:
            Dictionary with paths to generated reports and visualizations
        """
        from .advanced_visualizations import AdvancedOrganoidVisualizer

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        report_paths = {}

        # Generate standard reports
        standard_reports = self.generate_reports(results, output_dir)
        report_paths.update(standard_reports)

        # Generate advanced visualizations
        try:
            visualizer = AdvancedOrganoidVisualizer()

            # Create visualizations directory
            viz_dir = output_dir_path / 'advanced_visualizations'
            viz_dir.mkdir(exist_ok=True)

            print("🎨 Generating advanced visualizations...")

            # Generate comprehensive dashboard
            dashboard_plots = visualizer.create_comprehensive_analysis_dashboard(
                results, str(viz_dir), time_points)

            # Add visualization paths to report
            for plot_name, plot_path in dashboard_plots.items():
                report_paths[f'viz_{plot_name}'] = plot_path

            print(f"✅ Advanced visualizations generated: {len(dashboard_plots)} plots")

            # Create visualization summary
            viz_summary = {
                'timestamp': self.timestamp,
                'total_plots': len(dashboard_plots),
                'plot_list': list(dashboard_plots.keys()),
                'output_directory': str(viz_dir)
            }

            summary_path = viz_dir / 'visualization_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(viz_summary, f, indent=2, ensure_ascii=False)

            report_paths['visualization_summary'] = str(summary_path)

        except Exception as e:
            print(f"❌ Error generating advanced visualizations: {e}")
            report_paths['visualization_error'] = str(e)

        # Generate enhanced PDF report with visualizations
        try:
            enhanced_pdf_path = self._generate_enhanced_pdf_report(results, output_dir, report_paths)
            if enhanced_pdf_path:
                report_paths['enhanced_pdf'] = enhanced_pdf_path
                print(f"✅ Enhanced PDF report generated: {enhanced_pdf_path}")
        except Exception as e:
            print(f"❌ Error generating enhanced PDF report: {e}")
            report_paths['enhanced_pdf_error'] = str(e)

        return report_paths

    def _generate_enhanced_pdf_report(self, results: Dict[str, Any], output_dir: str,
                                    report_paths: Dict[str, str]) -> str:
        """
        Generate an enhanced PDF report that includes advanced visualizations

        Args:
            results: Analysis results
            output_dir: Output directory
            report_paths: Paths to generated visualizations

        Returns:
            Path to enhanced PDF report
        """
        if not REPORTLAB_AVAILABLE:
            print("❌ ReportLab not available for enhanced PDF generation")
            return None

        output_dir_path = Path(output_dir)
        pdf_path = output_dir_path / f"enhanced_organoid_analysis_{self.timestamp}.pdf"

        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)

        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )

        story.append(Paragraph("🧬 Comprehensive Kidney Organoid Cyst Analysis", title_style))
        story.append(Spacer(1, 12))

        # Analysis overview
        overview_style = ParagraphStyle('Overview', parent=styles['Normal'], fontSize=12, spaceAfter=12)

        params = results.get('parameters', {})
        cyst_summary = results.get('cyst_data_summary', {})

        overview_text = f"""
        <b>Analysis Parameters:</b><br/>
        • Total Organoids: {params.get('total_organoids', 'N/A')}<br/>
        • Time Lapse: {params.get('time_lapse_days', 'N/A')} days<br/>
        • Conversion Factor: {params.get('conversion_factor_um_per_pixel', 'N/A')} μm/pixel<br/>
        • Cysts Tracked: {cyst_summary.get('num_cysts_tracked', 0)}<br/>
        • Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        """

        story.append(Paragraph(overview_text, overview_style))
        story.append(Spacer(1, 20))

        # Section 1: Collective Outcome Analysis
        story.append(Paragraph("📊 Section 1: Collective Outcome Analysis", styles['Heading2']))

        # Include visualizations if available
        viz_plots = [
            ('viz_cyst_formation_efficiency', 'Cyst Formation Efficiency'),
            ('viz_cystic_index_timeseries', 'Cystic Index Time Series'),
            ('viz_cystic_index_boxplot', 'Cystic Index Distribution')
        ]

        for viz_key, viz_title in viz_plots:
            if viz_key in report_paths:
                try:
                    story.append(Paragraph(f"<b>{viz_title}:</b>", styles['Heading3']))
                    img = Image(report_paths[viz_key], width=400, height=267)  # 3:2 aspect ratio
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    print(f"Warning: Could not include {viz_title} in PDF: {e}")

        # Section 2: De Novo Cyst Formation
        story.append(Paragraph("🌱 Section 2: De Novo Cyst Formation Dynamics", styles['Heading2']))

        denovo_plots = [
            ('viz_cumulative_cyst_count', 'Cumulative Cyst Count'),
            ('viz_dual_axis_dynamics', 'Initiation vs Expansion Dynamics')
        ]

        for viz_key, viz_title in denovo_plots:
            if viz_key in report_paths:
                try:
                    story.append(Paragraph(f"<b>{viz_title}:</b>", styles['Heading3']))
                    img = Image(report_paths[viz_key], width=400, height=267)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    print(f"Warning: Could not include {viz_title} in PDF: {e}")

        # Section 3: Radial Expansion Heterogeneity
        story.append(Paragraph("📏 Section 3: Radial Expansion Heterogeneity", styles['Heading2']))

        expansion_plots = [
            ('viz_lasagna_plot', 'Growth Heterogeneity (Lasagna Plot)'),
            ('viz_velocity_vs_radius', 'Growth Mechanism Analysis')
        ]

        for viz_key, viz_title in expansion_plots:
            if viz_key in report_paths:
                try:
                    story.append(Paragraph(f"<b>{viz_title}:</b>", styles['Heading3']))
                    img = Image(report_paths[viz_key], width=400, height=267)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    print(f"Warning: Could not include {viz_title} in PDF: {e}")

        # Section 4: Morphological and Spatial Analysis
        story.append(Paragraph("🔬 Section 4: Morphological & Spatial Analysis", styles['Heading2']))

        morpho_plots = [
            ('viz_morphospace', 'Morphospace Analysis'),
            ('viz_spatial_density', 'Spatial Density Distribution')
        ]

        for viz_key, viz_title in morpho_plots:
            if viz_key in report_paths:
                try:
                    story.append(Paragraph(f"<b>{viz_title}:</b>", styles['Heading3']))
                    img = Image(report_paths[viz_key], width=400, height=267)
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    print(f"Warning: Could not include {viz_title} in PDF: {e}")

        # Detailed metrics results
        story.append(Paragraph("📋 Detailed Metrics Results", styles['Heading2']))

        metrics = results.get('metrics', {})
        for metric_name, metric_data in metrics.items():
            story.append(Paragraph(f"<b>{metric_name}:</b>", styles['Heading3']))

            metric_info = metric_data.get('info', {})
            metric_results = metric_data.get('results', {})

            if 'error' in metric_data:
                story.append(Paragraph(f"❌ Error: {metric_data['error']}", styles['Normal']))
            else:
                # Create summary text
                summary_text = f"<b>Description:</b> {metric_info.get('description', 'N/A')}<br/>"
                summary_text += f"<b>Unit:</b> {metric_info.get('unit', 'N/A')}<br/>"

                if 'value' in metric_results:
                    summary_text += f"<b>Value:</b> {metric_results['value']:.3f}<br/>"
                if 'mean_value' in metric_results:
                    summary_text += f"<b>Mean Value:</b> {metric_results['mean_value']:.3f}<br/>"

                story.append(Paragraph(summary_text, styles['Normal']))

            story.append(Spacer(1, 12))

        # Build PDF
        doc.build(story)

        return str(pdf_path)