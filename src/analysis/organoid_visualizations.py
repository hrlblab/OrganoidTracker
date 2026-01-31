"""
Advanced Organoid-Cyst Visualizations

Scientific visualization suite for organoid cyst analysis with publication-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings

from .organoid_cyst_data import ExperimentData

# Import configuration settings
try:
    from ...config import MATPLOTLIB_DPI, VISUALIZATION_FORMAT, FONT_SCALE_FACTOR, DISABLE_VISUALIZATION_TEXT, DISABLE_VISUALIZATION_TITLES
except ImportError:
    # Fallback values if config import fails
    MATPLOTLIB_DPI = 150
    VISUALIZATION_FORMAT = 'png'
    FONT_SCALE_FACTOR = 1.0
    DISABLE_VISUALIZATION_TEXT = False
    DISABLE_VISUALIZATION_TITLES = False

# Optional imports with graceful fallbacks
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ Seaborn not available. Using matplotlib styling.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️ Pandas not available. Using numpy for data processing.")


class OrganoidVisualizationSuite:
    """
    Comprehensive visualization suite for organoid-cyst analysis
    """

    def __init__(self, style: str = 'default', dpi: int = 300):
        self.dpi = dpi
        self.style = style
        self.figure_size = (12, 8)

        # Set up plotting style
        self._setup_plotting_style()

        # Color schemes for different visualization types
        self.color_schemes = {
            'cyst_lines': plt.cm.tab20,  # For individual cyst trajectories
            'organoid_groups': ['#1f77b4', '#ff7f0e', '#2ca02c'],  # Fast, medium, slow growth
            'time_series': '#2E86AB',  # Primary time series color
            'secondary': '#A23B72',   # Secondary/comparison color
            'accent': '#F18F01',      # Accent color
            'background': '#C73E1D'   # Background/reference color
        }

    def _setup_plotting_style(self):
        """Configure matplotlib styling for publication-quality plots"""
        if HAS_SEABORN:
            sns.set_style("whitegrid")
            sns.set_palette("husl")

        plt.style.use(self.style)

        # Configure matplotlib for better fonts and layouts
        base_config = {
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'font.family': 'sans-serif',
            'axes.grid': True,
            'grid.alpha': 0.3
        }
        
        if DISABLE_VISUALIZATION_TEXT:
            # Comprehensive text disabling (legacy mode)
            base_config.update({
                'xtick.bottom': False,
                'xtick.top': False,
                'ytick.left': False,
                'ytick.right': False,
                'xtick.labelbottom': False,
                'xtick.labeltop': False,
                'ytick.labelleft': False,
                'ytick.labelright': False,
                'axes.labelcolor': 'none',
                'axes.titlepad': 0,
                'legend.frameon': False,
                'figure.titlesize': 0,
                'axes.titlesize': 0,
                'axes.labelsize': 0,
                'xtick.labelsize': 0,
                'ytick.labelsize': 0,
                'legend.fontsize': 0
            })
        else:
            # Enable axis labels and legends with larger fonts
            base_config.update({
                'font.size': int(12 * FONT_SCALE_FACTOR),
                'axes.labelsize': int(12 * FONT_SCALE_FACTOR),
                'xtick.labelsize': int(10 * FONT_SCALE_FACTOR),
                'ytick.labelsize': int(10 * FONT_SCALE_FACTOR),
                'legend.fontsize': int(10 * FONT_SCALE_FACTOR),
                'xtick.labelbottom': True,
                'ytick.labelleft': True,
                'axes.grid': True,
                'grid.alpha': 0.3
            })
            
            # Handle titles separately
            if DISABLE_VISUALIZATION_TITLES:
                base_config.update({
                    'figure.titlesize': 0,
                    'axes.titlesize': 0,
                    'axes.titlepad': 0
                })
            else:
                base_config.update({
                    'axes.titlesize': int(14 * FONT_SCALE_FACTOR)
                })
        
        plt.rcParams.update(base_config)

    def _safe_set_text(self, ax, text_type: str, *args, **kwargs):
        """Safely set text elements based on configuration"""
        if text_type == 'title':
            # Only set title if titles are enabled
            if not DISABLE_VISUALIZATION_TITLES:
                ax.set_title(*args, **kwargs)
        elif text_type in ['xlabel', 'ylabel', 'legend', 'text']:
            # Set other text elements if general text is enabled
            if not DISABLE_VISUALIZATION_TEXT:
                if text_type == 'xlabel':
                    ax.set_xlabel(*args, **kwargs)
                elif text_type == 'ylabel':
                    ax.set_ylabel(*args, **kwargs)
                elif text_type == 'legend':
                    ax.legend(*args, **kwargs)
                elif text_type == 'text':
                    ax.text(*args, **kwargs)

    def _safe_set_ticks(self, ax, tick_type: str, *args, **kwargs):
        """Safely set tick elements only if text is enabled"""
        if not DISABLE_VISUALIZATION_TEXT:
            if tick_type == 'xticks':
                ax.set_xticks(*args, **kwargs)
            elif tick_type == 'yticks':
                ax.set_yticks(*args, **kwargs)
            elif tick_type == 'xticklabels':
                ax.set_xticklabels(*args, **kwargs)
            elif tick_type == 'yticklabels':
                ax.set_yticklabels(*args, **kwargs)

    def create_all_visualizations(
        self,
        experiment: ExperimentData,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Create all six required visualizations and return file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"🎨 Creating advanced visualizations in: {output_dir}")

        viz_paths = {}

        try:
            # A. Time vs % organoids with cysts
            viz_paths['organoids_with_cysts'] = self.plot_organoids_with_cysts_over_time(
                experiment, str(output_path / f"a_organoids_with_cysts_vs_time.{VISUALIZATION_FORMAT}")
            )

            # B. Time vs cyst count/organoid count ratio
            viz_paths['cyst_organoid_ratio'] = self.plot_cyst_organoid_ratio_over_time(
                experiment, str(output_path / f"b_cyst_organoid_ratio_vs_time.{VISUALIZATION_FORMAT}")
            )

            # C. Time vs areas of all cysts (multiple lines)
            viz_paths['cyst_areas_multiline'] = self.plot_cyst_areas_multiline(
                experiment, str(output_path / f"c_cyst_areas_vs_time.{VISUALIZATION_FORMAT}")
            )

            # D. Time vs circularity of all cysts (multiple lines)
            viz_paths['cyst_circularity_multiline'] = self.plot_cyst_circularity_multiline(
                experiment, str(output_path / f"d_cyst_circularity_vs_time.{VISUALIZATION_FORMAT}")
            )

            # E. Time vs circularity scatter plot (sized by area)
            viz_paths['circularity_scatter'] = self.plot_circularity_scatter(
                experiment, str(output_path / f"e_circularity_scatter.{VISUALIZATION_FORMAT}")
            )

            # F. Lasagna plot (organoid area heatmap)
            viz_paths['lasagna_plot'] = self.plot_lasagna_heatmap(
                experiment, str(output_path / f"f_lasagna_plot.{VISUALIZATION_FORMAT}")
            )

            print(f"✅ All visualizations created successfully")
            return viz_paths

        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return viz_paths

    def plot_organoids_with_cysts_over_time(
        self,
        experiment: ExperimentData,
        output_path: str
    ) -> str:
        """
        A. Time vs % of organoids having at least one cyst
        """
        print("📊 Creating plot A: % organoids with cysts vs time")

        try:
            # Calculate percentage for each frame
            time_points = []
            percentages = []

            for frame_idx in range(experiment.total_frames):
                time_days = experiment.get_time_at_frame(frame_idx)
                percentage = experiment.get_percentage_organoids_with_cysts_at_frame(frame_idx)

                time_points.append(time_days)
                percentages.append(percentage)

            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size)

            ax.plot(time_points, percentages,
                   color=self.color_schemes['time_series'],
                   linewidth=2.5,
                   marker='o',
                   markersize=4,
                   label='% Organoids with Cysts')

            # Titles removed per user request

            # Formatting
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add statistics annotation
            max_percentage = max(percentages) if percentages else 0
            final_percentage = percentages[-1] if percentages else 0

            stats_text = f'Final: {final_percentage:.1f}%\nMax: {max_percentage:.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            print(f"✅ Plot A saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error creating plot A: {e}")
            return None

    def plot_cyst_organoid_ratio_over_time(
        self,
        experiment: ExperimentData,
        output_path: str
    ) -> str:
        """
        B. Time vs cyst count/organoid count ratio
        """
        print("📊 Creating plot B: cyst/organoid ratio vs time")

        try:
            # Calculate ratio for each frame
            time_points = []
            ratios = []

            for frame_idx in range(experiment.total_frames):
                time_days = experiment.get_time_at_frame(frame_idx)
                ratio = experiment.get_cyst_to_organoid_ratio_at_frame(frame_idx)

                time_points.append(time_days)
                ratios.append(ratio)

            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size)

            ax.plot(time_points, ratios,
                   color=self.color_schemes['secondary'],
                   linewidth=2.5,
                   marker='s',
                   markersize=4,
                   label='Cysts per Organoid')

            # Titles removed per user request

            # Formatting
            ax.set_ylim(0, max(ratios) * 1.1 if ratios else 1)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add statistics
            max_ratio = max(ratios) if ratios else 0
            final_ratio = ratios[-1] if ratios else 0
            mean_ratio = np.mean(ratios) if ratios else 0

            stats_text = f'Final: {final_ratio:.2f}\nMax: {max_ratio:.2f}\nMean: {mean_ratio:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            print(f"✅ Plot B saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error creating plot B: {e}")
            return None

    def plot_cyst_areas_multiline(
        self,
        experiment: ExperimentData,
        output_path: str
    ) -> str:
        """
        C. Time vs areas of all cysts (multiple lines, different start frames)
        """
        print("📊 Creating plot C: cyst areas vs time (multi-line)")

        try:
            fig, ax = plt.subplots(figsize=self.figure_size)

            # Get all cysts and create color map
            all_cysts = experiment.get_all_cysts()
            n_cysts = len(all_cysts)

            if n_cysts == 0:
                # Create empty plot - titles removed per user request
                ax.text(0.5, 0.5, 'No cyst data available', transform=ax.transAxes,
                       ha='center', va='center', fontsize=int(14 * FONT_SCALE_FACTOR), alpha=0.6)
            else:
                # Create color map for cysts
                cmap = self.color_schemes['cyst_lines']
                colors_list = [cmap(i / max(1, n_cysts - 1)) for i in range(n_cysts)]

                plotted_lines = 0

                for cyst_idx, cyst in enumerate(all_cysts):
                    if not cyst.frame_data:
                        continue

                    # Extract time and area data for this cyst
                    time_points = []
                    areas = []

                    for frame_idx, frame_data in cyst.frame_data.items():
                        time_days = experiment.get_time_at_frame(frame_idx)
                        area_um2 = frame_data.get_area_um2(experiment.conversion_factor_um_per_pixel)

                        time_points.append(time_days)
                        areas.append(area_um2)

                    if len(time_points) >= 2:  # Only plot if we have at least 2 points
                        # Sort by time
                        sorted_data = sorted(zip(time_points, areas))
                        time_points, areas = zip(*sorted_data)

                        # Plot with unique color and transparency
                        alpha = 0.8 if n_cysts <= 10 else max(0.3, 1.0 / np.sqrt(n_cysts))

                        ax.plot(time_points, areas,
                               color=colors_list[cyst_idx],
                               linewidth=1.5,
                               alpha=alpha,
                               label=f'Cyst {cyst.cyst_id} (Org {cyst.organoid_id})' if n_cysts <= 15 else None)

                        plotted_lines += 1

                # Titles removed per user request

                # Legend handling
                if n_cysts <= 15 and plotted_lines > 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=int(8 * FONT_SCALE_FACTOR))
                elif plotted_lines > 0:
                    ax.text(0.02, 0.98, f'{plotted_lines} cyst trajectories',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            print(f"✅ Plot C saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error creating plot C: {e}")
            return None

    def plot_cyst_circularity_multiline(
        self,
        experiment: ExperimentData,
        output_path: str
    ) -> str:
        """
        D. Time vs circularity of all cysts (multiple lines like C)
        """
        print("📊 Creating plot D: cyst circularity vs time (multi-line)")

        try:
            fig, ax = plt.subplots(figsize=self.figure_size)

            # Get all cysts and create color map
            all_cysts = experiment.get_all_cysts()
            n_cysts = len(all_cysts)

            if n_cysts == 0:
                # Create empty plot - titles removed per user request
                ax.text(0.5, 0.5, 'No cyst data available', transform=ax.transAxes,
                       ha='center', va='center', fontsize=int(14 * FONT_SCALE_FACTOR), alpha=0.6)
            else:
                # Create color map for cysts
                cmap = self.color_schemes['cyst_lines']
                colors_list = [cmap(i / max(1, n_cysts - 1)) for i in range(n_cysts)]

                plotted_lines = 0

                for cyst_idx, cyst in enumerate(all_cysts):
                    if not cyst.frame_data:
                        continue

                    # Extract time and circularity data for this cyst
                    time_points = []
                    circularities = []

                    for frame_idx, frame_data in cyst.frame_data.items():
                        time_days = experiment.get_time_at_frame(frame_idx)
                        circularity = frame_data.circularity

                        time_points.append(time_days)
                        circularities.append(circularity)

                    if len(time_points) >= 2:  # Only plot if we have at least 2 points
                        # Sort by time
                        sorted_data = sorted(zip(time_points, circularities))
                        time_points, circularities = zip(*sorted_data)

                        # Plot with unique color and transparency
                        alpha = 0.8 if n_cysts <= 10 else max(0.3, 1.0 / np.sqrt(n_cysts))

                        ax.plot(time_points, circularities,
                               color=colors_list[cyst_idx],
                               linewidth=1.5,
                               alpha=alpha,
                               label=f'Cyst {cyst.cyst_id} (Org {cyst.organoid_id})' if n_cysts <= 15 else None)

                        plotted_lines += 1

                # Titles removed per user request
                ax.set_ylim(0, 1.05)

                # Add reference lines
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Circle')
                ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='High Circularity')

                # Legend handling
                if n_cysts <= 12 and plotted_lines > 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=int(8 * FONT_SCALE_FACTOR))
                elif plotted_lines > 0:
                    ax.text(0.02, 0.98, f'{plotted_lines} cyst trajectories',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            print(f"✅ Plot D saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error creating plot D: {e}")
            return None

    def plot_circularity_scatter(
        self,
        experiment: ExperimentData,
        output_path: str
    ) -> str:
        """
        E. Time vs circularity scatter plot (dot size = area, color = area intensity)
        """
        print("📊 Creating plot E: circularity scatter (sized by area)")

        try:
            # Collect all data points
            time_points = []
            circularities = []
            areas = []
            organoid_ids = []

            for organoid_id, organoid in experiment.organoids.items():
                for cyst in organoid.cysts.values():
                    for frame_idx, frame_data in cyst.frame_data.items():
                        time_days = experiment.get_time_at_frame(frame_idx)
                        area_um2 = frame_data.get_area_um2(experiment.conversion_factor_um_per_pixel)

                        time_points.append(time_days)
                        circularities.append(frame_data.circularity)
                        areas.append(area_um2)
                        organoid_ids.append(organoid_id)

            if not time_points:
                # Create empty plot - titles removed per user request
                fig, ax = plt.subplots(figsize=self.figure_size)
                ax.text(0.5, 0.5, 'No cyst data available', transform=ax.transAxes,
                       ha='center', va='center', fontsize=int(14 * FONT_SCALE_FACTOR), alpha=0.6)

                plt.tight_layout()
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return output_path

            # Convert to numpy arrays
            time_points = np.array(time_points)
            circularities = np.array(circularities)
            areas = np.array(areas)

            # Create plot with space for side legend
            fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(self.figure_size[0] + 2, self.figure_size[1]),
                                              gridspec_kw={'width_ratios': [4, 1]})

            # Normalize areas for coloring only (no size variation)
            min_area, max_area = np.min(areas), np.max(areas)

            if max_area > min_area:
                # Color: light blue to dark blue for larger areas
                color_values = (areas - min_area) / (max_area - min_area)
            else:
                color_values = np.zeros_like(areas)

            # Create scatter plot with fixed size, only color varies
            scatter = ax.scatter(time_points, circularities,
                               s=50,  # Fixed size for all points
                               c=color_values,
                               cmap='Blues',  # Light blue to dark blue
                               alpha=0.8,
                               edgecolors='black',
                               linewidth=0.5)

            # Add colorbar (always show, but without text labels when disabled)
            cbar = plt.colorbar(scatter, ax=ax)
            if DISABLE_VISUALIZATION_TEXT:
                cbar.set_ticks([])  # Remove tick marks and labels

            # Formatting - titles removed per user request
            ax.set_ylim(0, 1.05)

            # Add reference lines without labels (legends moved to side panel)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)

            # Grid without legend on main plot
            if not DISABLE_VISUALIZATION_TEXT:
                ax.grid(True, alpha=0.3)

            # Create visual legend panel for circularity references
            ax_legend.set_xlim(0, 1)
            ax_legend.set_ylim(0, 1)
            ax_legend.set_aspect('equal')

            # Remove all axes elements for clean legend panel
            ax_legend.set_xticks([])
            ax_legend.set_yticks([])
            for spine in ax_legend.spines.values():
                spine.set_visible(False)

            # Add visual legend: circularity reference lines
            line_length = 0.6
            x_pos = 0.2

            # Perfect circle reference (top) - red dashed line
            perfect_y = 0.7
            ax_legend.plot([x_pos, x_pos + line_length], [perfect_y, perfect_y],
                          color='red', linestyle='--', alpha=0.7, linewidth=3)

            # High circularity reference (bottom) - orange dashed line
            high_y = 0.3
            ax_legend.plot([x_pos, x_pos + line_length], [high_y, high_y],
                          color='orange', linestyle='--', alpha=0.7, linewidth=3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            print(f"✅ Plot E saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error creating plot E: {e}")
            return None

    def plot_lasagna_heatmap(
        self,
        experiment: ExperimentData,
        output_path: str
    ) -> str:
        """
        F. Lasagna plot: individual cyst area heatmap sorted by growth rate
        """
        print("📊 Creating plot F: cyst lasagna heatmap")

        try:
            # Get all individual cysts and calculate their growth rates
            all_cysts = experiment.get_all_cysts()
            
            if not all_cysts:
                # Create empty plot - titles removed per user request
                fig, ax = plt.subplots(figsize=self.figure_size)
                ax.text(0.5, 0.5, 'No cyst data available', transform=ax.transAxes,
                       ha='center', va='center', fontsize=int(14 * FONT_SCALE_FACTOR), alpha=0.6)

                plt.tight_layout()
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                return output_path

            # Calculate individual cyst growth rates and sort
            cyst_growth_data = []
            for cyst in all_cysts:
                growth_rate_per_frame = cyst.get_mean_area_growth_rate(experiment.conversion_factor_um_per_pixel)
                # Convert from μm²/frame to μm²/day using actual time lapse (accounting for Day 0)
                growth_rate_per_day = growth_rate_per_frame * experiment.time_lapse_days / max(1, experiment.total_frames - 1)
                cyst_growth_data.append({
                    'cyst': cyst,
                    'growth_rate': growth_rate_per_day,
                    'organoid_id': cyst.organoid_id,
                    'cyst_id': cyst.cyst_id
                })

            # Sort by individual cyst growth rate (highest to lowest) across all cysts
            # This shows the true growth rate ranking regardless of organoid grouping
            cyst_growth_data.sort(key=lambda x: -x['growth_rate'])
            
            sorted_cysts = [data['cyst'] for data in cyst_growth_data]
            sorted_growth_rates = [data['growth_rate'] for data in cyst_growth_data]
            sorted_labels = [(data['organoid_id'], data['cyst_id']) for data in cyst_growth_data]

            # Create data matrix for heatmap
            n_cysts = len(sorted_cysts)
            n_frames = experiment.total_frames

            heatmap_data = np.zeros((n_cysts, n_frames))

            for row_idx, cyst in enumerate(sorted_cysts):
                for frame_idx in range(n_frames):
                    # Get individual cyst area at this frame (0 if cyst not present)
                    area = cyst.get_area_at_frame(frame_idx, experiment.conversion_factor_um_per_pixel)
                    heatmap_data[row_idx, frame_idx] = area if area is not None else 0

            # Create time axis
            time_points = [experiment.get_time_at_frame(i) for i in range(n_frames)]

            # Create figure with custom layout - calculate dimensions for perfect squares
            # For perfect squares: figure width should be proportional to n_frames, height to n_cysts
            square_size = 0.5  # Size of each square in inches
            main_width = n_frames * square_size
            main_height = n_cysts * square_size
            bar_width = 2  # Fixed width for side bar
            legend_width = 1.5  # Width for visual legends

            total_width = main_width + bar_width + legend_width
            total_height = max(main_height, 4)  # Minimum height of 4 inches

            fig, (ax_main, ax_bar, ax_legend) = plt.subplots(1, 3, figsize=(total_width, total_height),
                                                 gridspec_kw={'width_ratios': [main_width, bar_width, legend_width]})

            # Main heatmap
            if np.max(heatmap_data) > 0:
                im = ax_main.imshow(heatmap_data,
                                  cmap='YlOrRd',
                                  aspect='equal',  # Equal aspect ratio for perfect squares
                                  interpolation='nearest')

                # Add colorbar with same height as heatmap (always show, but without text labels)
                cbar = plt.colorbar(im, ax=ax_main, shrink=1.0)
                if DISABLE_VISUALIZATION_TEXT:
                    cbar.set_ticks([])  # Remove tick marks and labels
            else:
                # All zeros - create placeholder
                im = ax_main.imshow(heatmap_data,
                                  cmap='gray',
                                  aspect='equal',  # Equal aspect ratio for perfect squares
                                  vmin=0, vmax=1)
                if not DISABLE_VISUALIZATION_TEXT:
                    ax_main.text(n_frames/2, n_cysts/2, 'No cyst data',
                               ha='center', va='center', fontsize=int(12 * FONT_SCALE_FACTOR), color='red')

            # Format main plot - titles removed per user request

            # Set custom ticks for time axis (only if text is enabled)
            if not DISABLE_VISUALIZATION_TEXT:
                time_tick_indices = np.linspace(0, n_frames-1, min(8, n_frames), dtype=int)
                ax_main.set_xticks(time_tick_indices)
                ax_main.set_xticklabels([f'{time_points[i]:.1f}' for i in time_tick_indices])

                # Set cyst labels (if not too many)
                if n_cysts <= 20:
                    ax_main.set_yticks(range(n_cysts))
                    ax_main.set_yticklabels([f'Org {org_id} Cyst {cyst_id}' for org_id, cyst_id in sorted_labels])
                else:
                    # Show every nth cyst
                    step = max(1, n_cysts // 10)
                    tick_indices = range(0, n_cysts, step)
                    ax_main.set_yticks(tick_indices)
                    ax_main.set_yticklabels([f'Org {sorted_labels[i][0]} Cyst {sorted_labels[i][1]}' for i in tick_indices])

            # Growth rate bar chart (individual cyst growth rates)
            if sorted_growth_rates:
                # Categorize individual cyst growth rates
                rates_array = np.array(sorted_growth_rates)
                if len(rates_array) > 1:
                    # Use percentiles for categorization
                    low_threshold = np.percentile(rates_array, 33)
                    high_threshold = np.percentile(rates_array, 67)

                    colors = []
                    for rate in sorted_growth_rates:
                        if rate <= low_threshold:
                            colors.append(self.color_schemes['organoid_groups'][2])  # Slow - green
                        elif rate <= high_threshold:
                            colors.append(self.color_schemes['organoid_groups'][1])  # Medium - orange
                        else:
                            colors.append(self.color_schemes['organoid_groups'][0])  # Fast - blue
                else:
                    colors = [self.color_schemes['organoid_groups'][1]] * len(sorted_growth_rates)

                # Y-positions should match heatmap row order (top = highest growth rate)
                # Since matplotlib puts y=0 at bottom, we need to reverse the positions
                y_positions = list(range(n_cysts-1, -1, -1))  # [n-1, n-2, ..., 1, 0] 
                ax_bar.barh(y_positions, sorted_growth_rates, color=colors, alpha=0.7)
                
                # Always show growth rate axis label and ticks (override text disabling for this specific chart)
                ax_bar.set_xlabel('Growth Rate (μm²/day)', fontsize=12, fontweight='bold')
                ax_bar.set_ylim(-0.5, n_cysts - 0.5)
                ax_bar.set_yticks([])  # Remove y-axis labels (shared with main plot)
                
                # Enable x-axis ticks and labels for growth rates
                ax_bar.tick_params(axis='x', which='both', bottom=True, top=False, 
                                 labelbottom=True, labelsize=10)

                # Legend removed per user request - keep only growth rate axis labels

            # Create visual legend panel for cyst growth categories (always show, without text)
            ax_legend.set_xlim(0, 1)
            ax_legend.set_ylim(0, 1)
            ax_legend.set_aspect('equal')

            # Remove all axes elements for clean legend panel
            ax_legend.set_xticks([])
            ax_legend.set_yticks([])
            for spine in ax_legend.spines.values():
                spine.set_visible(False)

            # Add visual legend: three color bars representing organoid types
            if len(set(colors)) > 1:
                # Create three color bars stacked vertically
                bar_height = 0.2
                bar_width = 0.6
                x_pos = 0.2

                # Fast growth (top) - blue
                fast_y = 0.7
                ax_legend.add_patch(plt.Rectangle((x_pos, fast_y), bar_width, bar_height,
                                                facecolor=self.color_schemes['organoid_groups'][0], alpha=0.7))

                # Medium growth (middle) - orange
                medium_y = 0.4
                ax_legend.add_patch(plt.Rectangle((x_pos, medium_y), bar_width, bar_height,
                                                facecolor=self.color_schemes['organoid_groups'][1], alpha=0.7))

                # Slow growth (bottom) - green
                slow_y = 0.1
                ax_legend.add_patch(plt.Rectangle((x_pos, slow_y), bar_width, bar_height,
                                                facecolor=self.color_schemes['organoid_groups'][2], alpha=0.7))

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            print(f"✅ Plot F saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error creating plot F: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_frame_comparison_visualization(
        self,
        original_frames: List,
        tracking_results: Dict,
        output_path: str
    ) -> Optional[str]:
        """
        Create a comprehensive frame comparison visualization showing:
        - All original frames in top row
        - All overlay frames in bottom row
        - Uses EXACT video generation code to eliminate implementation differences
        """
        try:
            import cv2

            total_frames = len(original_frames)
            print(f"🎨 Creating frame comparison visualization for {total_frames} frames")
            print(f"🔧 Using actual VideoOutputGenerator to eliminate implementation differences")

            # Object colors - EXACTLY match video generation (same as VideoOutputGenerator)
            object_colors = {
                0: (128, 128, 128), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
                4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255), 7: (255, 165, 0),
                8: (128, 0, 128), 9: (255, 192, 203), 10: (165, 42, 42), 11: (144, 238, 144),
                12: (135, 206, 235), 13: (221, 160, 221), 14: (240, 230, 140), 15: (255, 99, 71),
                16: (64, 224, 208), 17: (238, 130, 238), 18: (255, 182, 193), 19: (152, 251, 152),
                20: (245, 222, 179)
            }

            # Process all frames
            original_row_frames = []
            overlay_row_frames = []

            # Import video generation code to use exact same logic
            from ..utils.video_output import VideoOutputGenerator
            video_generator = VideoOutputGenerator()

            for frame_idx in range(total_frames):
                print(f"🎨 Processing frame {frame_idx}/{total_frames-1} with video generation code")

                # Get original frame
                original_frame = original_frames[frame_idx].copy()

                # Apply tracking masks - EXACTLY match video generation temporal logic
                mask_idx = total_frames - 1 - frame_idx  # Same as video generation

                # ✅ USE EXACT VIDEO GENERATION CODE for overlay creation
                overlay_frame = video_generator._generate_multi_object_frame(
                    frame=original_frame,
                    frame_idx=mask_idx,  # Use the correct mask index
                    video_segments=tracking_results,
                    object_colors=object_colors,
                    video_type='overlay',
                    alpha=0.4,  # Match video generation alpha
                    target_width=original_frame.shape[1],
                    target_height=original_frame.shape[0]
                )

                # Store frames
                original_row_frames.append(original_frame)
                overlay_row_frames.append(overlay_frame)

            # ✅ FIX: Apply final output reversal to match video generation exactly
            # Video generation does: mask_idx reversal + final output reversal
            # We need both steps to match perfectly
            print(f"🔄 Applying final output reversal to match video generation")
            overlay_row_frames.reverse()  # Same as video generation final step

            # ✅ NOTE: Original frames stay in natural order (Frame 0 = earliest)
            # Only overlay frames need reversal to match video generation temporal logic

            # Create the visualization with proper aspect ratio and spacing
            # Calculate optimal figure size to prevent stretching
            max_width = 16  # Maximum figure width
            frame_width = min(2.0, max_width / total_frames)  # Max 2 inches per frame
            fig_width = total_frames * frame_width
            fig_height = 6  # Increased height for better spacing

            fig, axes = plt.subplots(2, total_frames, figsize=(fig_width, fig_height))

            if total_frames == 1:
                axes = axes.reshape(2, 1)

            # Plot original frames (top row)
            for i, frame in enumerate(original_row_frames):
                axes[0, i].imshow(frame, aspect='equal')  # Maintain aspect ratio
                axes[0, i].set_title(f'{i}', fontsize=int(8 * FONT_SCALE_FACTOR))  # Smaller font, just number
                axes[0, i].axis('off')

            # Plot overlay frames (bottom row)
            for i, frame in enumerate(overlay_row_frames):
                axes[1, i].imshow(frame, aspect='equal')  # Maintain aspect ratio
                axes[1, i].set_title(f'{i}', fontsize=int(8 * FONT_SCALE_FACTOR))  # Smaller font, just number
                axes[1, i].axis('off')

            # Add row labels
            fig.text(0.02, 0.75, 'Original', rotation=90, fontsize=int(12 * FONT_SCALE_FACTOR), weight='bold', ha='center', va='center')
            fig.text(0.02, 0.25, 'Tracked', rotation=90, fontsize=int(12 * FONT_SCALE_FACTOR), weight='bold', ha='center', va='center')

            # Add title with proper spacing
            fig.suptitle(f'Frame-by-Frame Comparison: Original vs Tracked Cysts ({total_frames} frames)',
                        fontsize=int(12 * FONT_SCALE_FACTOR), weight='bold', y=0.92)  # Lower y position to avoid overlap

            # Add description
            description = (f"Top row: Original video frames (Frame 0 = earliest timepoint)\n"
                          f"Bottom row: Same frames with tracked cyst overlays (alpha=0.4)\n"
                          f"Temporal order: Frame 0 → Frame {total_frames-1} (matches video generation)")

            fig.text(0.5, 0.04, description, ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

            # Adjust layout with better spacing
            plt.tight_layout()
            plt.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.18)  # More space for title and description

            # Save the figure
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            print(f"✅ Frame comparison visualization saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error creating frame comparison visualization: {e}")
            import traceback
            traceback.print_exc()
            return None