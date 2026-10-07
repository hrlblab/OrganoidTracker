"""
Organoid-Cyst CSV Exporter

Exports detailed frame-by-frame data for organoid-cyst analysis in clear tabular format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import csv

from .organoid_cyst_data import ExperimentData


class OrganoidCSVExporter:
    """
    Handles CSV export of organoid-cyst data in various formats
    """

    def __init__(self):
        self.encoding = 'utf-8'

    def export_raw_data_table(
        self,
        experiment: ExperimentData,
        output_path: str
    ) -> str:
        """
        Export complete raw data table with all metrics for each cyst at each frame

        Format: [Organoid_ID, Cyst_ID, Frame, Time_Days, Area_um2, Circularity, Centroid_X, Centroid_Y]
        """
        print(f"📊 Exporting raw data table to: {output_path}")

        # Collect all data rows
        data_rows = []

        for organoid_id, organoid in experiment.organoids.items():
            for cyst_id, cyst in organoid.cysts.items():
                for frame_idx, frame_data in cyst.frame_data.items():
                    # Calculate time for this frame
                    time_days = experiment.get_time_at_frame(frame_idx)

                    # Calculate metrics in proper units
                    area_um2 = frame_data.get_area_um2(experiment.conversion_factor_um_per_pixel)

                    # Create data row
                    row = {
                        'Organoid_ID': organoid_id,
                        'Cyst_ID': cyst_id,
                        'Frame': frame_idx,
                        'Time_Days': round(time_days, 3),
                        'Area_um2': round(area_um2, 2),
                        'Circularity': round(frame_data.circularity, 4),
                        'Centroid_X': round(frame_data.centroid[0], 1),
                        'Centroid_Y': round(frame_data.centroid[1], 1),
                        'Radius_um': round(frame_data.get_radius_um(experiment.conversion_factor_um_per_pixel), 2)
                    }

                    data_rows.append(row)

        # Sort by Organoid_ID, Cyst_ID, Frame for clarity
        data_rows.sort(key=lambda x: (x['Organoid_ID'], x['Cyst_ID'], x['Frame']))

        # Export to CSV
        if data_rows:
            try:
                # Use pandas for better formatting control
                df = pd.DataFrame(data_rows)
                df.to_csv(output_path, index=False, encoding=self.encoding, float_format='%.4f')

                print(f"✅ Raw data exported: {len(data_rows)} rows")
                return output_path

            except ImportError:
                # Fallback to standard CSV if pandas not available
                print("⚠️ Pandas not available, using standard CSV writer")
                return self._export_csv_fallback(data_rows, output_path)

        else:
            print("⚠️ No data to export")
            # Create empty file with headers
            self._create_empty_csv(output_path)
            return output_path

    def export_summary_table(
        self,
        experiment: ExperimentData,
        output_path: str
    ) -> str:
        """
        Export summary table with aggregate metrics per cyst
        """
        print(f"📋 Exporting summary table to: {output_path}")

        summary_rows = []

        for organoid_id, organoid in experiment.organoids.items():
            for cyst_id, cyst in organoid.cysts.items():
                if not cyst.frame_data:
                    continue

                # Calculate aggregate metrics
                areas_um2 = [
                    frame_data.get_area_um2(experiment.conversion_factor_um_per_pixel)
                    for frame_data in cyst.frame_data.values()
                ]
                circularities = [
                    frame_data.circularity
                    for frame_data in cyst.frame_data.values()
                ]

                # Growth metrics
                growth_rate = cyst.get_mean_area_growth_rate(experiment.conversion_factor_um_per_pixel)

                row = {
                    'Organoid_ID': organoid_id,
                    'Cyst_ID': cyst_id,
                    'First_Frame': cyst.first_appearance_frame,
                    'Last_Frame': cyst.last_appearance_frame,
                    'Frames_Tracked': len(cyst.frame_data),
                    'Initial_Area_um2': round(areas_um2[0], 2) if areas_um2 else 0,
                    'Final_Area_um2': round(areas_um2[-1], 2) if areas_um2 else 0,
                    'Max_Area_um2': round(max(areas_um2), 2) if areas_um2 else 0,
                    'Mean_Area_um2': round(np.mean(areas_um2), 2) if areas_um2 else 0,
                    'Mean_Circularity': round(np.mean(circularities), 4) if circularities else 0,
                    'Growth_Rate_um2_per_day': round(growth_rate * experiment.time_lapse_days / max(1, experiment.total_frames - 1), 4)
                }

                summary_rows.append(row)

        # Sort by organoid and cyst ID
        summary_rows.sort(key=lambda x: (x['Organoid_ID'], x['Cyst_ID']))

        # Export to CSV
        if summary_rows:
            try:
                df = pd.DataFrame(summary_rows)
                df.to_csv(output_path, index=False, encoding=self.encoding, float_format='%.4f')

                print(f"✅ Summary exported: {len(summary_rows)} cysts")
                return output_path

            except ImportError:
                return self._export_csv_fallback(summary_rows, output_path)
        else:
            self._create_empty_csv(output_path, summary_headers=True)
            return output_path

    def export_organoid_summary(
        self,
        experiment: ExperimentData,
        output_path: str
    ) -> str:
        """
        Export organoid-level summary table
        """
        print(f"🔴 Exporting organoid summary to: {output_path}")

        organoid_rows = []

        for organoid_id, organoid in experiment.organoids.items():
            # Calculate organoid-level metrics
            total_cysts = len(organoid.cysts)

            if total_cysts > 0:
                # Aggregate area across all cysts per frame
                organoid_areas_per_frame = {}

                for cyst in organoid.cysts.values():
                    for frame_idx, frame_data in cyst.frame_data.items():
                        area_um2 = frame_data.get_area_um2(experiment.conversion_factor_um_per_pixel)

                        if frame_idx not in organoid_areas_per_frame:
                            organoid_areas_per_frame[frame_idx] = 0
                        organoid_areas_per_frame[frame_idx] += area_um2

                # Calculate growth metrics
                if organoid_areas_per_frame:
                    sorted_frames = sorted(organoid_areas_per_frame.keys())
                    initial_area = organoid_areas_per_frame[sorted_frames[0]]
                    final_area = organoid_areas_per_frame[sorted_frames[-1]]
                    max_area = max(organoid_areas_per_frame.values())
                    mean_area = np.mean(list(organoid_areas_per_frame.values()))

                    # Growth rate calculation
                    if len(sorted_frames) > 1:
                        time_span = experiment.get_time_at_frame(sorted_frames[-1]) - experiment.get_time_at_frame(sorted_frames[0])
                        growth_rate = (final_area - initial_area) / time_span if time_span > 0 else 0
                    else:
                        growth_rate = 0
                else:
                    initial_area = final_area = max_area = mean_area = growth_rate = 0
            else:
                initial_area = final_area = max_area = mean_area = growth_rate = 0

            row = {
                'Organoid_ID': organoid_id,
                'Marker_X': round(organoid.marker_point[0], 1),
                'Marker_Y': round(organoid.marker_point[1], 1),
                'Total_Cysts': total_cysts,
                'Initial_Total_Area_um2': round(initial_area, 2),
                'Final_Total_Area_um2': round(final_area, 2),
                'Max_Total_Area_um2': round(max_area, 2),
                'Mean_Total_Area_um2': round(mean_area, 2),
                'Growth_Rate_um2_per_day': round(growth_rate, 4)
            }

            organoid_rows.append(row)

        # Sort by organoid ID
        organoid_rows.sort(key=lambda x: x['Organoid_ID'])

        # Export to CSV
        if organoid_rows:
            try:
                df = pd.DataFrame(organoid_rows)
                df.to_csv(output_path, index=False, encoding=self.encoding, float_format='%.4f')

                print(f"✅ Organoid summary exported: {len(organoid_rows)} organoids")
                return output_path

            except ImportError:
                return self._export_csv_fallback(organoid_rows, output_path)
        else:
            self._create_empty_csv(output_path, organoid_headers=True)
            return output_path

    def _export_csv_fallback(self, data_rows: List[Dict], output_path: str) -> str:
        """
        Fallback CSV export using standard library
        """
        try:
            with open(output_path, 'w', newline='', encoding=self.encoding) as f:
                if data_rows:
                    writer = csv.DictWriter(f, fieldnames=data_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(data_rows)

            print(f"✅ CSV exported (fallback): {len(data_rows)} rows")
            return output_path

        except Exception as e:
            print(f"❌ CSV export failed: {e}")
            return None

    def _create_empty_csv(self, output_path: str, summary_headers: bool = False, organoid_headers: bool = False):
        """
        Create empty CSV file with appropriate headers
        """
        try:
            with open(output_path, 'w', newline='', encoding=self.encoding) as f:
                if organoid_headers:
                    headers = ['Organoid_ID', 'Marker_X', 'Marker_Y', 'Total_Cysts',
                              'Initial_Total_Area_um2', 'Final_Total_Area_um2',
                              'Max_Total_Area_um2', 'Mean_Total_Area_um2', 'Growth_Rate_um2_per_day']
                elif summary_headers:
                    headers = ['Organoid_ID', 'Cyst_ID', 'First_Frame', 'Last_Frame',
                              'Frames_Tracked', 'Initial_Area_um2', 'Final_Area_um2',
                              'Max_Area_um2', 'Mean_Area_um2', 'Mean_Circularity', 'Growth_Rate_um2_per_day']
                else:
                    headers = ['Organoid_ID', 'Cyst_ID', 'Frame', 'Time_Days',
                              'Area_um2', 'Circularity', 'Centroid_X', 'Centroid_Y', 'Radius_um']

                writer = csv.writer(f)
                writer.writerow(headers)

            print(f"📄 Empty CSV created with headers: {output_path}")

        except Exception as e:
            print(f"❌ Failed to create empty CSV: {e}")

    def _sanitize_for_csv(self, text: str) -> str:
        """
        Replace special characters that might cause CSV issues
        """
        replacements = {
            'μ': 'u',
            '±': '+/-',
            '°': 'deg',
            '²': '2',
            '³': '3',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma'
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text