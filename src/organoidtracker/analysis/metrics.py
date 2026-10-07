"""
Kidney organoid cyst analysis metrics

This module provides an expandable framework for calculating various metrics
related to kidney organoid cyst formation and growth analysis.
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from scipy import spatial
from skimage import measure


@dataclass
class AnalysisParameters:
    """Parameters provided by user for analysis calculations"""
    total_organoids: int
    time_lapse_days: float
    conversion_factor_um_per_pixel: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_organoids': self.total_organoids,
            'time_lapse_days': self.time_lapse_days,
            'conversion_factor_um_per_pixel': self.conversion_factor_um_per_pixel
        }


@dataclass
class CystData:
    """Enhanced data structure for individual cyst measurements"""
    object_id: int
    frame_indices: List[int]
    centroids: List[Tuple[float, float]]  # (x, y) coordinates
    areas_pixels: List[float]  # Area in pixels for each frame
    radii_pixels: List[float]  # Equivalent radius in pixels
    perimeters_pixels: List[float] = None  # Perimeter for each frame
    circularities: List[float] = None  # Circularity (4π×Area/Perimeter²)
    aspect_ratios: List[float] = None  # Major/minor axis ratio
    masks: List[np.ndarray] = None  # Binary masks for spatial analysis

    def __post_init__(self):
        """Initialize empty lists if not provided"""
        if self.perimeters_pixels is None:
            self.perimeters_pixels = []
        if self.circularities is None:
            self.circularities = []
        if self.aspect_ratios is None:
            self.aspect_ratios = []
        if self.masks is None:
            self.masks = []

    def get_radii_um(self, conversion_factor: float) -> List[float]:
        """Convert radii from pixels to micrometers"""
        return [r * conversion_factor for r in self.radii_pixels]

    def get_areas_um2(self, conversion_factor: float) -> List[float]:
        """Convert areas from pixels to square micrometers"""
        return [a * conversion_factor * conversion_factor for a in self.areas_pixels]

    def get_perimeters_um(self, conversion_factor: float) -> List[float]:
        """Convert perimeters from pixels to micrometers"""
        return [p * conversion_factor for p in self.perimeters_pixels]

    def get_mean_circularity(self) -> float:
        """Get mean circularity across all frames"""
        return np.mean(self.circularities) if self.circularities else 0.0

    def get_mean_aspect_ratio(self) -> float:
        """Get mean aspect ratio across all frames"""
        return np.mean(self.aspect_ratios) if self.aspect_ratios else 1.0


class BaseMetric(ABC):
    """Base class for all metrics calculations"""

    def __init__(self, name: str, description: str, unit: str):
        self.name = name
        self.description = description
        self.unit = unit

    @abstractmethod
    def calculate(self, cyst_data: List[CystData], params: AnalysisParameters) -> Dict[str, Any]:
        """Calculate the metric and return results"""
        pass

    def get_info(self) -> Dict[str, str]:
        """Get metric information"""
        return {
            'name': self.name,
            'description': self.description,
            'unit': self.unit
        }


class CystFormationEfficiency(BaseMetric):
    """Calculate cyst formation efficiency percentage"""

    def __init__(self):
        super().__init__(
            name="Cyst Formation Efficiency",
            description="Percentage of organoids that developed at least one cyst",
            unit="%"
        )

    def calculate(self, cyst_data: List[CystData], params: AnalysisParameters) -> Dict[str, Any]:
        """
        Calculate: (Number of Organoids with ≥1 Cyst / Total Number of Organoids) × 100
        """
        num_organoids_with_cysts = len(cyst_data)  # Each CystData represents one cyst/organoid
        total_organoids = params.total_organoids

        if total_organoids == 0:
            efficiency = 0.0
        else:
            efficiency = (num_organoids_with_cysts / total_organoids) * 100

        return {
            'value': efficiency,
            'organoids_with_cysts': num_organoids_with_cysts,
            'total_organoids': total_organoids,
            'formula': f"({num_organoids_with_cysts} / {total_organoids}) × 100"
        }


class DeNovoCystFormationRate(BaseMetric):
    """Calculate rate of de novo cyst formation"""

    def __init__(self):
        super().__init__(
            name="De Novo Cyst Formation Rate",
            description="Rate of new cyst formation over time",
            unit="cysts/day"
        )

    def calculate(self, cyst_data: List[CystData], params: AnalysisParameters) -> Dict[str, Any]:
        """
        Calculate: (N(cysts t final) - N(cysts t initial)) / (t final - t initial)
        """
        if not cyst_data or params.time_lapse_days == 0:
            return {
                'value': 0.0,
                'initial_cysts': 0,
                'final_cysts': 0,
                'time_period_days': params.time_lapse_days,
                'formula': "0 / 0 (no data)"
            }

        # For simplicity, assume initial cysts = 0 and final = number of tracked cysts
        # In future, this could be enhanced to track cyst appearance over time
        initial_cysts = 0  # Assuming we start tracking from cyst formation
        final_cysts = len(cyst_data)

        rate = (final_cysts - initial_cysts) / params.time_lapse_days

        return {
            'value': rate,
            'initial_cysts': initial_cysts,
            'final_cysts': final_cysts,
            'time_period_days': params.time_lapse_days,
            'formula': f"({final_cysts} - {initial_cysts}) / {params.time_lapse_days}"
        }


class RadialExpansionVelocity(BaseMetric):
    """Calculate radial expansion velocity for each cyst"""

    def __init__(self):
        super().__init__(
            name="Radial Expansion Velocity",
            description="Rate of cyst radius expansion over time",
            unit="um/day"
        )

    def calculate(self, cyst_data: List[CystData], params: AnalysisParameters) -> Dict[str, Any]:
        """
        Calculate: (r(t2) - r(t1)) / (t2 - t1) for each cyst
        """
        velocities = []
        cyst_details = []

        for cyst in cyst_data:
            if len(cyst.radii_pixels) < 2:
                # Need at least 2 time points to calculate velocity
                velocities.append(0.0)
                cyst_details.append({
                    'object_id': cyst.object_id,
                    'velocity_um_per_day': 0.0,
                    'initial_radius_um': 0.0,
                    'final_radius_um': 0.0,
                    'note': 'Insufficient time points'
                })
                continue

            # Convert radii to micrometers
            radii_um = cyst.get_radii_um(params.conversion_factor_um_per_pixel)

            # Calculate velocity using first and last time points
            initial_radius = radii_um[0]
            final_radius = radii_um[-1]

            # Velocity = change in radius / time period
            velocity = (final_radius - initial_radius) / params.time_lapse_days
            velocities.append(velocity)

            cyst_details.append({
                'object_id': cyst.object_id,
                'velocity_um_per_day': velocity,
                'initial_radius_um': initial_radius,
                'final_radius_um': final_radius,
                'radius_change_um': final_radius - initial_radius,
                'time_period_days': params.time_lapse_days
            })

        # Calculate statistics
        if velocities:
            mean_velocity = np.mean(velocities)
            std_velocity = np.std(velocities)
            max_velocity = np.max(velocities)
            min_velocity = np.min(velocities)
        else:
            mean_velocity = std_velocity = max_velocity = min_velocity = 0.0

        return {
            'mean_value': mean_velocity,
            'std_value': std_velocity,
            'max_value': max_velocity,
            'min_value': min_velocity,
            'individual_velocities': velocities,
            'cyst_details': cyst_details,
            'num_cysts': len(cyst_data)
        }


class CysticIndex(BaseMetric):
    """Calculate the Cystic Index: fraction of total area occupied by cysts"""

    def __init__(self):
        super().__init__(
            name="Cystic Index",
            description="Fraction of total organoid area occupied by cysts",
            unit="fraction (0-1)"
        )

    def calculate(self, cyst_data: List[CystData], params: AnalysisParameters) -> Dict[str, Any]:
        """
        Calculate: Total cyst area / Total organoid area
        """
        if not cyst_data:
            return {
                'value': 0.0,
                'total_cyst_area_pixels': 0.0,
                'total_organoid_area_pixels': 0.0,
                'cystic_index_per_frame': [],
                'formula': "0 / 0 (no cysts detected)"
            }

        # Calculate total cyst area for each frame
        frame_indices = set()
        for cyst in cyst_data:
            frame_indices.update(cyst.frame_indices)

        cystic_indices_per_frame = []
        total_cyst_area = 0
        total_organoid_area = 0

        for frame_idx in sorted(frame_indices):
            frame_cyst_area = 0
            frame_organoid_area = 0

            for cyst in cyst_data:
                if frame_idx in cyst.frame_indices:
                    idx_in_cyst = cyst.frame_indices.index(frame_idx)
                    cyst_area = cyst.areas_pixels[idx_in_cyst]
                    frame_cyst_area += cyst_area

                    # For simplicity, assume each cyst contributes to organoid area
                    # In practice, this might need estimation of total organoid boundaries
                    frame_organoid_area += cyst_area * 2  # Rough estimate: cyst area * expansion factor

            if frame_organoid_area > 0:
                frame_ci = frame_cyst_area / frame_organoid_area
                cystic_indices_per_frame.append({
                    'frame': frame_idx,
                    'cystic_index': frame_ci,
                    'cyst_area': frame_cyst_area,
                    'organoid_area': frame_organoid_area
                })
                total_cyst_area += frame_cyst_area
                total_organoid_area += frame_organoid_area

        # Overall cystic index
        overall_ci = total_cyst_area / total_organoid_area if total_organoid_area > 0 else 0.0

        return {
            'value': overall_ci,
            'total_cyst_area_pixels': total_cyst_area,
            'total_organoid_area_pixels': total_organoid_area,
            'cystic_index_per_frame': cystic_indices_per_frame,
            'mean_cystic_index': np.mean([ci['cystic_index'] for ci in cystic_indices_per_frame]) if cystic_indices_per_frame else 0.0,
            'std_cystic_index': np.std([ci['cystic_index'] for ci in cystic_indices_per_frame]) if cystic_indices_per_frame else 0.0,
            'formula': f"{total_cyst_area:.1f} / {total_organoid_area:.1f}"
        }


class MorphologicalAnalysis(BaseMetric):
    """Analyze morphological properties of cysts (circularity, aspect ratio, etc.)"""

    def __init__(self):
        super().__init__(
            name="Morphological Analysis",
            description="Shape and morphological characteristics of cysts",
            unit="dimensionless"
        )

    def calculate(self, cyst_data: List[CystData], params: AnalysisParameters) -> Dict[str, Any]:
        """
        Calculate morphological metrics for all cysts
        """
        if not cyst_data:
            return {
                'mean_circularity': 0.0,
                'mean_aspect_ratio': 1.0,
                'morphospace_data': [],
                'cyst_morphology': []
            }

        all_circularities = []
        all_aspect_ratios = []
        morphospace_data = []
        cyst_morphology = []

        for cyst in cyst_data:
            if cyst.circularities and cyst.aspect_ratios:
                cyst_mean_circularity = np.mean(cyst.circularities)
                cyst_mean_aspect_ratio = np.mean(cyst.aspect_ratios)

                all_circularities.extend(cyst.circularities)
                all_aspect_ratios.extend(cyst.aspect_ratios)

                # Morphospace data for visualization
                for i, frame_idx in enumerate(cyst.frame_indices):
                    if i < len(cyst.areas_pixels) and i < len(cyst.circularities):
                        morphospace_data.append({
                            'object_id': cyst.object_id,
                            'frame': frame_idx,
                            'area_pixels': cyst.areas_pixels[i],
                            'circularity': cyst.circularities[i],
                            'aspect_ratio': cyst.aspect_ratios[i] if i < len(cyst.aspect_ratios) else 1.0,
                            'radius_pixels': cyst.radii_pixels[i]
                        })

                cyst_morphology.append({
                    'object_id': cyst.object_id,
                    'mean_circularity': cyst_mean_circularity,
                    'mean_aspect_ratio': cyst_mean_aspect_ratio,
                    'circularity_std': np.std(cyst.circularities),
                    'aspect_ratio_std': np.std(cyst.aspect_ratios),
                    'num_timepoints': len(cyst.frame_indices)
                })

        return {
            'mean_circularity': np.mean(all_circularities) if all_circularities else 0.0,
            'std_circularity': np.std(all_circularities) if all_circularities else 0.0,
            'mean_aspect_ratio': np.mean(all_aspect_ratios) if all_aspect_ratios else 1.0,
            'std_aspect_ratio': np.std(all_aspect_ratios) if all_aspect_ratios else 0.0,
            'morphospace_data': morphospace_data,
            'cyst_morphology': cyst_morphology,
            'num_measurements': len(all_circularities)
        }


class SpatialOrganization(BaseMetric):
    """Analyze spatial organization and clustering of cysts"""

    def __init__(self):
        super().__init__(
            name="Spatial Organization",
            description="Spatial distribution and clustering analysis of cysts",
            unit="various"
        )

    def calculate(self, cyst_data: List[CystData], params: AnalysisParameters) -> Dict[str, Any]:
        """
        Calculate spatial organization metrics including clustering and density
        """
        if len(cyst_data) < 2:
            return {
                'clustering_coefficient': 0.0,
                'nearest_neighbor_distances': [],
                'spatial_density_map': None,
                'ripleys_k': None,
                'note': 'Insufficient cysts for spatial analysis (need ≥2)'
            }

        # Use final frame positions for spatial analysis
        final_positions = []
        for cyst in cyst_data:
            if cyst.centroids:
                final_positions.append(cyst.centroids[-1])  # Last position

        if len(final_positions) < 2:
            return {
                'clustering_coefficient': 0.0,
                'nearest_neighbor_distances': [],
                'spatial_density_map': None,
                'ripleys_k': None,
                'note': 'Insufficient position data'
            }

        positions = np.array(final_positions)

        # Calculate nearest neighbor distances
        try:
            from scipy import spatial
            distances = spatial.distance_matrix(positions, positions)
            np.fill_diagonal(distances, np.inf)  # Remove self-distances
            nn_distances = np.min(distances, axis=1)
        except ImportError:
            # Fallback to manual distance calculation
            distances = np.zeros((len(positions), len(positions)))
            for i in range(len(positions)):
                for j in range(len(positions)):
                    if i != j:
                        dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                        distances[i, j] = dist
                    else:
                        distances[i, j] = np.inf
            nn_distances = np.min(distances, axis=1)

        # Simple clustering coefficient (fraction of pairs closer than median distance)
        median_distance = np.median(nn_distances)
        close_pairs = np.sum(distances < median_distance) / 2  # Divide by 2 to avoid double counting
        total_pairs = len(positions) * (len(positions) - 1) / 2
        clustering_coefficient = close_pairs / total_pairs if total_pairs > 0 else 0.0

        # Simple spatial density calculation
        if len(positions) > 2:
            # Create a basic spatial density map
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]

            # Create grid for density map
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            if x_max > x_min and y_max > y_min:
                grid_size = 20  # 20x20 grid
                x_bins = np.linspace(x_min, x_max, grid_size)
                y_bins = np.linspace(y_min, y_max, grid_size)

                density_map, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
                density_info = {
                    'grid_shape': density_map.shape,
                    'max_density': np.max(density_map),
                    'mean_density': np.mean(density_map),
                    'x_range': (x_min, x_max),
                    'y_range': (y_min, y_max)
                }
            else:
                density_info = None
        else:
            density_info = None

        return {
            'clustering_coefficient': clustering_coefficient,
            'nearest_neighbor_distances': nn_distances.tolist(),
            'mean_nn_distance': np.mean(nn_distances),
            'std_nn_distance': np.std(nn_distances),
            'spatial_density_map': density_info,
            'num_cysts_analyzed': len(final_positions),
            'spatial_extent': {
                'x_range': (np.min(positions[:, 0]), np.max(positions[:, 0])),
                'y_range': (np.min(positions[:, 1]), np.max(positions[:, 1]))
            }
        }


class MetricsCalculator:
    """Main calculator that orchestrates all metrics calculations"""

    def __init__(self):
        self.metrics = [
            CystFormationEfficiency(),
            DeNovoCystFormationRate(),
            RadialExpansionVelocity(),
            CysticIndex(),
            MorphologicalAnalysis(),
            SpatialOrganization()
        ]

    def add_metric(self, metric: BaseMetric):
        """Add a new metric to the calculator"""
        self.metrics.append(metric)

    def get_available_metrics(self) -> List[Dict[str, str]]:
        """Get list of available metrics"""
        return [metric.get_info() for metric in self.metrics]

    def extract_cyst_data_from_tracking(self, video_segments: Dict, frames: List) -> List[CystData]:
        """
        Extract cyst data from tracking results

        Args:
            video_segments: Tracking results from SAM2
            frames: List of video frames

        Returns:
            List of CystData objects
        """
        cyst_data = []

        # Get all unique object IDs
        all_object_ids = set()
        for frame_idx, objects in video_segments.items():
            all_object_ids.update(objects.keys())

        # Remove background object (ID 0) if present
        all_object_ids.discard(0)

        for obj_id in all_object_ids:
            frame_indices = []
            centroids = []
            areas_pixels = []
            radii_pixels = []
            perimeters_pixels = []
            circularities = []
            aspect_ratios = []
            masks = []

            for frame_idx in sorted(video_segments.keys()):
                if obj_id in video_segments[frame_idx]:
                    mask = video_segments[frame_idx][obj_id]

                    # Convert mask to binary format
                    if hasattr(mask, 'cpu'):
                        mask_np = mask.cpu().numpy().squeeze()
                    else:
                        mask_np = mask.squeeze() if hasattr(mask, 'squeeze') else mask

                    # Convert logits to binary mask
                    if mask_np.dtype in [np.float32, np.float64]:
                        mask_binary = (1 / (1 + np.exp(-mask_np)) > 0.5).astype(np.uint8)
                    else:
                        mask_binary = (mask_np > 0.5).astype(np.uint8)

                    # Calculate geometric properties
                    moments = cv2.moments(mask_binary)
                    if moments['m00'] > 0:  # Avoid division by zero
                        # Centroid
                        cx = moments['m10'] / moments['m00']
                        cy = moments['m01'] / moments['m00']
                        centroids.append((cx, cy))

                        # Area
                        area = moments['m00']
                        areas_pixels.append(area)

                        # Equivalent radius (assuming circular cyst)
                        radius = np.sqrt(area / np.pi)
                        radii_pixels.append(radius)

                        # Perimeter calculation
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Use the largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            perimeter = cv2.arcLength(largest_contour, True)
                            perimeters_pixels.append(perimeter)

                            # Circularity: 4π×Area/Perimeter²
                            if perimeter > 0:
                                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                                circularities.append(min(circularity, 1.0))  # Cap at 1.0 for perfect circle
                            else:
                                circularities.append(0.0)

                            # Aspect ratio: ratio of major to minor axis
                            if len(largest_contour) >= 5:  # Need at least 5 points to fit ellipse
                                try:
                                    ellipse = cv2.fitEllipse(largest_contour)
                                    major_axis = max(ellipse[1])
                                    minor_axis = min(ellipse[1])
                                    if minor_axis > 0:
                                        aspect_ratio = major_axis / minor_axis
                                    else:
                                        aspect_ratio = 1.0
                                    aspect_ratios.append(aspect_ratio)
                                except:
                                    aspect_ratios.append(1.0)  # Default to circular
                            else:
                                aspect_ratios.append(1.0)
                        else:
                            perimeters_pixels.append(0.0)
                            circularities.append(0.0)
                            aspect_ratios.append(1.0)

                        # Store mask for spatial analysis
                        masks.append(mask_binary.copy())

                        frame_indices.append(frame_idx)

            if frame_indices:  # Only add if we have data
                cyst_data.append(CystData(
                    object_id=obj_id,
                    frame_indices=frame_indices,
                    centroids=centroids,
                    areas_pixels=areas_pixels,
                    radii_pixels=radii_pixels,
                    perimeters_pixels=perimeters_pixels,
                    circularities=circularities,
                    aspect_ratios=aspect_ratios,
                    masks=masks
                ))

        return cyst_data

    def calculate_all_metrics(self, video_segments: Dict, frames: List,
                            params: AnalysisParameters) -> Dict[str, Any]:
        """
        Calculate all metrics for the given data

        Args:
            video_segments: Tracking results
            frames: Video frames
            params: Analysis parameters

        Returns:
            Dictionary containing all calculated metrics
        """
        # Extract cyst data from tracking results
        cyst_data = self.extract_cyst_data_from_tracking(video_segments, frames)

        # Calculate all metrics
        results = {
            'parameters': params.to_dict(),
            'cyst_data_summary': {
                'num_cysts_tracked': len(cyst_data),
                'cyst_ids': [cyst.object_id for cyst in cyst_data]
            },
            'metrics': {}
        }

        for metric in self.metrics:
            try:
                metric_result = metric.calculate(cyst_data, params)
                results['metrics'][metric.name] = {
                    'info': metric.get_info(),
                    'results': metric_result
                }
            except Exception as e:
                results['metrics'][metric.name] = {
                    'info': metric.get_info(),
                    'error': str(e)
                }

        return results

    def save_analysis_data(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file for later use"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)