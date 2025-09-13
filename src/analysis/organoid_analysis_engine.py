"""
Organoid-Cyst Analysis Engine

This module handles the extraction of frame-by-frame data from SAM2 tracking results
and converts it into the structured organoid-cyst data format for analysis.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json

from .organoid_cyst_data import (
    ExperimentData, OrganoidData, CystTrajectory, CystFrameData
)


class OrganoidAnalysisEngine:
    """
    Main engine for extracting and analyzing organoid-cyst data from tracking results
    """

    def __init__(self, conversion_factor_um_per_pixel: float = 1.0):
        self.conversion_factor = conversion_factor_um_per_pixel
        self.debug_mode = False

    def extract_experiment_data_from_tracking(
        self,
        tracking_results: Dict[str, Any],
        organoid_data: Dict[int, Dict],  # From GUI: {organoid_id: {'point': (x,y), 'cysts': [...]}}
        time_lapse_days: float,
        total_frames: int
    ) -> ExperimentData:
        """
        Extract complete experiment data from SAM2 tracking results

        Args:
            tracking_results: Results from SAM2 video tracking
            organoid_data: Organoid-cyst relationships from GUI workflow
            time_lapse_days: Total experiment duration
            total_frames: Number of video frames

        Returns:
            ExperimentData: Complete structured experiment data
        """
        print(f"🔬 Extracting experiment data from {total_frames} frames...")

        # Create experiment container
        experiment = ExperimentData(
            total_frames=total_frames,
            time_lapse_days=time_lapse_days,
            conversion_factor_um_per_pixel=self.conversion_factor
        )

        # Process each organoid and its cysts
        for organoid_id, organoid_info in organoid_data.items():
            print(f"📍 Processing organoid {organoid_id} with {len(organoid_info['cysts'])} cysts...")

            # Create organoid data structure
            organoid = OrganoidData(
                organoid_id=organoid_id,
                marker_point=organoid_info['point']
            )

            # Process each cyst for this organoid
            for cyst_info in organoid_info['cysts']:
                cyst_id = cyst_info['cyst_id']
                print(f"  🔵 Processing cyst {cyst_id}...")

                # Extract cyst trajectory from tracking results
                cyst_trajectory = self._extract_cyst_trajectory(
                    cyst_id, organoid_id, tracking_results, total_frames
                )

                if cyst_trajectory and len(cyst_trajectory.frame_data) > 0:
                    organoid.add_cyst(cyst_trajectory)
                    print(f"    ✅ Added trajectory with {len(cyst_trajectory.frame_data)} frames")
                else:
                    print(f"    ⚠️ No valid trajectory data for cyst {cyst_id}")

            # Add organoid to experiment
            experiment.add_organoid(organoid)

        print(f"✅ Experiment extraction complete: {len(experiment.organoids)} organoids, {len(experiment.get_all_cysts())} cysts")
        return experiment

    def _extract_cyst_trajectory(
        self,
        cyst_id: int,
        organoid_id: int,
        tracking_results: Dict[str, Any],
        total_frames: int
    ) -> Optional[CystTrajectory]:
        """
        Extract trajectory data for a single cyst from tracking results
        """
        # Create cyst trajectory
        trajectory = CystTrajectory(
            cyst_id=cyst_id,
            organoid_id=organoid_id
        )

        # Extract mask data for this cyst across all frames
        try:
            # Handle different tracking result formats
            if 'video_segments' in tracking_results:
                masks_data = tracking_results['video_segments']
            elif 'masks' in tracking_results:
                masks_data = tracking_results['masks']
            elif isinstance(tracking_results, dict) and all(isinstance(k, int) for k in tracking_results.keys()):
                # Direct SAM2 format: {frame_idx: {obj_id: mask}}
                masks_data = tracking_results
                if self.debug_mode:
                    print(f"✅ Using direct SAM2 format for cyst {cyst_id}")
            else:
                print(f"⚠️ Unknown tracking results format for cyst {cyst_id}")
                if self.debug_mode:
                    print(f"   Expected 'video_segments' or 'masks' keys, or direct SAM2 format")
                    if isinstance(tracking_results, dict):
                        print(f"   Available keys: {list(tracking_results.keys())}")
                return None

            # Process each frame
            for frame_idx in range(total_frames):
                try:
                    # Get mask for this cyst at this frame
                    mask = self._get_mask_for_cyst_frame(masks_data, cyst_id, frame_idx)

                    if mask is not None and np.any(mask):
                        # Calculate cyst metrics from mask
                        frame_data = self._calculate_cyst_metrics_from_mask(
                            mask, frame_idx
                        )

                        if frame_data:
                            trajectory.add_frame_data(frame_data)

                except Exception as e:
                    if self.debug_mode:
                        print(f"    Warning: Error processing frame {frame_idx} for cyst {cyst_id}: {e}")
                    continue

            return trajectory if len(trajectory.frame_data) > 0 else None

        except Exception as e:
            print(f"❌ Error extracting trajectory for cyst {cyst_id}: {e}")
            return None

    def _get_mask_for_cyst_frame(
        self,
        masks_data: Any,
        cyst_id: int,
        frame_idx: int
    ) -> Optional[np.ndarray]:
        """
        Extract mask for specific cyst at specific frame from tracking results
        """
        try:
            mask = None

            # Handle different mask data formats from SAM2
            if isinstance(masks_data, dict):
                # Format: {frame_idx: {obj_id: mask}}
                if frame_idx in masks_data and cyst_id in masks_data[frame_idx]:
                    mask = masks_data[frame_idx][cyst_id]

                # Alternative format: {obj_id: {frame_idx: mask}}
                elif cyst_id in masks_data and frame_idx in masks_data[cyst_id]:
                    mask = masks_data[cyst_id][frame_idx]

            elif isinstance(masks_data, list) and len(masks_data) > frame_idx:
                # Format: [frame_data, ...] where frame_data has object masks
                frame_data = masks_data[frame_idx]
                if isinstance(frame_data, dict) and cyst_id in frame_data:
                    mask = frame_data[cyst_id]

            # Convert PyTorch tensor to NumPy array if needed
            if mask is not None:
                # Handle PyTorch tensors from SAM2
                if hasattr(mask, 'cpu') and hasattr(mask, 'numpy'):
                    # It's a PyTorch tensor
                    mask = mask.cpu().numpy()

                    # SAM2 masks are usually (1, H, W) - squeeze to (H, W)
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)

                    # Convert to binary mask (SAM2 returns logits/probabilities)
                    mask = (mask > 0.5).astype(np.uint8)

                elif not isinstance(mask, np.ndarray):
                    # Try to convert other array-like objects to numpy
                    mask = np.array(mask)

                return mask

            return None

        except Exception as e:
            if self.debug_mode:
                print(f"Error getting mask for cyst {cyst_id}, frame {frame_idx}: {e}")
            return None

    def _calculate_cyst_metrics_from_mask(
        self,
        mask: np.ndarray,
        frame_idx: int
    ) -> Optional[CystFrameData]:
        """
        Calculate cyst metrics (area, circularity, centroid) from binary mask
        """
        try:
            # Validate mask dimensions - must be 2D
            if mask.ndim != 2:
                if self.debug_mode:
                    print(f"Invalid mask at frame {frame_idx}: Expected 2D array, got {mask.ndim}D")
                return None

            # Validate mask size - must have reasonable dimensions
            if mask.shape[0] < 1 or mask.shape[1] < 1:
                if self.debug_mode:
                    print(f"Invalid mask at frame {frame_idx}: Empty dimensions {mask.shape}")
                return None

            # Validate mask size - reject degenerate cases
            if mask.shape[0] == 1 and mask.shape[1] == 1:
                if self.debug_mode:
                    print(f"Invalid mask at frame {frame_idx}: Single pixel mask rejected")
                return None

            # Reject linear masks (lines that are essentially 1-dimensional)
            if mask.shape[0] == 1 or mask.shape[1] == 1:
                if self.debug_mode:
                    print(f"Invalid mask at frame {frame_idx}: Linear mask rejected {mask.shape}")
                return None

            # Reject very thin masks (aspect ratio too extreme)
            aspect_ratio = max(mask.shape[0], mask.shape[1]) / min(mask.shape[0], mask.shape[1])
            if aspect_ratio > 10:  # More than 10:1 aspect ratio
                if self.debug_mode:
                    print(f"Invalid mask at frame {frame_idx}: Extreme aspect ratio {aspect_ratio:.1f}")
                return None

            # Ensure mask is binary
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8)

            # Calculate moments for basic metrics
            moments = cv2.moments(mask)

            if moments['m00'] <= 0:  # No area
                return None

            # Additional validation: reject masks that are too small or linear
            area_pixels = float(moments['m00'])

            # Reject very small areas (likely noise or artifacts)
            if area_pixels < 9:  # Less than 3x3 pixels minimum
                if self.debug_mode:
                    print(f"Mask at frame {frame_idx}: Area too small ({area_pixels} pixels)")
                return None

            # Calculate centroid
            centroid_x = moments['m10'] / moments['m00']
            centroid_y = moments['m01'] / moments['m00']
            centroid = (float(centroid_x), float(centroid_y))

            # Validate centroid is within mask bounds
            if (centroid_x < 0 or centroid_x >= mask.shape[1] or
                centroid_y < 0 or centroid_y >= mask.shape[0]):
                if self.debug_mode:
                    print(f"Invalid mask at frame {frame_idx}: Centroid outside bounds")
                return None

            # Calculate circularity using contour
            circularity = self._calculate_circularity(mask)

            # Validate circularity bounds
            if circularity < 0 or circularity > 1.0001:  # Allow small floating point error
                if self.debug_mode:
                    print(f"Invalid circularity at frame {frame_idx}: {circularity}")
                # Clamp to valid range
                circularity = max(0.0, min(1.0, circularity))

            # Create frame data
            frame_data = CystFrameData(
                frame_index=frame_idx,
                area_pixels=area_pixels,
                circularity=circularity,
                centroid=centroid,
                mask=mask.copy()
            )

            return frame_data

        except Exception as e:
            if self.debug_mode:
                print(f"Error calculating metrics from mask at frame {frame_idx}: {e}")
            return None

    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """
        Calculate circularity (4π × Area / Perimeter²) from binary mask
        """
        try:
            # Validate input mask
            if mask.ndim != 2:
                if self.debug_mode:
                    print(f"Circularity: Invalid mask dimensions: {mask.ndim}D")
                return 0.0

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                if self.debug_mode:
                    print("Circularity: No contours found")
                return 0.0

            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Validate contour has sufficient points
            if len(largest_contour) < 3:
                if self.debug_mode:
                    print(f"Circularity: Insufficient contour points: {len(largest_contour)}")
                return 0.0

            # Calculate area and perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            # Validate measurements
            if perimeter <= 0 or area <= 0:
                if self.debug_mode:
                    print(f"Circularity: Invalid measurements - area: {area}, perimeter: {perimeter}")
                return 0.0

            # Additional validation for degenerate shapes
            if area < 1.0:  # Very small area
                return 0.0

            if perimeter < 2.0:  # Very small perimeter
                return 0.0

            # Calculate circularity: 4π × Area / Perimeter²
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

            # Validate result
            if np.isnan(circularity) or np.isinf(circularity):
                if self.debug_mode:
                    print(f"Circularity: Invalid result: {circularity}")
                return 0.0

            # Clamp to [0, 1] (perfect circle = 1)
            return max(0.0, min(circularity, 1.0))

        except Exception as e:
            if self.debug_mode:
                print(f"Error calculating circularity: {e}")
            return 0.0

    def save_experiment_data(self, experiment: ExperimentData, output_path: str):
        """
        Save experiment data to JSON file for debugging and validation
        """
        try:
            # Create summary data (not including full masks)
            summary = {
                'experiment_info': experiment.to_dict(),
                'organoids': []
            }

            for organoid_id, organoid in experiment.organoids.items():
                organoid_info = {
                    'organoid_id': organoid_id,
                    'marker_point': organoid.marker_point,
                    'cyst_count': len(organoid.cysts),
                    'cysts': []
                }

                for cyst_id, cyst in organoid.cysts.items():
                    cyst_info = {
                        'cyst_id': cyst_id,
                        'frames_count': len(cyst.frame_data),
                        'first_frame': cyst.first_appearance_frame,
                        'last_frame': cyst.last_appearance_frame,
                        'mean_area_um2': np.mean([
                            frame_data.get_area_um2(experiment.conversion_factor_um_per_pixel)
                            for frame_data in cyst.frame_data.values()
                        ]) if cyst.frame_data else 0,
                        'mean_circularity': np.mean([
                            frame_data.circularity
                            for frame_data in cyst.frame_data.values()
                        ]) if cyst.frame_data else 0,
                    }
                    organoid_info['cysts'].append(cyst_info)

                summary['organoids'].append(organoid_info)

            # Save to file
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"💾 Experiment data summary saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error saving experiment data: {e}")


class OrganoidAnalysisValidator:
    """
    Validation and quality checks for organoid analysis data
    """

    @staticmethod
    def validate_experiment_data(experiment: ExperimentData) -> Dict[str, Any]:
        """
        Validate experiment data and return quality metrics
        """
        validation_results = {
            'total_organoids': len(experiment.organoids),
            'total_cysts': len(experiment.get_all_cysts()),
            'frames_analyzed': experiment.total_frames,
            'issues': [],
            'warnings': []
        }

        # Check for empty organoids
        empty_organoids = [
            org_id for org_id, org in experiment.organoids.items()
            if len(org.cysts) == 0
        ]
        if empty_organoids:
            validation_results['warnings'].append(f"Empty organoids: {empty_organoids}")

        # Check for cysts with insufficient data
        short_cysts = []
        for organoid in experiment.organoids.values():
            for cyst in organoid.cysts.values():
                if len(cyst.frame_data) < 3:  # Less than 3 frames
                    short_cysts.append(f"Organoid {cyst.organoid_id}, Cyst {cyst.cyst_id}")

        if short_cysts:
            validation_results['warnings'].append(f"Cysts with <3 frames: {short_cysts}")

        # Check time coverage
        all_cysts = experiment.get_all_cysts()
        if all_cysts:
            max_frames_tracked = max(len(cyst.frame_data) for cyst in all_cysts)
            coverage = max_frames_tracked / experiment.total_frames * 100
            validation_results['max_tracking_coverage_percent'] = coverage

            if coverage < 50:
                validation_results['warnings'].append(f"Low tracking coverage: {coverage:.1f}%")

        return validation_results