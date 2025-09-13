"""
New data structure for organoid-cyst tracking analysis

This module provides the enhanced data model for tracking individual organoids
and their associated cysts across video frames.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import json
from pathlib import Path


@dataclass
class CystFrameData:
    """Data for a single cyst at a specific frame"""
    frame_index: int
    area_pixels: float
    circularity: float
    centroid: Tuple[float, float]  # (x, y)
    mask: Optional[np.ndarray] = None
    radius_pixels: Optional[float] = None
    perimeter_pixels: Optional[float] = None

    def __post_init__(self):
        """Calculate radius if not provided"""
        if self.radius_pixels is None and self.area_pixels > 0:
            self.radius_pixels = np.sqrt(self.area_pixels / np.pi)

    def get_area_um2(self, conversion_factor: float) -> float:
        """Convert area to square micrometers"""
        return self.area_pixels * conversion_factor * conversion_factor

    def get_radius_um(self, conversion_factor: float) -> float:
        """Convert radius to micrometers"""
        return self.radius_pixels * conversion_factor if self.radius_pixels else 0.0


@dataclass
class CystTrajectory:
    """Complete trajectory of a single cyst across frames"""
    cyst_id: int
    organoid_id: int
    frame_data: Dict[int, CystFrameData] = field(default_factory=dict)  # frame_index -> CystFrameData
    first_appearance_frame: Optional[int] = None
    last_appearance_frame: Optional[int] = None

    def add_frame_data(self, frame_data: CystFrameData):
        """Add data for a specific frame"""
        self.frame_data[frame_data.frame_index] = frame_data

        # Update appearance tracking
        if self.first_appearance_frame is None or frame_data.frame_index < self.first_appearance_frame:
            self.first_appearance_frame = frame_data.frame_index
        if self.last_appearance_frame is None or frame_data.frame_index > self.last_appearance_frame:
            self.last_appearance_frame = frame_data.frame_index

    def get_area_at_frame(self, frame_index: int, conversion_factor: float = 1.0) -> Optional[float]:
        """Get area at specific frame (in μm² if conversion_factor provided)"""
        if frame_index in self.frame_data:
            return self.frame_data[frame_index].get_area_um2(conversion_factor)
        return None

    def get_circularity_at_frame(self, frame_index: int) -> Optional[float]:
        """Get circularity at specific frame"""
        if frame_index in self.frame_data:
            return self.frame_data[frame_index].circularity
        return None

    def exists_at_frame(self, frame_index: int) -> bool:
        """Check if cyst exists at given frame"""
        return frame_index in self.frame_data

    def get_trajectory_length(self) -> int:
        """Get number of frames this cyst appears in"""
        return len(self.frame_data)

    def get_mean_area_growth_rate(self, conversion_factor: float = 1.0) -> float:
        """Calculate mean area growth rate (μm²/frame)"""
        if len(self.frame_data) < 2:
            return 0.0

        frames = sorted(self.frame_data.keys())
        total_growth = 0.0
        growth_periods = 0

        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            prev_area = self.get_area_at_frame(prev_frame, conversion_factor)
            curr_area = self.get_area_at_frame(curr_frame, conversion_factor)

            if prev_area is not None and curr_area is not None:
                growth = (curr_area - prev_area) / (curr_frame - prev_frame)
                total_growth += growth
                growth_periods += 1

        return total_growth / growth_periods if growth_periods > 0 else 0.0


@dataclass
class OrganoidData:
    """Data for a single organoid and all its cysts"""
    organoid_id: int
    marker_point: Tuple[float, float]  # Initial click point for identification
    cysts: Dict[int, CystTrajectory] = field(default_factory=dict)  # cyst_id -> CystTrajectory
    next_cyst_id: int = 1

    def add_cyst(self, cyst_trajectory: CystTrajectory):
        """Add a cyst trajectory to this organoid"""
        self.cysts[cyst_trajectory.cyst_id] = cyst_trajectory
        # Update next_cyst_id to avoid conflicts
        if cyst_trajectory.cyst_id >= self.next_cyst_id:
            self.next_cyst_id = cyst_trajectory.cyst_id + 1

    def get_new_cyst_id(self) -> int:
        """Get the next available cyst ID for this organoid"""
        cyst_id = self.next_cyst_id
        self.next_cyst_id += 1
        return cyst_id

    def get_cyst_count_at_frame(self, frame_index: int) -> int:
        """Get number of cysts at specific frame"""
        return sum(1 for cyst in self.cysts.values() if cyst.exists_at_frame(frame_index))

    def get_total_area_at_frame(self, frame_index: int, conversion_factor: float = 1.0) -> float:
        """Get total cyst area for this organoid at specific frame"""
        total_area = 0.0
        for cyst in self.cysts.values():
            area = cyst.get_area_at_frame(frame_index, conversion_factor)
            if area is not None:
                total_area += area
        return total_area

    def has_cysts_at_frame(self, frame_index: int) -> bool:
        """Check if organoid has any cysts at given frame"""
        return any(cyst.exists_at_frame(frame_index) for cyst in self.cysts.values())

    def get_first_cyst_appearance_frame(self) -> Optional[int]:
        """Get the frame where the first cyst appeared"""
        if not self.cysts:
            return None
        return min(cyst.first_appearance_frame for cyst in self.cysts.values()
                  if cyst.first_appearance_frame is not None)

    def get_mean_growth_rate(self, conversion_factor: float = 1.0) -> float:
        """Calculate mean growth rate across all cysts"""
        if not self.cysts:
            return 0.0

        growth_rates = [cyst.get_mean_area_growth_rate(conversion_factor)
                       for cyst in self.cysts.values()]
        return np.mean(growth_rates) if growth_rates else 0.0


@dataclass
class ExperimentData:
    """Complete experiment data with all organoids and their cysts"""
    total_frames: int
    time_lapse_days: float
    conversion_factor_um_per_pixel: float
    organoids: Dict[int, OrganoidData] = field(default_factory=dict)  # organoid_id -> OrganoidData
    next_organoid_id: int = 1
    frame_timestamps: List[float] = field(default_factory=list)  # Time for each frame

    def __post_init__(self):
        """Initialize frame timestamps if not provided"""
        if not self.frame_timestamps and self.total_frames > 0:
            # Evenly distribute time across frames
            time_per_frame = self.time_lapse_days / max(1, self.total_frames - 1)
            self.frame_timestamps = [i * time_per_frame for i in range(self.total_frames)]

    def add_organoid(self, organoid_data: OrganoidData):
        """Add an organoid to the experiment"""
        self.organoids[organoid_data.organoid_id] = organoid_data
        # Update next_organoid_id to avoid conflicts
        if organoid_data.organoid_id >= self.next_organoid_id:
            self.next_organoid_id = organoid_data.organoid_id + 1

    def get_new_organoid_id(self) -> int:
        """Get the next available organoid ID"""
        organoid_id = self.next_organoid_id
        self.next_organoid_id += 1
        return organoid_id

    def get_total_organoid_count(self) -> int:
        """Get total number of organoids"""
        return len(self.organoids)

    def get_total_cyst_count_at_frame(self, frame_index: int) -> int:
        """Get total number of cysts across all organoids at specific frame"""
        return sum(organoid.get_cyst_count_at_frame(frame_index)
                  for organoid in self.organoids.values())

    def get_organoids_with_cysts_at_frame(self, frame_index: int) -> int:
        """Get number of organoids that have at least one cyst at specific frame"""
        return sum(1 for organoid in self.organoids.values()
                  if organoid.has_cysts_at_frame(frame_index))

    def get_percentage_organoids_with_cysts_at_frame(self, frame_index: int) -> float:
        """Get percentage of organoids with cysts at specific frame"""
        total_organoids = self.get_total_organoid_count()
        if total_organoids == 0:
            return 0.0
        organoids_with_cysts = self.get_organoids_with_cysts_at_frame(frame_index)
        return (organoids_with_cysts / total_organoids) * 100.0

    def get_cyst_to_organoid_ratio_at_frame(self, frame_index: int) -> float:
        """Get ratio of cysts to organoids at specific frame"""
        total_organoids = self.get_total_organoid_count()
        if total_organoids == 0:
            return 0.0
        total_cysts = self.get_total_cyst_count_at_frame(frame_index)
        return total_cysts / total_organoids

    def get_all_cysts(self) -> List[CystTrajectory]:
        """Get all cyst trajectories from all organoids"""
        all_cysts = []
        for organoid in self.organoids.values():
            all_cysts.extend(organoid.cysts.values())
        return all_cysts

    def get_time_at_frame(self, frame_index: int) -> float:
        """Get time (in days) at specific frame"""
        if 0 <= frame_index < len(self.frame_timestamps):
            return self.frame_timestamps[frame_index]
        return 0.0

    def sort_organoids_by_growth_rate(self) -> List[Tuple[int, float]]:
        """Sort organoids by their growth rate (returns list of (organoid_id, growth_rate))"""
        organoid_growth_rates = []
        for organoid_id, organoid in self.organoids.items():
            growth_rate = organoid.get_mean_growth_rate(self.conversion_factor_um_per_pixel)
            organoid_growth_rates.append((organoid_id, growth_rate))

        # Sort by growth rate (descending - fastest first)
        return sorted(organoid_growth_rates, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        # Note: This is a simplified version - full implementation would handle numpy arrays
        return {
            'total_frames': self.total_frames,
            'time_lapse_days': self.time_lapse_days,
            'conversion_factor_um_per_pixel': self.conversion_factor_um_per_pixel,
            'total_organoids': self.get_total_organoid_count(),
            'frame_timestamps': self.frame_timestamps,
            'organoid_count': len(self.organoids),
            'total_cyst_count': len(self.get_all_cysts())
        }

    def save_to_json(self, filepath: str):
        """Save experiment metadata to JSON (not including full trajectory data)"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Helper functions for data extraction
def extract_experiment_data_from_tracking(tracking_results: Dict[str, Any],
                                        time_lapse_days: float,
                                        conversion_factor: float) -> ExperimentData:
    """
    Extract experiment data from SAM2 tracking results using the new organoid-cyst model

    Args:
        tracking_results: Results from SAM2 tracking
        time_lapse_days: Duration of experiment in days
        conversion_factor: μm per pixel conversion factor

    Returns:
        ExperimentData: Structured data ready for analysis
    """
    # This will be implemented when we integrate with the GUI
    # For now, return empty structure
    return ExperimentData(
        total_frames=0,
        time_lapse_days=time_lapse_days,
        conversion_factor_um_per_pixel=conversion_factor
    )