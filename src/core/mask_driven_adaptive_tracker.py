#!/usr/bin/env python3
"""
Mask-Driven Adaptive Bounding Box Tracker
Uses mask centroid and size history to predict object position and scale changes
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import time
from pathlib import Path


@dataclass
class MaskFeatures:
    """Extracted features from a segmentation mask"""
    centroid: Tuple[float, float]
    area: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    aspect_ratio: float
    compactness: float  # 4π*area/perimeter²
    confidence: float
    frame_idx: int


@dataclass
class PredictionResult:
    """Result of position and size prediction"""
    predicted_centroid: Tuple[float, float]
    predicted_scale_factor: float
    position_confidence: float
    size_confidence: float
    strategy_used: str


@dataclass
class AdaptationDecision:
    """Decision for bbox adaptation"""
    should_update: bool
    new_bbox: Tuple[int, int, int, int]
    strategy: str
    confidence: float
    reasoning: str


class MaskDrivenAdaptiveTracker:
    """
    Adaptive bbox tracker that uses mask analysis for predictions
    
    Key Features:
    - Centroid-based position prediction using velocity tracking
    - Size-based scale prediction using trend analysis
    - Confidence-weighted decision making
    - No assumptions about object growth patterns
    """
    
    def __init__(self, initial_bbox: Tuple[int, int, int, int], obj_id: int, config: Optional[Dict] = None):
        """
        Initialize mask-driven adaptive tracker
        
        Args:
            initial_bbox: Initial bounding box (x1, y1, x2, y2)
            obj_id: Object identifier
            config: Configuration parameters
        """
        self.obj_id = obj_id
        self.initial_bbox = initial_bbox
        self.current_bbox = initial_bbox
        
        # Configuration with mathematically sound defaults
        default_config = {
            # History management
            'max_history_length': 10,           # Keep last N frames of data
            'min_history_for_prediction': 3,   # Need at least 3 points for trend
            
            # Centroid prediction
            'velocity_smoothing_factor': 0.7,  # Exponential smoothing (0=no smoothing, 1=max smoothing)
            'max_position_extrapolation': 50,  # Max pixels to extrapolate position
            
            # Size prediction  
            'size_smoothing_factor': 0.8,      # Size changes more conservatively
            'min_scale_factor': 0.5,           # Don't shrink below 50%
            'max_scale_factor': 2.0,           # Don't grow above 200%
            'size_change_threshold': 0.1,      # 10% change threshold for significant size change
            
            # Confidence thresholds
            'high_confidence_threshold': 0.8,  # Trust current mask completely
            'medium_confidence_threshold': 0.5, # Blend current + prediction
            'low_confidence_threshold': 0.2,   # Use prediction only
            
            # Fallback behavior
            'conservative_expansion_factor': 1.2, # 20% expansion when no reliable data
            'min_bbox_size': 20,               # Minimum bbox dimension
            'max_bbox_size': 1000,             # Maximum bbox dimension
            
            # Edge case tolerance (for problematic objects)
            'max_consecutive_failures': 10,    # Max consecutive low-quality masks before giving up
            'failure_recovery_mode': 'gradual_shrink', # How to handle persistent failures
            'allow_object_loss': True,         # Allow objects to be marked as lost
            'lost_object_bbox_factor': 0.8,    # Shrink lost object bbox to 80% of last good
            
            # Debug and logging
            'enable_logging': True,
            'enable_detailed_math_logging': False,
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Historical data storage
        self.mask_features_history: List[MaskFeatures] = []
        self.prediction_history: List[PredictionResult] = []
        self.adaptation_decisions: List[AdaptationDecision] = []
        
        # Derived data caches (computed from history)
        self._cached_velocity = None
        self._cached_size_trend = None
        self._cache_valid_frame = -1
        
        # Performance tracking
        self.total_updates = 0
        self.successful_predictions = 0
        self.fallback_uses = 0
        
        # Edge case handling
        self.consecutive_failures = 0
        self.is_object_lost = False
        self.last_good_bbox = initial_bbox
        self.last_good_frame = 0
        
        if self.config['enable_logging']:
            print(f"🎯 MaskDrivenAdaptiveTracker initialized for object {obj_id}")
            print(f"   Initial bbox: {initial_bbox}")
    
    def update_bbox(self, mask: np.ndarray, frame_idx: int, confidence: float = None) -> Tuple[int, int, int, int]:
        """
        Update bounding box based on mask analysis and historical data
        
        Args:
            mask: Current segmentation mask (2D binary array)
            frame_idx: Current frame index
            confidence: Optional confidence score for current mask
            
        Returns:
            Updated bounding box (x1, y1, x2, y2)
        """
        self.total_updates += 1
        
        # 1. Extract features from current mask
        mask_features = self._extract_mask_features(mask, frame_idx, confidence)
        
        # 2. Update historical data
        self._update_history(mask_features)
        
        # 3. Make adaptation decision
        decision = self._make_adaptation_decision(mask_features)
        
        # 4. Update current bbox
        previous_bbox = self.current_bbox
        self.current_bbox = decision.new_bbox
        
        # 5. Record decision
        self.adaptation_decisions.append(decision)
        
        # 6. Logging
        if self.config['enable_logging']:
            self._log_adaptation(previous_bbox, decision, mask_features)
        
        return self.current_bbox
    
    def _extract_mask_features(self, mask: np.ndarray, frame_idx: int, confidence: float) -> MaskFeatures:
        """
        Extract mathematical features from segmentation mask
        
        Args:
            mask: Binary segmentation mask
            frame_idx: Current frame index  
            confidence: Mask confidence score
            
        Returns:
            MaskFeatures object with extracted features
        """
        # Ensure mask is 2D binary
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        
        # Handle empty mask
        if np.sum(mask) == 0:
            # Use current bbox center as fallback
            cx = (self.current_bbox[0] + self.current_bbox[2]) / 2
            cy = (self.current_bbox[1] + self.current_bbox[3]) / 2
            return MaskFeatures(
                centroid=(cx, cy),
                area=0.0,
                bbox=self.current_bbox,
                aspect_ratio=1.0,
                compactness=0.0,
                confidence=confidence or 0.0,
                frame_idx=frame_idx
            )
        
        # Calculate centroid using moments (more accurate than mean)
        moments = cv2.moments(mask)
        if moments['m00'] > 0:  # Avoid division by zero
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            # Fallback to coordinate mean
            coords = np.where(mask > 0)
            cx = np.mean(coords[1])
            cy = np.mean(coords[0])
        
        # Calculate area
        area = float(np.sum(mask > 0))
        
        # Calculate tight bounding box
        coords = np.where(mask > 0)
        x_min, x_max = int(coords[1].min()), int(coords[1].max())
        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        tight_bbox = (x_min, y_min, x_max + 1, y_max + 1)  # +1 for inclusive bounds
        
        # Calculate aspect ratio
        bbox_width = tight_bbox[2] - tight_bbox[0]
        bbox_height = tight_bbox[3] - tight_bbox[1]
        aspect_ratio = bbox_width / max(bbox_height, 1)  # Avoid division by zero
        
        # Calculate compactness (mathematical measure of shape regularity)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
            else:
                compactness = 0.0
        else:
            compactness = 0.0
        
        return MaskFeatures(
            centroid=(cx, cy),
            area=area,
            bbox=tight_bbox,
            aspect_ratio=aspect_ratio,
            compactness=compactness,
            confidence=confidence or 0.5,
            frame_idx=frame_idx
        )
    
    def _update_history(self, features: MaskFeatures):
        """Update historical data with new mask features"""
        self.mask_features_history.append(features)
        
        # Maintain history length limit
        max_length = self.config['max_history_length']
        if len(self.mask_features_history) > max_length:
            self.mask_features_history = self.mask_features_history[-max_length:]
        
        # Invalidate cached computations
        self._cache_valid_frame = -1
    
    def _make_adaptation_decision(self, current_features: MaskFeatures) -> AdaptationDecision:
        """
        Make bbox adaptation decision based on current features and history
        
        Args:
            current_features: Features extracted from current mask
            
        Returns:
            AdaptationDecision with new bbox and reasoning
        """
        confidence = current_features.confidence
        
        # EDGE CASE TOLERANCE: Check for problematic object patterns
        if self._is_problematic_object(current_features):
            return self._handle_problematic_object(current_features)
        
        # Strategy selection based on confidence and data availability
        if confidence >= self.config['high_confidence_threshold']:
            self.consecutive_failures = 0  # Reset failure counter on good mask
            self.last_good_bbox = self.current_bbox
            self.last_good_frame = current_features.frame_idx
            return self._fit_to_current_mask(current_features)
        
        elif confidence >= self.config['medium_confidence_threshold'] and self._has_sufficient_history():
            self.consecutive_failures = 0  # Reset failure counter
            return self._blend_current_and_predicted(current_features)
        
        elif confidence >= self.config['low_confidence_threshold'] and self._has_sufficient_history():
            self.consecutive_failures += 1
            return self._use_prediction_only(current_features)
        
        else:
            self.consecutive_failures += 1
            return self._conservative_fallback(current_features)
    
    def _fit_to_current_mask(self, features: MaskFeatures) -> AdaptationDecision:
        """Fit bbox tightly to current high-confidence mask"""
        if features.area == 0:
            return AdaptationDecision(
                should_update=False,
                new_bbox=self.current_bbox,
                strategy="maintain_current",
                confidence=features.confidence,
                reasoning="empty_mask_high_confidence"
            )
        
        # Add padding around tight bbox
        padding_factor = 0.15  # 15% padding
        x1, y1, x2, y2 = features.bbox
        width, height = x2 - x1, y2 - y1
        
        padding_x = max(5, int(width * padding_factor))
        padding_y = max(5, int(height * padding_factor))
        
        new_bbox = (
            max(0, x1 - padding_x),
            max(0, y1 - padding_y),
            x2 + padding_x,
            y2 + padding_y
        )
        
        return AdaptationDecision(
            should_update=True,
            new_bbox=new_bbox,
            strategy="fit_to_mask",
            confidence=features.confidence,
            reasoning="high_confidence_mask"
        )
    
    def _blend_current_and_predicted(self, current_features: MaskFeatures) -> AdaptationDecision:
        """Blend current mask information with historical predictions"""
        if not self._has_sufficient_history():
            return self._fit_to_current_mask(current_features)
        
        # Get prediction from history
        prediction = self._predict_from_history()
        
        # Blend current and predicted centroids (weighted by confidence)
        current_weight = current_features.confidence
        prediction_weight = (prediction.position_confidence + prediction.size_confidence) / 2
        
        total_weight = current_weight + prediction_weight
        if total_weight == 0:
            return self._conservative_fallback(current_features)
        
        # Weighted centroid
        current_cx, current_cy = current_features.centroid
        pred_cx, pred_cy = prediction.predicted_centroid
        
        blended_cx = (current_cx * current_weight + pred_cx * prediction_weight) / total_weight
        blended_cy = (current_cy * current_weight + pred_cy * prediction_weight) / total_weight
        
        # Blend scale factors
        if current_features.area > 0:
            current_bbox_area = (self.current_bbox[2] - self.current_bbox[0]) * (self.current_bbox[3] - self.current_bbox[1])
            current_scale = np.sqrt(current_features.area / max(current_bbox_area, 1))
        else:
            current_scale = 1.0
        
        blended_scale = (current_scale * current_weight + prediction.predicted_scale_factor * prediction_weight) / total_weight
        
        # Create blended bbox
        new_bbox = self._create_bbox_from_center_and_scale(
            (blended_cx, blended_cy),
            blended_scale
        )
        
        return AdaptationDecision(
            should_update=True,
            new_bbox=new_bbox,
            strategy="blend_current_predicted",
            confidence=(current_weight + prediction_weight) / 2,
            reasoning=f"blended_weights_c{current_weight:.2f}_p{prediction_weight:.2f}"
        )
    
    def _use_prediction_only(self, current_features: MaskFeatures) -> AdaptationDecision:
        """Use historical prediction when current mask is unreliable"""
        prediction = self._predict_from_history()
        
        new_bbox = self._create_bbox_from_center_and_scale(
            prediction.predicted_centroid,
            prediction.predicted_scale_factor
        )
        
        pred_confidence = (prediction.position_confidence + prediction.size_confidence) / 2
        
        return AdaptationDecision(
            should_update=True,
            new_bbox=new_bbox,
            strategy="prediction_only", 
            confidence=pred_confidence,
            reasoning=f"low_current_conf_{current_features.confidence:.2f}_using_prediction"
        )
    
    def _conservative_fallback(self, current_features: MaskFeatures) -> AdaptationDecision:
        """Conservative expansion when no reliable data is available"""
        self.fallback_uses += 1
        
        # Small conservative expansion
        expansion_factor = self.config['conservative_expansion_factor']
        cx = (self.current_bbox[0] + self.current_bbox[2]) / 2
        cy = (self.current_bbox[1] + self.current_bbox[3]) / 2
        
        new_bbox = self._create_bbox_from_center_and_scale((cx, cy), expansion_factor)
        
        return AdaptationDecision(
            should_update=True,
            new_bbox=new_bbox,
            strategy="conservative_fallback",
            confidence=0.3,  # Low confidence for fallback
            reasoning="insufficient_reliable_data"
        )
    
    def _predict_from_history(self) -> PredictionResult:
        """
        Predict next position and scale from historical data
        
        Returns:
            PredictionResult with position and size predictions
        """
        if not self._has_sufficient_history():
            # Return current state as "prediction"
            cx = (self.current_bbox[0] + self.current_bbox[2]) / 2
            cy = (self.current_bbox[1] + self.current_bbox[3]) / 2
            return PredictionResult(
                predicted_centroid=(cx, cy),
                predicted_scale_factor=1.0,
                position_confidence=0.0,
                size_confidence=0.0,
                strategy_used="insufficient_history"
            )
        
        # Use cached computations if available
        if self._cache_valid_frame == len(self.mask_features_history):
            # Cache is valid, use cached values
            pass
        else:
            # Recompute cached values
            self._cached_velocity = self._compute_velocity()
            self._cached_size_trend = self._compute_size_trend()
            self._cache_valid_frame = len(self.mask_features_history)
        
        # Position prediction using velocity
        last_centroid = self.mask_features_history[-1].centroid
        predicted_centroid = (
            last_centroid[0] + self._cached_velocity[0],
            last_centroid[1] + self._cached_velocity[1]
        )
        
        # Size prediction using trend
        predicted_scale_factor = self._cached_size_trend
        
        # Calculate confidence based on historical consistency
        position_confidence = self._calculate_velocity_confidence()
        size_confidence = self._calculate_size_confidence()
        
        return PredictionResult(
            predicted_centroid=predicted_centroid,
            predicted_scale_factor=predicted_scale_factor,
            position_confidence=position_confidence,
            size_confidence=size_confidence,
            strategy_used="history_based_prediction"
        )
    
    def _compute_velocity(self) -> Tuple[float, float]:
        """
        Compute smoothed velocity from centroid history
        
        Returns:
            (vx, vy) velocity in pixels per frame
        """
        if len(self.mask_features_history) < 2:
            return (0.0, 0.0)
        
        # Calculate velocity between consecutive frames
        velocities = []
        for i in range(1, len(self.mask_features_history)):
            prev_features = self.mask_features_history[i-1]
            curr_features = self.mask_features_history[i]
            
            # Calculate frame difference (handle non-consecutive frames)
            frame_diff = max(1, curr_features.frame_idx - prev_features.frame_idx)
            
            # Velocity = change in position / time
            vx = (curr_features.centroid[0] - prev_features.centroid[0]) / frame_diff
            vy = (curr_features.centroid[1] - prev_features.centroid[1]) / frame_diff
            
            velocities.append((vx, vy))
        
        # Apply exponential smoothing
        if len(velocities) == 1:
            return velocities[0]
        
        smoothing = self.config['velocity_smoothing_factor']
        smoothed_vx = velocities[0][0]
        smoothed_vy = velocities[0][1]
        
        for vx, vy in velocities[1:]:
            smoothed_vx = smoothing * smoothed_vx + (1 - smoothing) * vx
            smoothed_vy = smoothing * smoothed_vy + (1 - smoothing) * vy
        
        # Clamp to maximum extrapolation distance
        max_extrap = self.config['max_position_extrapolation']
        magnitude = np.sqrt(smoothed_vx**2 + smoothed_vy**2)
        if magnitude > max_extrap:
            scale = max_extrap / magnitude
            smoothed_vx *= scale
            smoothed_vy *= scale
        
        return (smoothed_vx, smoothed_vy)
    
    def _compute_size_trend(self) -> float:
        """
        Compute size trend from area history
        
        Returns:
            Scale factor for next frame (1.0 = no change)
        """
        if len(self.mask_features_history) < 2:
            return 1.0
        
        # Calculate size ratios between consecutive frames
        size_ratios = []
        for i in range(1, len(self.mask_features_history)):
            prev_area = self.mask_features_history[i-1].area
            curr_area = self.mask_features_history[i].area
            
            if prev_area > 0 and curr_area > 0:
                ratio = curr_area / prev_area
                size_ratios.append(ratio)
        
        if not size_ratios:
            return 1.0
        
        # Apply exponential smoothing to size ratios
        smoothing = self.config['size_smoothing_factor']
        smoothed_ratio = size_ratios[0]
        
        for ratio in size_ratios[1:]:
            smoothed_ratio = smoothing * smoothed_ratio + (1 - smoothing) * ratio
        
        # Clamp to reasonable bounds
        smoothed_ratio = np.clip(
            smoothed_ratio,
            self.config['min_scale_factor'],
            self.config['max_scale_factor']
        )
        
        return smoothed_ratio
    
    def _calculate_velocity_confidence(self) -> float:
        """Calculate confidence in velocity prediction based on consistency"""
        if len(self.mask_features_history) < 3:
            return 0.5
        
        # Calculate variance in recent velocities
        recent_velocities = []
        for i in range(max(1, len(self.mask_features_history) - 5), len(self.mask_features_history)):
            if i < len(self.mask_features_history):
                prev = self.mask_features_history[i-1]
                curr = self.mask_features_history[i]
                frame_diff = max(1, curr.frame_idx - prev.frame_idx)
                vx = (curr.centroid[0] - prev.centroid[0]) / frame_diff
                vy = (curr.centroid[1] - prev.centroid[1]) / frame_diff
                recent_velocities.append((vx, vy))
        
        if len(recent_velocities) < 2:
            return 0.5
        
        # Calculate variance
        vx_values = [v[0] for v in recent_velocities]
        vy_values = [v[1] for v in recent_velocities]
        
        vx_var = np.var(vx_values)
        vy_var = np.var(vy_values)
        
        # Lower variance = higher confidence
        avg_variance = (vx_var + vy_var) / 2
        confidence = 1.0 / (1.0 + avg_variance)  # Confidence decreases with variance
        
        return np.clip(confidence, 0.1, 0.9)
    
    def _calculate_size_confidence(self) -> float:
        """Calculate confidence in size prediction based on consistency"""
        if len(self.mask_features_history) < 3:
            return 0.5
        
        # Calculate variance in recent size changes
        recent_areas = [f.area for f in self.mask_features_history[-5:]]
        if len(recent_areas) < 2:
            return 0.5
        
        # Calculate relative variance (coefficient of variation)
        mean_area = np.mean(recent_areas)
        if mean_area == 0:
            return 0.1
        
        std_area = np.std(recent_areas)
        cv = std_area / mean_area  # Coefficient of variation
        
        # Lower CV = higher confidence
        confidence = 1.0 / (1.0 + cv)
        
        return np.clip(confidence, 0.1, 0.9)
    
    def _create_bbox_from_center_and_scale(self, center: Tuple[float, float], scale_factor: float) -> Tuple[int, int, int, int]:
        """
        Create bbox from center position and scale factor
        
        Args:
            center: (cx, cy) center coordinates
            scale_factor: Factor to scale current bbox size
            
        Returns:
            (x1, y1, x2, y2) bounding box
        """
        cx, cy = center
        
        # Current bbox dimensions
        curr_width = self.current_bbox[2] - self.current_bbox[0]
        curr_height = self.current_bbox[3] - self.current_bbox[1]
        
        # New dimensions
        new_width = max(self.config['min_bbox_size'], 
                       min(self.config['max_bbox_size'], 
                           int(curr_width * scale_factor)))
        new_height = max(self.config['min_bbox_size'],
                        min(self.config['max_bbox_size'],
                            int(curr_height * scale_factor)))
        
        # Create bbox centered at cx, cy
        x1 = int(cx - new_width / 2)
        y1 = int(cy - new_height / 2)
        x2 = x1 + new_width
        y2 = y1 + new_height
        
        # Ensure non-negative coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        return (x1, y1, x2, y2)
    
    def _has_sufficient_history(self) -> bool:
        """Check if we have enough history for reliable predictions"""
        min_history = self.config['min_history_for_prediction']
        return len(self.mask_features_history) >= min_history
    
    def _log_adaptation(self, previous_bbox: Tuple[int, int, int, int], 
                       decision: AdaptationDecision, features: MaskFeatures):
        """Log adaptation decision details"""
        if not self.config['enable_logging']:
            return
        
        print(f"   📦 Object {self.obj_id} Frame {features.frame_idx}: {decision.strategy}")
        print(f"      Decision: {decision.reasoning} (confidence: {decision.confidence:.3f})")
        print(f"      Bbox: {previous_bbox} → {decision.new_bbox}")
        print(f"      Mask: centroid=({features.centroid[0]:.1f},{features.centroid[1]:.1f}), area={features.area:.0f}")
        
        if self.config['enable_detailed_math_logging'] and self._has_sufficient_history():
            velocity = self._cached_velocity if (hasattr(self, '_cached_velocity') and self._cached_velocity is not None) else (0, 0)
            size_trend = self._cached_size_trend if (hasattr(self, '_cached_size_trend') and self._cached_size_trend is not None) else 1.0
            print(f"      Math: velocity=({velocity[0]:.2f},{velocity[1]:.2f}), size_trend={size_trend:.3f}")
    
    def get_current_bbox(self) -> Tuple[int, int, int, int]:
        """Get current bounding box"""
        return self.current_bbox
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        if not self.mask_features_history:
            return {"status": "no_data"}
        
        # Calculate performance metrics
        strategy_counts = {}
        for decision in self.adaptation_decisions:
            strategy = decision.strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Calculate confidence statistics
        confidences = [f.confidence for f in self.mask_features_history]
        
        stats = {
            "obj_id": self.obj_id,
            "total_frames": len(self.mask_features_history),
            "total_updates": self.total_updates,
            "fallback_uses": self.fallback_uses,
            "strategy_distribution": strategy_counts,
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "prediction_success_rate": self.successful_predictions / max(self.total_updates, 1),
            "current_bbox": self.current_bbox,
            "initial_bbox": self.initial_bbox,
            "has_sufficient_history": self._has_sufficient_history(),
            "recent_strategies": [d.strategy for d in self.adaptation_decisions[-5:]]
        }
        
        return stats
    
    def reset_state(self):
        """Reset tracker state"""
        self.current_bbox = self.initial_bbox
        self.mask_features_history.clear()
        self.prediction_history.clear()
        self.adaptation_decisions.clear()
        self._cached_velocity = None
        self._cached_size_trend = None
        self._cache_valid_frame = -1
        self.total_updates = 0
        self.successful_predictions = 0
        self.fallback_uses = 0
        
        if self.config['enable_logging']:
            print(f"🔄 MaskDrivenAdaptiveTracker reset for object {self.obj_id}")
    
    def _is_problematic_object(self, current_features: MaskFeatures) -> bool:
        """
        Detect if this object is exhibiting problematic tracking patterns
        
        Args:
            current_features: Current mask features
            
        Returns:
            True if object shows problematic patterns requiring special handling
        """
        # Check for too many consecutive failures
        if self.consecutive_failures >= self.config['max_consecutive_failures']:
            return True
        
        # Check for extremely low area masks repeatedly
        if (current_features.area < self.config.get('min_mask_area', 50) and 
            self.consecutive_failures >= 3):
            return True
        
        # Check for bbox size explosion patterns
        current_area = (self.current_bbox[2] - self.current_bbox[0]) * (self.current_bbox[3] - self.current_bbox[1])
        initial_area = (self.initial_bbox[2] - self.initial_bbox[0]) * (self.initial_bbox[3] - self.initial_bbox[1])
        
        if current_area > initial_area * 10:  # More than 10x original size
            return True
        
        return False
    
    def _handle_problematic_object(self, current_features: MaskFeatures) -> AdaptationDecision:
        """
        Handle objects that exhibit problematic tracking patterns with tolerance
        
        Args:
            current_features: Current mask features
            
        Returns:
            AdaptationDecision with tolerance-based handling
        """
        if not self.is_object_lost and self.config['allow_object_loss']:
            # Mark object as lost and use last good bbox
            self.is_object_lost = True
            
            # Shrink to indicate uncertainty
            factor = self.config['lost_object_bbox_factor']
            x1, y1, x2, y2 = self.last_good_bbox
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            
            new_width = int(width * factor)
            new_height = int(height * factor)
            
            new_bbox = (
                int(center_x - new_width / 2),
                int(center_y - new_height / 2),
                int(center_x + new_width / 2),
                int(center_y + new_height / 2)
            )
            
            if self.config['enable_logging']:
                print(f"   🔍 Object {self.obj_id} marked as LOST after {self.consecutive_failures} consecutive failures")
                print(f"   📦 Using last good bbox with {factor:.0%} size: {self.last_good_bbox} → {new_bbox}")
            
            return AdaptationDecision(
                should_update=True,
                new_bbox=new_bbox,
                strategy="object_lost_tolerance",
                confidence=0.1,
                reasoning=f"lost_after_{self.consecutive_failures}_failures"
            )
        
        elif self.is_object_lost:
            # Object already marked as lost, maintain current bbox with gradual shrink
            if self.config['failure_recovery_mode'] == 'gradual_shrink':
                factor = max(0.5, 0.98)  # Shrink by 2% each frame, minimum 50%
                
                x1, y1, x2, y2 = self.current_bbox
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                
                new_width = max(self.config['min_bbox_size'], int(width * factor))
                new_height = max(self.config['min_bbox_size'], int(height * factor))
                
                new_bbox = (
                    int(center_x - new_width / 2),
                    int(center_y - new_height / 2),
                    int(center_x + new_width / 2),
                    int(center_y + new_height / 2)
                )
                
                return AdaptationDecision(
                    should_update=True,
                    new_bbox=new_bbox,
                    strategy="lost_object_gradual_shrink",
                    confidence=0.1,
                    reasoning="maintaining_lost_object_with_shrink"
                )
            else:
                # Maintain current bbox
                return AdaptationDecision(
                    should_update=False,
                    new_bbox=self.current_bbox,
                    strategy="lost_object_maintain",
                    confidence=0.1,
                    reasoning="maintaining_lost_object_bbox"
                )
        
        else:
            # Fallback to conservative behavior if loss not allowed
            return self._conservative_fallback(current_features)