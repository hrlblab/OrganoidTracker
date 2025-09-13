#!/usr/bin/env python3
"""
Regular SAM2 Video Tracker
Implementation of BaseVideoTracker for standard SAM2 model
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Any

# Import base classes
from .base_model import BaseVideoTracker, ModelCapabilities, ModelMetadata
from .mask_driven_adaptive_tracker import MaskDrivenAdaptiveTracker

# Suppress warnings
warnings.filterwarnings("ignore")

# Import SAM2
try:
    # Import from the reorganized models directory
    import sys
    from pathlib import Path
    sam2_path = Path(__file__).parent.parent.parent / "models" / "sam2"
    sys.path.insert(0, str(sam2_path))

    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
    print("✅ SAM2 loaded successfully from models/sam2/")
except ImportError as e:
    print(f"⚠️  SAM2 not available: {e}")
    SAM2_AVAILABLE = False
    # Create dummy function to prevent errors
    def build_sam2_video_predictor(*args, **kwargs):
        raise ImportError("SAM2 not available")


class SAM2Tracker(BaseVideoTracker):
    """
    Regular SAM2 Video Object Tracker
    Implements BaseVideoTracker interface for standard SAM2 model
    """

    def __init__(self, model_config='sam2_hiera_b', checkpoint_path=None,
                 device='cuda', enable_adaptive_tracking=True, enable_reverse_tracking=True, **kwargs):
        """
        Initialize the SAM2 tracker

        Args:
            model_config: SAM2 model configuration
            checkpoint_path: Path to model checkpoint (None for auto-detection)
            device: Device to run inference on
            enable_adaptive_tracking: Enable adaptive bounding box tracking
            enable_reverse_tracking: Enable reverse temporal tracking (last→first frame)
        """
        super().__init__("SAM2", **kwargs)

        self.model_config = model_config
        self.enable_adaptive_tracking = enable_adaptive_tracking
        self.enable_reverse_tracking = enable_reverse_tracking
        
        # Adaptive tracking components
        self.adaptive_trackers: Dict[int, MaskDrivenAdaptiveTracker] = {}
        
        # Load adaptive config from main config
        try:
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from config import (
                ENABLE_ADAPTIVE_TRACKING, ADAPTIVE_BBOX_UPDATE_THRESHOLD, ADAPTIVE_EXPANSION_FACTOR,
                ADAPTIVE_MIN_MASK_AREA, ADAPTIVE_MAX_EXPANSION_FACTOR, ADAPTIVE_CONFIDENCE_THRESHOLD,
                ADAPTIVE_LOW_CONFIDENCE_THRESHOLD, ADAPTIVE_PADDING_FACTOR, ADAPTIVE_VELOCITY_SMOOTHING,
                ADAPTIVE_ENABLE_LOGGING, ADAPTIVE_ENABLE_VELOCITY_PREDICTION
            )
            
            self.enable_adaptive_tracking = enable_adaptive_tracking and ENABLE_ADAPTIVE_TRACKING
            self.adaptive_config = {
                'bbox_update_threshold': ADAPTIVE_BBOX_UPDATE_THRESHOLD,
                'expansion_factor': ADAPTIVE_EXPANSION_FACTOR,
                'min_mask_area': ADAPTIVE_MIN_MASK_AREA,
                'max_expansion_factor': ADAPTIVE_MAX_EXPANSION_FACTOR,
                'confidence_threshold': ADAPTIVE_CONFIDENCE_THRESHOLD,
                'low_confidence_threshold': ADAPTIVE_LOW_CONFIDENCE_THRESHOLD,
                'padding_factor': ADAPTIVE_PADDING_FACTOR,
                'velocity_smoothing': ADAPTIVE_VELOCITY_SMOOTHING,
                'enable_logging': ADAPTIVE_ENABLE_LOGGING,
                'enable_velocity_prediction': ADAPTIVE_ENABLE_VELOCITY_PREDICTION,
                'is_backward_tracking': False,  # Will be set when video is loaded
                **kwargs.get('adaptive_config', {})  # Allow override
            }
        except ImportError:
            # Fallback to defaults if config not available
            self.enable_adaptive_tracking = enable_adaptive_tracking
            self.adaptive_config = kwargs.get('adaptive_config', {})

        # Auto-determine checkpoint path if not provided
        if checkpoint_path is None:
            # Get absolute path to checkpoints directory
            app_root = Path(__file__).parent.parent.parent
            checkpoints_dir = app_root / "checkpoints"

            # Map model config to checkpoint file for regular SAM2
            checkpoint_map = {
                'sam2_hiera_s': 'sam2.1_hiera_small.pt',
                'sam2_hiera_b': 'sam2.1_hiera_base_plus.pt',
                'sam2_hiera_l': 'sam2.1_hiera_large.pt',
                'sam2_hiera_t': 'sam2.1_hiera_tiny.pt'
            }

            checkpoint_file = checkpoint_map.get(model_config, 'sam2.1_hiera_small.pt')
            checkpoint_path = str(checkpoints_dir / checkpoint_file)
            print(f"   Auto-selected SAM2 checkpoint: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.device_name = device

        # Model-specific attributes
        self.predictor = None
        self.video_tensor = None
        self.inference_state = None

        # Auto-detect device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("⚠️  CUDA not available, using CPU")

        self.device = torch.device(device)

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        """Get model metadata for registration"""
        return ModelMetadata(
            name="sam2",
            display_name="SAM2",
            description="Segment Anything Model 2 for general purpose video object segmentation",
            capabilities=[
                ModelCapabilities.CLICK_PROMPTS,
                ModelCapabilities.BBOX_PROMPTS,
                ModelCapabilities.MULTI_OBJECT,
                ModelCapabilities.REAL_TIME
            ],
            requirements=["PyTorch", "SAM2 package", "CUDA (recommended)"]
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if SAM2 is available"""
        return SAM2_AVAILABLE

    def load_model(self, **kwargs) -> bool:
        """Load and initialize the SAM2 model"""
        if not SAM2_AVAILABLE:
            print("❌ SAM2 not available. Please install sam2 package.")
            return False

        try:
            print(f"🔄 Loading SAM2 model...")

            # Add the SAM2 model directory to sys.path
            from hydra import initialize_config_dir, compose
            from hydra.core.global_hydra import GlobalHydra
            from pathlib import Path

            models_path = Path(__file__).parent.parent.parent / "models"
            sam2_path = models_path / "sam2"
            sam2_path_str = str(sam2_path.absolute())
            if sam2_path_str not in sys.path:
                sys.path.insert(0, sam2_path_str)
                print(f"   Added to Python path: {sam2_path_str}")

            # Get absolute path to the SAM2 config directory
            config_dir = sam2_path / "configs"
            config_dir_abs = str(config_dir.absolute())

            print(f"   Using config directory: {config_dir_abs}")
            print(f"   Config file: {self.model_config}")
            print(f"   Checkpoint: {self.checkpoint_path}")

            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()

            # Initialize Hydra with the SAM2 config directory
            with initialize_config_dir(config_dir=config_dir_abs, version_base=None):
                from build_sam import build_sam2_video_predictor

                # Use SAM2 build system with optional improved tracking configs
                # Check if improved tracking is enabled
                try:
                    from config import SAM2_USE_IMPROVED_CONFIG
                    use_improved_config = SAM2_USE_IMPROVED_CONFIG
                except ImportError:
                    use_improved_config = False

                if use_improved_config and self.model_config == 'sam2_hiera_b':
                    # Use improved tracking configuration for hiera_b model
                    config_map = {
                        'sam2_hiera_s': 'sam2.1/sam2.1_hiera_s.yaml',
                        'sam2_hiera_b': 'sam2.1/sam2.1_hiera_b+_improved_tracking.yaml',  # IMPROVED CONFIG
                        'sam2_hiera_l': 'sam2.1/sam2.1_hiera_l.yaml',
                        'sam2_hiera_t': 'sam2.1/sam2.1_hiera_t.yaml'
                    }
                    print(f"   🚀 Using IMPROVED tracking configuration for {self.model_config}")
                else:
                    # Use standard configurations
                    config_map = {
                        'sam2_hiera_s': 'sam2.1/sam2.1_hiera_s.yaml',
                        'sam2_hiera_b': 'sam2.1/sam2.1_hiera_b+.yaml',
                        'sam2_hiera_l': 'sam2.1/sam2.1_hiera_l.yaml',
                        'sam2_hiera_t': 'sam2.1/sam2.1_hiera_t.yaml'
                    }

                config_file = config_map.get(self.model_config, 'sam2.1/sam2.1_hiera_s.yaml')
                print(f"   Using config: {config_file}")

                self.predictor = build_sam2_video_predictor(
                    config_file=config_file,
                    ckpt_path=self.checkpoint_path,
                    device=self.device_name,
                    mode="eval"
                )

            self._patch_device_operations()
            self.is_loaded = True
            self.is_initialized = True
            print("✅ SAM2 model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading SAM2: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return False

    def _patch_device_operations(self):
        """Patch device operations for compatibility"""
        original_add_new_points = self.predictor.add_new_points
        original_add_new_bbox = getattr(self.predictor, 'add_new_bbox', None)
        original_add_new_points_or_box = getattr(self.predictor, 'add_new_points_or_box', None)

        def patched_add_new_points(inference_state, frame_idx, obj_id, points, labels, **kwargs):
            if hasattr(points, 'to'):
                points = points.to(self.device)
            if hasattr(labels, 'to'):
                labels = labels.to(self.device)
            return original_add_new_points(inference_state, frame_idx, obj_id, points, labels, **kwargs)

        self.predictor.add_new_points = patched_add_new_points

        if original_add_new_bbox:
            def patched_add_new_bbox(inference_state, frame_idx, obj_id, bbox, **kwargs):
                if hasattr(bbox, 'to'):
                    bbox = bbox.to(self.device)
                return original_add_new_bbox(inference_state, frame_idx, obj_id, bbox, **kwargs)

            self.predictor.add_new_bbox = patched_add_new_bbox

        # Patch the add_new_points_or_box method for bounding box support
        if original_add_new_points_or_box:
            def patched_add_new_points_or_box(inference_state, frame_idx, obj_id, points=None, labels=None, box=None, **kwargs):
                # AGGRESSIVE DEVICE PATCHING: Temporarily patch torch.zeros and torch.tensor
                # to ensure all tensors created inside SAM2 are on the correct device

                # Store original functions
                original_torch_zeros = torch.zeros
                original_torch_tensor = torch.tensor

                def device_aware_zeros(*args, **kwargs):
                    if 'device' not in kwargs:
                        kwargs['device'] = self.device
                    return original_torch_zeros(*args, **kwargs)

                def device_aware_tensor(*args, **kwargs):
                    if 'device' not in kwargs and len(args) > 0:
                        # Only add device if we're creating a new tensor (not converting)
                        kwargs['device'] = self.device
                    return original_torch_tensor(*args, **kwargs)

                # Temporarily replace torch functions
                torch.zeros = device_aware_zeros
                torch.tensor = device_aware_tensor

                try:
                    # Ensure input tensors are on correct device
                    if points is not None and hasattr(points, 'to'):
                        points = points.to(self.device)
                    if labels is not None and hasattr(labels, 'to'):
                        labels = labels.to(self.device)
                    if box is not None and hasattr(box, 'to'):
                        box = box.to(self.device)

                    # Call original method with device-aware tensor creation
                    result = original_add_new_points_or_box(inference_state, frame_idx, obj_id, points=points, labels=labels, box=box, **kwargs)
                    return result

                finally:
                    # Always restore original functions
                    torch.zeros = original_torch_zeros
                    torch.tensor = original_torch_tensor

            self.predictor.add_new_points_or_box = patched_add_new_points_or_box

    def _debug_inference_state_devices(self):
        """Debug method to check device placement of inference state"""
        if not isinstance(self.inference_state, dict):
            return

        for key, value in self.inference_state.items():
            if hasattr(value, 'device'):
                print(f"   • inference_state['{key}'] device: {value.device}")
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'device'):
                        print(f"   • inference_state['{key}']['{subkey}'] device: {subvalue.device}")
                    elif isinstance(subvalue, (list, tuple)) and len(subvalue) > 0:
                        for i, item in enumerate(subvalue[:3]):  # Check first 3 items
                            if hasattr(item, 'device'):
                                print(f"   • inference_state['{key}']['{subkey}'][{i}] device: {item.device}")
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                for i, item in enumerate(value[:3]):  # Check first 3 items
                    if hasattr(item, 'device'):
                        print(f"   • inference_state['{key}'][{i}] device: {item.device}")

    def _move_inference_state_to_device(self):
        """Move all tensors in inference state to the correct device"""
        if not isinstance(self.inference_state, dict):
            return

        def move_to_device(obj, path=""):
            if hasattr(obj, 'to') and hasattr(obj, 'device'):
                if obj.device != self.device:
                    print(f"   🔄 Moving {path} from {obj.device} to {self.device}")
                    return obj.to(self.device)
                return obj
            elif isinstance(obj, dict):
                return {k: move_to_device(v, f"{path}.{k}") for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_to_device(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(item, f"{path}[{i}]") for i, item in enumerate(obj))
            else:
                return obj

        print(f"   🔧 Recursively moving inference state to {self.device}")
        self.inference_state = move_to_device(self.inference_state, "inference_state")

    def load_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Load video for tracking"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print(f"📹 Loading video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

        cap.release()

        # Store video path for later reuse (e.g., in clear_prompts)
        self.current_video_path = video_path

        # CONFIGURABLE PREPROCESSING: Reverse video based on user setting
        original_frames = frames.copy()  # Keep original order for reference
        
        if self.enable_reverse_tracking:
            # REVERSE TRACKING: Process from last frame to first (biological use case)
            frames.reverse()
            print(f"🔄 Preprocessed video: reversed for backwards tracking")
            print(f"📊 Video preprocessing: Original {len(original_frames)} frames -> Reversed {len(frames)} frames")
            self.is_reversed_video = True
            
            # UPDATE ADAPTIVE TRACKING CONFIG: Enable backward tracking mode
            if hasattr(self, 'adaptive_config'):
                self.adaptive_config['is_backward_tracking'] = True
                print(f"🔄 Adaptive tracking: enabled backward mode for reverse video processing")
        else:
            # FORWARD TRACKING: Process from first frame to last (standard use case)
            print(f"▶️ Processing video: forward tracking mode")
            print(f"📊 Video processing: {len(frames)} frames in original temporal order")
            self.is_reversed_video = False
            
            # UPDATE ADAPTIVE TRACKING CONFIG: Disable backward tracking mode
            if hasattr(self, 'adaptive_config'):
                self.adaptive_config['is_backward_tracking'] = False
                print(f"▶️ Adaptive tracking: enabled forward mode for normal video processing")

        # Store video data
        self.video_frames = frames
        self.original_frames = original_frames  # Keep original order for debugging
        self.video_path = video_path

        # Prepare video tensor
        self._prepare_video_tensor()

        # Initialize inference state using video path (let SAM2 handle video loading)
        try:
            print(f"🔍 Initializing inference state on device: {self.device}")
            self.inference_state = self.predictor.init_state(
                video_path=video_path,
                offload_video_to_cpu=False,
                offload_state_to_cpu=False
            )
            print(f"✅ Inference state initialized")

            # Debug: Check inference state device placement
            print(f"🔍 Checking inference state device placement...")
            self._debug_inference_state_devices()

            # CRITICAL FIX: Move inference state to correct device
            print(f"🔧 Moving inference state to device: {self.device}")
            self._move_inference_state_to_device()

            print(f"🔍 After device move - checking devices again...")
            self._debug_inference_state_devices()

        except Exception as e:
            print(f"Failed to initialize with video_path, trying alternative method: {e}")
            # Fallback: this might not work in current SAM2 version
            try:
                self.inference_state = self.predictor.init_state(video_path=str(video_path))
                print(f"🔧 Moving inference state to device: {self.device}")
                self._move_inference_state_to_device()
            except Exception as e2:
                print(f"All initialization methods failed: {e2}")
                raise e2

        video_info = {
            'num_frames': len(frames),
            'fps': self.fps,
            'dimensions': frames[0].shape[:2] if frames else (0, 0),
            'total_frames': total_frames
        }

        print(f"✅ Loaded {len(frames)} frames at {self.fps} FPS")
        return video_info

    def _prepare_video_tensor(self, target_size=1024):
        """Convert video frames to SAM2 tensor format"""
        processed_frames = []

        for frame in self.video_frames:
            # Resize frame
            frame_pil = Image.fromarray(frame)
            frame_pil = frame_pil.resize((target_size, target_size))

            # Convert to tensor
            frame_tensor = torch.tensor(np.array(frame_pil)).permute(2, 0, 1).float()
            processed_frames.append(frame_tensor)

        # Stack and move to device
        self.video_tensor = torch.stack(processed_frames).to(self.device)

    def get_first_frame(self) -> np.ndarray:
        """Get the first frame for GUI click prompting (preprocessed video first frame)"""
        if self.video_frames is None:
            raise ValueError("No video loaded. Call load_video() first.")

        # Return the first frame of preprocessed (reversed) video
        # This represents the last frame of the original video for backwards tracking
        return self.video_frames[0].copy()

    def add_click_prompt(self, x: int, y: int, label: int = 1, obj_id: int = 1, frame_idx: int = 0) -> bool:
        """Add a click prompt for object tracking"""
        try:
            print(f"🎯 Adding click prompt at ({x}, {y}) with label {label}")
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"🔍 DEBUG: Prompt added to frame_idx={frame_idx} (preprocessed video)")
                if hasattr(self, 'is_reversed_video') and self.is_reversed_video:
                    original_frame_idx = len(self.video_frames) - 1 - frame_idx
                    print(f"🔍 DEBUG: This corresponds to original frame {original_frame_idx}")

            # Convert coordinates to tensor
            points = torch.tensor([[x, y]], dtype=torch.float32).to(self.device)
            labels = torch.tensor([label], dtype=torch.int32).to(self.device)

            # Add prompt to SAM2
            self.predictor.add_new_points(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels
            )

            # Store prompt for reference
            if obj_id not in self.prompts:
                self.prompts[obj_id] = []

            self.prompts[obj_id].append({
                'frame_idx': frame_idx,
                'x': x,
                'y': y,
                'label': label,
                'type': 'click'
            })

            return True
        except Exception as e:
            print(f"❌ Error adding click prompt: {str(e)}")
            return False

    def add_bbox_prompt(self, x1: int, y1: int, x2: int, y2: int, obj_id: int = 1, frame_idx: int = 0) -> bool:
        """Add a bounding box prompt for object tracking"""
        if self.predictor is None:
            print("⚠️  No model loaded yet")
            return False

        if not hasattr(self.predictor, 'add_new_points_or_box'):
            print("⚠️  Bounding box prompts not supported in this SAM2 version")
            return False

        try:
            # Initialize adaptive tracker for this object if enabled
            if self.enable_adaptive_tracking and obj_id not in self.adaptive_trackers:
                initial_bbox = (x1, y1, x2, y2)
                self.adaptive_trackers[obj_id] = MaskDrivenAdaptiveTracker(
                    initial_bbox=initial_bbox,
                    obj_id=obj_id,
                    config=self.adaptive_config
                )
                print(f"🎯 Initialized adaptive tracker for object {obj_id}")

            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"📦 Adding bbox prompt: ({x1}, {y1}) to ({x2}, {y2}) for object {obj_id}")

            # Deep device debugging
            print(f"🔍 DEVICE DEBUG:")
            print(f"   • Tracker device: {self.device}")
            print(f"   • Inference state type: {type(self.inference_state)}")

            # Check inference state device
            if hasattr(self.inference_state, 'device'):
                print(f"   • Inference state device: {self.inference_state.device}")

            # Check inference state contents for device mismatches
            if isinstance(self.inference_state, dict):
                for key, value in self.inference_state.items():
                    if hasattr(value, 'device'):
                        print(f"   • inference_state['{key}'] device: {value.device}")
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        if hasattr(value[0], 'device'):
                            print(f"   • inference_state['{key}'][0] device: {value[0].device}")

            # Convert to tensor (format: [x1, y1, x2, y2])
            # Device handling is now done by the patched method
            bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            print(f"   • bbox tensor device (before patching): {bbox.device}")

            # Add to SAM2 using the correct method (device patching handles device placement)
            print(f"🔄 Calling add_new_points_or_box...")
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=bbox,  # Use 'box' parameter, not 'bbox'
                clear_old_points=True
            )
            print(f"✅ add_new_points_or_box completed successfully")

            # Store prompt
            if obj_id not in self.prompts:
                self.prompts[obj_id] = []

            self.prompts[obj_id].append({
                'frame_idx': frame_idx,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'type': 'bbox'
            })

            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"✅ Successfully added bbox prompt for object {obj_id}")

            return True
        except Exception as e:
            print(f"❌ Error adding bbox prompt: {str(e)}")
            return False

    def run_tracking(self, progress_callback: Optional[callable] = None) -> Dict[int, Dict[int, Any]]:
        """Run object tracking across all video frames"""
        print("🔄 Running object tracking...")

        if not self.prompts:
            raise ValueError("No prompts added. Add click or bbox prompts first.")

        video_segments = {}
        total_frames = len(self.video_frames)

        try:
            frame_count = 0
            # Quality filtering parameters for better disappearing object handling
            try:
                from config import SAM2_IMPROVED_TRACKING, SAM2_MIN_MASK_AREA, SAM2_MIN_CONFIDENCE, SAM2_MEMORY_FRAMES
                enable_quality_filtering = SAM2_IMPROVED_TRACKING
                min_mask_area = SAM2_MIN_MASK_AREA
                min_confidence = SAM2_MIN_CONFIDENCE
                memory_frames = SAM2_MEMORY_FRAMES
            except ImportError:
                # Fallback to default values if config not available
                enable_quality_filtering = True
                min_mask_area = 50
                min_confidence = 0.3
                memory_frames = 2

            print(f"   🔍 Quality filtering: {'enabled' if enable_quality_filtering else 'disabled'}")
            if enable_quality_filtering:
                print(f"   🔍 Thresholds: min_area={min_mask_area}px, min_confidence={min_confidence:.1f}")
            if memory_frames >= 1000:
                print(f"   🧠 Memory dependence: ALL previous frames (unlimited)")
            else:
                print(f"   🧠 Memory dependence: {memory_frames} previous frames")

            # Track processed frames for memory management
            processed_frames = []

            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                frame_masks = {}

                for i, out_obj_id in enumerate(out_obj_ids):
                    mask_logits = out_mask_logits[i]

                    # Apply quality filtering to reduce false tracking of disappeared objects
                    if enable_quality_filtering:
                        # Clean mask first (keep only largest connected component)
                        cleaned_mask = self._clean_mask_to_largest_component(mask_logits)

                        if cleaned_mask is not None and self._is_mask_quality_acceptable(cleaned_mask, min_mask_area, min_confidence):
                            frame_masks[out_obj_id] = cleaned_mask
                            
                            # Adaptive tracking: Update bounding box if enabled
                            if self.enable_adaptive_tracking and out_obj_id in self.adaptive_trackers:
                                self._update_adaptive_bbox(out_obj_id, cleaned_mask, out_frame_idx)
                                
                        else:
                            print(f"   🔍 Frame {out_frame_idx}, Object {out_obj_id}: Low quality mask filtered out")
                            
                            # Adaptive tracking: Handle low quality mask
                            if self.enable_adaptive_tracking and out_obj_id in self.adaptive_trackers:
                                self._handle_low_quality_tracking(out_obj_id, out_frame_idx)
                    else:
                        # No filtering - use all masks
                        frame_masks[out_obj_id] = mask_logits
                        
                        # Adaptive tracking for unfiltered masks
                        if self.enable_adaptive_tracking and out_obj_id in self.adaptive_trackers:
                            # Pass the original mask_logits - conversion will happen in _update_adaptive_bbox
                            self._update_adaptive_bbox(out_obj_id, mask_logits, out_frame_idx)

                # Only store frame if it has valid masks
                if frame_masks:
                    video_segments[out_frame_idx] = frame_masks

                # Memory management: limit dependence to recent frames only
                processed_frames.append(out_frame_idx)
                if len(processed_frames) > memory_frames:
                    # Clear older memory beyond the specified number of frames
                    old_frame = processed_frames.pop(0)
                    self._clear_old_frame_memory(old_frame)

                frame_count += 1
                if progress_callback:
                    progress_callback(frame_count, total_frames, f"Processing frame {frame_count}/{total_frames}")

        except Exception as e:
            print(f"⚠️  Warning during tracking: {e}")
            print("Continuing with available results...")

        if not video_segments:
            raise RuntimeError("No tracking results obtained. Check model compatibility.")

        print(f"✅ Tracking completed for {len(video_segments)} frames")
        return video_segments

    def _clean_mask_to_largest_component(self, mask_logits):
        """
        Clean mask by keeping only the largest connected component.
        This ensures each cyst is represented as a single, connected circular object.

        Args:
            mask_logits: Raw mask logits from SAM2

        Returns:
            Cleaned mask with only the largest connected component, or None if cleaning fails
        """
        try:
            import torch
            import numpy as np
            import cv2

            # Convert to numpy if it's a tensor
            if isinstance(mask_logits, torch.Tensor):
                mask_np = mask_logits.cpu().numpy()
                is_tensor = True
                original_device = mask_logits.device
            else:
                mask_np = mask_logits
                is_tensor = False

            # Handle different dimensionalities
            original_shape = mask_np.shape
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            # Convert logits to binary mask
            if mask_np.dtype != bool:
                # Apply sigmoid to convert logits to probabilities, then threshold
                mask_probs = 1 / (1 + np.exp(-mask_np))
                binary_mask = mask_probs > 0.5
                # Keep the original logits for the cleaned mask
                use_probs = True
            else:
                binary_mask = mask_np
                use_probs = False

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask.astype(np.uint8), connectivity=8
            )

            if num_labels <= 1:  # Only background, no components found
                return None

            # Find the largest component (excluding background which is label 0)
            component_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            largest_component_idx = np.argmax(component_areas) + 1  # +1 because we skipped background

            # Create mask with only the largest component
            largest_component_mask = (labels == largest_component_idx)

            if use_probs:
                # Apply the largest component mask to the original probabilities/logits
                if mask_np.dtype == bool:
                    # If original was boolean, keep it boolean
                    cleaned_mask = largest_component_mask
                else:
                    # Apply component mask to original logits, set others to very negative
                    cleaned_mask = np.where(largest_component_mask, mask_np, -20.0)
            else:
                cleaned_mask = largest_component_mask

            # Restore original shape if needed
            if len(original_shape) > 2:
                for _ in range(len(original_shape) - 2):
                    cleaned_mask = np.expand_dims(cleaned_mask, axis=0)

            # Convert back to tensor if input was tensor
            if is_tensor:
                cleaned_mask = torch.tensor(cleaned_mask, device=original_device, dtype=mask_logits.dtype)

            # Log the cleaning result
            original_area = np.count_nonzero(binary_mask)
            cleaned_area = np.count_nonzero(largest_component_mask)
            removed_components = num_labels - 2  # -1 for background, -1 for largest

            if removed_components > 0:
                print(f"      🧹 Cleaned mask: kept largest component ({cleaned_area}px), "
                      f"removed {removed_components} smaller components ({original_area - cleaned_area}px)")

            return cleaned_mask

        except Exception as e:
            print(f"      Warning: Error cleaning mask: {e}")
            return mask_logits  # Return original if cleaning fails

    def _clear_old_frame_memory(self, frame_idx: int):
        """
        Clear memory for an old frame to reduce dependence on distant history.
        This helps with disappearing objects by not persisting old memory too long.

        Args:
            frame_idx: Frame index to clear from memory
        """
        try:
            if hasattr(self, 'inference_state') and self.inference_state is not None:
                # Clear non-conditioning frame outputs for this frame from all objects
                for obj_idx in self.inference_state.get("output_dict_per_obj", {}):
                    obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                    # Remove old non-conditioning memory
                    if "non_cond_frame_outputs" in obj_output_dict:
                        obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

                    # Also clear from temp output dict
                    temp_output_dict_per_obj = self.inference_state.get("temp_output_dict_per_obj", {})
                    if obj_idx in temp_output_dict_per_obj:
                        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

                print(f"      🧠 Cleared memory for frame {frame_idx} (reducing memory dependence)")
        except Exception as e:
            print(f"      Warning: Error clearing frame memory for frame {frame_idx}: {e}")

    def _is_mask_quality_acceptable(self, mask_logits, min_area: int = 50, min_confidence: float = 0.3) -> bool:
        """
        Check if a mask meets quality criteria to reduce false tracking of disappeared objects.

        Args:
            mask_logits: Raw mask logits from SAM2
            min_area: Minimum mask area in pixels
            min_confidence: Minimum confidence score (0-1)

        Returns:
            bool: True if mask quality is acceptable
        """
        try:
            import torch
            import numpy as np

            # Convert to numpy if it's a tensor
            if isinstance(mask_logits, torch.Tensor):
                mask_np = mask_logits.cpu().numpy()
            else:
                mask_np = mask_logits

            # Handle different dimensionalities
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            # Convert logits to probabilities
            if mask_np.dtype != bool:
                # Apply sigmoid to convert logits to probabilities
                mask_probs = 1 / (1 + np.exp(-mask_np))
                binary_mask = mask_probs > 0.5
                max_confidence = np.max(mask_probs)
            else:
                binary_mask = mask_np
                max_confidence = 1.0  # Boolean masks are considered fully confident

            # Check area criterion
            mask_area = np.count_nonzero(binary_mask)
            area_ok = mask_area >= min_area

            # Check confidence criterion
            confidence_ok = max_confidence >= min_confidence

            # Additional quality checks
            # Check if mask is not just noise (has some structure)
            if mask_area > 0:
                # Calculate mask compactness (area vs perimeter ratio)
                # More compact objects are less likely to be noise
                try:
                    import cv2
                    contours, _ = cv2.findContours(
                        binary_mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        perimeter = cv2.arcLength(largest_contour, True)

                        if perimeter > 0:
                            compactness = 4 * np.pi * area / (perimeter * perimeter)
                            # Compactness check for cyst-like circular shapes
                            structure_ok = compactness > 0.15  # Stricter compactness requirement
                        else:
                            structure_ok = True
                    else:
                        structure_ok = False
                except:
                    structure_ok = True  # If CV2 fails, don't filter based on structure
            else:
                structure_ok = False

            result = area_ok and confidence_ok and structure_ok

            if not result:
                try:
                    print(f"      Quality check: area={mask_area}>={min_area}? {area_ok}, "
                          f"confidence={max_confidence:.3f}>={min_confidence}? {confidence_ok}, "
                          f"structure_ok? {structure_ok}")
                    if 'compactness' in locals():
                        print(f"      Structure details: compactness={compactness:.3f}>0.15? {structure_ok}")
                except:
                    print(f"      Quality check: area={area_ok}, confidence={confidence_ok}, structure={structure_ok}")

            return result

        except Exception as e:
            print(f"      Warning: Error in mask quality check: {e}")
            return True  # If quality check fails, don't filter (be conservative)

    def get_frame_mask(self, frame_idx: int, obj_id: int = 1, video_segments: Optional[Dict] = None) -> np.ndarray:
        """Get binary mask for a specific frame and object"""
        if video_segments is None:
            raise ValueError("No tracking results provided")

        if frame_idx not in video_segments or obj_id not in video_segments[frame_idx]:
            # Return empty mask
            h, w = self.video_frames[frame_idx].shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # Get mask from results
        mask = video_segments[frame_idx][obj_id].cpu().numpy()
        if mask.ndim > 2:
            mask = mask.squeeze()

        # Convert to binary
        mask_binary = (mask > 0.5).astype(np.uint8)

        # Resize to original frame size if needed
        h, w = self.video_frames[frame_idx].shape[:2]
        if mask_binary.shape != (h, w):
            mask_binary = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        return mask_binary

    def get_frame_overlay(self, frame_idx: int, obj_id: int = 1, video_segments: Optional[Dict] = None,
                         color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.3) -> np.ndarray:
        """Get frame with segmentation overlay"""
        frame = self.video_frames[frame_idx].copy()
        mask = self.get_frame_mask(frame_idx, obj_id, video_segments)

        if np.any(mask > 0):
            frame[mask > 0] = frame[mask > 0] * (1 - alpha) + np.array(color) * alpha

        return frame.astype(np.uint8)

    def clear_prompts(self, obj_id: Optional[int] = None) -> bool:
        """Clear prompts for a specific object or all objects"""
        try:
            if obj_id is None:
                # Clear all prompts - full reset needed
                self.prompts.clear()

                # Reinitialize inference state for full clear
                if self.current_video_path is not None:
                    try:
                        self.inference_state = self.predictor.init_state(
                            video_path=self.current_video_path,
                            offload_video_to_cpu=False,
                            offload_state_to_cpu=False
                        )
                    except Exception as e:
                        print(f"Failed to reinitialize state in clear_prompts: {e}")
                        # Try simpler version
                        self.inference_state = self.predictor.init_state(video_path=str(self.current_video_path))

                print("🧹 All prompts cleared")
            else:
                # Clear prompts for specific object only - DO NOT reinitialize state
                if obj_id in self.prompts:
                    del self.prompts[obj_id]

                    # Re-add all remaining prompts to the existing inference state
                    if self.inference_state is not None:
                        try:
                            # Reset inference state and re-add all remaining prompts
                            self.inference_state = self.predictor.init_state(
                                video_path=self.current_video_path,
                                offload_video_to_cpu=False,
                                offload_state_to_cpu=False
                            )

                            # Re-add all remaining prompts
                            for remaining_obj_id, prompts_list in self.prompts.items():
                                for prompt in prompts_list:
                                    if prompt['type'] == 'bbox':
                                        # Extract bbox coordinates from prompt structure
                                        x1, y1, x2, y2 = prompt['x1'], prompt['y1'], prompt['x2'], prompt['y2']
                                        frame_idx = prompt.get('frame_idx', 0)

                                        # Use torch tensor for consistency
                                        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

                                        self.predictor.add_new_points_or_box(
                                            inference_state=self.inference_state,
                                            frame_idx=frame_idx,
                                            obj_id=remaining_obj_id,
                                            box=bbox,
                                            clear_old_points=True
                                        )
                                    elif prompt['type'] == 'click':
                                        # Extract click coordinates from prompt structure
                                        point = prompt['point']
                                        label = prompt.get('label', 1)
                                        frame_idx = prompt.get('frame_idx', 0)

                                        self.predictor.add_new_points_or_box(
                                            inference_state=self.inference_state,
                                            frame_idx=frame_idx,
                                            obj_id=remaining_obj_id,
                                            points=np.array([point], dtype=np.float32),
                                            labels=np.array([label], dtype=np.int32)
                                        )
                        except Exception as e:
                            print(f"Warning: Failed to re-add remaining prompts after clearing object {obj_id}: {e}")

                print(f"🧹 Prompts cleared for object {obj_id}")

            return True
        except Exception as e:
            print(f"Error clearing prompts: {e}")
            return False

    def get_prompt_count(self, obj_id: Optional[int] = None) -> int:
        """Get the number of prompts for a specific object or all objects"""
        if obj_id is None:
            # Count all prompts across all objects
            total_count = 0
            for prompts_list in self.prompts.values():
                total_count += len(prompts_list)
            return total_count
        else:
            # Count prompts for specific object
            return len(self.prompts.get(obj_id, []))
    
    def _update_adaptive_bbox(self, obj_id: int, mask, frame_idx: int):
        """Update adaptive bounding box based on current mask"""
        try:
            if obj_id not in self.adaptive_trackers:
                return
            
            adaptive_tracker = self.adaptive_trackers[obj_id]
            
            # Convert tensor to numpy if needed
            if hasattr(mask, 'cpu'):
                # PyTorch tensor
                mask_np = mask.cpu().numpy()
            else:
                # Already numpy array
                mask_np = mask
            
            # Ensure 2D array
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            
            # Convert to binary if needed
            if mask_np.dtype != bool and mask_np.dtype != np.uint8:
                mask_np = (mask_np > 0.5).astype(np.uint8)
            elif mask_np.dtype == bool:
                mask_np = mask_np.astype(np.uint8)
            
            # Calculate confidence from mask quality
            mask_area = int(np.sum(mask_np > 0))
            confidence = min(1.0, mask_area / 200.0)  # Rough confidence estimation
            
            # Update the adaptive tracker (pass numpy array)
            updated_bbox = adaptive_tracker.update_bbox(mask_np, frame_idx, confidence)
            
            # Check if bbox was significantly updated
            decision = adaptive_tracker.adaptation_decisions[-1] if adaptive_tracker.adaptation_decisions else None
            if decision and decision.should_update:
                # FRAME BOUNDS CHECK: Ensure next frame exists before applying update
                next_frame_idx = frame_idx + 1
                if hasattr(self, 'video_frames') and next_frame_idx < len(self.video_frames):
                    # Apply updated bbox as new prompt for next frame
                    self._apply_adaptive_bbox_update(obj_id, updated_bbox, next_frame_idx)
                elif self.adaptive_config.get('enable_logging', False):
                    print(f"   ⚠️  Skipping bbox update for object {obj_id}: next frame {next_frame_idx} is beyond video end ({len(self.video_frames) if hasattr(self, 'video_frames') else 'unknown'} frames)")
                
        except Exception as e:
            print(f"Warning: Error in adaptive bbox update for object {obj_id}: {e}")
            if self.adaptive_config.get('enable_logging', False):
                import traceback
                print(f"   Debug: mask type: {type(mask)}, shape: {getattr(mask, 'shape', 'N/A')}")
                traceback.print_exc()
    
    def _handle_low_quality_tracking(self, obj_id: int, frame_idx: int):
        """Handle low quality tracking by expanding search region"""
        try:
            if obj_id not in self.adaptive_trackers:
                return
            
            adaptive_tracker = self.adaptive_trackers[obj_id]
            
            # Create empty mask to trigger expansion (use frame size if available)
            if hasattr(self, 'video_frames') and self.video_frames:
                h, w = self.video_frames[0].shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                empty_mask = np.zeros((1024, 1024), dtype=np.uint8)  # Default SAM2 size
            
            # Force update with very low confidence (triggers conservative fallback)
            updated_bbox = adaptive_tracker.update_bbox(empty_mask, frame_idx, confidence=0.1)
            
            # FRAME BOUNDS CHECK: Ensure next frame exists before applying recovery update
            next_frame_idx = frame_idx + 1
            if hasattr(self, 'video_frames') and next_frame_idx < len(self.video_frames):
                # Apply expanded bbox for recovery
                self._apply_adaptive_bbox_update(obj_id, updated_bbox, next_frame_idx)
            elif self.adaptive_config.get('enable_logging', False):
                print(f"   ⚠️  Skipping recovery bbox update for object {obj_id}: next frame {next_frame_idx} is beyond video end ({len(self.video_frames) if hasattr(self, 'video_frames') else 'unknown'} frames)")
            
        except Exception as e:
            print(f"Warning: Error in low quality tracking handling for object {obj_id}: {e}")
    
    def _apply_adaptive_bbox_update(self, obj_id: int, new_bbox: Tuple[int, int, int, int], frame_idx: int):
        """Apply updated bounding box as new prompt for future tracking"""
        try:
            if not hasattr(self.predictor, 'add_new_points_or_box'):
                return
            
            x1, y1, x2, y2 = new_bbox
            
            # COMPREHENSIVE BBOX SANITY CHECKS
            # Check for reasonable dimensions
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                if self.adaptive_config.get('enable_logging', False):
                    print(f"   ❌ Invalid bbox dimensions for object {obj_id}: {width}x{height} (bbox: {new_bbox})")
                return
            
            # Check for astronomically large values (likely mathematical error)
            MAX_REASONABLE_SIZE = 5000  # 5000 pixels max for any dimension
            MAX_REASONABLE_AREA = 2000000  # 2M pixel area max
            
            if width > MAX_REASONABLE_SIZE or height > MAX_REASONABLE_SIZE:
                if self.adaptive_config.get('enable_logging', False):
                    print(f"   ❌ Bbox too large for object {obj_id}: {width}x{height} exceeds {MAX_REASONABLE_SIZE}px limit")
                return
            
            if width * height > MAX_REASONABLE_AREA:
                if self.adaptive_config.get('enable_logging', False):
                    print(f"   ❌ Bbox area too large for object {obj_id}: {width * height}px² exceeds {MAX_REASONABLE_AREA}px² limit")
                return
            
            # Check for negative coordinates (should be handled but double-check)
            if x1 < 0 or y1 < 0:
                if self.adaptive_config.get('enable_logging', False):
                    print(f"   ⚠️  Negative coordinates for object {obj_id}: ({x1}, {y1})")
                # Clamp to 0
                x1 = max(0, x1)
                y1 = max(0, y1)
            
            # Check against video frame dimensions if available
            if hasattr(self, 'video_frames') and self.video_frames:
                frame_height, frame_width = self.video_frames[0].shape[:2]
                
                if x2 > frame_width or y2 > frame_height:
                    if self.adaptive_config.get('enable_logging', False):
                        print(f"   ⚠️  Bbox extends beyond frame for object {obj_id}: bbox({x1},{y1},{x2},{y2}) vs frame({frame_width}x{frame_height})")
                    # Clamp to frame boundaries
                    x2 = min(x2, frame_width)
                    y2 = min(y2, frame_height)
                    
                    # Recheck dimensions after clamping
                    width = x2 - x1
                    height = y2 - y1
                    if width <= 0 or height <= 0:
                        if self.adaptive_config.get('enable_logging', False):
                            print(f"   ❌ Bbox became invalid after frame clamping for object {obj_id}")
                        return
            
            # Update bbox values after validation/clamping
            new_bbox = (x1, y1, x2, y2)
            
            if self.adaptive_config.get('enable_logging', False):
                print(f"   ✅ Applying validated bbox for object {obj_id}: {new_bbox} ({width}x{height}px)")
            
            # Create bbox tensor properly
            import torch
            box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=self.device)
            
            # Apply device compatibility fixes
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box
            )
            
            # Update stored prompts
            if obj_id not in self.prompts:
                self.prompts[obj_id] = []
            
            self.prompts[obj_id].append({
                'frame_idx': frame_idx,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'type': 'adaptive_bbox'
            })
            
        except Exception as e:
            print(f"Warning: Error applying adaptive bbox update for object {obj_id}: {e}")
    
    def get_adaptive_tracking_stats(self) -> Dict[int, Dict]:
        """Get adaptive tracking statistics for all objects"""
        stats = {}
        for obj_id, tracker in self.adaptive_trackers.items():
            stats[obj_id] = tracker.get_tracking_stats()
        return stats