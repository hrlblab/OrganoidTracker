#!/usr/bin/env python3
"""
Medical-SAM2 Core Inference Module
Clean API for GUI-based object tracking in videos
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import warnings
import json
from typing import Dict, List, Tuple, Optional, Any

# Import base classes
from .base_model import BaseVideoTracker, ModelCapabilities, ModelMetadata

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.projections")

# Patch SAM2 C++ extension issues
def patch_connected_components():
    """Monkey patch for C++ extension compatibility"""
    try:
        # Use the models_path defined at module level
        if str(models_path) not in sys.path:
            sys.path.insert(0, str(models_path))

        import medical_sam2.sam2_train.utils.misc as misc_module

        def fallback_get_connected_components(mask):
            try:
                from scipy import ndimage
                mask_np = mask.cpu().numpy()
                batch_size = mask_np.shape[0]

                labels_list = []
                areas_list = []

                for i in range(batch_size):
                    labeled_array, num_features = ndimage.label(mask_np[i, 0])
                    areas = np.zeros_like(labeled_array)
                    for label_id in range(1, num_features + 1):
                        component_mask = labeled_array == label_id
                        area = np.sum(component_mask)
                        areas[component_mask] = area

                    labels_list.append(labeled_array)
                    areas_list.append(areas)

                return torch.tensor(np.stack(labels_list)).unsqueeze(1), torch.tensor(np.stack(areas_list)).unsqueeze(1)

            except Exception as e:
                print(f"Fallback connected components failed: {e}")
                return mask, mask

        # Replace the problematic function
        misc_module.get_connected_components = fallback_get_connected_components

    except ImportError:
        pass

# Add path to models directory for imports
models_path = Path(__file__).parent.parent.parent / "models"
if str(models_path) not in sys.path:
    sys.path.insert(0, str(models_path))

# Apply patch
patch_connected_components()

# Import SAM2 after patching
try:
    from medical_sam2.sam2_train.build_sam import build_sam2_video_predictor
except ImportError:
    # Fallback to original path if available
    from sam2_train.build_sam import build_sam2_video_predictor


class MedicalSAM2Tracker(BaseVideoTracker):
    """
    Medical-SAM2 Video Object Tracker
    Implements BaseVideoTracker interface for Medical-SAM2 model
    """

    def __init__(self, model_config='sam2_hiera_b', checkpoint_path=None,
                 device='cuda', **kwargs):
        """
        Initialize the Medical-SAM2 tracker

        Args:
            model_config: SAM2 model configuration
            checkpoint_path: Path to model checkpoint (None for auto-detection)
            device: Device to run inference on
        """
        super().__init__("Medical-SAM2", **kwargs)

        self.model_config = model_config

        # Auto-determine checkpoint path if not provided
        if checkpoint_path is None:
            # Get absolute path to checkpoints directory
            app_root = Path(__file__).parent.parent.parent
            checkpoints_dir = app_root / "checkpoints"

            # For Medical-SAM2, use the specific pretrained checkpoint
            # The Medical-SAM2 uses its own checkpoint regardless of model size config
            checkpoint_path = str(checkpoints_dir / "MedSAM2_pretrain.pth")
            print(f"   Using Medical-SAM2 checkpoint: {checkpoint_path}")

        # For Medical-SAM2, the checkpoint was actually trained with small model dimensions
        # Revert to using the original small configuration
        # self.model_config = 'sam2_hiera_b'  # Use our custom base config
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
            name="medical_sam2",
            display_name="Medical-SAM2",
            description="Medical-optimized SAM2 for precise medical image and video segmentation",
            capabilities=[
                ModelCapabilities.CLICK_PROMPTS,
                ModelCapabilities.BBOX_PROMPTS,
                ModelCapabilities.MULTI_OBJECT,
                ModelCapabilities.MEDICAL_OPTIMIZED
            ],
            requirements=["PyTorch", "CUDA (recommended)"]
        )

    def load_model(self, **kwargs) -> bool:
        """Load and initialize the Medical-SAM2 model"""
        try:
            print(f"🔄 Loading Medical-SAM2 model...")

            # Set up the config path for Medical-SAM2
            # The config files are in models/medical_sam2/sam2_train/
            import os
            from hydra import initialize_config_dir, compose
            from hydra.core.global_hydra import GlobalHydra
            from pathlib import Path

            # Get absolute path to the config directory
            config_dir = models_path / "medical_sam2" / "sam2_train"
            config_dir_abs = str(config_dir.absolute())

            # Add medical_sam2 directory to sys.path so sam2_train can be imported
            medical_sam2_path = models_path / "medical_sam2"
            medical_sam2_path_str = str(medical_sam2_path.absolute())
            if medical_sam2_path_str not in sys.path:
                sys.path.insert(0, medical_sam2_path_str)
                print(f"   Added to Python path: {medical_sam2_path_str}")

            print(f"   Using config directory: {config_dir_abs}")
            print(f"   Config file: {self.model_config}")

            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()

            # Initialize Hydra with the medical_sam2 config directory
            with initialize_config_dir(config_dir=config_dir_abs, version_base=None):
                self.predictor = build_sam2_video_predictor(
                    config_file=self.model_config,
                    ckpt_path=self.checkpoint_path,
                    device=self.device_name,
                    mode="eval"
                )

            self._patch_device_operations()
            self.is_loaded = True
            self.is_initialized = True
            print("✅ Medical-SAM2 model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading Medical-SAM2: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            return False

    def _patch_device_operations(self):
        """Patch hardcoded CUDA operations for device compatibility"""
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

        # PREPROCESSING: Reverse video at pipeline start for backwards tracking
        # Now everything processes normally as if it's a regular video
        original_frames = frames.copy()  # Keep original order for reference
        frames.reverse()
        print(f"🔄 Preprocessed video: reversed for backwards tracking")
        print(f"📊 Video preprocessing: Original {len(original_frames)} frames -> Reversed {len(frames)} frames")

        # Store video data (reversed - but everything treats this as "normal" now)
        self.video_frames = frames
        self.original_frames = original_frames  # Keep original order for debugging
        self.video_path = video_path
        self.is_reversed_video = True  # Flag to remember this is preprocessed

        # Prepare video tensor for SAM2
        self._prepare_video_tensor()

        # Initialize inference state
        print(f"🔍 Initializing Medical-SAM2 inference state on device: {self.device}")
        print(f"   • video_tensor device: {self.video_tensor.device if hasattr(self.video_tensor, 'device') else 'N/A'}")

        self.inference_state = self.predictor.val_init_state(imgs_tensor=self.video_tensor)
        print(f"✅ Medical-SAM2 inference state initialized")

        # Debug: Check inference state device placement
        print(f"🔍 Checking Medical-SAM2 inference state device placement...")
        self._debug_inference_state_devices()

        # CRITICAL FIX: Move inference state to correct device
        print(f"🔧 Moving Medical-SAM2 inference state to device: {self.device}")
        self._move_inference_state_to_device()

        print(f"🔍 After device move - checking Medical-SAM2 devices again...")
        self._debug_inference_state_devices()

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
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"📦 Adding Medical-SAM2 bbox prompt: ({x1}, {y1}) to ({x2}, {y2}) for object {obj_id}")

            # Deep device debugging for Medical-SAM2
            print(f"🔍 MEDICAL-SAM2 DEVICE DEBUG:")
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
            print(f"🔄 Calling Medical-SAM2 add_new_points_or_box...")
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=bbox,  # Use 'box' parameter, not 'bbox'
                clear_old_points=True
            )
            print(f"✅ Medical-SAM2 add_new_points_or_box completed successfully")

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
        print(f"🎯 Number of prompts: {len(self.prompts)}")
        for obj_id, obj_prompts in self.prompts.items():
            print(f"   Object {obj_id}: {len(obj_prompts)} prompts")

        if not self.prompts:
            raise ValueError("No prompts added. Add click or bbox prompts first.")

        video_segments = {}
        total_frames = len(self.video_frames)

        try:
            frame_count = 0
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: out_mask_logits[i]
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

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

    def get_frame_mask(self, frame_idx: int, obj_id: int = 1, video_segments: Optional[Dict] = None) -> np.ndarray:
        """Get binary mask for a specific frame and object"""
        if video_segments is None:
            raise ValueError("No tracking results provided")

        if frame_idx not in video_segments or obj_id not in video_segments[frame_idx]:
            # Return empty mask
            h, w = self.video_frames[frame_idx].shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # Get mask from results
        mask = video_segments[frame_idx][obj_id]

        # Handle tensor conversion
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        elif hasattr(mask, 'numpy'):
            mask = mask.numpy()

        if mask.ndim > 2:
            mask = mask.squeeze()

        # CRITICAL FIX: Convert logits to probabilities using sigmoid
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask)
        else:
            mask_tensor = mask

        # Apply sigmoid to convert logits to probabilities
        mask_prob = torch.sigmoid(mask_tensor).numpy()

        # Convert to binary with proper threshold
        mask_binary = (mask_prob > 0.5).astype(np.uint8)

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
        """Clear all prompts or prompts for a specific object"""
        try:
            if obj_id is not None:
                # Clear prompts for specific object
                if obj_id in self.prompts:
                    del self.prompts[obj_id]
                    print(f"✅ Cleared prompts for object {obj_id}")
                else:
                    print(f"⚠️  No prompts found for object {obj_id}")
            else:
                # Clear all prompts
                self.prompts.clear()
                print("✅ Cleared all prompts")

            # Reset inference state if we have one
            if hasattr(self, 'inference_state') and self.inference_state is not None:
                # Re-initialize inference state to clear any cached data
                self.inference_state = self.predictor.init_state(self.video_tensor.to(self.device))

            return True
        except Exception as e:
            print(f"❌ Error clearing prompts: {str(e)}")
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