#!/usr/bin/env python3
"""
SAM2 Video Tracker
Implementation of BaseVideoTracker for the SAM 2 and SAM 2.1 model families.

Frame contract
--------------
Every frame index that enters or leaves this module is a *chronological* index into
``self.video_frames``, the unique frames of the video in acquisition order (consecutive
near-identical frames produced by video generation are collapsed into one time point).
With reverse tracking enabled, the user annotates the last chronological frame; prompts
are placed on that frame and propagation runs backwards through time using the
predictor's native reverse mode. Results are keyed by chronological frame index in both
directions, so the output and analysis code never reorder anything.
"""

import hashlib
import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..paths import checkpoints_dir
from .base_model import BaseVideoTracker, ModelCapabilities, ModelMetadata
from .masks import PackedMask
from .tracking_result import TrackingResult

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Hydra configs owned by the application (variants of the vendored SAM2 configs).
APP_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

try:
    from sam2.build_sam import build_sam2_video_predictor  # noqa: F401  availability probe

    from .sam2_frames import build_frame_predictor

    SAM2_AVAILABLE = True
    logger.info("SAM2 package imported")
except ImportError as e:  # pragma: no cover - environment dependent
    logger.warning(f"SAM2 not available: {e}")
    SAM2_AVAILABLE = False

    def build_frame_predictor(*args, **kwargs):
        raise ImportError("SAM2 not available")


# Checkpoint families: "2.1" is the default; "2" is the original SAM 2 release used for the
# paper's figures. Both are served by the vendored code with their own config directory.
CHECKPOINT_FAMILIES = {
    "2.1": {"prefix": "sam2.1", "config_dir": "sam2.1"},
    "2": {"prefix": "sam2", "config_dir": "sam2"},
}
# model_config -> (config size tag, checkpoint size name)
SIZE_NAMES = {
    "sam2_hiera_t": ("t", "tiny"),
    "sam2_hiera_s": ("s", "small"),
    "sam2_hiera_b": ("b+", "base_plus"),
    "sam2_hiera_l": ("l", "large"),
}
DEFAULT_MODEL_CONFIG = "sam2_hiera_s"


def checkpoint_filename(model_config: str, family: str = "2.1") -> str:
    """Checkpoint file name for a model size and checkpoint family."""
    fam = CHECKPOINT_FAMILIES[str(family)]
    size = SIZE_NAMES.get(model_config, SIZE_NAMES[DEFAULT_MODEL_CONFIG])
    return f"{fam['prefix']}_hiera_{size[1]}.pt"


def config_filename(model_config: str, family: str = "2.1", improved: bool = False) -> str:
    """Hydra config name for a model size and family.

    Standard configs are composed from the vendored ``sam2`` package under the names upstream
    uses (``configs/<family>/<name>.yaml``); the application's own variants are composed from
    ``APP_CONFIG_DIR``. ``hydra_search_path`` picks the matching Hydra context.
    """
    fam = CHECKPOINT_FAMILIES[str(family)]
    size = SIZE_NAMES.get(model_config, SIZE_NAMES[DEFAULT_MODEL_CONFIG])
    if improved and str(family) == "2.1" and model_config == "sam2_hiera_b":
        return "sam2.1_hiera_b+_improved_tracking.yaml"
    return f"configs/{fam['config_dir']}/{fam['prefix']}_hiera_{size[0]}.yaml"


def hydra_search_path(config_file: str):
    """Hydra initialization context in which ``config_file`` can be composed.

    The vendored ``sam2`` package initializes the global Hydra instance when it is imported;
    callers clear it before entering this context.
    """
    from hydra import initialize_config_dir, initialize_config_module

    if (APP_CONFIG_DIR / config_file).is_file():
        return initialize_config_dir(config_dir=str(APP_CONFIG_DIR), version_base="1.2")
    return initialize_config_module("sam2", version_base="1.2")


def _app_setting(name: str, default):
    """Read a constant from the application's config module, falling back to a default."""
    try:
        from .. import config as app_config

        return getattr(app_config, name, default)
    except Exception:
        return default


def _sha256_file(path: str, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_to_binary(mask) -> np.ndarray:
    """Binary uint8 (H, W) from a PackedMask, a bool/uint8 array, or float logits (> 0)."""
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = arr.squeeze()
    return (arr > 0).astype(np.uint8)


class SAM2Tracker(BaseVideoTracker):
    """
    SAM2 Video Object Tracker
    Implements BaseVideoTracker for the SAM 2 and SAM 2.1 model families
    """

    def __init__(
        self,
        model_config="sam2_hiera_b",
        checkpoint_path=None,
        device="cuda",
        enable_reverse_tracking=True,
        checkpoint_family=None,
        **kwargs,
    ):
        """
        Initialize the SAM2 tracker

        Args:
            model_config: Model size ('sam2_hiera_t', 'sam2_hiera_s', 'sam2_hiera_b', 'sam2_hiera_l')
            checkpoint_path: Path to model checkpoint (None for auto-detection)
            device: Device to run inference on ('cuda' or 'cpu'); falls back to CPU without CUDA
            enable_reverse_tracking: Annotate the last frame and track backwards in time
            checkpoint_family: '2.1' (default) or '2' (original SAM 2 checkpoints)
        """
        super().__init__("SAM2", **kwargs)

        self.model_config = model_config
        self.enable_reverse_tracking = bool(enable_reverse_tracking)
        self.checkpoint_family = str(checkpoint_family or _app_setting("SAM2_CHECKPOINT_FAMILY", "2.1"))
        if self.checkpoint_family not in CHECKPOINT_FAMILIES:
            raise ValueError(
                f"Unknown checkpoint family {self.checkpoint_family!r}; use one of {sorted(CHECKPOINT_FAMILIES)}"
            )

        # Resolve the device once; every later operation uses the same resolved device.
        requested = str(device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            requested = "cpu"
        self.device = torch.device(requested)
        self.device_name = requested

        if checkpoint_path is None:
            checkpoint_path = str(checkpoints_dir() / checkpoint_filename(model_config, self.checkpoint_family))
            logger.debug(f"Auto-selected SAM2 checkpoint: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256: str | None = None

        # Duplicate-frame collapse (video generation can repeat frames)
        self.collapse_duplicates = bool(_app_setting("COLLAPSE_DUPLICATE_FRAMES", True))
        self.duplicate_threshold = float(_app_setting("DUPLICATE_FRAME_MAD_THRESHOLD", 1.0))

        # Model and video state
        self.predictor = None
        self.inference_state = None
        self.video_frames: list[np.ndarray] | None = None  # unique frames, chronological
        self.original_frames: list[np.ndarray] | None = None  # same list; kept for consumers
        self.frame_map: list[int] = []  # unique index -> decoded index
        self.decoded_frame_count = 0
        self.duplicate_frames_removed = 0
        self.annotation_frame_index = 0
        self.video_sha256: str | None = None
        self.current_video_path: str | None = None
        self.is_reversed_video = self.enable_reverse_tracking
        self.debug_mode = False

    # ------------------------------------------------------------------ metadata
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
                ModelCapabilities.REAL_TIME,
            ],
            requirements=["PyTorch", "SAM2 package", "CUDA (recommended)"],
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if SAM2 is available"""
        return SAM2_AVAILABLE

    # ------------------------------------------------------------------ model
    def load_model(self, **kwargs) -> bool:
        """Load and initialize the SAM2 model"""
        if not SAM2_AVAILABLE:
            logger.error("SAM2 not available. Please install sam2 package.")
            return False

        try:
            from hydra.core.global_hydra import GlobalHydra

            if not Path(self.checkpoint_path).is_file():
                logger.error(
                    f"Checkpoint not found: {self.checkpoint_path}. "
                    f"Download it with: bash checkpoints/download_ckpts.sh {self.checkpoint_family}"
                )
                return False

            improved = bool(_app_setting("SAM2_USE_IMPROVED_CONFIG", False))
            config_file = config_filename(self.model_config, self.checkpoint_family, improved)

            logger.info("Loading SAM2 model...")
            logger.debug(
                f"Family: SAM {self.checkpoint_family}   Size: {self.model_config}   Device: {self.device_name}"
            )
            logger.debug(f"Config: {config_file}")
            logger.debug(f"Checkpoint: {self.checkpoint_path}")

            GlobalHydra.instance().clear()
            with hydra_search_path(config_file):
                self.predictor = build_frame_predictor(config_file, self.checkpoint_path, self.device_name)

            self.checkpoint_sha256 = _sha256_file(self.checkpoint_path)
            self.is_loaded = True
            self.is_initialized = True
            logger.info("SAM2 model loaded successfully!")
            return True
        except Exception as e:
            import traceback

            logger.error(f"Error loading SAM2: {str(e)}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return False

    # ------------------------------------------------------------------ video
    def _unique_frame_indices(self, frames: list[np.ndarray]) -> list[int]:
        """Indices of frames that differ from their predecessor (mean absolute grayscale difference)."""
        keep = [0]
        previous = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY).astype(np.int16)
        for index in range(1, len(frames)):
            current = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY).astype(np.int16)
            if float(np.abs(current - previous).mean()) >= self.duplicate_threshold:
                keep.append(index)
            previous = current
        return keep

    def load_video(self, video_path: str, max_frames: int | None = None) -> dict[str, Any]:
        """Decode the video, collapse duplicated frames, and initialize the predictor state."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()

        if not frames:
            raise ValueError(f"No frames could be decoded from: {video_path}")

        decoded = len(frames)
        keep = self._unique_frame_indices(frames) if self.collapse_duplicates else list(range(decoded))

        self.frame_map = keep
        self.video_frames = [frames[i] for i in keep]
        self.original_frames = self.video_frames
        self.decoded_frame_count = decoded
        self.duplicate_frames_removed = decoded - len(keep)
        self.video_path = video_path
        self.current_video_path = video_path
        self.video_sha256 = _sha256_file(video_path)
        self.prompts = {}

        n = len(self.video_frames)
        self.annotation_frame_index = n - 1 if self.enable_reverse_tracking else 0

        if self.duplicate_frames_removed:
            removed = [i for i in range(decoded) if i not in set(keep)]
            logger.info(
                f"Collapsed {self.duplicate_frames_removed} duplicated frame(s) {removed}: {decoded} decoded -> {n} unique time points"
            )
        direction = "reverse" if self.enable_reverse_tracking else "forward"
        logger.info(
            f"Tracking direction: {direction}; annotation frame = chronological frame {self.annotation_frame_index}"
        )

        logger.info(f"Initializing inference state on device: {self.device}")
        self.inference_state = self.predictor.init_state_from_frames(
            self.video_frames, offload_video_to_cpu=False, offload_state_to_cpu=False
        )
        logger.info(f"Loaded {n} unique frames ({decoded} decoded) at nominal {self.fps} FPS")

        return {
            "num_frames": n,
            "decoded_frames": decoded,
            "duplicate_frames_removed": self.duplicate_frames_removed,
            "frame_map": list(keep),
            "fps": self.fps,
            "dimensions": self.video_frames[0].shape[:2],
            "total_frames": decoded,
            "reported_frame_count": reported_frame_count,
            "direction": direction,
            "annotation_frame_index": self.annotation_frame_index,
        }

    def get_annotation_frame(self) -> np.ndarray:
        """The frame the user annotates: the last chronological frame in reverse mode."""
        if self.video_frames is None:
            raise ValueError("No video loaded. Call load_video() first.")
        return self.video_frames[self.annotation_frame_index].copy()

    def get_first_frame(self) -> np.ndarray:
        """Legacy name used by the GUI for the annotation frame."""
        return self.get_annotation_frame()

    def _to_chronological(self, display_frame_idx: int) -> int:
        """Map a display index (0 = annotation frame) to a chronological frame index."""
        n = len(self.video_frames) if self.video_frames else 0
        if not 0 <= display_frame_idx < max(n, 1):
            raise ValueError(f"frame index {display_frame_idx} out of range for {n} frames")
        return n - 1 - display_frame_idx if self.enable_reverse_tracking else display_frame_idx

    # ------------------------------------------------------------------ prompts
    def add_click_prompt(self, x: int, y: int, label: int = 1, obj_id: int = 1, frame_idx: int = 0) -> bool:
        """Add a click prompt (frame_idx is a display index; 0 is the annotation frame)."""
        try:
            chronological = self._to_chronological(frame_idx)
            logger.info(f"Adding click prompt at ({x}, {y}) with label {label} on chronological frame {chronological}")
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=chronological,
                obj_id=obj_id,
                points=np.array([[x, y]], dtype=np.float32),
                labels=np.array([label], dtype=np.int32),
                clear_old_points=False,
            )
            self.prompts.setdefault(obj_id, []).append(
                {
                    "frame_idx": chronological,
                    "display_frame_idx": frame_idx,
                    "x": x,
                    "y": y,
                    "label": label,
                    "type": "click",
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error adding click prompt: {str(e)}")
            return False

    def add_bbox_prompt(self, x1: int, y1: int, x2: int, y2: int, obj_id: int = 1, frame_idx: int = 0) -> bool:
        """Add a bounding box prompt (frame_idx is a display index; 0 is the annotation frame)."""
        if self.predictor is None or self.inference_state is None:
            logger.warning("No model or video loaded yet")
            return False
        try:
            chronological = self._to_chronological(frame_idx)
            if self.debug_mode:
                logger.info(
                    f"Adding bbox prompt ({x1}, {y1})-({x2}, {y2}) for object {obj_id} on chronological frame {chronological}"
                )
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=chronological,
                obj_id=obj_id,
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                clear_old_points=True,
            )
            self.prompts.setdefault(obj_id, []).append(
                {
                    "frame_idx": chronological,
                    "display_frame_idx": frame_idx,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "type": "bbox",
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error adding bbox prompt: {str(e)}")
            return False

    # ------------------------------------------------------------------ tracking
    def run_tracking(self, progress_callback: Callable[..., None] | None = None) -> TrackingResult:
        """Propagate the prompts through the video and return masks keyed by chronological frame."""
        logger.info("Running object tracking...")
        if not self.prompts:
            raise ValueError("No prompts added. Add click or bbox prompts first.")
        if self.inference_state is None or not self.video_frames:
            raise RuntimeError("No video loaded. Call load_video() first.")

        total_frames = len(self.video_frames)
        result = TrackingResult(
            status=TrackingResult.COMPLETED,
            frames_total=total_frames,
            frames_done=0,
            direction="reverse" if self.enable_reverse_tracking else "forward",
            annotation_frame=self.annotation_frame_index,
            frame_map=self.frame_map,
        )

        enable_quality_filtering = bool(_app_setting("SAM2_IMPROVED_TRACKING", True))
        min_mask_area = int(_app_setting("SAM2_MIN_MASK_AREA", 50))
        min_confidence = float(_app_setting("SAM2_MIN_CONFIDENCE", 0.3))
        memory_frames = int(_app_setting("SAM2_MEMORY_FRAMES", 2))

        logger.debug(f"Quality filtering: {'enabled' if enable_quality_filtering else 'disabled'}")
        if enable_quality_filtering:
            logger.debug(f"Thresholds: min_area={min_mask_area}px, min_confidence={min_confidence:.1f}")
        if memory_frames >= 1000:
            logger.debug("Memory dependence: ALL previous frames (unlimited)")
        else:
            logger.debug(f"Memory dependence: {memory_frames} previous frames")

        processed_frames: list[int] = []
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=self.annotation_frame_index,
                reverse=self.enable_reverse_tracking,
            ):
                frame_idx = int(out_frame_idx)
                frame_masks = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask_logits = out_mask_logits[i]
                    if enable_quality_filtering:
                        cleaned = self._clean_mask_to_largest_component(mask_logits)
                        if cleaned is not None and self._is_mask_quality_acceptable(
                            cleaned, min_mask_area, min_confidence
                        ):
                            frame_masks[out_obj_id] = PackedMask.from_logits(cleaned)
                        else:
                            logger.debug(f"Frame {frame_idx}, Object {out_obj_id}: Low quality mask filtered out")
                    else:
                        frame_masks[out_obj_id] = PackedMask.from_logits(mask_logits)

                if frame_masks:
                    result[frame_idx] = frame_masks
                result.presence[frame_idx] = self._presence_scores(frame_idx, out_obj_ids)

                processed_frames.append(frame_idx)
                if len(processed_frames) > memory_frames:
                    self._clear_old_frame_memory(processed_frames.pop(0))

                result.frames_done += 1
                if progress_callback:
                    progress_callback(
                        result.frames_done, total_frames, f"Processing frame {result.frames_done}/{total_frames}"
                    )
        except Exception as e:
            result.status = TrackingResult.PARTIAL
            result.error = f"{type(e).__name__}: {e}"
            logger.warning(f"Tracking stopped after {result.frames_done}/{total_frames} frames: {result.error}")

        if not result:
            if result.error:
                raise RuntimeError(f"Tracking failed before producing any masks: {result.error}")
            raise RuntimeError("No tracking results obtained. Check model compatibility.")

        logger.info(f"Tracking {result.summary()}")
        return result

    def _presence_scores(self, frame_idx: int, obj_ids) -> dict[int, float]:
        """SAM2's object presence logits for a frame (positive means the object is present)."""
        scores: dict[int, float] = {}
        try:
            state = self.inference_state
            for obj_id in obj_ids:
                obj_idx = state["obj_id_to_idx"][obj_id]
                outputs = state["output_dict_per_obj"][obj_idx]
                out = outputs["cond_frame_outputs"].get(frame_idx) or outputs["non_cond_frame_outputs"].get(frame_idx)
                if out is not None and out.get("object_score_logits") is not None:
                    scores[int(obj_id)] = float(out["object_score_logits"].reshape(-1)[0].item())
        except Exception:
            pass
        return scores

    def _clean_mask_to_largest_component(self, mask_logits) -> np.ndarray | None:
        """
        Keep only the largest connected component of a mask so each cyst is one object.

        Returns the cleaned logits as a float32 (H, W) array (other components set to -20),
        or None when the mask has no foreground.
        """
        try:
            if isinstance(mask_logits, torch.Tensor):
                mask_np = mask_logits.detach().float().cpu().numpy()
            else:
                mask_np = np.asarray(mask_logits, dtype=np.float32)
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            binary_mask = mask_np > 0  # sigmoid(logit) > 0.5
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_mask.astype(np.uint8), connectivity=8
            )
            if num_labels <= 1:
                return None

            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_component_idx = int(np.argmax(component_areas)) + 1
            largest_component_mask = labels == largest_component_idx
            cleaned = np.where(largest_component_mask, mask_np, np.float32(-20.0)).astype(np.float32)

            removed_components = num_labels - 2
            if removed_components > 0:
                original_area = int(np.count_nonzero(binary_mask))
                cleaned_area = int(np.count_nonzero(largest_component_mask))
                logger.debug(
                    f"Cleaned mask: kept largest component ({cleaned_area}px), "
                    f"removed {removed_components} smaller components ({original_area - cleaned_area}px)"
                )
            return cleaned
        except Exception as e:
            logger.warning(f"Error cleaning mask: {e}")
            if isinstance(mask_logits, torch.Tensor):
                return mask_logits.detach().float().cpu().numpy().squeeze()
            return np.asarray(mask_logits, dtype=np.float32).squeeze()

    def _clear_old_frame_memory(self, frame_idx: int):
        """Drop an old frame's non-conditioning outputs to limit memory dependence."""
        try:
            if self.inference_state is not None:
                for obj_idx in self.inference_state.get("output_dict_per_obj", {}):
                    obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                    if "non_cond_frame_outputs" in obj_output_dict:
                        obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
                    temp_output_dict_per_obj = self.inference_state.get("temp_output_dict_per_obj", {})
                    if obj_idx in temp_output_dict_per_obj:
                        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)
                logger.debug(f"Cleared memory for frame {frame_idx} (reducing memory dependence)")
        except Exception as e:
            logger.warning(f"Error clearing frame memory for frame {frame_idx}: {e}")

    def _is_mask_quality_acceptable(self, mask_logits, min_area: int = 50, min_confidence: float = 0.3) -> bool:
        """Area, confidence and compactness checks on a logits array (or tensor)."""
        try:
            if isinstance(mask_logits, torch.Tensor):
                mask_np = mask_logits.detach().float().cpu().numpy()
            else:
                mask_np = np.asarray(mask_logits)
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            if mask_np.dtype != bool:
                mask_probs = 1 / (1 + np.exp(-mask_np.astype(np.float32)))
                binary_mask = mask_probs > 0.5
                max_confidence = float(np.max(mask_probs))
            else:
                binary_mask = mask_np
                max_confidence = 1.0

            mask_area = int(np.count_nonzero(binary_mask))
            area_ok = mask_area >= min_area
            confidence_ok = max_confidence >= min_confidence

            structure_ok = False
            compactness = None
            if mask_area > 0:
                try:
                    contours, _ = cv2.findContours(
                        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        perimeter = cv2.arcLength(largest_contour, True)
                        if perimeter > 0:
                            compactness = 4 * np.pi * area / (perimeter * perimeter)
                            structure_ok = compactness > 0.15
                        else:
                            structure_ok = True
                except Exception:
                    structure_ok = True

            result = area_ok and confidence_ok and structure_ok
            if not result:
                detail = f", compactness={compactness:.3f}>0.15? {structure_ok}" if compactness is not None else ""
                logger.debug(
                    f"Quality check: area={mask_area}>={min_area}? {area_ok}, "
                    f"confidence={max_confidence:.3f}>={min_confidence}? {confidence_ok}{detail}"
                )
            return result
        except Exception as e:
            logger.warning(f"Error in mask quality check: {e}")
            return True

    # ------------------------------------------------------------------ results access
    def get_frame_mask(self, frame_idx: int, obj_id: int = 1, video_segments: dict | None = None) -> np.ndarray:
        """Binary mask (uint8, frame resolution) for a chronological frame and object."""
        if video_segments is None:
            raise ValueError("No tracking results provided")
        h, w = self.video_frames[frame_idx].shape[:2]
        if frame_idx not in video_segments or obj_id not in video_segments[frame_idx]:
            return np.zeros((h, w), dtype=np.uint8)
        mask_binary = _mask_to_binary(video_segments[frame_idx][obj_id])
        if mask_binary.shape != (h, w):
            mask_binary = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask_binary

    def get_frame_overlay(
        self,
        frame_idx: int,
        obj_id: int = 1,
        video_segments: dict | None = None,
        color: tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.3,
    ) -> np.ndarray:
        """Frame with the object's mask blended in."""
        frame = self.video_frames[frame_idx].copy()
        mask = self.get_frame_mask(frame_idx, obj_id, video_segments)
        if np.any(mask > 0):
            frame[mask > 0] = frame[mask > 0] * (1 - alpha) + np.array(color) * alpha
        return frame.astype(np.uint8)

    # ------------------------------------------------------------------ prompt management
    def _reset_inference_state(self):
        if self.video_frames:
            self.inference_state = self.predictor.init_state_from_frames(
                self.video_frames, offload_video_to_cpu=False, offload_state_to_cpu=False
            )

    def clear_prompts(self, obj_id: int | None = None) -> bool:
        """Clear all prompts, or one object's prompts and re-apply the others."""
        try:
            if obj_id is None:
                self.prompts.clear()
                self._reset_inference_state()
                logger.info("All prompts cleared")
                return True

            self.prompts.pop(obj_id, None)
            self._reset_inference_state()
            for remaining_obj_id, prompts_list in self.prompts.items():
                for prompt in prompts_list:
                    if prompt["type"] == "bbox":
                        self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=prompt["frame_idx"],
                            obj_id=remaining_obj_id,
                            box=np.array([prompt["x1"], prompt["y1"], prompt["x2"], prompt["y2"]], dtype=np.float32),
                            clear_old_points=True,
                        )
                    elif prompt["type"] == "click":
                        self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=prompt["frame_idx"],
                            obj_id=remaining_obj_id,
                            points=np.array([[prompt["x"], prompt["y"]]], dtype=np.float32),
                            labels=np.array([prompt.get("label", 1)], dtype=np.int32),
                            clear_old_points=False,
                        )
            logger.info(f"Prompts cleared for object {obj_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing prompts: {e}")
            return False

    def get_prompt_count(self, obj_id: int | None = None) -> int:
        """Number of prompts for one object or in total"""
        if obj_id is None:
            return sum(len(prompts_list) for prompts_list in self.prompts.values())
        return len(self.prompts.get(obj_id, []))

    def provenance(self) -> dict[str, Any]:
        """Facts needed to reproduce a run."""
        return {
            "backend": "sam2-vendored",
            "checkpoint_family": self.checkpoint_family,
            "model_config": self.model_config,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "video_path": self.current_video_path,
            "video_sha256": self.video_sha256,
            "decoded_frames": self.decoded_frame_count,
            "unique_frames": len(self.video_frames) if self.video_frames else 0,
            "frame_map": list(self.frame_map),
            "duplicate_threshold": self.duplicate_threshold,
            "direction": "reverse" if self.enable_reverse_tracking else "forward",
            "annotation_frame_index": self.annotation_frame_index,
        }
