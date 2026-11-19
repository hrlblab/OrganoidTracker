"""Optional Transformers SAM 2 backend using the existing frame and result contract.

The reference adapter still owns decode, prompt bookkeeping, quality filtering, memory
limits and output masks. This bridge replaces inference only. Frames use the reference's
PIL resize/normalization so a backend comparison does not also change preprocessing.
Local Hugging Face snapshots are required; loading never downloads or executes remote code.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional

from ..paths import checkpoints_dir
from .base_model import ModelMetadata
from .sam2_frames import frames_to_tensor
from .sam2_tracker import SIZE_NAMES, SAM2Tracker, _app_setting, _sha256_file

logger = logging.getLogger(__name__)


class _PredictorBridge:
    """Translate the small predictor boundary used by ``SAM2Tracker`` to Transformers."""

    def __init__(self, model: Any, processor: Any, device: torch.device) -> None:
        self.model, self.processor, self.device = model, processor, device

    def init_state_from_frames(self, frames, **kwargs) -> dict[str, Any]:
        from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoInferenceSession

        images = frames_to_tensor(frames, self.model.config.image_size, self.device)
        session = Sam2VideoInferenceSession(
            video=cast(Any, images),  # upstream annotates FloatTensor rather than torch.Tensor
            video_height=frames[0].shape[0],
            video_width=frames[0].shape[1],
            inference_device=self.device,
            inference_state_device=self.device,
            video_storage_device=self.device,
            dtype=torch.float32,
        )
        # The history is shared, so the reference adapter's memory limit and presence
        # extraction operate on the actual inference state, not a copied history.
        return {
            "session": session,
            "output_dict_per_obj": session.output_dict_per_obj,
            "obj_id_to_idx": {},
            "obj_ids": [],
        }

    def add_new_points_or_box(
        self, inference_state, frame_idx, obj_id, box=None, points=None, labels=None, clear_old_points=True
    ) -> None:
        session = inference_state["session"]
        pending = list(session.obj_with_new_inputs)
        self.processor.add_inputs_to_inference_session(
            session,
            frame_idx,
            [obj_id],
            input_boxes=None if box is None else [[box.tolist()]],
            input_points=None if points is None else [[points.tolist()]],
            input_labels=None if labels is None else [[labels.tolist()]],
            clear_old_inputs=clear_old_points,
        )
        # Adding each box separately must not forget the earlier pending objects.
        session.obj_with_new_inputs = list(dict.fromkeys(pending + session.obj_with_new_inputs))
        inference_state["obj_ids"] = session.obj_ids
        inference_state["obj_id_to_idx"] = {oid: session.obj_id_to_idx(oid) for oid in session.obj_ids}

    def propagate_in_video(self, inference_state, start_frame_idx, reverse):
        session = inference_state["session"]
        if reverse and start_frame_idx == 0:
            # Transformers' reverse iterator excludes this one-frame boundary.
            outputs = [self.model(session, frame_idx=0, reverse=True)]
        else:
            outputs = self.model.propagate_in_video_iterator(session, start_frame_idx=start_frame_idx, reverse=reverse)
        for output in outputs:
            masks = functional.interpolate(
                output.pred_masks.float(),
                size=(session.video_height, session.video_width),
                mode="bilinear",
                align_corners=False,
            )
            yield output.frame_idx, output.object_ids, masks


class HfSam2Adapter(SAM2Tracker):
    """SAM 2 / 2.1 from a local Transformers snapshot; opt in with backend ``hf-sam2``."""

    def __init__(
        self,
        model_config="sam2_hiera_b",
        checkpoint_path=None,
        device="cuda",
        enable_reverse_tracking=True,
        checkpoint_family=None,
        **kwargs,
    ):
        super().__init__(
            model_config=model_config,
            checkpoint_path=checkpoint_path,
            device=device,
            enable_reverse_tracking=enable_reverse_tracking,
            checkpoint_family=checkpoint_family,
            **kwargs,
        )
        if checkpoint_path is None:
            size = SIZE_NAMES[self.model_config][1].replace("_", "-")
            self.checkpoint_path = str(checkpoints_dir() / "hf" / f"sam{self.checkpoint_family}-hiera-{size}")
        self.snapshot_hashes: dict[str, str] = {}

    @classmethod
    def is_available(cls) -> bool:
        return find_spec("transformers") is not None

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        metadata = super().get_metadata()
        metadata.name = "hf-sam2"
        metadata.display_name = "SAM2 (Transformers, experimental)"
        metadata.description = "Optional SAM 2 inference through a local Hugging Face snapshot"
        metadata.requirements = ["Transformers", "local model snapshot", "CUDA (recommended)"]
        return metadata

    def load_model(self, **kwargs) -> bool:
        try:
            from transformers import Sam2VideoModel, Sam2VideoProcessor

            directory = Path(self.checkpoint_path)
            required = (
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
            )
            if not all((directory / name).is_file() for name in required):
                raise ValueError(f"{directory}: expected a local Hugging Face SAM 2 snapshot ({', '.join(required)})")
            if (
                self.model_config == "sam2_hiera_b"
                and self.checkpoint_family == "2.1"
                and _app_setting("SAM2_USE_IMPROVED_CONFIG", False)
            ):
                raise ValueError("The application-specific improved base-plus config is not supported by hf-sam2")
            model: Any = Sam2VideoModel.from_pretrained(directory, local_files_only=True)
            model = model.to(self.device).eval()
            backbone = model.config.vision_config.backbone_config
            expected = {
                "sam2_hiera_t": (96, 12),
                "sam2_hiera_s": (96, 16),
                "sam2_hiera_b": (112, 24),
                "sam2_hiera_l": (144, 48),
            }[self.model_config]
            if (backbone.hidden_size, sum(backbone.blocks_per_stage)) != expected:
                raise ValueError("snapshot model size does not match tracking.model_config")
            if bool(model.config.enable_temporal_pos_encoding_for_object_pointers) != (self.checkpoint_family == "2.1"):
                raise ValueError("snapshot model family does not match tracking.checkpoint_family")
            processor = Sam2VideoProcessor.from_pretrained(directory, local_files_only=True)
            self.predictor = _PredictorBridge(model, processor, self.device)
            names = sorted({*required, *(path.name for path in directory.glob("*.json"))})
            self.snapshot_hashes = {name: _sha256_file(str(directory / name)) for name in names}
            self.checkpoint_sha256 = self.snapshot_hashes["model.safetensors"]
            self.config_file = str(directory / "config.json")
            self.is_loaded = self.is_initialized = True
            return True
        except Exception as error:
            self.is_loaded = self.is_initialized = False
            logger.error("Cannot load Transformers SAM 2: %s", error)
            return False

    def run_tracking(self, progress_callback=None, should_stop=None):
        # Rebuild from prompts for every propagation, including after cancellation. Cached
        # non-conditioning outputs must not change the next run's inference history.
        if self.prompts and self.video_frames:
            self._reset_inference_state()
            for obj_id, prompts in self.prompts.items():
                for prompt in prompts:
                    box = None
                    points = labels = None
                    if prompt["type"] == "bbox":
                        box = np.array([prompt[k] for k in ("x1", "y1", "x2", "y2")], dtype=np.float32)
                    else:
                        points = np.array([[prompt["x"], prompt["y"]]], dtype=np.float32)
                        labels = np.array([prompt.get("label", 1)], dtype=np.int32)
                    self.predictor.add_new_points_or_box(
                        self.inference_state,
                        prompt["frame_idx"],
                        obj_id,
                        box=box,
                        points=points,
                        labels=labels,
                        clear_old_points=box is not None,
                    )
        return super().run_tracking(progress_callback, should_stop)

    def provenance(self) -> dict[str, Any]:
        import transformers

        result = super().provenance()
        result.update(
            backend="sam2-transformers",
            transformers_version=transformers.__version__,
            snapshot_files=dict(self.snapshot_hashes),
            preprocessing="reference-pil-bicubic",
            hole_filling=False,
        )
        return result
