"""Feed in-memory frames to the vendored SAM2 video predictor.

Upstream ``SAM2VideoPredictor.init_state`` only accepts an MP4 path (decoded with decord)
or a folder of JPEG files. The application already decodes the video with OpenCV, so the
predictor built here constructs the same inference state directly from those frames. There
is no second decode and no decord dependency, and the frame order the predictor sees is
exactly the chronological order the application works with.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

from sam2.sam2_video_predictor import SAM2VideoPredictor

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

# Matches the postprocessing overrides in sam2.build_sam.build_sam2_video_predictor.
_POSTPROCESSING_OVERRIDES = [
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    "++model.binarize_mask_from_pts_for_mem_enc=true",
    "++model.fill_hole_area=8",
]


def frames_to_tensor(frames: Sequence[np.ndarray], image_size: int, device) -> torch.Tensor:
    """Resize RGB uint8 frames to ``image_size`` square and normalize as SAM2 expects.

    Uses PIL's default resampling for RGB images (bicubic), the same path upstream takes for
    JPEG frame folders in ``sam2.utils.misc._load_img_as_tensor``.
    """
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
    images = torch.zeros(len(frames), 3, image_size, image_size, dtype=torch.float32)
    for index, frame in enumerate(frames):
        resized = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB").resize((image_size, image_size))
        images[index] = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
    images = images.to(device)
    images -= mean.to(device)
    images /= std.to(device)
    return images


class FramePredictor(SAM2VideoPredictor):
    """``SAM2VideoPredictor`` that can start from frames already held in memory."""

    @torch.inference_mode()
    def init_state_from_frames(
        self,
        frames: Sequence[np.ndarray],
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict:
        """Build an inference state from RGB uint8 frames in chronological order."""
        if len(frames) == 0:
            raise ValueError("at least one frame is required")
        video_height, video_width = frames[0].shape[:2]
        compute_device = self.device
        image_device = torch.device("cpu") if offload_video_to_cpu else compute_device
        images = frames_to_tensor(frames, self.image_size, image_device)
        return self._build_inference_state(
            images, video_height, video_width, offload_video_to_cpu, offload_state_to_cpu
        )

    def _build_inference_state(
        self, images, video_height, video_width, offload_video_to_cpu, offload_state_to_cpu
    ) -> dict:
        """Mirror of ``SAM2VideoPredictor.init_state`` after frame loading (vendored lines 58-99)."""
        compute_device = self.device
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        inference_state["storage_device"] = torch.device("cpu") if offload_state_to_cpu else compute_device
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state


def build_frame_predictor(config_file: str, ckpt_path: str, device: str, apply_postprocessing: bool = True) -> FramePredictor:
    """Instantiate a ``FramePredictor`` from a SAM2 Hydra config and checkpoint.

    Must be called inside an initialized Hydra context (the tracker opens one on the
    vendored config directory). Mirrors ``sam2.build_sam.build_sam2_video_predictor`` with
    the predictor class replaced.
    """
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from sam2.build_sam import _load_checkpoint

    overrides: List[str] = [f"++model._target_={FramePredictor.__module__}.{FramePredictor.__name__}"]
    if apply_postprocessing:
        overrides += _POSTPROCESSING_OVERRIDES
    cfg = compose(config_name=config_file, overrides=overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    return model
