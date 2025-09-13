# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

# Only initialize if not already initialized
if not GlobalHydra.instance().is_initialized():
    try:
        initialize_config_module("sam2_train", version_base="1.2")
    except Exception as e:
        # If initialization fails, continue anyway
        print(f"Warning: Hydra initialization in sam2_train failed: {e}")
        pass
