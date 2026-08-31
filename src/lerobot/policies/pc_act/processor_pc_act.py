#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch

from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.factory import make_default_pre_post_processors
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_pc_act import PCACTConfig


def make_pc_act_pre_post_processors(
    config: PCACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[torch.Tensor, torch.Tensor],
]:
    """Default scaffold, same reasoning as pc_diffusion: rename, batch, device, normalize in;
    unnormalize out. Clouds are IDENTITY-normalised and rescaled inside the model."""
    return make_default_pre_post_processors(config, dataset_stats)


__all__ = [
    "POLICY_POSTPROCESSOR_DEFAULT_NAME",
    "POLICY_PREPROCESSOR_DEFAULT_NAME",
    "make_pc_act_pre_post_processors",
]
