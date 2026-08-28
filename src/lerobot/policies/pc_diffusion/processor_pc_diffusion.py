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

from .configuration_pc_diffusion import PCDiffusionConfig


def make_pc_diffusion_pre_post_processors(
    config: PCDiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[torch.Tensor, torch.Tensor],
]:
    """Build the pre/post processing pipelines for `pc_diffusion`.

    The default scaffold is sufficient here -- rename, add batch dim, move to device, normalize on
    the way in; unnormalize and move to CPU on the way out. Point clouds need no bespoke step:
    `AddBatchDimensionObservationStep` handles rank-2 clouds, and `NormalizerProcessorStep`
    broadcasts the per-axis `(C,)` stats across `(B, S, N, C)` correctly.
    """
    return make_default_pre_post_processors(config, dataset_stats)


__all__ = [
    "POLICY_POSTPROCESSOR_DEFAULT_NAME",
    "POLICY_PREPROCESSOR_DEFAULT_NAME",
    "make_pc_diffusion_pre_post_processors",
]
