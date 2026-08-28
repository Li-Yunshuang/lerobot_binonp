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

from .configuration_pcd_diffusion import PcdDiffusionConfig


def make_pcd_diffusion_pre_post_processors(
    config: PcdDiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[torch.Tensor, torch.Tensor],
]:
    """Build the pre/post pipelines, leaving the point clouds untouched.

    The clouds are normalised inside the model (centred on a shared frame, then scaled), so the
    processor must not touch them. Two things guarantee that:

    * `normalization_mapping["POINT_CLOUD"] = IDENTITY` -- `_apply_transform` returns the tensor
      unchanged for an IDENTITY mode.
    * the pose keys and cloud stats are dropped below, so even a mis-set mapping cannot normalise
      them: `_NormalizationMixin` also leaves a key alone when it is absent from `_tensor_stats`.

    Belt and braces on purpose: a per-element MIN_MAX over an (N, 3) cloud is silently wrong --
    it makes normalisation depend on point index order -- and nothing downstream would flag it.
    """
    if dataset_stats is not None:
        drop = {
            config.pointcloud_key,
            config.goal_pointcloud_key,
            config.object_poses_key,
            config.goal_object_poses_key,
        }
        dataset_stats = {k: v for k, v in dataset_stats.items() if k not in drop}

    return make_default_pre_post_processors(config, dataset_stats)


__all__ = ["make_pcd_diffusion_pre_post_processors"]
